"""PaddleOCR engine wrapper.

Caches one PaddleOCR instance per ``(lang, use_angle_cls)`` key and reuses it.
All OCR predict calls are serialized with a threading.Lock because PaddleOCR's
predictors are not safe to call concurrently from multiple threads.

Targets the PaddleOCR 2.7.3 API:
    PaddleOCR(use_angle_cls=<bool>, lang=<code>, show_log=False)
    result = instance.ocr(img_bgr, cls=use_angle_cls)

Result structure (PaddleOCR 2.7.x):
    [
        [
            [box, (text, conf)],
            ...
        ]
    ]
where ``result[0]`` may be ``None`` when no text is detected.
"""

import threading

import numpy as np


class OCREngine:
    """Singleton-style PaddleOCR wrapper with per-(lang, angle) caching."""

    def __init__(self) -> None:
        self._instances = {}
        self._instances_lock = threading.Lock()
        # Serializes all .ocr() predict calls across threads.
        self._predict_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Instance management
    # ------------------------------------------------------------------
    def _get_instance(self, lang: str, use_angle_cls: bool):
        """Return a cached PaddleOCR instance, creating it on first use."""
        key = (lang, bool(use_angle_cls))
        # Fast path: already created.
        instance = self._instances.get(key)
        if instance is not None:
            return instance

        with self._instances_lock:
            # Re-check inside the lock to avoid double construction.
            instance = self._instances.get(key)
            if instance is None:
                # Imported lazily so importing this module is cheap and does
                # not trigger PaddleOCR/Paddle initialization at import time.
                from paddleocr import PaddleOCR

                # Accuracy-oriented inference tuning (all valid PaddleOCR
                # 2.7.3 args, no extra model downloads):
                #  - det_limit_side_len 1536 (vs default 960): lets the
                #    detector use our high-DPI render instead of shrinking it,
                #    so small/dense text is found.
                #  - det_db_unclip_ratio 1.8 (vs 1.5): expands detected boxes
                #    so characters at box edges aren't clipped before recog.
                #  - det_db_box_thresh 0.5 (vs 0.6): recovers fainter text.
                #  - use_dilation: connects broken strokes in noisy scans.
                instance = PaddleOCR(
                    use_angle_cls=bool(use_angle_cls),
                    lang=lang,
                    show_log=False,
                    det_limit_side_len=1536,
                    det_limit_type="max",
                    det_db_unclip_ratio=1.8,
                    det_db_box_thresh=0.5,
                    use_dilation=True,
                )
                self._instances[key] = instance
            return instance

    def detect_boxes(self, img_bgr: np.ndarray, *, lang: str = "en") -> list:
        """Return text-line boxes (4-point polygons) for the image.

        Used by the handwriting pipeline, which detects lines with PaddleOCR but
        recognizes them with a handwriting model. We reuse the normal OCR call
        and keep only its boxes — PaddleOCR 2.7.3's detection-only (``rec=False``)
        path has a numpy truth-value bug, so we avoid it.
        """
        lines = self.ocr_lines(img_bgr, lang=lang, use_angle_cls=False)
        return [line["box"] for line in lines]

    def warmup(self, lang: str = "en") -> None:
        """Preload a PaddleOCR instance (downloads weights on first run).

        Called at application startup so the first real request does not pay
        the model-load / weight-download cost. Runs a tiny dummy inference to
        force lazy predictor initialization.
        """
        try:
            # Construction (which on first run downloads model weights from the
            # network) is the failure-prone step, so it MUST be inside the guard:
            # a first-run download failure / corrupt model cache must not abort
            # server startup. Lazy construction then retries on the first real
            # OCR request, where jobs.py turns any failure into a per-page error.
            instance = self._get_instance(lang, use_angle_cls=False)
            dummy = np.full((32, 32, 3), 255, dtype=np.uint8)
            with self._predict_lock:
                instance.ocr(dummy, cls=False)
        except Exception:
            # Warmup is best-effort; a failure here must not crash startup.
            pass

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------
    def _run(self, img_bgr: np.ndarray, lang: str, use_angle_cls: bool):
        """Run PaddleOCR under the predict lock and return the raw lines list.

        Returns the list of ``[box, (text, conf)]`` entries (possibly empty).
        """
        instance = self._get_instance(lang, use_angle_cls)
        with self._predict_lock:
            result = instance.ocr(img_bgr, cls=bool(use_angle_cls))

        if not result:
            return []
        page = result[0]
        if page is None:
            return []
        return page

    @staticmethod
    def _line_y(box) -> float:
        """Top-y coordinate of a detection box (4 [x, y] points)."""
        return min(float(pt[1]) for pt in box)

    @staticmethod
    def _line_x(box) -> float:
        """Left-x coordinate of a detection box (4 [x, y] points)."""
        return min(float(pt[0]) for pt in box)

    @staticmethod
    def _line_height(box) -> float:
        ys = [float(pt[1]) for pt in box]
        return max(ys) - min(ys)

    def ocr_lines(
        self, img_bgr: np.ndarray, *, lang: str, use_angle_cls: bool
    ) -> list:
        """Return detected text lines in reading order.

        Each element is ``{"text": str, "confidence": float, "box": list}``.
        Lines are grouped into rows by their vertical position, and within a
        row sorted left-to-right, approximating natural reading order.

        NOTE: this assumes a SINGLE-column layout. On a multi-column scan,
        side-by-side lines at the same vertical position are merged into one row
        and emitted left-to-right, so the two columns come out interleaved. Real
        column reconstruction (XY-cut / gutter detection) is not implemented yet.
        Born-digital multi-column PDFs take the text-layer path (textlayer.py,
        same single-column caveat) or the Gemini path.
        """
        raw = self._run(img_bgr, lang, use_angle_cls)

        items = []
        for entry in raw:
            try:
                box, payload = entry[0], entry[1]
                text, conf = payload[0], payload[1]
                if text is None or not box:
                    continue
                # Compute geometry inside the guard so a single malformed
                # detection box is skipped rather than aborting the whole page.
                y = self._line_y(box)
                x = self._line_x(box)
                h = self._line_height(box)
            except (TypeError, IndexError, ValueError):
                continue
            items.append(
                {
                    "text": str(text),
                    "confidence": float(conf),
                    "box": box,
                    "_y": y,
                    "_x": x,
                    "_h": h,
                }
            )

        if not items:
            return []

        # Group items into rows: two items share a row if their top-y values
        # are within a tolerance derived from the median glyph height.
        heights = [it["_h"] for it in items if it["_h"] > 0]
        median_h = float(np.median(heights)) if heights else 12.0
        tol = max(median_h * 0.6, 6.0)

        # Sort primarily by y so we can sweep rows top-to-bottom.
        items.sort(key=lambda it: (it["_y"], it["_x"]))

        rows = []
        current = [items[0]]
        current_y = items[0]["_y"]
        for it in items[1:]:
            if abs(it["_y"] - current_y) <= tol:
                current.append(it)
            else:
                rows.append(current)
                current = [it]
                current_y = it["_y"]
        rows.append(current)

        ordered = []
        for row in rows:
            row.sort(key=lambda it: it["_x"])
            for it in row:
                ordered.append(
                    {
                        "text": it["text"],
                        "confidence": it["confidence"],
                        "box": it["box"],
                    }
                )
        return ordered

    def ocr_text(
        self, img_bgr: np.ndarray, *, lang: str, use_angle_cls: bool
    ) -> str:
        """Return all detected text joined in reading order by newlines."""
        text, _ = self.ocr_text_conf(
            img_bgr, lang=lang, use_angle_cls=use_angle_cls
        )
        return text

    def ocr_text_conf(
        self, img_bgr: np.ndarray, *, lang: str, use_angle_cls: bool
    ):
        """Return ``(text, mean_confidence)`` for the image.

        ``mean_confidence`` is the average per-line recognition confidence in
        ``[0, 1]`` (or ``None`` if nothing was detected). It lets the UI flag
        pages the OCR engine itself is unsure about.
        """
        lines = self.ocr_lines(
            img_bgr, lang=lang, use_angle_cls=use_angle_cls
        )
        text = "\n".join(line["text"] for line in lines)
        if not lines:
            return text, None
        conf = sum(line["confidence"] for line in lines) / len(lines)
        return text, float(conf)


# Module-level singleton.
engine = OCREngine()
