"""Local handwriting OCR using HuggingFace VisionEncoderDecoder models (offline, free).

PaddleOCR's printed-text recogniser is weak on handwriting. This module pairs
PaddleOCR's *detector* (to find text-line boxes) with a TrOCR-family handwriting
*recogniser* (a local HuggingFace transformers model) to read each line.

The recogniser is selected by language + an optional 'large' quality flag:

  en  (default) -> microsoft/trocr-base-handwritten     (English, fast-ish)
  en  + large   -> microsoft/trocr-large-handwritten    (English, slower, more accurate)
  hi  (Hindi)   -> sabaridsnfuji/Hindi_Offline_Handwritten_OCR (Hindi, EXPERIMENTAL)
  ne  (Nepali / generic Devanagari) -> aayushpuri01/TrOCR-Devanagari

It is OPTIONAL and heavy: it needs ``transformers`` + ``torch`` installed and a
one-time model download (hundreds of MB). It is slow on CPU and opt-in per
document. If the deps/model are missing it raises a clear, actionable error
instead of crashing, and unknown languages fall back to the English model.

Backward compatibility: ``lang='en'`` + ``large=False`` (the defaults, and the
module-level ``hw_engine`` singleton) behave byte-for-byte like the original
single-model implementation.
"""

import os
import threading

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Model registry. Each entry describes one recogniser.
#
#  id        : HuggingFace repo for the VisionEncoderDecoderModel weights.
#  processor : how to build the TrOCRProcessor:
#                "bundled"  -> TrOCRProcessor.from_pretrained(id)  (id ships the
#                              preprocessor_config + tokenizer)
#                ("manual", feat_repo, tok_repo) -> build from a ViT feature
#                              extractor repo + a decoder tokenizer repo, because
#                              the weights repo ships ONLY config + safetensors.
#  max_new_tokens : decode cap (the Hindi model is trained for <=64 chars/crop).
#  experimental   : True for models whose quality on this task is unvalidated.
# ----------------------------------------------------------------------------
_MODELS = {
    "en": {
        "id": "microsoft/trocr-base-handwritten",
        "processor": "bundled",
        "max_new_tokens": 96,
        "experimental": False,
    },
    "en_large": {
        "id": "microsoft/trocr-large-handwritten",
        "processor": "bundled",
        "max_new_tokens": 96,
        "experimental": False,
    },
    "hi": {
        # Only genuinely Hindi-trained option found. Repo ships weights only, so
        # the processor must be assembled from the encoder + decoder base repos.
        # Quality is modest and UNVALIDATED here — treat as experimental.
        "id": "sabaridsnfuji/Hindi_Offline_Handwritten_OCR",
        "processor": ("manual",
                      "google/vit-base-patch16-224-in21k",
                      "surajp/RoBERTa-hindi-guj-san"),
        "max_new_tokens": 64,  # model is trained for <=64 chars per crop
        "experimental": True,
    },
    "ne": {
        # Nepali Devanagari; bundled processor, clean drop-in. Usable as a
        # generic Devanagari fallback (reads many Hindi glyphs, Nepali-biased).
        "id": "aayushpuri01/TrOCR-Devanagari",
        "processor": "bundled",
        "max_new_tokens": 64,
        "experimental": True,
    },
}

# Default fallback when a requested language has no handwriting model.
_DEFAULT_KEY = "en"

# Backwards-compat constant some callers/tests may import.
MODEL_NAME = _MODELS[_DEFAULT_KEY]["id"]


def _resolve_key(lang: str, large: bool) -> str:
    """Map a (lang, large) request to a registry key, falling back to English.

    Devanagari-family languages share script; 'hi'/'ne'/'devanagari' all route
    to a Devanagari model. 'large' currently only applies to English (no large
    Devanagari model exists), so it is ignored for non-English.
    """
    lang = (lang or "en").strip().lower()
    if lang in ("en", "english"):
        return "en_large" if large else "en"
    if lang in ("hi", "hin", "hindi", "mr", "marathi", "sa"):
        return "hi"
    if lang in ("ne", "nep", "nepali", "devanagari"):
        return "ne"
    return _DEFAULT_KEY


def is_available() -> bool:
    """True if the optional handwriting dependencies are importable."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def _dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _crop_quad(img: np.ndarray, box) -> np.ndarray:
    """Perspective-warp a 4-point text box to an upright rectangle crop."""
    pts = np.array(box, dtype="float32")
    w = int(max(_dist(pts[0], pts[1]), _dist(pts[2], pts[3])))
    h = int(max(_dist(pts[1], pts[2]), _dist(pts[3], pts[0])))
    w = max(w, 1)
    h = max(h, 1)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, matrix, (w, h))


class HandwritingEngine:
    """Lazy-loaded TrOCR-family wrapper for ONE registry entry.

    Thread-safe; one model instance reused. Construct via the module-level
    ``get_engine(lang, large)`` factory so identical models are shared.
    """

    def __init__(self, key: str = _DEFAULT_KEY) -> None:
        self._key = key if key in _MODELS else _DEFAULT_KEY
        self._cfg = _MODELS[self._key]
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._loaded = False
        self._error = None

    @property
    def model_id(self) -> str:
        return self._cfg["id"]

    @property
    def experimental(self) -> bool:
        return bool(self._cfg.get("experimental"))

    def _build_processor(self):
        from transformers import TrOCRProcessor

        proc = self._cfg["processor"]
        if proc == "bundled":
            return TrOCRProcessor.from_pretrained(self._cfg["id"])
        # ("manual", feature_extractor_repo, tokenizer_repo)
        _, feat_repo, tok_repo = proc
        from transformers import AutoTokenizer

        # AutoImageProcessor is the modern name; in transformers 4.40.2
        # AutoFeatureExtractor still resolves it fine as a fallback.
        try:
            from transformers import AutoImageProcessor as _FeatLoader
        except Exception:  # very old fallback
            from transformers import AutoFeatureExtractor as _FeatLoader

        feature_extractor = _FeatLoader.from_pretrained(feat_repo)
        tokenizer = AutoTokenizer.from_pretrained(tok_repo)
        # TrOCRProcessor accepts an image processor as `image_processor=`; older
        # transformers used `feature_extractor=`. 4.40.2 accepts both kwargs.
        try:
            return TrOCRProcessor(image_processor=feature_extractor,
                                  tokenizer=tokenizer)
        except TypeError:
            return TrOCRProcessor(feature_extractor=feature_extractor,
                                  tokenizer=tokenizer)

    def _ensure(self) -> None:
        if self._loaded:
            if self._error:
                raise RuntimeError(self._error)
            return
        with self._lock:
            if self._loaded:
                if self._error:
                    raise RuntimeError(self._error)
                return
            try:
                import torch
                from transformers import VisionEncoderDecoderModel

                torch.set_num_threads(max(1, os.cpu_count() or 1))
                self._processor = self._build_processor()
                self._model = VisionEncoderDecoderModel.from_pretrained(
                    self._cfg["id"]
                )
                self._model.eval()
                self._loaded = True
            except (ImportError, ModuleNotFoundError) as exc:
                # Missing deps are DETERMINISTIC — they won't fix themselves at
                # runtime, so cache the failure permanently (fail fast).
                self._error = (
                    "Handwriting model '{}' unavailable ({}). Enable it with: "
                    "pip install transformers torch  (first use downloads the "
                    "model, a few hundred MB; non-English models also fetch a "
                    "tokenizer/feature-extractor repo).".format(
                        self._cfg["id"], exc
                    )
                )
                self._loaded = True
                raise RuntimeError(self._error)
            except Exception as exc:
                # TRANSIENT failure (network drop / partial HF download / HF
                # outage / disk full). Do NOT poison the engine: leave _loaded
                # False so the next request retries from_pretrained (which can
                # resume the partial download). Otherwise the engine would be
                # dead for the whole process until a server restart.
                raise RuntimeError(
                    "Handwriting model '{}' could not be loaded ({}). This is "
                    "often a transient network/download issue — try again; a "
                    "later request will retry the download.".format(
                        self._cfg["id"], exc
                    )
                ) from exc

    def _recognize(self, crop_bgr: np.ndarray) -> str:
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pixel_values = self._processor(
            images=pil, return_tensors="pt"
        ).pixel_values
        with torch.no_grad():
            generated = self._model.generate(
                pixel_values, max_new_tokens=self._cfg["max_new_tokens"]
            )
        return self._processor.batch_decode(
            generated, skip_special_tokens=True
        )[0]

    def ocr_text(self, img_bgr: np.ndarray, boxes: list) -> str:
        """Recognize handwriting in ``img_bgr`` given detected line ``boxes``.

        Boxes are ordered into reading order (top-to-bottom, left-to-right),
        each cropped and passed through the recogniser, joined by newlines.
        """
        self._ensure()
        if not boxes:
            return ""

        items = []
        for box in boxes:
            # Skip any malformed box (not a 4-point quad) so a single bad
            # detection can't abort the whole page's handwriting OCR.
            if not box or len(box) != 4:
                continue
            try:
                ys = [p[1] for p in box]
                xs = [p[0] for p in box]
            except (TypeError, IndexError):
                continue
            items.append({
                "box": box,
                "cy": sum(ys) / len(ys),
                "cx": sum(xs) / len(xs),
                "h": max(ys) - min(ys),
            })
        if not items:
            return ""
        heights = [it["h"] for it in items if it["h"] > 0]
        tol = max((float(np.median(heights)) if heights else 12.0) * 0.6, 8.0)
        items.sort(key=lambda it: (it["cy"], it["cx"]))

        rows = [[items[0]]]
        cy = items[0]["cy"]
        for it in items[1:]:
            if abs(it["cy"] - cy) <= tol:
                rows[-1].append(it)
            else:
                rows.append([it])
                cy = it["cy"]

        lines = []
        for row in rows:
            for it in sorted(row, key=lambda x: x["cx"]):
                try:
                    crop = _crop_quad(img_bgr, it["box"])
                    if crop.size == 0:
                        continue
                    text = self._recognize(crop).strip()
                except Exception:
                    text = ""
                if text:
                    lines.append(text)
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# Factory: one cached engine per registry key (so 'en', 'en_large' and a
# Devanagari model can coexist without reloading). Thread-safe.
# ----------------------------------------------------------------------------
_engines: dict = {}
_engines_lock = threading.Lock()


def get_engine(lang: str = "en", large: bool = False) -> "HandwritingEngine":
    """Return a cached handwriting engine for the requested (lang, large)."""
    key = _resolve_key(lang, large)
    eng = _engines.get(key)
    if eng is not None:
        return eng
    with _engines_lock:
        eng = _engines.get(key)
        if eng is None:
            eng = HandwritingEngine(key)
            _engines[key] = eng
        return eng


# Backwards-compatible module-level singleton (English base), so existing
# `from pipeline.handwriting import hw_engine` callers keep working unchanged.
hw_engine = get_engine("en", large=False)
