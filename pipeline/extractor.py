"""Per-page extraction: text-layer when present, else render + OCR.

Implements the project contract exactly:
  DPI_MAX, DPI_FAST
  render_page_image(page, dpi) -> BGR uint8 ndarray
  extract_page(page, *, mode, lang, preprocess, min_chars=50) -> dict
"""

import numpy as np

from pipeline.preprocess import preprocess_image
from pipeline.ocr_engine import engine
from pipeline.textlayer import has_text_layer, extract_text_layer

# fitz is PyMuPDF; imported lazily-safe at module import.
import fitz  # type: ignore


DPI_MAX = 300
DPI_FAST = 150


def render_page_image(page, dpi: int) -> np.ndarray:
    """Render a fitz.Page to an HxWx3 BGR uint8 ndarray at the given DPI.

    Uses a pixmap with zoom = dpi/72. Handles RGB, grayscale and alpha
    channels, always returning a 3-channel BGR image (OpenCV/PaddleOCR
    convention).
    """
    zoom = float(dpi) / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    # alpha=False keeps things simple, but some pages/pixmaps can still carry
    # channel counts other than 3; normalise defensively below.
    pix = page.get_pixmap(matrix=matrix, alpha=False)

    # Build an ndarray view from the pixmap samples.
    buf = np.frombuffer(pix.samples, dtype=np.uint8)
    n = pix.n  # bytes per pixel (channels)
    arr = buf.reshape(pix.height, pix.width, n)

    if n == 1:
        # Grayscale -> BGR
        bgr = np.repeat(arr, 3, axis=2)
    elif n == 3:
        # PyMuPDF emits RGB; convert to BGR for OpenCV/PaddleOCR.
        bgr = arr[:, :, ::-1]
    elif n == 4:
        # RGBA -> drop alpha, RGB -> BGR.
        rgb = arr[:, :, :3]
        bgr = rgb[:, :, ::-1]
    else:
        # CMYK or other exotic layouts: re-render forcing an RGB colorspace.
        pix = page.get_pixmap(matrix=matrix, alpha=False, colorspace=fitz.csRGB)
        buf = np.frombuffer(pix.samples, dtype=np.uint8)
        arr = buf.reshape(pix.height, pix.width, pix.n)
        bgr = arr[:, :, :3][:, :, ::-1]

    # Ensure a contiguous, owned uint8 array (the buffer view is read-only).
    return np.ascontiguousarray(bgr, dtype=np.uint8)


def extract_page(
    page,
    *,
    mode: str,
    lang: str,
    preprocess: bool,
    binarize: bool = False,
    handwriting: bool = False,
    online: bool = False,
    online_key: str = "",
    online_model: str = "",
    force_ocr: bool = False,
    min_chars: int = 50,
) -> dict:
    """Extract text from a single page.

    If the page has a usable text layer, return it directly (exact, so no
    confidence is reported). Otherwise render the page to an image, optionally
    preprocess (deskew + CLAHE + denoise), optionally binarize, and OCR it.

    ``preprocess`` and ``binarize`` are independent: ``preprocess`` enables the
    grayscale clean-up pipeline; ``binarize`` additionally Sauvola-thresholds
    the image (only meaningful when ``preprocess`` is on).

    Routing precedence for a page that needs reading: text-layer (unless
    ``force_ocr``) -> online Gemini (if ``online`` + key) -> local handwriting
    (if ``handwriting``) -> local OCR.

    Returns a uniform dict from EVERY branch:
    ``{"source": "text"|"ocr"|"online"|"handwriting", "text": str,
    "confidence": float|None, "lines": list|None}``. ``confidence`` and
    ``lines`` (per-line ``[{text, confidence}]``) are populated only for local
    ``"ocr"``; the other sources set both to ``None`` so consumers never have to
    guard for a missing key.
    """
    # "Force OCR" skips the text-layer shortcut so even born-digital pages are
    # rendered + OCR'd (or sent online). Use it when the embedded text is
    # garbled/wrong, or when text is trapped inside images on a text page.
    if not force_ocr and has_text_layer(page, min_chars=min_chars):
        return {
            "source": "text",
            "text": extract_text_layer(page),
            "confidence": None,
            "lines": None,
        }

    dpi = DPI_MAX if mode == "max" else DPI_FAST
    img = render_page_image(page, dpi)

    # Online vision path (opt-in): send the RAW page image to Google Gemini and
    # let it transcribe (handwriting + printed). Takes precedence over the local
    # OCR/handwriting engines for any page that needs OCR. The page bytes leave
    # this machine — gated on an explicit api key, never the default. Errors
    # propagate to the per-page handler with an actionable message.
    if online:
        from pipeline import online_ocr
        if online_ocr.is_configured(online_key):
            text = online_ocr.transcribe_image_bgr(
                img, api_key=online_key, model=(online_model or None)
            )
            return {"source": "online", "text": text, "confidence": None, "lines": None}

    # Handwriting path: detect lines with PaddleOCR, recognise with TrOCR.
    # Crops are taken from the natural render (no binarization, which TrOCR
    # dislikes). Slow on CPU; opt-in per document.
    if handwriting:
        from pipeline.handwriting import get_engine
        # Line DETECTION stays on the script-agnostic "en" DB detector (it finds
        # text regions regardless of script, and avoids pulling an extra
        # PaddleOCR language model). Only the RECOGNISER is language-aware.
        boxes = engine.detect_boxes(img, lang="en")
        hw = get_engine(lang=lang)
        text = hw.ocr_text(img, boxes)
        return {"source": "handwriting", "text": text, "confidence": None, "lines": None}

    if preprocess:
        img = preprocess_image(img, mode=mode, binarize=binarize)

    use_angle_cls = (mode == "max")
    lines = engine.ocr_lines(img, lang=lang, use_angle_cls=use_angle_cls)
    text = "\n".join(ln["text"] for ln in lines)
    conf = (sum(ln["confidence"] for ln in lines) / len(lines)) if lines else None
    # Per-line confidence so the UI can flag exactly which lines to double-check.
    # Order matches the joined text (one entry per "\n"-separated line).
    line_conf = [
        {"text": ln["text"], "confidence": round(float(ln["confidence"]), 4)}
        for ln in lines
    ]
    return {"source": "ocr", "text": text, "confidence": conf, "lines": line_conf}
