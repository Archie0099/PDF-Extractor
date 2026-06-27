"""Regression: OCREngine.warmup() must never raise.

warmup() is called from app.py's startup event. Its docstring promises that
"a failure here must not crash startup", but the model-construction call
(`_get_instance`, which downloads PaddleOCR weights on first run) used to sit
OUTSIDE the try/except — so a first-run network failure or a corrupt model
cache propagated out of warmup, out of the async startup handler, and aborted
the whole server boot (including the text-layer-only and Gemini paths that need
no PaddleOCR at all).
"""

from pipeline.ocr_engine import OCREngine


def test_warmup_swallows_construction_failure():
    eng = OCREngine()

    def boom(*a, **k):
        raise RuntimeError("simulated first-run weight download failure")

    eng._get_instance = boom  # type: ignore[assignment]
    # Must NOT raise — startup depends on this being best-effort.
    eng.warmup("en")


def test_warmup_swallows_predict_failure():
    eng = OCREngine()

    class FakeInstance:
        def ocr(self, *a, **k):
            raise RuntimeError("simulated predict failure")

    eng._get_instance = lambda *a, **k: FakeInstance()  # type: ignore[assignment]
    eng.warmup("en")  # must not raise
