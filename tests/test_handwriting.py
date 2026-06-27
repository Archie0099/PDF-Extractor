"""Tests for pipeline.handwriting engine load/error semantics.

Regression: a TRANSIENT model-load failure (network drop / partial HF download)
must NOT permanently poison the engine — the next request should retry. Only a
deterministic missing-deps failure should be cached. Also: a malformed detection
box must be skipped, not crash the whole page.
"""

import numpy as np

from pipeline.handwriting import HandwritingEngine


def test_transient_failure_is_not_cached(monkeypatch):
    eng = HandwritingEngine("en")
    calls = {"n": 0}

    def flaky_processor():
        calls["n"] += 1
        raise RuntimeError("simulated transient HF download drop")

    monkeypatch.setattr(eng, "_build_processor", flaky_processor)

    # First call fails...
    import pytest
    with pytest.raises(RuntimeError):
        eng._ensure()
    assert eng._loaded is False, "transient failure must NOT poison the engine"

    # ...and the SECOND call retries (does not short-circuit on a cached error).
    with pytest.raises(RuntimeError):
        eng._ensure()
    assert calls["n"] == 2, "engine should retry the load after a transient failure"


def test_missing_deps_failure_is_cached(monkeypatch):
    eng = HandwritingEngine("en")
    calls = {"n": 0}

    def missing_dep():
        calls["n"] += 1
        raise ImportError("No module named 'transformers'")

    monkeypatch.setattr(eng, "_build_processor", missing_dep)

    import pytest
    with pytest.raises(RuntimeError):
        eng._ensure()
    assert eng._loaded is True, "deterministic missing-deps failure should be cached"

    # Second call short-circuits on the cached error (no retry).
    with pytest.raises(RuntimeError):
        eng._ensure()
    assert calls["n"] == 1, "missing-deps failure must not be retried every call"


def test_ocr_text_skips_malformed_box(monkeypatch):
    eng = HandwritingEngine("en")
    monkeypatch.setattr(eng, "_ensure", lambda: None)
    monkeypatch.setattr(eng, "_recognize", lambda crop: "ok")

    img = np.full((100, 200, 3), 255, dtype=np.uint8)
    good = [[10, 10], [90, 10], [90, 30], [10, 30]]   # valid 4-point quad
    bad = [[10, 50], [90, 50], [90, 70]]              # malformed 3-point box
    # Must not raise; the good box is recognized, the bad one skipped.
    out = eng.ocr_text(img, [good, bad])
    assert "ok" in out
