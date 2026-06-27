"""Tests for pipeline.online_ocr — the optional Gemini client.

These run fully offline (no network): they exercise key sanitization, response
parsing, and the documented "only ever raise a single-line RuntimeError, never
echo the key" contract.
"""

import numpy as np
import pytest

from pipeline import online_ocr


# ----------------------------- key sanitization -----------------------------
def test_clean_key_accepts_normal_key():
    assert online_ocr._clean_key("  AIzaSyABC123def  ") == "AIzaSyABC123def"
    # The 'AQ.'-style keys are also valid ASCII.
    assert online_ocr._clean_key("AQ.Ab8-xyz_123") == "AQ.Ab8-xyz_123"


@pytest.mark.parametrize("bad", [
    "AIzaSy\nDEF",        # interior newline (key pasted wrapped across 2 lines)
    "AIzaSy\rDEF",        # carriage return
    "AIzaSy\x00DEF",      # NUL
    "AIzaSycafé",         # non-ASCII
])
def test_clean_key_rejects_malformed_without_echoing_key(bad):
    with pytest.raises(RuntimeError) as ei:
        online_ocr._clean_key(bad)
    msg = str(ei.value)
    # Must NOT contain the key value (the secret part), and must be actionable.
    assert "DEF" not in msg and "café" not in msg
    assert "aistudio.google.com" in msg


def test_transcribe_with_malformed_key_raises_runtimeerror_not_valueerror():
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    # A CR/LF key used to make http.client raise a raw ValueError whose message
    # INCLUDED the full key -> echoed into per-page error text. Now: clean
    # RuntimeError, no network call, no key in the message.
    with pytest.raises(RuntimeError) as ei:
        online_ocr.transcribe_image_bgr(img, api_key="SECRETKEYPART\nx")
    assert "SECRETKEYPART" not in str(ei.value)


# ----------------------------- response parsing -----------------------------
def test_extract_text_handles_null_text_part():
    # An explicit {"text": null} part must not raise TypeError on the join.
    payload = {"candidates": [{"finishReason": "STOP", "content": {"parts": [
        {"text": None}, {"text": "real transcription"},
    ]}}]}
    assert online_ocr._extract_text(payload) == "real transcription"


def test_extract_text_concatenates_parts():
    payload = {"candidates": [{"finishReason": "STOP", "content": {"parts": [
        {"text": "a"}, {"text": "b"}, {"text": "c"},
    ]}}]}
    assert online_ocr._extract_text(payload) == "abc"


def test_extract_text_blocked_raises_runtimeerror():
    with pytest.raises(RuntimeError):
        online_ocr._extract_text({"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []})


def test_resolve_model_strips_prefix():
    assert online_ocr._resolve_model("models/gemini-2.5-flash") == "gemini-2.5-flash"
    assert online_ocr._resolve_model(None) == online_ocr.DEFAULT_MODEL
