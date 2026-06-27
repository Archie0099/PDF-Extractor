"""Tests for pipeline.analyze.

Regression: the cross-config 'agree' signal was mathematically inert (it reduced
to |toks[n]|/|union|, which REWARDS a config that hallucinates many unique
tokens). It must instead measure genuine corroboration by OTHER configs. Plus a
ZeroDivision guard for a direct max_pages<=0 call.
"""

import io

import fitz

from pipeline.analyze import _score_page, suggest_settings


def test_agree_rewards_corroboration_not_unique_tokens():
    # Hold conf / coverage / mass EQUAL across all configs so ONLY the agree
    # signal differs. 'a' and 'b' corroborate each other; 'c' has unique tokens
    # nobody else confirms. With the fixed metric 'c' must score strictly lower.
    # (Under the OLD inert metric, agree reduced to |toks|/|union| = 0.5 for all
    # three, so 'c' would TIE 'a'/'b' — this test fails on the old code.)
    raw = {
        "a": {"mean_conf": 0.9, "n_conf": 2, "mass": 10, "conf_text": "alpha bravo"},
        "b": {"mean_conf": 0.9, "n_conf": 2, "mass": 10, "conf_text": "alpha bravo"},
        "c": {"mean_conf": 0.9, "n_conf": 2, "mass": 10, "conf_text": "gamma delta"},
    }
    scores = _score_page(raw)
    assert scores["a"] > scores["c"]
    assert scores["b"] > scores["c"]
    assert abs(scores["a"] - scores["b"]) < 1e-9


def test_agree_component_is_zero_for_fully_unique_config():
    # Two configs with completely disjoint confident tokens -> neither is
    # corroborated -> agree term contributes nothing (no self-credit).
    raw = {
        "a": {"mean_conf": 0.9, "n_conf": 1, "mass": 4, "conf_text": "apple"},
        "b": {"mean_conf": 0.9, "n_conf": 1, "mass": 4, "conf_text": "orange"},
    }
    scores = _score_page(raw)
    # Both equal on conf/coverage/mass and both have agree 0 -> equal composite.
    assert abs(scores["a"] - scores["b"]) < 1e-9


def test_score_page_all_zero_no_division_error():
    raw = {
        "a": {"mean_conf": None, "n_conf": 0, "mass": 0, "conf_text": ""},
        "b": {"mean_conf": None, "n_conf": 0, "mass": 0, "conf_text": ""},
    }
    out = _score_page(raw)
    assert out == {"a": 0.0, "b": 0.0}


def _image_only_pdf():
    """A 1-page PDF whose page has NO text layer (forces the OCR-eligible path)."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (600, 200), "white")
    ImageDraw.Draw(img).text((20, 80), "scanned only", fill="black")
    buf = io.BytesIO(); img.save(buf, format="PNG")
    doc = fitz.open()
    page = doc.new_page(width=600, height=200)
    page.insert_image(fitz.Rect(0, 0, 600, 200), stream=buf.getvalue())
    data = doc.tobytes(); doc.close()
    return data


def test_suggest_settings_max_pages_zero_returns_fallback_not_crash():
    # max_pages=0 on an OCR-eligible PDF used to hit ZeroDivisionError (swallowed
    # into a generic fallback). Now it returns the fallback WITHOUT running OCR.
    result = suggest_settings(_image_only_pdf(), lang="en", max_pages=0)
    assert isinstance(result, dict)
    assert result["ok"] is True
    assert result["decision"] == "inconclusive"
