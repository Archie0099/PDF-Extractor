"""Tests for pipeline.postprocess — the header/footer stripper DELETES text,
so its safety contract is critical.

Two data-safety properties this guards:
  * Data loss: bare multi-group numeric lines (ISO dates, currency amounts,
    numeric codes) sitting in the top/bottom band must NOT be collapsed to one
    digit-skeleton key and stripped from every page.
  * Confidence mapping: a surviving line must keep its OWN confidence, not a
    removed identical-text duplicate's.

Plus the core safety invariants (never empty a page, only strip true running
lines, keep numbered headings / labeled data rows, no-op on short docs).
"""

import pytest

from pipeline.postprocess import (
    strip_running_headers_footers,
    realign_line_confidences,
    _key,
)


# ----------------------------- data loss ------------------------------------
def test_key_does_not_collapse_distinct_dates():
    assert _key("2026-06-01") != _key("2026-06-02")
    assert _key("2026-06-01") != _key("2026-12-31")


def test_key_does_not_collapse_distinct_currency():
    assert _key("1,234.56") != _key("2,000.00")


def test_bare_date_header_is_not_stripped():
    pages = [
        "2026-06-01\nMorning shift produced 12 units\nNotes nominal",
        "2026-06-02\nMorning shift produced 15 units\nNotes minor delay",
        "2026-06-03\nMorning shift produced 9 units\nNotes machine reset",
        "2026-06-04\nMorning shift produced 20 units\nNotes overtime",
    ]
    out = strip_running_headers_footers(pages)
    for original, cleaned in zip(pages, out):
        date = original.split("\n")[0]
        assert date in cleaned, f"distinct date {date!r} was wrongly stripped"


def test_bare_currency_footer_is_not_stripped():
    pages = [
        "ACME LTD\nItem A\n1,234.56",
        "ACME LTD\nItem B\n2,000.00",
        "ACME LTD\nItem C\n3,500.99",
    ]
    out = strip_running_headers_footers(pages)
    for original, cleaned in zip(pages, out):
        amount = original.split("\n")[-1]
        assert amount in cleaned, f"distinct amount {amount!r} was wrongly stripped"
    # The genuinely-constant header "ACME LTD" SHOULD still be stripped.
    assert all("ACME LTD" not in c for c in out)


# ------------------- page numbers STILL collapse (no regression) ------------
@pytest.mark.parametrize("a,b", [
    ("Page 1", "Page 2"),
    ("Page 1 of 10", "Page 7 of 10"),
    ("12", "13"),
    ("- 1 -", "- 2 -"),
])
def test_real_page_numbers_still_share_a_key(a, b):
    assert _key(a) == _key(b)


def test_running_page_numbers_and_constant_banner_are_stripped():
    pages = [
        "JOURNAL OF THINGS\nUnique body of page one alpha\nPage 1",
        "JOURNAL OF THINGS\nUnique body of page two bravo\nPage 2",
        "JOURNAL OF THINGS\nUnique body of page three charlie\nPage 3",
        "JOURNAL OF THINGS\nUnique body of page four delta\nPage 4",
    ]
    out = strip_running_headers_footers(pages)
    joined = "\n".join(out)
    assert "JOURNAL OF THINGS" not in joined        # constant banner stripped
    assert "Page 1" not in joined and "Page 4" not in joined  # page numbers stripped
    assert "Unique body of page three charlie" in joined      # body kept


# ----------------------------- safety invariants ----------------------------
def test_never_empties_a_page():
    pages = ["SAME\nSAME\nSAME"] * 3
    out = strip_running_headers_footers(pages)
    assert all(c.strip() for c in out)  # no page reduced to nothing


def test_noop_under_min_pages():
    pages = ["HDR\nbody\nFTR", "HDR\nother\nFTR"]  # only 2 pages
    assert strip_running_headers_footers(pages) == pages


def test_numbered_heading_kept():
    # "Section 3" must never be collapsed with "Section 4" and stripped.
    assert _key("Section 3") != _key("Section 4")


# -------------------------- confidence mapping ------------------------------
def test_realign_index_mapping_for_duplicate_text_via_job_path():
    # A page whose top "DUP" header is stripped but whose body "DUP" survives.
    # Distinct per-page footers so the footer is NOT a running line.
    orig_lines = [
        {"text": "DUP", "confidence": 0.50},   # header duplicate (will be stripped)
        {"text": "DUP", "confidence": 0.99},   # body duplicate (survives)
        {"text": "foot one", "confidence": 0.70},
    ]
    pages = ["DUP\nDUP\nfoot one", "DUP\nDUP\nfoot two", "DUP\nDUP\nfoot three"]
    cleaned, kept = strip_running_headers_footers(pages, return_kept=True)
    assert cleaned[0] == "DUP\nfoot one"  # top header DUP gone, body DUP + footer kept

    # Index-threaded mapping (what jobs.py uses): exact, even for duplicate text.
    mapped = [orig_lines[i] for i in kept[0] if 0 <= i < len(orig_lines)]
    assert [m["text"] for m in mapped] == cleaned[0].split("\n")
    assert mapped[0]["confidence"] == 0.99  # survivor keeps its OWN confidence

    # Prove the bug this fixes: the old text-equality matcher mis-binds the
    # survivor to the REMOVED header's 0.50.
    text_matched = realign_line_confidences(cleaned[0], orig_lines)
    assert text_matched[0]["confidence"] == 0.50  # the defect we route around


def test_realign_text_matcher_still_safe_on_unique_lines():
    orig = [
        {"text": "HDR", "confidence": 0.9},
        {"text": "body", "confidence": 0.7},
        {"text": "FTR", "confidence": 0.95},
    ]
    out = realign_line_confidences("body", orig)
    assert out == [{"text": "body", "confidence": 0.7}]


def test_realign_empty_returns_none():
    assert realign_line_confidences("anything", []) is None


def test_return_kept_indices_align_with_text():
    pages = [
        "HDR\nunique one\nFTR",
        "HDR\nunique two\nFTR",
        "HDR\nunique three\nFTR",
    ]
    cleaned, kept = strip_running_headers_footers(pages, return_kept=True)
    for original, c, k in zip(pages, cleaned, kept):
        orig_lines = original.split("\n")
        assert [orig_lines[i] for i in k] == c.split("\n")
