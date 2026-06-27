"""Cross-page cleanup: strip running headers / footers and page numbers.

Operates on the FULL list of per-page texts after extraction. A line is treated
as a running header/footer only if a normalized version of it recurs in the top
OR bottom band of a strong majority of pages — and it is removed ONLY from that
band, never from a page body.

Safety properties (each guards a real failure mode found in review):
  * The top and bottom bands are DISJOINT and capped at half the page's
    non-blank lines, so on short/sparse pages a body line can never be counted
    as both header and footer (which previously emptied invoice/form pages).
  * Digit normalization (so "Page 1"/"Page 2"/bare "12" match) is applied ONLY
    to page-number-like lines; every other line is compared verbatim, so a
    numbered heading ("Section 3 …") or a numeric data row ("Total: 1,234.56")
    is never collapsed and stripped.
  * A line must recur on a strong majority of pages (default 70%), so a real
    heading that merely repeats on a couple of pages of a short doc is kept.
  * A page is never reduced to nothing: if every candidate line would be
    stripped, the page is left untouched.
  * No-ops entirely on documents with fewer than ``min_pages`` pages.
"""

import math
import re
from collections import Counter

_DIGITS = re.compile(r"\d+")
_WS = re.compile(r"\s+")
# Words that commonly accompany a page number ("Page 1 of 10", "pg. 3").
_PAGEWORDS = re.compile(r"\b(?:page|pg|p|of|no)\b")


def _key(line: str) -> str:
    """Normalized match key.

    Page-number-like lines (only page-words + digits + punctuation remain) get
    their digits mapped to ``#`` so "Page 1"/"Page 2"/bare "12" all match. Every
    other line is compared verbatim — so numbered headings and numeric data rows
    are NOT collapsed together and erased.

    The digit→``#`` collapse is gated to lines that are *plausibly a page
    number*: at most ONE digit group, OR an explicit page-word ("Page", "of",
    …). A bare MULTI-group value — an ISO date ``2026-06-01`` (``#-#-#``), a
    currency amount ``1,234.56`` (``#,#.#``), a numeric code — is genuinely
    distinct per page; collapsing its skeleton would let the recurrence test
    delete real data from every page (e.g. a per-page date header or an invoice
    amount footer). Those are compared verbatim so they are never stripped.
    """
    base = _WS.sub(" ", line.strip().lower()).strip()
    probe = re.sub(r"[^\w\s]", "", _PAGEWORDS.sub(" ", base))
    if _DIGITS.sub("", probe).strip() == "":  # nothing but page-words + digits
        if len(_DIGITS.findall(base)) <= 1 or _PAGEWORDS.search(base):
            return _WS.sub(" ", _DIGITS.sub("#", base)).strip()
    return base


def realign_line_confidences(cleaned_text, orig_lines):
    """Map a stripped page's surviving lines back to their confidences.

    Stripping only REMOVES whole lines (it never edits or reorders the survivors),
    so a two-pointer walk over the original per-line list recovers each surviving
    line's confidence. Returns a new ``[{text, confidence}]`` list aligned to
    ``cleaned_text`` (lines that can't be matched get ``confidence=None``), or
    ``None`` if there were no original lines.
    """
    if not orig_lines:
        return None
    cleaned = (cleaned_text or "").split("\n")
    out, oi = [], 0
    for cl in cleaned:
        while oi < len(orig_lines) and (orig_lines[oi].get("text", "") != cl):
            oi += 1
        if oi < len(orig_lines):
            out.append(orig_lines[oi])
            oi += 1
        else:
            out.append({"text": cl, "confidence": None})
    return out


def _bands(nonblank, band):
    """Disjoint (top, bottom) index lists, capped at half the non-blank lines.

    Capping at ``len // 2`` guarantees the two bands never share an index, so
    the page interior is never a header/footer candidate.
    """
    b = min(band, len(nonblank) // 2)
    if b <= 0:
        return [], []
    return nonblank[:b], nonblank[len(nonblank) - b:]


def strip_running_headers_footers(
    page_texts,
    *,
    band: int = 3,
    min_frac: float = 0.7,
    min_pages: int = 3,
    return_kept: bool = False,
):
    """Return cleaned copies of ``page_texts`` with running headers/footers gone.

    Parameters
    ----------
    page_texts : list[str]
        One extracted-text string per page, in page order.
    band : int
        Max non-blank lines at the top and bottom of each page to consider as
        header/footer candidates (capped per page so top/bottom never overlap).
    min_frac : float
        A line must recur in its band on at least ``ceil(min_frac * n_pages)``
        pages (min 2) to count as a running header/footer.
    min_pages : int
        Documents shorter than this are returned unchanged.
    return_kept : bool
        If True, return ``(cleaned_texts, kept_indices)`` where
        ``kept_indices[p]`` is the list of ORIGINAL line indices (into
        ``page_texts[p].split("\\n")``) that survived on page ``p``. This lets a
        caller realign per-line metadata (OCR confidences) to the EXACT
        surviving line — re-deriving the mapping by text equality misbinds
        identical-text duplicate lines. Default False keeps the legacy
        list-of-strings return.
    """
    n = len(page_texts)

    def _all_indices():
        return [list(range(len((t or "").split("\n")))) for t in page_texts]

    if n < min_pages:
        cleaned = list(page_texts)
        return (cleaned, _all_indices()) if return_kept else cleaned

    pages_lines = []
    for txt in page_texts:
        lines = (txt or "").split("\n")
        nonblank = [i for i, ln in enumerate(lines) if ln.strip()]
        pages_lines.append((lines, nonblank))

    top_counter, bot_counter = Counter(), Counter()
    for lines, nonblank in pages_lines:
        top, bot = _bands(nonblank, band)
        for k in {_key(lines[i]) for i in top}:
            if k:
                top_counter[k] += 1
        for k in {_key(lines[i]) for i in bot}:
            if k:
                bot_counter[k] += 1

    thresh = max(2, math.ceil(min_frac * n))
    running_top = {k for k, c in top_counter.items() if c >= thresh}
    running_bot = {k for k, c in bot_counter.items() if c >= thresh}
    if not running_top and not running_bot:
        cleaned = list(page_texts)
        return (cleaned, _all_indices()) if return_kept else cleaned

    out, kept_all = [], []
    for lines, nonblank in pages_lines:
        top, bot = _bands(nonblank, band)
        strip_idx = set()
        for i in top:
            if _key(lines[i]) in running_top:
                strip_idx.add(i)
        for i in bot:
            if _key(lines[i]) in running_bot:
                strip_idx.add(i)
        # Never reduce a page to nothing — if everything would go, keep it as-is.
        if nonblank and set(nonblank).issubset(strip_idx):
            out.append("\n".join(lines).strip("\n"))
            kept_all.append(list(range(len(lines))))  # nothing actually removed
            continue
        kept_idx = [i for i in range(len(lines)) if i not in strip_idx]
        kept = [lines[i] for i in kept_idx]
        cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept).strip("\n"))
        out.append(cleaned)
        kept_all.append(kept_idx)
    return (out, kept_all) if return_kept else out
