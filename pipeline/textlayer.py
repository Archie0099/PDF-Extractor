"""PyMuPDF text-layer extraction with reading-order reconstruction.

These helpers operate on ``fitz.Page`` objects. ``has_text_layer`` decides
whether the embedded text layer is rich enough to use directly, and
``extract_text_layer`` reconstructs paragraphs in natural reading order from
the page's structured ``"dict"`` output.
"""


def has_text_layer(page, *, min_chars: int = 50) -> bool:
    """Return True if the page's text layer has enough alphanumeric content.

    Counts alphanumeric characters in the stripped text layer and compares
    against ``min_chars``. Whitespace and punctuation-only layers (common in
    scanned PDFs with stray marks) are treated as having no usable text.
    """
    try:
        text = page.get_text("text") or ""
    except Exception:
        return False

    alnum = sum(1 for ch in text if ch.isalnum())
    return alnum >= min_chars


def extract_text_layer(page) -> str:
    """Reconstruct page text in reading order from the structured dict.

    Blocks are sorted by their bounding box ``(y0, x0)``. Within each block,
    line order is preserved and the spans of a line are concatenated. Lines are
    joined by ``"\n"`` and blocks are separated by ``"\n\n"`` to approximate
    paragraph structure.

    NOTE: the flat ``(y0, x0)`` sort assumes a SINGLE-column layout. On a
    multi-column page, blocks at similar vertical positions in different columns
    interleave, so the reconstructed text zig-zags between columns (every
    character is captured, but the reading order is scrambled). Column-aware
    reordering is not implemented yet.
    """
    try:
        data = page.get_text("dict")
    except Exception:
        return ""

    blocks = data.get("blocks", []) if isinstance(data, dict) else []

    text_blocks = []
    for block in blocks:
        # Only text blocks have "lines"; image blocks (type 1) are skipped.
        if block.get("type", 0) != 0:
            continue
        lines = block.get("lines")
        if not lines:
            continue

        bbox = block.get("bbox", [0.0, 0.0, 0.0, 0.0])
        y0 = float(bbox[1]) if len(bbox) > 1 else 0.0
        x0 = float(bbox[0]) if len(bbox) > 0 else 0.0
        text_blocks.append((y0, x0, lines))

    # Sort blocks top-to-bottom, then left-to-right.
    text_blocks.sort(key=lambda b: (b[0], b[1]))

    block_texts = []
    for _y0, _x0, lines in text_blocks:
        line_texts = []
        for line in lines:
            spans = line.get("spans", [])
            line_str = "".join(span.get("text", "") for span in spans)
            line_texts.append(line_str)
        block_str = "\n".join(line_texts).strip("\n")
        if block_str.strip():
            block_texts.append(block_str)

    return "\n\n".join(block_texts)
