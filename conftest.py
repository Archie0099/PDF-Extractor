"""Pytest configuration + shared fixtures for the PDF Extractor test suite.

Run from the project root with the venv interpreter:

    .venv\\Scripts\\python.exe -m pytest -q

Pure-function tests (postprocess / preprocess / online_ocr / analyze / export)
are fast and need no model load. The API tests build a FastAPI ``TestClient``
once per session — that triggers the PaddleOCR warmup (cached weights), so the
first API test is slow.
"""

import os
import sys

import pytest

# Make the project root importable (``import app`` / ``import pipeline...``)
# regardless of where pytest is invoked from.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _born_digital_pdf(text_lines=None) -> bytes:
    """A single-page born-digital PDF with a real (selectable) text layer."""
    import fitz

    lines = text_lines or [
        "Hello world from the text layer of this document.",
        "Second line with plenty of alphanumeric content xyz 123.",
    ]
    doc = fitz.open()
    page = doc.new_page()
    y = 100
    for ln in lines:
        page.insert_text((72, y), ln, fontsize=16)
        y += 28
    data = doc.tobytes()
    doc.close()
    return data


def _multipage_with_running_headers(n_pages=4) -> bytes:
    """A multi-page born-digital PDF with a running header, footer + page no."""
    import fitz

    doc = fitz.open()
    for i in range(1, n_pages + 1):
        p = doc.new_page(width=400, height=600)
        p.insert_text((50, 40), "ACME CORP CONFIDENTIAL REPORT", fontsize=11)
        p.insert_text((50, 90), f"Unique body content of page {i} alpha bravo.", fontsize=12)
        p.insert_text((50, 120), f"Second body line for page {i} charlie delta.", fontsize=12)
        p.insert_text((50, 560), "Copyright 2026 ACME - all rights reserved", fontsize=10)
        p.insert_text((190, 580), f"Page {i}", fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def born_digital_pdf():
    return _born_digital_pdf()


@pytest.fixture
def multipage_headers_pdf():
    return _multipage_with_running_headers(4)


@pytest.fixture(scope="session")
def client():
    """A FastAPI TestClient with app startup/shutdown run (warms PaddleOCR once)."""
    from fastapi.testclient import TestClient
    import app as appmod

    with TestClient(appmod.app) as c:
        yield c
