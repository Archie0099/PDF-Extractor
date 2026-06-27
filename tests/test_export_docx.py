"""Tests for pipeline.export_docx.build_docx.

Regression: a malformed ``documents`` payload (non-list, or list/dict elements
of the wrong type, or a non-string ``text``) used to raise an unhandled
exception inside ``build_docx`` -> HTTP 500 from POST /api/export/docx. The
body comes from an untrusted request, so build_docx must never crash on it.
"""

import zipfile
import io

import pytest

from pipeline.export_docx import build_docx


def _is_docx(data: bytes) -> bool:
    # A .docx is a zip with a word/document.xml entry.
    if not data:
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            return "word/document.xml" in z.namelist()
    except zipfile.BadZipFile:
        return False


def test_well_formed_roundtrips():
    docs = [{
        "filename": "report.pdf",
        "pages": [
            {"page": 1, "source": "text", "confidence": None, "text": "hello\nworld"},
            {"page": 2, "source": "ocr", "confidence": 0.91, "text": "scanned line"},
        ],
    }]
    data = build_docx(docs)
    assert _is_docx(data)


def test_empty_list():
    assert _is_docx(build_docx([]))


@pytest.mark.parametrize("bad", [
    "notalist",                                   # documents is a string
    None,                                         # documents is None
    42,                                           # documents is an int
    ["a", "b"],                                   # list of strings
    [{"filename": "f", "pages": "oops"}],         # pages is a string
    [{"filename": "f", "pages": ["x"]}],          # a page is a string
    [{"filename": "f", "pages": [{"page": 1, "source": "ocr", "text": 123}]}],  # text is int
    [{"filename": None, "pages": [{"page": 1, "text": None}]}],  # None filename/text
])
def test_malformed_payloads_never_crash(bad):
    # The whole point: these must return valid .docx bytes, not raise.
    data = build_docx(bad)
    assert _is_docx(data)


@pytest.mark.parametrize("ctrl", ["\x00", "\x07", "\x0b", "\x0c", "\x1b", "\x1f"])
def test_xml_illegal_control_chars_dont_crash(ctrl):
    # Form-feed (\x0c) is organically reachable from a PDF text layer; all of
    # these are XML-illegal and used to 500 the whole Word export.
    docs = [{"filename": "f" + ctrl, "pages": [
        {"page": 1, "source": "text", "text": "before" + ctrl + "after"},
    ]}]
    data = build_docx(docs)
    assert _is_docx(data)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml").decode("utf-8", "replace")
    assert ctrl not in xml            # the control char was stripped
    assert "beforeafter" in xml       # surrounding text preserved


def test_legal_whitespace_preserved():
    # Tab / newline / carriage-return are XML-legal and must NOT be stripped by
    # the control-char sanitizer (only the illegal C0 controls are removed).
    from pipeline.export_docx import _xml_safe
    assert _xml_safe("tab\there\r\nline") == "tab\there\r\nline"
    data = build_docx([{"filename": "f", "pages": [
        {"page": 1, "source": "text", "text": "tab\there\nnext line"},
    ]}])
    assert _is_docx(data)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml").decode("utf-8", "replace")
    assert "next line" in xml  # content preserved across the newline split


@pytest.mark.parametrize("conf", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_confidence_doesnt_crash(conf):
    # json.loads accepts bare NaN/Infinity -> round() would raise -> 500.
    docs = [{"filename": "f", "pages": [
        {"page": 1, "source": "ocr", "confidence": conf, "text": "hi"},
    ]}]
    assert _is_docx(build_docx(docs))


def test_blank_page_gets_placeholder():
    docs = [{"filename": "x", "pages": [{"page": 1, "source": "text", "text": "  "}]}]
    data = build_docx(docs)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml").decode("utf-8", "replace")
    assert "no text on this page" in xml
