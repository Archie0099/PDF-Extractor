"""Regression: all PyMuPDF (fitz) access must happen on the single OCR worker
thread, never on the event-loop thread.

PyMuPDF/MuPDF is not thread-safe even across separate Document objects, and the
app already routes page rendering through a single-worker executor for exactly
this reason. But create_job's validation open and run()'s doc open/close used to
run on the event-loop thread, so a concurrent multi-file upload could have the
event loop inside fitz.open while the worker thread was inside get_pixmap — two
threads in MuPDF at once. This test pins every fitz.open to the 'ocr-worker'
thread by recording the calling thread name through the real upload->run path.
"""

import threading
import time

import fitz
import pytest


@pytest.fixture
def record_fitz_threads(monkeypatch):
    """Patch fitz.open to record the name of the thread it runs on."""
    seen = []
    real_open = fitz.open

    def spy_open(*args, **kwargs):
        seen.append(threading.current_thread().name)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(fitz, "open", spy_open)
    return seen


def _born_digital():
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((72, 100), "Thread safety regression test content here.", fontsize=16)
    p.insert_text((72, 130), "Second line of alphanumeric body text xyz 123.", fontsize=16)
    data = doc.tobytes()
    doc.close()
    return data


def test_all_fitz_open_calls_run_on_the_ocr_worker(client, record_fitz_threads):
    # Upload (create_job opens), let run() open/close, then render a page image
    # (render_page_png opens) — every one must be on the single ocr-worker thread.
    pdf = _born_digital()
    record_fitz_threads.clear()  # drop the test's own PDF-construction open()
    r = client.post(
        "/api/upload",
        files={"file": ("t.pdf", pdf, "application/pdf")},
        data={"mode": "fast", "lang": "en", "preprocess": "false"},
    )
    assert r.status_code == 200
    jid = r.json()["job_id"]
    for _ in range(60):
        st = client.get(f"/api/jobs/{jid}").json()["status"]
        if st in ("done", "error", "cancelled"):
            break
        time.sleep(0.2)
    assert st == "done"

    # Trigger an on-demand render too (this fitz.open lives in render_page_png).
    img = client.get(f"/api/jobs/{jid}/pages/1/image?dpi=120")
    assert img.status_code == 200

    assert record_fitz_threads, "expected at least one fitz.open during the flow"
    offenders = [t for t in record_fitz_threads if not t.startswith("ocr-worker")]
    assert not offenders, f"fitz.open ran off the OCR worker thread: {offenders}"
