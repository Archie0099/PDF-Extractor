"""Endpoint-level contract tests via FastAPI TestClient.

These reuse the session-scoped ``client`` fixture (warms PaddleOCR once).
"""

import io

import fitz
from fastapi.testclient import TestClient


def test_startup_survives_warmup_failure(monkeypatch):
    """If model warmup raises, the server must still boot (text-layer + Gemini
    paths need no PaddleOCR). Build a fresh client with a poisoned warmup."""
    import app as appmod

    def boom(*a, **k):
        raise RuntimeError("simulated warmup failure")

    monkeypatch.setattr(appmod.engine, "warmup", boom)
    # TestClient(...) as a context manager runs the startup event; it must not
    # raise even though warmup is now guaranteed to fail.
    with TestClient(appmod.app) as c:
        assert c.get("/").status_code == 200


def test_docx_endpoint_never_500s_on_hostile_body(client):
    hostile_bodies = [
        {"documents": "notalist"},
        {},
        {"documents": [123, "x", None]},
        {"documents": [{"filename": "f", "pages": [{"page": 1, "source": "ocr", "text": "a\x0cb"}]}]},
    ]
    for body in hostile_bodies:
        r = client.post("/api/export/docx", json=body)
        assert r.status_code in (200, 400), f"got {r.status_code} for {body}"

    # NaN/Infinity confidence: json.loads (Starlette) accepts bare NaN, so send a
    # RAW body the way a hand-crafted client would (httpx's json= would reject it).
    raw = '{"documents":[{"filename":"f","pages":[{"page":1,"source":"ocr","confidence":NaN,"text":"hi"}]}]}'
    r = client.post("/api/export/docx", content=raw,
                    headers={"Content-Type": "application/json"})
    assert r.status_code in (200, 400)
    assert r.status_code != 500


def test_page_image_dpi_is_clamped(client):
    # Upload a born-digital page so there's something to render.
    doc = fitz.open(); p = doc.new_page()
    p.insert_text((72, 100), "Render me with a clamped dpi please.", fontsize=16)
    pdf = doc.tobytes(); doc.close()
    jid = client.post("/api/upload", files={"file": ("d.pdf", pdf, "application/pdf")},
                      data={"mode": "fast", "preprocess": "false"}).json()["job_id"]
    # 0, negative, and absurdly-large dpi must all be clamped to a valid render.
    for dpi in (0, -5, 999999):
        r = client.get(f"/api/jobs/{jid}/pages/1/image?dpi={dpi}")
        assert r.status_code == 200
        assert r.content[:4] == b"\x89PNG"
    # Out-of-range page -> 404, not 500.
    assert client.get(f"/api/jobs/{jid}/pages/999/image").status_code == 404


def test_unknown_job_endpoints_404(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/nope/pages/1/image").status_code == 404
    assert client.post("/api/jobs/nope/cancel").status_code == 404
    # delete is idempotent -> always 200
    assert client.delete("/api/jobs/nope").status_code == 200


def test_upload_rejects_online_without_key(client):
    doc = fitz.open(); doc.new_page().insert_text((72, 100), "x" * 80, fontsize=14)
    pdf = doc.tobytes(); doc.close()
    r = client.post("/api/upload", files={"file": ("d.pdf", pdf, "application/pdf")},
                    data={"online": "true", "online_key": ""})
    assert r.status_code == 400
    assert "key" in r.json()["detail"].lower()
