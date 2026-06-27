"""Tests for the knowledge-graph feature (pipeline.kg + the /api/graph endpoints).

Everything runs OFFLINE: the Gemini response parsers are exercised with canned
dicts, the graph + hybrid retrieval are pure Python, and the build/query
endpoints are driven with kg's two network calls monkeypatched — mirroring the
"never touch the network, never echo the key" approach in tests/test_online_ocr.py.
"""

import io
import json
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from pipeline import kg


# --------------------------- triple-response parser ---------------------------
def test_parse_triple_payload_json_object():
    inner = json.dumps({
        "entities": [{"name": "Archit", "type": "person"}],
        "triples": [{"subject": "Archit", "predicate": "studies at", "object": "BITS"}],
    })
    payload = {"candidates": [{"content": {"parts": [{"text": inner}]}}]}
    out = kg._parse_triple_payload(payload)
    assert out["entities"] == [{"name": "Archit", "type": "PERSON"}]  # type upper-cased
    assert out["triples"][0]["predicate"] == "studies at"


def test_parse_triple_payload_strips_code_fence():
    inner = "```json\n" + json.dumps({"triples": [{"subject": "A", "predicate": "r", "object": "B"}]}) + "\n```"
    payload = {"candidates": [{"content": {"parts": [{"text": inner}]}}]}
    assert kg._parse_triple_payload(payload)["triples"] == [
        {"subject": "A", "predicate": "r", "object": "B"}
    ]


def test_parse_triple_payload_accepts_bare_array():
    inner = json.dumps([{"subject": "A", "predicate": "r", "object": "B"}])
    payload = {"candidates": [{"content": {"parts": [{"text": inner}]}}]}
    assert kg._parse_triple_payload(payload)["triples"][0]["object"] == "B"


def test_parse_triple_payload_drops_incomplete_triples():
    inner = json.dumps({"triples": [
        {"subject": "A", "predicate": "", "object": "B"},   # missing predicate -> dropped
        {"subject": "A", "predicate": "r", "object": "B"},
    ]})
    payload = {"candidates": [{"content": {"parts": [{"text": inner}]}}]}
    assert len(kg._parse_triple_payload(payload)["triples"]) == 1


def test_parse_triple_payload_empty_on_garbage():
    # A non-JSON candidate must not raise — one odd page can't fail the build.
    payload = {"candidates": [{"content": {"parts": [{"text": "not json at all"}]}}]}
    assert kg._parse_triple_payload(payload) == {"entities": [], "triples": []}


def test_parse_triple_payload_blocked_raises():
    with pytest.raises(RuntimeError):
        kg._parse_triple_payload({"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []})


# ----------------------------- embedding parser ------------------------------
def test_parse_embed_payload_ok():
    payload = {"embeddings": [{"values": [1.0, 2.0]}, {"values": [3.0, 4.0]}]}
    assert kg._parse_embed_payload(payload, 2) == [[1.0, 2.0], [3.0, 4.0]]


def test_parse_embed_payload_count_mismatch_raises():
    # A silent count mismatch would misalign vectors with nodes — must raise.
    with pytest.raises(RuntimeError):
        kg._parse_embed_payload({"embeddings": [{"values": [1.0]}]}, 2)


def test_parse_embed_payload_empty_vector_raises():
    with pytest.raises(RuntimeError):
        kg._parse_embed_payload({"embeddings": [{"values": []}]}, 1)


# --------------------------- key safety (reused) -----------------------------
def test_extract_triples_malformed_key_never_echoed():
    # A CR/LF key must raise a clean, KEY-FREE RuntimeError (no network call).
    with pytest.raises(RuntimeError) as ei:
        kg.extract_triples("some real text", api_key="SECRETKEYPART\nx")
    assert "SECRETKEYPART" not in str(ei.value)


def test_extract_triples_empty_text_no_network():
    # Empty text returns immediately, before any HTTP — safe to call with junk key.
    assert kg.extract_triples("   ", api_key="AIzaWhatever") == {"entities": [], "triples": []}


def test_post_json_maps_http_error_without_key(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 401, "denied", {}, io.BytesIO(b'{"error":{"message":"bad key"}}')
        )
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError) as ei:
        kg._post_json("https://x", {}, "MYSECRETKEY", timeout=5)
    assert "MYSECRETKEY" not in str(ei.value)


def test_list_embedding_models_filters_by_method(monkeypatch):
    body = {"models": [
        {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-embedding-001", "supportedGenerationMethods": ["embedContent", "batchEmbedContents"]},
        {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
    ]}

    class _Resp:
        def __init__(self, d): self._d = d
        def read(self): return self._d
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=None: _Resp(json.dumps(body).encode()),
    )
    out = kg._list_embedding_models("AIzaX")
    assert "gemini-embedding-001" in out and "text-embedding-004" in out
    assert "gemini-2.5-flash" not in out  # generateContent-only -> excluded


def test_pick_embedding_model_prefers_newest_available(monkeypatch):
    monkeypatch.setattr(kg, "_EMBED_MODEL_RESOLVED", {})
    monkeypatch.setattr(kg, "_list_embedding_models", lambda *a, **k: ["text-embedding-004", "gemini-embedding-001"])
    assert kg._pick_embedding_model("AIzaX") == "gemini-embedding-001"


def test_pick_embedding_model_explicit_request_wins(monkeypatch):
    monkeypatch.setattr(kg, "_list_embedding_models", lambda *a, **k: [])
    assert kg._pick_embedding_model("AIzaX", "models/my-embed") == "my-embed"


def test_pick_embedding_model_falls_back_when_discovery_fails(monkeypatch):
    monkeypatch.setattr(kg, "_EMBED_MODEL_RESOLVED", {})

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(kg, "_list_embedding_models", boom)
    # No usable list -> the documented default (never crashes the build).
    assert kg._pick_embedding_model("AIzaX") == kg.DEFAULT_EMBED_MODEL


def test_pick_embedding_model_caches_per_key(monkeypatch):
    # The cache must be keyed by the API key — on the multi-user Space, one
    # visitor's key must not pin the embedding model for everyone.
    monkeypatch.setattr(kg, "_EMBED_MODEL_RESOLVED", {})
    seen = []

    def fake_list(api_key, *, timeout=30):
        seen.append(api_key)
        return ["gemini-embedding-001"] if api_key == "KEYA" else ["text-embedding-004"]

    monkeypatch.setattr(kg, "_list_embedding_models", fake_list)
    assert kg._pick_embedding_model("KEYA") == "gemini-embedding-001"
    assert kg._pick_embedding_model("KEYB") == "text-embedding-004"  # not pinned by KEYA
    assert kg._pick_embedding_model("KEYA") == "gemini-embedding-001"  # cached
    assert seen.count("KEYA") == 1  # second KEYA call served from cache


def test_post_json_retries_on_429_then_succeeds(monkeypatch):
    calls = {"n": 0}

    class _Resp:
        def __init__(self, data): self._d = data
        def read(self): return self._d
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(
                req.full_url, 429, "rate", {}, io.BytesIO(b'{"error":{"message":"slow down"}}')
            )
        return _Resp(b'{"ok":true}')

    monkeypatch.setattr(kg.time, "sleep", lambda *_: None)  # no real backoff delay
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert kg._post_json("https://x", {"a": 1}, "AIzaX", timeout=5) == {"ok": True}
    assert calls["n"] == 3  # two 429s + one success


def test_post_json_error_carries_status_code(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 404, "nf", {}, io.BytesIO(b'{"error":{"message":"not found"}}')
        )
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(kg.GeminiHTTPError) as ei:
        kg._post_json("https://x", {}, "AIzaX", timeout=5)
    assert ei.value.code == 404  # lets callers branch (e.g. batch->single fallback)


def test_embed_texts_falls_back_to_embedcontent_on_batch_404(monkeypatch):
    # If the model doesn't expose batchEmbedContents (a 404), embed_texts must
    # degrade to per-item embedContent rather than aborting the whole build.
    seen = {"batch": 0, "single": 0}

    def fake_post(url, body, api_key, *, timeout):
        if "batchEmbedContents" in url:
            seen["batch"] += 1
            raise kg.GeminiHTTPError("not found", code=404)
        seen["single"] += 1
        return {"embedding": {"values": [0.1, 0.2, 0.3]}}

    monkeypatch.setattr(kg, "_post_json", fake_post)
    out = kg.embed_texts(["a", "b"], api_key="AIzaX", model="some-model")
    assert out.shape == (2, 3)
    assert seen["batch"] == 1 and seen["single"] == 2  # tried batch once, then per-item


# ------------------------------ graph + search -------------------------------
def _fake_gemini(monkeypatch):
    def fake_extract(text, *, api_key, model=None, timeout=120):
        return {
            "entities": [
                {"name": "Archit", "type": "PERSON"},
                {"name": "BITS Pilani", "type": "ORG"},
                {"name": "Hyderabad", "type": "PLACE"},
            ],
            "triples": [
                {"subject": "Archit", "predicate": "studies at", "object": "BITS Pilani"},
                {"subject": "BITS Pilani", "predicate": "located in", "object": "Hyderabad"},
            ],
        }

    def fake_embed(texts, *, api_key, model=None, task_type="RETRIEVAL_DOCUMENT", timeout=120):
        rng = np.random.default_rng(len(texts) + 1)
        return rng.standard_normal((len(texts), 8)).astype("float32")

    monkeypatch.setattr(kg, "extract_triples", fake_extract)
    monkeypatch.setattr(kg, "embed_texts", fake_embed)
    # Keep build_graph fully offline: don't let model discovery hit the network.
    monkeypatch.setattr(kg, "_pick_embedding_model", lambda *a, **k: "gemini-embedding-001")


def test_build_graph_attaches_page_provenance(monkeypatch):
    _fake_gemini(monkeypatch)
    g = kg.build_graph([(1, "Archit studies at BITS Pilani in Hyderabad"), (2, "")], api_key="AIzaX")
    assert g.num_nodes == 3 and g.num_edges == 2
    # Page 2 was empty -> skipped; every triple carries its real source page.
    assert all(t["page"] == 1 for t in g.triples)
    assert g.has_vectors  # nodes were embedded


def test_search_lexical_finds_and_explains():
    g = kg.KnowledgeGraph()
    g._add_triple("Archit", "studies at", "BITS Pilani", 1)
    g._add_triple("BITS Pilani", "located in", "Hyderabad", 1)
    res = g.search("where is BITS located", hops=2)
    assert res["mode"] == "lexical"  # no vectors -> lexical seeds
    assert "BITS Pilani" in [a["entity"] for a in res["answers"]]
    # Explainability: an answer carries supporting facts with page provenance.
    assert any(a["facts"] and a["facts"][0]["page"] == 1 for a in res["answers"])


def test_search_empty_query_returns_no_answers():
    g = kg.KnowledgeGraph()
    g._add_triple("A", "r", "B", 1)
    assert g.search("   ")["answers"] == []


def test_search_traversal_path_carries_pages():
    g = kg.KnowledgeGraph()
    g._add_triple("Archit", "studies at", "BITS Pilani", 1)
    g._add_triple("BITS Pilani", "located in", "Hyderabad", 2)
    res = g.search("Archit", hops=2, top_k=1, max_results=10)
    hyd = next((a for a in res["answers"] if a["entity"] == "Hyderabad"), None)
    assert hyd is not None and hyd["hops"] == 2
    # The supporting path threads pages 1 -> 2 (the chain of facts).
    assert hyd["path_pages"] == [1, 2]
    assert [t["predicate"] for t in hyd["path"]] == ["studies at", "located in"]


def test_search_semantic_mode_with_vectors():
    g = kg.KnowledgeGraph()
    g._add_triple("Archit", "studies at", "BITS Pilani", 1)
    g._add_triple("BITS Pilani", "located in", "Hyderabad", 1)
    # Give each node a vector; query vector equals BITS's vector -> BITS ranks top.
    vecs = np.eye(3, dtype="float32")
    g.set_vectors(vecs, model="text-embedding-004")
    bits_id = g._index[kg._norm_key("BITS Pilani")]
    res = g.search("anything", query_vec=vecs[bits_id], hops=1, top_k=1)
    assert res["mode"] == "semantic"
    assert res["answers"][0]["entity"] == "BITS Pilani"


def test_viz_and_export_dicts():
    g = kg.KnowledgeGraph()
    g._add_triple("A", "r", "B", 1)
    g._add_triple("B", "s", "C", 2)
    viz = g.to_viz_dict()
    assert {n["name"] for n in viz["nodes"]} == {"A", "B", "C"}
    assert len(viz["edges"]) == 2 and viz["truncated"] is False
    exp = g.to_export_dict()
    assert exp["node_count"] == 3 and exp["triple_count"] == 2
    assert exp["triples"][0]["page"] == 1


def test_search_reports_lexical_when_query_vector_dim_mismatches():
    # Vectors exist, but a wrong-dimension query vector forces a lexical fallback;
    # the reported mode must say so (diagnosability bug fix).
    g = kg.KnowledgeGraph()
    g._add_triple("Archit", "studies at", "BITS Pilani", 1)
    g.set_vectors(np.eye(g.num_nodes, dtype="float32"))
    res = g.search("BITS", query_vec=np.ones((99,), dtype="float32"), top_k=1)
    assert res["mode"] == "lexical"  # NOT "semantic" — it actually fell back
    assert "BITS Pilani" in [a["entity"] for a in res["answers"]]


# ---------------------- build isolation & robustness -------------------------
def test_build_graph_isolates_a_failing_page(monkeypatch):
    def fake_extract(text, *, api_key, model=None, timeout=120):
        if "BOOM" in text:
            raise RuntimeError("rate limited on this page")
        return {"entities": [{"name": "X", "type": "OTHER"}],
                "triples": [{"subject": "X", "predicate": "r", "object": "Y"}]}

    monkeypatch.setattr(kg, "extract_triples", fake_extract)
    monkeypatch.setattr(kg, "embed_texts", lambda texts, **k: np.zeros((len(texts), 4), "float32"))
    monkeypatch.setattr(kg, "_pick_embedding_model", lambda *a, **k: "m")
    g = kg.build_graph([(1, "good"), (2, "BOOM bad"), (3, "good")], api_key="AIzaX")
    assert g.pages_built == 2 and g.pages_failed == 1  # one page skipped, not whole build
    assert g.num_nodes >= 2  # the good pages still produced a graph


def test_build_graph_extracts_pages_concurrently(monkeypatch):
    # Per-page Gemini calls must fan across the shared pool, not run strictly
    # sequentially — verified by observing >1 worker thread doing the work while
    # the result stays correct and in page order.
    import threading

    seen_threads = set()

    def fake_extract(text, *, api_key, model=None, timeout=120):
        seen_threads.add(threading.current_thread().name)
        time.sleep(0.05)  # hold the worker so siblings run in parallel
        return {"entities": [{"name": text, "type": "OTHER"}], "triples": []}

    monkeypatch.setattr(kg, "extract_triples", fake_extract)
    monkeypatch.setattr(kg, "_pick_embedding_model", lambda *a, **k: "m")
    monkeypatch.setattr(kg, "embed_texts", lambda texts, **k: np.zeros((len(texts), 4), "float32"))
    pages = [(i, "page%d" % i) for i in range(1, 9)]
    g = kg.build_graph(pages, api_key="AIzaX")
    assert g.pages_built == 8 and g.num_nodes == 8
    assert len(seen_threads) > 1  # ran on multiple pool workers (concurrent)
    # Graph construction stayed in page order (deterministic node ids).
    assert [n["name"] for n in g.nodes] == ["page%d" % i for i in range(1, 9)]


def test_build_graph_raises_when_every_page_fails(monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("invalid api key")
    monkeypatch.setattr(kg, "extract_triples", _raise)
    with pytest.raises(RuntimeError) as ei:
        kg.build_graph([(1, "a"), (2, "b")], api_key="AIzaX")
    assert "invalid api key" in str(ei.value)  # systemic failure surfaced, not silent empty graph


def test_build_graph_keeps_lexical_graph_when_embeddings_fail(monkeypatch):
    def fake_extract(text, *, api_key, model=None, timeout=120):
        return {"entities": [{"name": "Archit", "type": "PERSON"}],
                "triples": [{"subject": "Archit", "predicate": "studies at", "object": "BITS"}]}

    def boom_embed(texts, **k):
        raise RuntimeError("embedding quota exceeded")

    monkeypatch.setattr(kg, "extract_triples", fake_extract)
    monkeypatch.setattr(kg, "embed_texts", boom_embed)
    monkeypatch.setattr(kg, "_pick_embedding_model", lambda *a, **k: "m")
    g = kg.build_graph([(1, "text")], api_key="AIzaX")
    # The (costly) triple extraction is NOT discarded; graph works lexically.
    assert g.num_nodes == 2 and not g.has_vectors
    assert g.embed_error and "quota" in g.embed_error
    res = g.search("BITS")
    assert res["mode"] == "lexical" and res["answers"]


def test_parse_triple_payload_preserves_inner_backticks():
    # Precise fence-strip must not eat backticks inside the JSON content.
    inner = json.dumps({"triples": [{"subject": "code", "predicate": "uses", "object": "`ls`"}]})
    payload = {"candidates": [{"content": {"parts": [{"text": "```json\n" + inner + "\n```"}]}}]}
    out = kg._parse_triple_payload(payload)
    assert out["triples"][0]["object"] == "`ls`"


# ------------------------- no-leak / serialization ---------------------------
def test_job_to_dict_excludes_graph_and_key():
    """The graph (and the API key) must never appear in the client-facing dict —
    to_dict() is an explicit allow-list, so a new field is hidden by default."""
    from pipeline.jobs import Job

    j = Job(
        job_id="x", filename="f", pdf_bytes=b"", mode="max", lang="en", preprocess=True,
        binarize=False, handwriting=False, online=False, online_key="SECRETKEY",
        online_model="", force_ocr=False, remove_headers=True, total_pages=0,
    )
    j.knowledge_graph = kg.KnowledgeGraph()
    j.kg_status = "ready"
    j.kg_error = "oops"
    d = j.to_dict()
    assert "knowledge_graph" not in d and "kg_status" not in d and "kg_error" not in d
    assert "SECRETKEY" not in json.dumps(d)


# --------------------------------- endpoints ---------------------------------
def _upload_and_finish(client, pdf):
    jid = client.post(
        "/api/upload",
        files={"file": ("d.pdf", pdf, "application/pdf")},
        data={"mode": "fast", "preprocess": "false"},
    ).json()["job_id"]
    for _ in range(150):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert j["status"] == "done"
    return jid


def test_capabilities_includes_kg_flag(client):
    assert client.get("/api/capabilities").json().get("knowledge_graph") is True


def test_graph_endpoints_build_query_get(client, monkeypatch, born_digital_pdf):
    _fake_gemini(monkeypatch)
    jid = _upload_and_finish(client, born_digital_pdf)

    rb = client.post("/api/graph/build", data={"job_id": jid, "online_key": "AIzaTESTKEY"})
    assert rb.status_code == 200
    bd = rb.json()
    assert bd["nodes"] == 3 and bd["edges"] == 2 and bd["has_vectors"] is True
    # Build telemetry is surfaced (so partial builds / degraded search aren't silent).
    assert bd["pages_built"] >= 1 and bd["pages_failed"] == 0 and bd["embed_error"] is None

    # The built graph must NOT leak into the job snapshot.
    jd = client.get(f"/api/jobs/{jid}").json()
    assert "knowledge_graph" not in jd and "kg_status" not in jd

    rq = client.post("/api/graph/query", data={"job_id": jid, "query": "BITS", "online_key": "AIzaTESTKEY"})
    assert rq.status_code == 200 and rq.json()["answers"]

    rv = client.get(f"/api/graph/{jid}")
    assert rv.status_code == 200 and rv.json()["nodes"]
    rf = client.get(f"/api/graph/{jid}?full=1")
    assert rf.json()["triple_count"] == 2 and rf.json()["node_count"] == 3


def test_graph_build_failure_does_not_strand_status(client, monkeypatch, born_digital_pdf):
    # If a build fails, kg_status must be reset to "error" (not stuck on
    # "building"), so a retry is NOT 409-locked.
    def _raise(*a, **k):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(kg, "extract_triples", _raise)
    jid = _upload_and_finish(client, born_digital_pdf)
    r1 = client.post("/api/graph/build", data={"job_id": jid, "online_key": "AIzaTESTKEY"})
    assert r1.status_code == 400 and "quota" in r1.json()["detail"].lower()
    # Retry must reach the build again (400), not be rejected as "already building" (409).
    r2 = client.post("/api/graph/build", data={"job_id": jid, "online_key": "AIzaTESTKEY"})
    assert r2.status_code == 400


def test_graph_build_requires_key(client, monkeypatch, born_digital_pdf):
    monkeypatch.delenv("DEMO_GEMINI_KEY", raising=False)
    jid = _upload_and_finish(client, born_digital_pdf)
    r = client.post("/api/graph/build", data={"job_id": jid})  # no key, no demo key
    assert r.status_code == 400 and "key" in r.json()["detail"].lower()


def test_graph_query_before_build_returns_409(client, born_digital_pdf):
    jid = _upload_and_finish(client, born_digital_pdf)
    r = client.post("/api/graph/query", data={"job_id": jid, "query": "x"})
    assert r.status_code == 409


def test_graph_build_unknown_job_404(client):
    assert client.post("/api/graph/build", data={"job_id": "nope"}).status_code == 404
    assert client.post("/api/graph/query", data={"job_id": "nope", "query": "x"}).status_code == 404
    assert client.get("/api/graph/nope").status_code == 404
