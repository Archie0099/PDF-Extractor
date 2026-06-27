"""Optional knowledge-graph + embedding search over EXTRACTED text (Gemini).

After a PDF is extracted, this module turns its per-page text into a small
**knowledge graph** — typed entities (nodes) joined by ``(subject, predicate,
object)`` triples (edges) — embeds every node into a vector space, and answers a
query by combining **semantic vector search** (conceptually-related entities)
with **explicit graph traversal** (following the facts). The answer is
*explainable*: every result carries the supporting triples and the page each
came from.

DESIGN / IDENTITY
- This is an OPT-IN, online feature. Like ``online_ocr.py`` it sends text to
  Google's Gemini API, so it is OFF by default and only runs when the caller
  passes a Gemini API key. The page text leaves this machine.
- ZERO new pip dependencies: the REST calls use the Python standard library
  only (``urllib.request`` + ``json``); the graph is a plain in-memory
  adjacency structure; ``numpy`` (already a core dep) holds the node vectors.
  This keeps the feature working on the lean hosted build too (no torch /
  sentence-transformers / networkx required).
- The security-critical key sanitization and the HTTP error mapping are REUSED
  from ``online_ocr`` so the "never echo the key, only raise a single-line
  RuntimeError" contract holds here as well.

Importing this module performs NO network call and has no import-time side
effects.

Public API
----------
``extract_triples(text, *, api_key, model=None, timeout=...) -> dict``
    Pull ``{"entities": [...], "triples": [...]}`` out of one page of text.
``embed_texts(texts, *, api_key, model=None, task_type=..., timeout=...) -> np.ndarray``
    Embed a list of strings to an ``(n, dim)`` float32 matrix.
``build_graph(pages, *, api_key, ...) -> KnowledgeGraph``
    Build a per-document graph from ``[(page_number, text), ...]``.
``KnowledgeGraph``
    Holds the graph + node vectors; ``.search(query, ...)`` does hybrid
    retrieval and returns ranked, explainable results.
"""

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Reuse the verified, security-tested helpers from the OCR client so the key is
# sanitized identically and HTTP errors never leak it.
from pipeline.online_ocr import (
    _API_BASE,
    _API_KEY_HEADER,
    _clean_key,
    _friendly_http_error,
    is_configured,
)

# ---------------------------------------------------------------------------
# Gemini REST endpoints (same base/host as online_ocr; different methods).
# ---------------------------------------------------------------------------
_GENERATE_URL = _API_BASE + "/{model}:generateContent"
_BATCH_EMBED_URL = _API_BASE + "/{model}:batchEmbedContents"
_EMBED_URL = _API_BASE + "/{model}:embedContent"  # single-item fallback

# Triple extraction must default to a FREE-TIER model (Flash). Pro/preview
# models return HTTP 429 limit:0 on the free tier — see online_ocr notes.
DEFAULT_TRIPLE_MODEL = "gemini-2.5-flash"

# Embeddings: the wire format wants the "models/" prefix in the per-request
# "model" field. The embedding model id has CHANGED over time — Google retired
# `text-embedding-004` on v1beta (it now 404s) in favour of `gemini-embedding-001`.
# So we DON'T trust a single hard-coded id: build_graph DISCOVERS the model the
# key can actually use via ModelService.ListModels (`_pick_embedding_model`),
# trying this preference order and falling back to the default only if discovery
# fails. Order newest→oldest so a key that still has an old model also works.
DEFAULT_EMBED_MODEL = "gemini-embedding-001"
EMBED_MODEL_PREFERENCE = (
    "gemini-embedding-001",   # current GA model (default ~3072-dim; we L2-normalize)
    "text-embedding-004",     # legacy 768-dim (retired on v1beta for many keys)
    "embedding-001",          # older fallback
)

# Gemini caps batchEmbedContents at 100 requests per call; chunk to stay under.
_EMBED_BATCH = 100

# Defensive cap on per-page text sent for triple extraction. A single PDF page
# is tiny next to Flash's context window, but a pathological page (e.g. a giant
# embedded text dump) shouldn't balloon the request.
_MAX_PAGE_CHARS = 100_000

# Transient-failure backoff. Free-tier embedding/generate limits are real, so a
# 429/503 is retried a few times with exponential backoff before giving up.
_RETRY_STATUSES = frozenset({429, 503})
_RETRY_ATTEMPTS = 4
_RETRY_BASE_DELAY = 1.5  # seconds; doubled each attempt

# Build scalability: per-page triple extraction is one (I/O-bound) Gemini call
# each, so they're fanned across this SHARED, bounded pool instead of running
# strictly sequentially — a multi-page build is ~Nx faster on a key with real
# throughput. The pool is module-level on purpose: being shared, it also caps the
# TOTAL number of concurrent Gemini calls across simultaneous builds/users, so one
# big document can't blow the rate limit for everyone. Overshoot of the provider's
# rate limit is absorbed by the 429/503 backoff in _post_json. Graph mutation stays
# single-threaded (results are collected, then applied in deterministic page order).
_BUILD_CONCURRENCY = 5
_EXTRACT_POOL = ThreadPoolExecutor(
    max_workers=_BUILD_CONCURRENCY, thread_name_prefix="kg-extract"
)

# Instruction for triple extraction. We ask for typed entities AND triples and
# force a JSON response via responseSchema for deterministic parsing.
_TRIPLE_PROMPT = (
    "You are building a knowledge graph from a document. From the TEXT below, "
    "extract the key entities and the factual relationships between them.\n"
    "- entities: the important named things (people, organizations, places, "
    "dates, IDs, amounts, concepts). Give each a short canonical name and a "
    "TYPE (e.g. PERSON, ORG, PLACE, DATE, ID, AMOUNT, CONCEPT, OTHER).\n"
    "- triples: factual (subject, predicate, object) statements stated or "
    "clearly implied by the text. Use concise predicates (e.g. 'works at', "
    "'located in', 'has id', 'dated'). Subject and object should be entity "
    "names. Do NOT invent facts that are not supported by the text.\n"
    "Return only what the text supports; if the text is empty or has no "
    "extractable facts, return empty lists.\n\nTEXT:\n"
)

# Response schema for generateContent — keeps output a strict JSON object.
_TRIPLE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "entities": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "type": {"type": "STRING"},
                },
                "required": ["name"],
            },
        },
        "triples": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "subject": {"type": "STRING"},
                    "predicate": {"type": "STRING"},
                    "object": {"type": "STRING"},
                },
                "required": ["subject", "predicate", "object"],
            },
        },
    },
    "required": ["triples"],
}


# ---------------------------------------------------------------------------
# Low-level HTTP (stdlib) — mirrors online_ocr's request/error handling.
# ---------------------------------------------------------------------------
class GeminiHTTPError(RuntimeError):
    """A friendly, KEY-FREE Gemini HTTP error that also carries the status code.

    Subclasses RuntimeError so every existing ``except RuntimeError`` still
    catches it; the ``.code`` lets callers branch (e.g. fall back from
    batchEmbedContents to embedContent on a 404 method-not-found).
    """

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


def _post_json(url: str, body: dict, api_key: str, *, timeout: float) -> dict:
    """POST a JSON body to Gemini and return the parsed JSON response.

    ``api_key`` must already be sanitized via ``_clean_key``. Retries 429/503
    with exponential backoff, then maps any HTTP/network/non-JSON failure to a
    single-line RuntimeError (never echoing the key — ``_friendly_http_error``
    reads only the API's own error body).
    """
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            _API_KEY_HEADER: api_key,
        },
    )

    last_http_err: Optional[urllib.error.HTTPError] = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            break
        except urllib.error.HTTPError as err:
            last_http_err = err
            if err.code in _RETRY_STATUSES and attempt < _RETRY_ATTEMPTS - 1:
                time.sleep(_RETRY_BASE_DELAY * (2 ** attempt))
                continue
            raise GeminiHTTPError(str(_friendly_http_error(err)), code=err.code) from None
        except urllib.error.URLError as err:
            raise RuntimeError(
                "Could not reach the Gemini API ({}). Check your internet "
                "connection.".format(err.reason)
            ) from None
        except TimeoutError:
            raise RuntimeError(
                "The Gemini request timed out after {}s.".format(timeout)
            ) from None
    else:  # pragma: no cover - loop always breaks or raises
        raise GeminiHTTPError(
            str(_friendly_http_error(last_http_err)),
            code=getattr(last_http_err, "code", None),
        )

    try:
        payload = json.loads(raw.decode("utf-8", "replace"))
    except (ValueError, AttributeError) as err:
        raise RuntimeError(
            "Gemini returned a response that was not valid JSON ({}). The "
            "service may be having problems; retry shortly.".format(err)
        ) from None
    if not isinstance(payload, dict):
        raise RuntimeError(
            "Gemini returned an unexpected response (not a JSON object); "
            "retry shortly."
        )
    return payload


def _resolve_model(model, default: str) -> str:
    """Return a clean bare model id (no ``models/`` prefix), defaulting when unset."""
    name = (str(model).strip() if model else "") or default
    if name.startswith("models/"):
        name = name[len("models/"):]
    return name or default


# Process-level cache of the discovered embedding model, keyed by a HASH of the
# API key (never the key itself). Different keys/projects can have access to
# different models — on the multi-user demo Space the first visitor's key must
# NOT pin the model for everyone. Only a SUCCESSFUL discovery is cached, so a
# transient failure that fell back to the default is retried next time.
_EMBED_MODEL_RESOLVED: dict = {}


def _list_embedding_models(api_key, *, timeout=30) -> list:
    """Return bare ids of models the key can use for embedding.

    Calls ``GET /v1beta/models`` and keeps those whose
    ``supportedGenerationMethods`` include ``batchEmbedContents`` (what we call)
    or ``embedContent``. Used to pick a model that actually exists, instead of
    hard-coding an id that Google may retire (the cause of the text-embedding-004
    404). Mirrors ``online_ocr.list_models`` but filters for embedding support.
    """
    api_key = _clean_key(api_key)
    request = urllib.request.Request(
        _API_BASE, method="GET", headers={_API_KEY_HEADER: api_key}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as err:
        raise _friendly_http_error(err) from None
    except urllib.error.URLError as err:
        raise RuntimeError(
            "Could not reach the Gemini API ({}). Check your internet "
            "connection.".format(err.reason)
        ) from None
    except TimeoutError:
        raise RuntimeError(
            "The Gemini request timed out after {}s.".format(timeout)
        ) from None
    try:
        payload = json.loads(raw.decode("utf-8", "replace"))
    except (ValueError, AttributeError):
        return []
    out = []
    for entry in (payload.get("models") or []):
        if not isinstance(entry, dict):
            continue
        methods = entry.get("supportedGenerationMethods") or []
        if "batchEmbedContents" in methods or "embedContent" in methods:
            name = str(entry.get("name") or "")
            if name.startswith("models/"):
                name = name[len("models/"):]
            if name:
                out.append(name)
    return out


def _pick_embedding_model(api_key, requested=None, *, timeout=30) -> str:
    """Choose an embedding model id the key can actually use.

    An explicit ``requested`` model wins. Otherwise discover the available
    embedding models and pick by :data:`EMBED_MODEL_PREFERENCE`; on a transient
    discovery failure fall back to :data:`DEFAULT_EMBED_MODEL` (without caching,
    so it's retried). A successful discovery is cached for the process.
    """
    if requested:
        return _resolve_model(requested, DEFAULT_EMBED_MODEL)
    key_hash = hashlib.sha256(_clean_key(api_key).encode("utf-8")).hexdigest()
    cached = _EMBED_MODEL_RESOLVED.get(key_hash)
    if cached:
        return cached
    try:
        available = set(_list_embedding_models(api_key, timeout=timeout))
    except RuntimeError:
        available = set()
    chosen = next((m for m in EMBED_MODEL_PREFERENCE if m in available), None)
    if not chosen and available:
        # An unfamiliar but embedding-capable model — prefer one that looks like
        # an embedding model, else just take the first deterministically.
        chosen = next(
            (m for m in sorted(available) if "embedding" in m.lower()),
            sorted(available)[0],
        )
    if chosen:
        _EMBED_MODEL_RESOLVED[key_hash] = chosen  # cache only a real discovery
        return chosen
    return DEFAULT_EMBED_MODEL


# ---------------------------------------------------------------------------
# Response parsers (pure functions — unit-tested with canned dicts, no network).
# ---------------------------------------------------------------------------
def _response_text(payload: dict) -> str:
    """Concatenate the text parts of a generateContent response.

    Handles a prompt-level block or a non-STOP finishReason by raising an
    actionable RuntimeError, and tolerates an explicit ``{"text": null}`` part.
    """
    feedback = payload.get("promptFeedback") or {}
    block_reason = feedback.get("blockReason")
    candidates = payload.get("candidates") or []
    if not candidates:
        if block_reason:
            raise RuntimeError(
                "Gemini blocked the request (blockReason={}). The text tripped "
                "a safety filter; knowledge-graph extraction is unavailable for "
                "this page.".format(block_reason)
            )
        raise RuntimeError(
            "Gemini returned no candidates for knowledge-graph extraction."
        )
    candidate = candidates[0] or {}
    content = candidate.get("content") or {}
    parts = content.get("parts") or []
    return "".join(
        str(part.get("text") or "") for part in parts if isinstance(part, dict)
    )


def _parse_triple_payload(payload: dict) -> dict:
    """Parse a generateContent response into ``{"entities": [...], "triples": [...]}``.

    The model is asked to return a JSON object (responseMimeType=application/json),
    so the candidate text is itself a JSON string. We parse it and coerce every
    field to clean strings, dropping malformed/empty rows. Never raises on a
    merely-empty or slightly-malformed result — returns empty lists instead, so
    one odd page can't fail the whole build.
    """
    text = _response_text(payload).strip()
    if not text:
        return {"entities": [], "triples": []}
    # Strip a ```json ... ``` fence if the model wrapped its JSON in one — but
    # only the fence markers, so backticks legitimately inside the content
    # (e.g. a value with a code span) are preserved.
    if text.startswith("```"):
        text = text[3:]
        if text[:4].lower() == "json":
            text = text[4:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return {"entities": [], "triples": []}
    if not isinstance(obj, dict):
        # Some models return a bare array of triples.
        obj = {"triples": obj} if isinstance(obj, list) else {}

    entities = []
    for e in obj.get("entities") or []:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name") or "").strip()
        if not name:
            continue
        etype = str(e.get("type") or "").strip().upper() or "OTHER"
        entities.append({"name": name, "type": etype})

    triples = []
    for t in obj.get("triples") or []:
        if not isinstance(t, dict):
            continue
        subj = str(t.get("subject") or "").strip()
        pred = str(t.get("predicate") or "").strip()
        obj_ = str(t.get("object") or "").strip()
        if subj and pred and obj_:
            triples.append({"subject": subj, "predicate": pred, "object": obj_})
    return {"entities": entities, "triples": triples}


def _parse_embed_payload(payload: dict, expected: int) -> list:
    """Parse a batchEmbedContents response into a list of float vectors.

    Response shape: ``{"embeddings": [{"values": [...]}, ...]}`` in request
    order. Raises if the count doesn't match what we sent (a silent mismatch
    would misalign vectors with nodes).
    """
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list):
        raise RuntimeError(
            "Gemini embedding response had no 'embeddings' list; cannot build "
            "the vector index."
        )
    if len(embeddings) != expected:
        raise RuntimeError(
            "Gemini returned {} embeddings for {} inputs (count mismatch).".format(
                len(embeddings), expected
            )
        )
    out = []
    for item in embeddings:
        values = (item or {}).get("values") if isinstance(item, dict) else None
        if not isinstance(values, list) or not values:
            raise RuntimeError("Gemini returned an empty embedding vector.")
        out.append([float(v) for v in values])
    return out


def _parse_single_embed_payload(payload: dict) -> list:
    """Parse a single-item embedContent response: ``{"embedding": {"values": [...]}}``."""
    emb = payload.get("embedding") if isinstance(payload, dict) else None
    values = emb.get("values") if isinstance(emb, dict) else None
    if not isinstance(values, list) or not values:
        raise RuntimeError("Gemini returned an empty embedding vector.")
    return [float(v) for v in values]


def _embed_item(wire_model: str, text: str, task_type: str) -> dict:
    """One embedding request item (same shape for batch list and single body)."""
    return {
        "model": wire_model,
        "content": {"parts": [{"text": text}]},
        "taskType": task_type,
    }


# ---------------------------------------------------------------------------
# Public extraction / embedding calls.
# ---------------------------------------------------------------------------
def extract_triples(text: str, *, api_key, model=None, timeout: float = 120) -> dict:
    """Extract typed entities + ``(subject, predicate, object)`` triples from text.

    Returns ``{"entities": [{"name","type"}], "triples":
    [{"subject","predicate","object"}]}``. Provenance (the source page) is NOT
    returned by the model — the caller attaches it.
    """
    if not is_configured(api_key):
        raise RuntimeError(
            "Knowledge-graph extraction needs a Gemini API key but none was "
            "provided. Get a free key from https://aistudio.google.com/."
        )
    clean = str(text or "").strip()
    if not clean:
        return {"entities": [], "triples": []}
    if len(clean) > _MAX_PAGE_CHARS:
        clean = clean[:_MAX_PAGE_CHARS]

    api_key = _clean_key(api_key)
    model_id = _resolve_model(model, DEFAULT_TRIPLE_MODEL)
    body = {
        "contents": [{"parts": [{"text": _TRIPLE_PROMPT + clean}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": _TRIPLE_SCHEMA,
        },
    }
    payload = _post_json(
        _GENERATE_URL.format(model=model_id), body, api_key, timeout=timeout
    )
    return _parse_triple_payload(payload)


def embed_texts(
    texts,
    *,
    api_key,
    model=None,
    task_type: str = "RETRIEVAL_DOCUMENT",
    timeout: float = 120,
) -> np.ndarray:
    """Embed a list of strings into an ``(n, dim)`` float32 matrix via Gemini.

    ``task_type`` should be ``RETRIEVAL_DOCUMENT`` for the graph's nodes and
    ``RETRIEVAL_QUERY`` for the search query — the embedding model uses it to
    tune the vectors for retrieval, a real accuracy win. Batches at 100 inputs
    per request (Gemini's cap) and concatenates in order. Pass an explicit
    ``model`` (typically resolved by :func:`_pick_embedding_model`); the default
    is only a last resort since Google retires embedding-model ids over time.
    """
    if not is_configured(api_key):
        raise RuntimeError(
            "Knowledge-graph search needs a Gemini API key but none was provided."
        )
    items = [str(t or "") for t in texts]
    if not items:
        return np.zeros((0, 0), dtype=np.float32)

    api_key = _clean_key(api_key)
    model_id = _resolve_model(model, DEFAULT_EMBED_MODEL)
    wire_model = "models/" + model_id
    batch_url = _BATCH_EMBED_URL.format(model=model_id)
    single_url = _EMBED_URL.format(model=model_id)

    vectors: list = []
    use_batch = True
    for start in range(0, len(items), _EMBED_BATCH):
        chunk = items[start:start + _EMBED_BATCH]
        if use_batch:
            body = {"requests": [_embed_item(wire_model, t, task_type) for t in chunk]}
            try:
                payload = _post_json(batch_url, body, api_key, timeout=timeout)
                vectors.extend(_parse_embed_payload(payload, len(chunk)))
                continue
            except GeminiHTTPError as exc:
                # 404 = this model/version doesn't expose batchEmbedContents.
                # Degrade to per-item embedContent instead of failing the build.
                if exc.code != 404:
                    raise
                use_batch = False
        for t in chunk:
            payload = _post_json(single_url, _embed_item(wire_model, t, task_type),
                                 api_key, timeout=timeout)
            vectors.append(_parse_single_embed_payload(payload))

    return np.asarray(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# Graph data structures + builder.
# ---------------------------------------------------------------------------
def _norm_key(name: str) -> str:
    """Canonical key for entity de-duplication (case/space-insensitive)."""
    return re.sub(r"\s+", " ", str(name or "").strip()).casefold()


_WORD_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def _tokens(text: str) -> set:
    return {t for t in _WORD_RE.findall(str(text or "").casefold()) if len(t) > 1}


@dataclass
class KnowledgeGraph:
    """A per-document knowledge graph + node embeddings, held in RAM.

    Nodes are entities; edges are ``(subject, predicate, object)`` triples, each
    carrying the page it came from (provenance). Node vectors live in a parallel
    float32 matrix (L2-normalized for cosine). Everything is plain Python +
    numpy — no networkx, no external store — so it evicts with its ``Job`` and
    serializes only via the explicit ``to_*`` methods (never leaked by accident).
    """

    # Parallel arrays indexed by node id (0..n-1).
    nodes: list = field(default_factory=list)        # [{name, type, pages, degree}]
    triples: list = field(default_factory=list)      # [{subject_id, predicate, object_id, subject, object, page}]
    adjacency: dict = field(default_factory=dict)     # node_id -> [(neighbor_id, triple_index)]
    _index: dict = field(default_factory=dict, repr=False)  # norm_key -> node_id
    vectors: Optional[np.ndarray] = field(default=None, repr=False)  # (n, dim) L2-normalized
    embed_model: Optional[str] = None
    # Build telemetry (surfaced so partial builds / degraded search aren't silent).
    pages_built: int = 0      # pages whose triple extraction succeeded
    pages_failed: int = 0     # pages skipped due to a per-page Gemini failure
    embed_error: Optional[str] = None  # set if embedding failed -> lexical-only

    # --- construction helpers ------------------------------------------------
    def _add_node(self, name: str, etype: str = "OTHER", page: Optional[int] = None) -> int:
        key = _norm_key(name)
        if not key:
            return -1
        nid = self._index.get(key)
        if nid is None:
            nid = len(self.nodes)
            self._index[key] = nid
            self.nodes.append(
                {"name": str(name).strip(), "type": (etype or "OTHER"), "pages": [], "degree": 0}
            )
            self.adjacency[nid] = []
        node = self.nodes[nid]
        # Prefer a known type over OTHER if a later mention supplies one.
        if (not node["type"] or node["type"] == "OTHER") and etype and etype != "OTHER":
            node["type"] = etype
        if page is not None and page not in node["pages"]:
            node["pages"].append(page)
        return nid

    def _add_triple(self, subject: str, predicate: str, obj: str, page: Optional[int]) -> None:
        s_id = self._add_node(subject, page=page)
        o_id = self._add_node(obj, page=page)
        if s_id < 0 or o_id < 0 or s_id == o_id:
            return
        tindex = len(self.triples)
        self.triples.append(
            {
                "subject_id": s_id,
                "object_id": o_id,
                "predicate": str(predicate).strip(),
                "subject": self.nodes[s_id]["name"],
                "object": self.nodes[o_id]["name"],
                "page": page,
            }
        )
        # Undirected adjacency for traversal; direction is kept in the triple.
        self.adjacency[s_id].append((o_id, tindex))
        self.adjacency[o_id].append((s_id, tindex))
        self.nodes[s_id]["degree"] += 1
        self.nodes[o_id]["degree"] += 1

    def _node_embed_text(self, nid: int) -> str:
        """Text used to embed a node: its name/type + a little fact context."""
        node = self.nodes[nid]
        bits = ["{} ({})".format(node["name"], node["type"])]
        # Up to a few connected facts give the embedding semantic context.
        ctx = [
            "{} {} {}".format(t["subject"], t["predicate"], t["object"])
            for _, tindex in self.adjacency[nid][:3]
            for t in (self.triples[tindex],)
        ]
        if ctx:
            bits.append(". ".join(ctx))
        return ". ".join(bits)

    def set_vectors(self, vectors: Optional[np.ndarray], model: Optional[str] = None) -> None:
        """Attach (and L2-normalize) the node-vector matrix."""
        if vectors is None or getattr(vectors, "size", 0) == 0:
            self.vectors = None
            return
        v = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = v / norms
        self.embed_model = model

    # --- public properties ---------------------------------------------------
    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.triples)

    @property
    def has_vectors(self) -> bool:
        return self.vectors is not None and self.vectors.shape[0] == len(self.nodes)

    # --- retrieval -----------------------------------------------------------
    def _semantic_scores(self, query: str, query_vec: Optional[np.ndarray]):
        """Per-node relevance to the query in [0,1].

        Returns ``(scores, used_vectors)``. Uses cosine similarity when vectors +
        a usable query vector are available; otherwise falls back to lexical
        token overlap (``used_vectors=False``) so the caller can report the mode
        accurately even when it silently fell back (e.g. a dim mismatch).
        """
        n = len(self.nodes)
        if n == 0:
            return np.zeros((0,), dtype=np.float32), False
        if self.has_vectors and query_vec is not None and getattr(query_vec, "size", 0):
            q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
            qn = np.linalg.norm(q)
            if qn > 0 and q.shape[0] == self.vectors.shape[1]:
                sims = self.vectors @ (q / qn)  # cosine; both are L2-normalized
                return ((sims + 1.0) / 2.0).astype(np.float32), True  # [-1,1]->[0,1]
        # Lexical fallback.
        q_tokens = _tokens(query)
        scores = np.zeros((n,), dtype=np.float32)
        if not q_tokens:
            return scores, False
        for i, node in enumerate(self.nodes):
            nt = _tokens(node["name"])
            if nt:
                scores[i] = len(q_tokens & nt) / float(len(q_tokens | nt))
        return scores, False

    def _multi_source_paths(self, seeds: list, hops: int) -> dict:
        """Multi-source BFS from ``seeds`` up to ``hops``.

        Returns ``{node_id: (distance, source_seed_id, [triple_index,...])}``
        where the triple list is the chain of edges from the nearest seed to the
        node (empty for the seeds themselves).
        """
        result: dict = {}
        prev: dict = {}
        queue: deque = deque()
        for s in seeds:
            if s not in result:
                result[s] = (0, s, [])
                queue.append(s)
        while queue:
            u = queue.popleft()
            dist, src, _ = result[u]
            if dist >= hops:
                continue
            for nb_id, tindex in self.adjacency.get(u, []):
                if nb_id not in result:
                    prev[nb_id] = (u, tindex)
                    result[nb_id] = (dist + 1, src, [])
                    queue.append(nb_id)
        # Reconstruct triple chains via predecessors.
        for nid in list(result.keys()):
            dist, src, _ = result[nid]
            chain = []
            cur = nid
            while cur in prev:
                u, tindex = prev[cur]
                chain.append(tindex)
                cur = u
            chain.reverse()
            result[nid] = (dist, src, chain)
        return result

    def search(
        self,
        query: str,
        *,
        api_key=None,
        query_vec: Optional[np.ndarray] = None,
        embed_model=None,
        top_k: int = 6,
        hops: int = 2,
        max_results: int = 10,
        timeout: float = 60,
    ) -> dict:
        """Hybrid retrieval: semantic seeds + graph traversal -> explainable answers.

        1. Score every node's semantic relevance to ``query`` (cosine over
           embeddings, else lexical).
        2. Take the ``top_k`` strongest as SEED nodes.
        3. Traverse up to ``hops`` from the seeds to gather connected entities +
           the linking triples.
        4. Rank candidates by ``semantic + graph-proximity`` and return each
           answer with its supporting path (chain of triples) and page refs.

        If ``query_vec`` is given it is used directly; otherwise, when an
        ``api_key`` is supplied and the graph has vectors, the query is embedded
        via Gemini (RETRIEVAL_QUERY). With neither, search degrades to lexical.
        """
        query = str(query or "").strip()
        out = {
            "query": query,
            "answers": [],
            "seeds": [],
            "triples": [],
            "mode": "lexical",
        }
        if not query or not self.nodes:
            return out

        # Obtain a query vector if we can (don't fail search if embedding fails).
        if query_vec is None and self.has_vectors and is_configured(api_key):
            try:
                mat = embed_texts(
                    [query],
                    api_key=api_key,
                    model=embed_model or self.embed_model,
                    task_type="RETRIEVAL_QUERY",
                    timeout=timeout,
                )
                if mat.size:
                    query_vec = mat[0]
            except RuntimeError:
                query_vec = None

        sem, used_vectors = self._semantic_scores(query, query_vec)
        out["mode"] = "semantic" if used_vectors else "lexical"
        if sem.size == 0 or float(sem.max()) <= 0:
            return out

        # Seeds = strongest semantic matches.
        order = np.argsort(-sem)
        seeds = [int(i) for i in order[:max(1, top_k)] if sem[int(i)] > 0]
        out["seeds"] = [self.nodes[i]["name"] for i in seeds]

        reach = self._multi_source_paths(seeds, hops=max(0, hops))

        # Rank reachable candidates.
        W_SEM, W_GRAPH, DECAY = 0.6, 0.4, 0.55
        scored = []
        for nid, (dist, src, chain) in reach.items():
            sim = float(sem[nid])
            graph_term = float(sem[src]) * (DECAY ** dist)
            score = W_SEM * sim + W_GRAPH * graph_term
            scored.append((score, dist, nid, src, chain))
        scored.sort(key=lambda x: (-x[0], x[1]))

        used_triples: dict = {}
        for score, dist, nid, src, chain in scored[:max(1, max_results)]:
            node = self.nodes[nid]
            path_triples = [self._triple_view(t) for t in chain]
            path_pages = sorted({t["page"] for t in path_triples if t["page"] is not None})
            # The entity's own incident facts make even a seed (empty path)
            # explainable; cap a few of the highest-information ones.
            fact_idx = [tindex for _, tindex in self.adjacency.get(nid, [])][:6]
            facts = [self._triple_view(t) for t in fact_idx]
            for t in list(chain) + fact_idx:
                used_triples[t] = self._triple_view(t)
            out["answers"].append(
                {
                    "entity": node["name"],
                    "type": node["type"],
                    "score": round(float(score), 4),
                    "similarity": round(float(sem[nid]), 4),
                    "hops": int(dist),
                    "pages": sorted(node["pages"]),
                    "path": path_triples,
                    "path_pages": path_pages,
                    "facts": facts,
                }
            )

        # Flat, de-duplicated list of every supporting triple referenced above.
        out["triples"] = list(used_triples.values())
        return out

    def _triple_view(self, tindex: int) -> dict:
        t = self.triples[tindex]
        return {
            "subject": t["subject"],
            "predicate": t["predicate"],
            "object": t["object"],
            "page": t["page"],
        }

    # --- serialization (explicit; the graph is NEVER auto-serialized) --------
    def to_viz_dict(self, max_nodes: int = 120) -> dict:
        """Compact node/edge lists for the vanilla-canvas visualization.

        Caps to the highest-degree ``max_nodes`` so a huge graph stays drawable
        (and the dropped count is reported, never silently truncated).
        """
        order = sorted(
            range(len(self.nodes)), key=lambda i: self.nodes[i]["degree"], reverse=True
        )
        keep = set(order[:max_nodes])
        nodes = [
            {
                "id": i,
                "name": self.nodes[i]["name"],
                "type": self.nodes[i]["type"],
                "pages": sorted(self.nodes[i]["pages"]),
                "degree": self.nodes[i]["degree"],
            }
            for i in sorted(keep)
        ]
        edges = [
            {
                "source": t["subject_id"],
                "target": t["object_id"],
                "predicate": t["predicate"],
                "page": t["page"],
            }
            for t in self.triples
            if t["subject_id"] in keep and t["object_id"] in keep
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(self.nodes),
            "total_edges": len(self.triples),
            "truncated": len(self.nodes) > len(keep),
        }

    def to_export_dict(self) -> dict:
        """Full graph for the .kg.json export (entities + triples + provenance)."""
        return {
            "entities": [
                {"name": n["name"], "type": n["type"], "pages": sorted(n["pages"])}
                for n in self.nodes
            ],
            "triples": [self._triple_view(i) for i in range(len(self.triples))],
            "node_count": len(self.nodes),
            "triple_count": len(self.triples),
            "embed_model": self.embed_model,
        }


def build_graph(
    pages,
    *,
    api_key,
    triple_model=None,
    embed_model=None,
    max_pages: Optional[int] = None,
    embed: bool = True,
    timeout: float = 120,
) -> KnowledgeGraph:
    """Build a per-document :class:`KnowledgeGraph` from extracted page text.

    Parameters
    ----------
    pages : list[tuple[int, str]]
        ``(page_number, text)`` pairs — typically
        ``[(p.page, p.text) for p in job.pages if p.status == "done"]``.
    api_key : str
        A Gemini API key (required; this is the opt-in online feature).
    triple_model, embed_model : str, optional
        Model overrides; default to Flash + a discovered embedding model
        (``_pick_embedding_model``, prefers gemini-embedding-001).
    max_pages : int, optional
        Cap the number of non-empty pages processed (cost control). None = all.
    embed : bool
        If True (default) also embed the nodes so semantic search works; if
        False the graph supports lexical search only (cheaper).

    Per-page triple extraction is fanned across a shared bounded pool
    (``_EXTRACT_POOL``), so a multi-page build is much faster on a key with real
    throughput; the shared pool also caps total concurrent Gemini calls. Each
    page is isolated — a single failed page is skipped and counted
    (``pages_failed``); only an all-pages failure raises. Embedding failure is
    non-fatal (``embed_error`` set, lexical search retained).

    Returns a graph (possibly empty if the document yields no facts). Raises a
    single-line RuntimeError only for a missing key or a hard Gemini failure.
    """
    if not is_configured(api_key):
        raise RuntimeError(
            "Building a knowledge graph needs a Gemini API key. Enter your free "
            "key from https://aistudio.google.com/, or use the demo key."
        )

    usable = [(int(pn), str(txt or "")) for pn, txt in pages if str(txt or "").strip()]
    if max_pages is not None and max_pages > 0:
        usable = usable[:max_pages]

    graph = KnowledgeGraph()
    last_error: Optional[Exception] = None

    # Fan the per-page Gemini calls across the shared pool (concurrency), then
    # build the graph single-threaded below in PAGE ORDER so node ids stay
    # deterministic and the adjacency dicts are never mutated from two threads.
    def _extract_one(text):
        return extract_triples(text, api_key=api_key, model=triple_model, timeout=timeout)

    futures = [(page_no, _EXTRACT_POOL.submit(_extract_one, text)) for page_no, text in usable]
    results: dict = {}
    for page_no, fut in futures:
        # Isolate each page: one rate-limited / safety-blocked page (common on the
        # free tier) must NOT discard every other page (and its API spend).
        try:
            results[page_no] = fut.result()
        except RuntimeError as exc:
            results[page_no] = exc
            last_error = exc

    for page_no, _text in usable:
        data = results.get(page_no)
        if not isinstance(data, dict):  # a skipped page (exception) or missing
            graph.pages_failed += 1
            continue
        graph.pages_built += 1
        # Register typed entities first so types are known, then triples.
        for e in data.get("entities", []):
            graph._add_node(e["name"], e.get("type", "OTHER"), page=page_no)
        for t in data.get("triples", []):
            graph._add_triple(t["subject"], t["predicate"], t["object"], page=page_no)

    # If EVERY page failed, this is a systemic failure (bad key / quota / network),
    # not "a document with no facts" — surface it instead of returning an empty graph.
    if usable and graph.pages_failed == len(usable):
        raise last_error or RuntimeError(
            "Knowledge-graph extraction failed for every page."
        )

    if embed and graph.num_nodes:
        # Resolve the embedding model the key can actually use (Google retires
        # ids over time), so search uses the SAME model the nodes were built with.
        # Embeddings are the SEMANTIC half: if they fail (quota / retired model),
        # keep the graph for LEXICAL search rather than throwing away the (costly)
        # triple extraction. The degradation is surfaced via has_vectors/embed_error.
        try:
            resolved = _pick_embedding_model(api_key, embed_model, timeout=min(timeout, 30))
            node_texts = [graph._node_embed_text(i) for i in range(graph.num_nodes)]
            vectors = embed_texts(
                node_texts,
                api_key=api_key,
                model=resolved,
                task_type="RETRIEVAL_DOCUMENT",
                timeout=timeout,
            )
            graph.set_vectors(vectors, model=resolved)
        except RuntimeError as exc:
            graph.embed_error = str(exc)

    return graph
