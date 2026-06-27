"""PDF text extraction app — FastAPI backend.

Run locally with:
    uvicorn app:app --reload
(serves on http://127.0.0.1:8000)

Local and CPU-only by default. An online OCR path (Google Gemini) is optional
and off unless you supply your own API key.
"""

import io
import json
import os
import asyncio
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pipeline.jobs import manager
from pipeline.ocr_engine import engine
from pipeline.export_docx import build_docx
from pipeline.analyze import suggest_settings

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_HTML = STATIC_DIR / "index.html"

app = FastAPI(title="Local PDF Text Extractor")

# Serve static assets (CSS/JS) locally — no external CDN/network dependency.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Hold strong references to fire-and-forget background tasks: asyncio keeps only
# a weak reference to a Task, so a bare create_task() result can in principle be
# GC'd mid-flight ("Task was destroyed but it is pending"). Keeping them here and
# discarding on completion makes the lifetime explicit.
_BG_TASKS: set = set()


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)


def _parse_bool(value: str) -> bool:
    """Parse a form string into a bool. true/1/yes (case-insensitive) -> True."""
    return str(value).strip().lower() in ("true", "1", "yes")


@app.on_event("startup")
async def _startup() -> None:
    # Preload the English PaddleOCR model at startup so the first request is fast
    # (first ever run still downloads model weights from disk cache). This is
    # best-effort: a failed/slow first-run download must NEVER abort startup —
    # text-layer extraction and the online Gemini path don't need PaddleOCR at
    # all. Run it in the OCR worker thread so a slow download can't block the
    # event loop, and swallow any error (OCR lazy-loads on first real use).
    from pipeline.jobs import _EXECUTOR

    async def _safe_warmup() -> None:
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(_EXECUTOR, engine.warmup, "en")
        except Exception:
            pass

    _spawn(_safe_warmup())


@app.get("/")
async def index():
    """Serve the single-page front-end."""
    return FileResponse(str(INDEX_HTML))


# Cache the (deterministic) optional-feature probe so torch/transformers are
# imported at most once.
_CAPABILITIES: dict = {}


@app.get("/api/capabilities")
async def capabilities():
    """Report which optional features are usable in THIS deployment.

    Local handwriting (TrOCR) needs torch + transformers, which the lean hosted
    build omits. The front-end reads this to disable the handwriting toggle
    gracefully and point users to online OCR, instead of letting a page fail
    mid-extraction with a missing-dependency error.
    """
    if "handwriting" not in _CAPABILITIES:
        from pipeline.jobs import _EXECUTOR
        from pipeline.handwriting import is_available

        loop = asyncio.get_running_loop()
        try:
            _CAPABILITIES["handwriting"] = await loop.run_in_executor(
                _EXECUTOR, is_available
            )
        except Exception:
            _CAPABILITIES["handwriting"] = False
    return {
        "handwriting": bool(_CAPABILITIES["handwriting"]),
        # True when this deployment provides a shared Gemini key (a Space secret),
        # so the front-end can offer online OCR without the visitor's own key.
        "online_demo": bool(os.environ.get("DEMO_GEMINI_KEY", "").strip()),
        # Knowledge-graph search is always supported by the backend (stdlib +
        # numpy, no torch/networkx). It needs a Gemini key at build time — the
        # user's own key or, where present, the demo key (online_demo above).
        "knowledge_graph": True,
    }


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    mode: str = Form("max"),
    lang: str = Form("en"),
    preprocess: str = Form("true"),
    binarize: str = Form("false"),
    handwriting: str = Form("false"),
    online: str = Form("false"),
    online_key: str = Form(""),
    online_model: str = Form(""),
    force_ocr: str = Form("false"),
    remove_headers: str = Form("true"),
):
    """Accept a PDF upload, create a job, and kick off background processing."""
    pre = _parse_bool(preprocess)
    binar = _parse_bool(binarize)
    handw = _parse_bool(handwriting)
    onl = _parse_bool(online)
    okey = (online_key or "").strip()
    omodel = (online_model or "").strip()
    force = _parse_bool(force_ocr)
    rm_headers = _parse_bool(remove_headers)

    # Cheap validations BEFORE buffering the whole upload into memory, so a
    # request that will be rejected anyway doesn't pay a full file read.
    #
    # A deployment may provide a shared demo key via the DEMO_GEMINI_KEY env var
    # (set as a Space secret, never committed to the repo). Fall back to it when
    # the visitor turns on online OCR without supplying their own key, so the
    # hosted demo works out of the box. The user's own key always takes
    # precedence; this only fills in a blank.
    if onl and not okey:
        okey = os.environ.get("DEMO_GEMINI_KEY", "").strip()
    if onl and not okey:
        raise HTTPException(
            status_code=400,
            detail=(
                "Online OCR needs a Gemini API key. Enter your free key from "
                "aistudio.google.com, or turn Online OCR off."
            ),
        )

    if handw:
        from pipeline.handwriting import is_available
        if not is_available():
            raise HTTPException(
                status_code=400,
                detail=(
                    "Handwriting mode needs extra packages. In the project venv "
                    "run: pip install transformers torch  (first use also "
                    "downloads the TrOCR model, a few hundred MB)."
                ),
            )

    pdf_bytes = await file.read()

    # create_job opens the PDF with PyMuPDF to validate + count pages. Run it on
    # the single OCR worker thread so ALL fitz access stays on one thread —
    # PyMuPDF is not thread-safe even across separate Documents, and the event
    # loop could otherwise call fitz.open here concurrently with another job's
    # render on the worker thread (multi-file uploads run concurrently).
    from pipeline.jobs import _EXECUTOR
    loop = asyncio.get_running_loop()
    try:
        job = await loop.run_in_executor(
            _EXECUTOR,
            lambda: manager.create_job(
                file.filename or "document.pdf",
                pdf_bytes,
                mode=mode,
                lang=lang,
                preprocess=pre,
                binarize=binar,
                handwriting=handw,
                online=onl,
                online_key=okey,
                online_model=omodel,
                force_ocr=force,
                remove_headers=rm_headers,
            ),
        )
    except ValueError as exc:
        # encrypted / corrupt / otherwise unreadable PDF
        raise HTTPException(status_code=400, detail=str(exc))

    # Launch processing as a background task; progress is streamed over SSE.
    _spawn(manager.run(job))

    return {
        "job_id": job.job_id,
        "filename": job.filename,
        "total_pages": job.total_pages,
        "status": job.status,
    }


@app.post("/api/online/validate")
async def online_validate(api_key: str = Form(...)):
    """Validate a Gemini API key and return the vision models it can use.

    Lets the UI confirm the key works (and populate the model dropdown) before
    the user runs a whole document. Never stores the key server-side.
    """
    from pipeline import online_ocr

    def _check():
        all_models = online_ocr.list_models(api_key)
        supported = [m for m in online_ocr.SUPPORTED_MODELS if m in set(all_models)]
        return {
            "ok": True,
            "supported": supported,
            "recommended": online_ocr.best_available_model(all_models),
        }

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _check)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:  # never surface a bare 500 from a key check
        raise HTTPException(
            status_code=502, detail="Online key check failed unexpectedly."
        )


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    lang: str = Form("en"),
):
    """Sample 1-2 pages and recommend the best settings for THIS document.

    Opt-in and fast. Runs the CPU-bound sweep in the OCR worker thread pool so
    it never stalls the event loop / SSE streams, and so it never runs OCR
    concurrently with a live extraction job (PaddleOCR is not thread-safe).
    """
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="empty file")
    from pipeline.jobs import _EXECUTOR
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _EXECUTOR, lambda: suggest_settings(pdf_bytes, lang=lang)
    )
    return result


@app.post("/api/export/docx")
async def export_docx(payload: dict = Body(...)):
    """Build a .docx from the posted extracted documents and stream it back."""
    documents = payload.get("documents", []) if isinstance(payload, dict) else []
    loop = asyncio.get_running_loop()
    # build_docx is hardened to never raise on a malformed/hostile payload, but
    # keep a belt-and-suspenders guard so a single bad export can't 500.
    try:
        data = await loop.run_in_executor(None, build_docx, documents)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not build the Word document.")
    return StreamingResponse(
        io.BytesIO(data),
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
        headers={"Content-Disposition": 'attachment; filename="extraction.docx"'},
    )


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Return the current full state of a job."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_dict()


@app.get("/api/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    """Stream per-page progress events as Server-Sent Events."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    async def event_generator():
        async for event in manager.events(job_id):
            yield f"data: {json.dumps(event)}\n\n"
            etype = event.get("type")
            if etype in ("done", "error", "cancelled"):
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Request cancellation of a running job."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    manager.cancel(job_id)
    return {"status": "cancelling"}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Remove a job and free its in-memory PDF bytes. Idempotent (always 200)."""
    manager.delete(job_id)
    return {"status": "deleted"}


@app.get("/api/jobs/{job_id}/pages/{n}/image")
async def page_image(job_id: str, n: int, dpi: int = 150):
    """Render a 1-based page to PNG and stream it back."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    # Clamp DPI to a sane range so a degenerate ?dpi= (0, negative, or huge)
    # can't crash the renderer (fitz error / giant allocation) into a 500.
    dpi = max(36, min(600, dpi))
    try:
        # PNG raster + encode is CPU-bound; run it off the event loop. Use the
        # SAME single-worker OCR executor so PyMuPDF rendering never runs
        # concurrently with an OCR page render or another image render (fitz is
        # not formally thread-safe even across separate Document objects).
        from pipeline.jobs import _EXECUTOR
        loop = asyncio.get_running_loop()
        png = await loop.run_in_executor(
            _EXECUTOR, manager.render_page_png, job_id, n, dpi
        )
    except (KeyError, IndexError, ValueError):
        raise HTTPException(status_code=404, detail="page not found")
    except Exception:
        # A PyMuPDF render error (e.g. a page with a malformed embedded image
        # raises fitz.FileDataError / RuntimeError) must not surface as a bare
        # 500 — the frontend <img> onerror handles a non-image response.
        raise HTTPException(status_code=404, detail="could not render page")
    return StreamingResponse(io.BytesIO(png), media_type="image/png")


# ---------------------------------------------------------------------------
# Knowledge graph (opt-in, online) — build a per-document graph from the
# already-extracted text, then answer queries with hybrid semantic + graph-
# traversal retrieval. These endpoints touch NO fitz/PaddleOCR, so they run on
# the DEFAULT thread pool (run_in_executor(None, ...)) — putting them on the
# single OCR worker would needlessly serialize them behind live extraction.
# ---------------------------------------------------------------------------
def _resolve_kg_key(explicit: str, job) -> str:
    """Pick the Gemini key for KG: explicit (browser) -> the job's own key -> demo."""
    return (
        (explicit or "").strip()
        or (getattr(job, "online_key", "") or "").strip()
        or os.environ.get("DEMO_GEMINI_KEY", "").strip()
    )


@app.post("/api/graph/build")
async def graph_build(
    job_id: str = Form(...),
    online_key: str = Form(""),
    online_model: str = Form(""),
    embed: str = Form("true"),
    max_pages: str = Form(""),
):
    """Build a knowledge graph from a finished job's extracted text."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status not in ("done", "error"):
        raise HTTPException(
            status_code=409,
            detail="Extraction is still running; wait for it to finish first.",
        )
    if job.kg_status == "building":
        raise HTTPException(
            status_code=409,
            detail="A knowledge graph is already being built for this document.",
        )

    okey = _resolve_kg_key(online_key, job)
    if not okey:
        raise HTTPException(
            status_code=400,
            detail=(
                "Building a knowledge graph needs a Gemini API key. Enter your "
                "free key from aistudio.google.com (the same key online OCR uses)."
            ),
        )

    pages = [
        (p.page, p.text)
        for p in job.pages
        if p.status == "done" and (p.text or "").strip()
    ]
    if not pages:
        raise HTTPException(
            status_code=400, detail="No extracted text to build a graph from."
        )

    mp = None
    if (max_pages or "").strip():
        try:
            mp = max(0, int(max_pages))
        except ValueError:
            mp = None
    do_embed = _parse_bool(embed)

    from pipeline import kg as kgmod

    job.kg_status = "building"
    job.kg_error = None
    loop = asyncio.get_running_loop()
    try:
        graph = await loop.run_in_executor(
            None,
            lambda: kgmod.build_graph(
                pages,
                api_key=okey,
                triple_model=(online_model or None),
                max_pages=mp,
                embed=do_embed,
            ),
        )
        # Set the terminal state INSIDE the try so the finally below sees it.
        job.knowledge_graph = graph
        job.kg_status = "ready"
    except RuntimeError as exc:
        # A Gemini/key/quota/network failure — actionable, single-line message.
        job.kg_status = "error"
        job.kg_error = str(exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        job.kg_status = "error"
        job.kg_error = "Knowledge-graph build failed unexpectedly."
        raise HTTPException(
            status_code=502, detail="Knowledge-graph build failed unexpectedly."
        )
    finally:
        # A client disconnect raises asyncio.CancelledError — a BaseException
        # that bypasses the except clauses above (they catch only Exception).
        # Without this, kg_status would stay stuck on "building" and 409-lock
        # every future rebuild of this document until it's evicted. Reset it.
        if job.kg_status == "building":
            job.kg_status = "error"
            job.kg_error = "Build was interrupted before it finished; try again."

    return {
        "job_id": job_id,
        "status": "ready",
        "nodes": graph.num_nodes,
        "edges": graph.num_edges,
        "has_vectors": graph.has_vectors,
        "pages_built": graph.pages_built,
        "pages_failed": graph.pages_failed,
        "embed_error": graph.embed_error,
    }


@app.post("/api/graph/query")
async def graph_query(
    job_id: str = Form(...),
    query: str = Form(...),
    online_key: str = Form(""),
    top_k: int = Form(6),
    hops: int = Form(2),
):
    """Answer a query against a built graph: ranked entities + supporting paths."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    graph = job.knowledge_graph
    if graph is None:
        raise HTTPException(
            status_code=409, detail="No knowledge graph has been built for this document yet."
        )
    q = (query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Enter a question to search the graph.")

    okey = _resolve_kg_key(online_key, job)  # optional: enables semantic search
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: graph.search(
                q,
                api_key=(okey or None),
                top_k=max(1, min(20, int(top_k))),
                hops=max(0, min(3, int(hops))),
            ),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=502, detail="Graph query failed unexpectedly.")
    return result


@app.get("/api/graph/{job_id}")
async def graph_get(job_id: str, full: bool = False):
    """Return the built graph.

    Default: compact node/edge lists for the canvas visualization (capped).
    ``?full=1``: the complete entities + triples + provenance for .json export.
    """
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    graph = job.knowledge_graph
    if graph is None:
        return {
            "status": job.kg_status,
            "error": job.kg_error,
            "nodes": [],
            "edges": [],
            "total_nodes": 0,
            "total_edges": 0,
            "truncated": False,
        }
    if full:
        data = graph.to_export_dict()
        data["status"] = "ready"
        return data
    data = graph.to_viz_dict()
    data["status"] = "ready"
    return data
