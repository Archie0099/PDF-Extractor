"""Async job manager for PDF extraction.

One job per uploaded file. Heavy CPU work (rendering + OCR) is serialized
through a SINGLE-worker ThreadPoolExecutor via loop.run_in_executor so OCR
never runs concurrently. Per-job progress is pushed to an asyncio.Queue that
is consumed by the SSE endpoint.

Implements the project contract exactly:
  Job dataclass + to_dict()
  JobManager.create_job / get / run / cancel / events / render_page_png
  manager = JobManager()  module-level singleton
"""

import asyncio
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import fitz  # type: ignore

from pipeline.extractor import extract_page


# Single-worker executor: serializes all heavy CPU work (OCR is not
# thread-safe and we want deterministic, low-memory CPU usage).
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr-worker")


# --- Job retention / eviction policy (in-memory only; single-user local) ------
# A finished job keeps its full pdf_bytes in RAM. To bound memory across a long
# session we evict the oldest / expired *finished, unsubscribed* jobs. Jobs that
# are still pending/processing, or that have a live SSE subscriber, are NEVER
# eviction candidates regardless of these limits.
MAX_JOBS = 50                       # keep at most this many evictable jobs
JOB_TTL_SECONDS = 2 * 60 * 60       # evict finished+idle jobs older than this
_SWEEP_MIN_INTERVAL = 5.0           # throttle: don't sweep on get() more than
                                    # once per this many seconds (monotonic)

_TERMINAL_STATES = ("done", "error", "cancelled")


@dataclass
class PageResult:
    page: int  # 1-based
    source: Optional[str] = None  # "text" | "ocr" | None
    text: str = ""
    status: str = "pending"  # "pending" | "done" | "error"
    error: Optional[str] = None
    confidence: Optional[float] = None  # mean OCR confidence [0,1]; None for text-layer
    lines: Optional[list] = None  # per-line [{text, confidence}] for OCR pages (else None)

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "source": self.source,
            "text": self.text,
            "status": self.status,
            "error": self.error,
            "confidence": self.confidence,
            "lines": self.lines,
        }


@dataclass
class Job:
    job_id: str
    filename: str
    pdf_bytes: bytes
    mode: str
    lang: str
    preprocess: bool
    binarize: bool
    handwriting: bool
    online: bool
    online_key: str  # NOT serialized in to_dict() — never leaked to the API
    online_model: str
    force_ocr: bool
    remove_headers: bool
    total_pages: int
    pages: list = field(default_factory=list)  # list[PageResult]
    status: str = "pending"  # pending|processing|done|error|cancelled
    processed_pages: int = 0
    error: Optional[str] = None

    # Optional knowledge graph (opt-in, online). Built on demand from this job's
    # extracted pages and attached here, so it is reclaimed with the job by the
    # TTL/LRU eviction below and — because to_dict() is an explicit allow-list —
    # is NEVER serialized to the client or broadcast over SSE (same protection
    # the API key gets). Held loosely as ``object`` to keep jobs.py free of the
    # kg / numpy import at module load.
    knowledge_graph: object = field(default=None, repr=False)
    kg_status: str = field(default="none", repr=False)  # none|building|ready|error
    kg_error: Optional[str] = field(default=None, repr=False)

    # Internals (not serialized). One asyncio.Queue per live SSE subscriber so
    # progress is broadcast (fan-out), not consumed once — reconnects are safe.
    subscribers: list = field(default_factory=list, repr=False)
    cancel_event: "threading.Event" = field(default_factory=threading.Event, repr=False)
    # Eviction bookkeeping (monotonic clock; immune to wall-clock changes).
    created_at: float = field(default_factory=time.monotonic, repr=False)
    last_access: float = field(default_factory=time.monotonic, repr=False)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "status": self.status,
            "total_pages": self.total_pages,
            "processed_pages": self.processed_pages,
            "pages": [p.to_dict() for p in self.pages],
            "error": self.error,
        }


def _try_repair(pdf_bytes: bytes):
    """Attempt to repair a PDF fitz can't open, using pikepdf. Returns clean
    bytes on success, or None if repair is unavailable/failed."""
    try:
        import io as _io
        import pikepdf  # optional; only used as a fallback

        out = _io.BytesIO()
        with pikepdf.open(_io.BytesIO(pdf_bytes)) as pdf:
            pdf.save(out)
        return out.getvalue()
    except Exception:
        return None


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        # Throttle for get()-triggered sweeps (guarded by self._lock).
        self._last_sweep: float = 0.0

    # --- eviction internals --------------------------------------------------
    @staticmethod
    def _is_evictable(job: "Job") -> bool:
        """A job may be evicted only if it has reached a terminal state AND has
        no live SSE subscriber. pending/processing or subscribed jobs are never
        touched, so eviction can't break an in-flight run or an open stream.

        ``job.subscribers`` is mutated only on the event loop (events()); reading
        its length here under self._lock is a plain, atomic CPython list read.
        A momentarily-stale read is safe: worst case we *skip* evicting a job
        that just lost its last subscriber, deferring it to the next sweep.
        """
        return job.status in _TERMINAL_STATES and not job.subscribers

    def _evict_locked(self, now: float) -> None:
        """Evict expired, then over-cap, evictable jobs. MUST hold self._lock.

        1. TTL: drop evictable jobs whose last_access is older than the TTL.
        2. Cap: if more than MAX_JOBS evictable jobs remain, drop the
           least-recently-accessed ones until at most MAX_JOBS are left.
        Non-evictable (active/subscribed) jobs are never removed and do not
        count against the cap as targets — they're simply left in place.
        """
        self._last_sweep = now

        # Pass 1 — TTL on evictable jobs.
        if JOB_TTL_SECONDS > 0:
            expired = [
                jid
                for jid, job in self._jobs.items()
                if self._is_evictable(job)
                and (now - job.last_access) > JOB_TTL_SECONDS
            ]
            for jid in expired:
                del self._jobs[jid]

        # Pass 2 — count cap on remaining evictable jobs (LRU by last_access).
        if MAX_JOBS >= 0:
            evictable = [
                (job.last_access, jid)
                for jid, job in self._jobs.items()
                if self._is_evictable(job)
            ]
            overflow = len(evictable) - MAX_JOBS
            if overflow > 0:
                evictable.sort()  # oldest last_access first
                for _, jid in evictable[:overflow]:
                    job = self._jobs.get(jid)
                    if job is not None and self._is_evictable(job):
                        del self._jobs[jid]

    def create_job(
        self,
        filename: str,
        pdf_bytes: bytes,
        *,
        mode: str,
        lang: str,
        preprocess: bool,
        binarize: bool = False,
        handwriting: bool = False,
        online: bool = False,
        online_key: str = "",
        online_model: str = "",
        force_ocr: bool = False,
        remove_headers: bool = True,
    ) -> Job:
        """Open the PDF to validate it and count pages, then register a Job.

        Raises ValueError("encrypted") for password-protected PDFs and
        ValueError(<msg>) for corrupt / unreadable PDFs.
        """
        usable_bytes = pdf_bytes
        try:
            doc = fitz.open(stream=usable_bytes, filetype="pdf")
        except Exception as exc:  # corrupt / not a PDF — try to repair once
            repaired = _try_repair(pdf_bytes)
            if repaired is None:
                raise ValueError(f"corrupt: {exc}") from exc
            usable_bytes = repaired
            try:
                doc = fitz.open(stream=usable_bytes, filetype="pdf")
            except Exception as exc2:
                raise ValueError(f"corrupt: {exc2}") from exc2

        try:
            if doc.needs_pass:
                raise ValueError("encrypted")
            try:
                total_pages = doc.page_count
            except Exception as exc:
                raise ValueError(f"corrupt: {exc}") from exc
            if total_pages <= 0:
                raise ValueError("corrupt: no pages")
        finally:
            doc.close()

        job_id = uuid.uuid4().hex
        job = Job(
            job_id=job_id,
            filename=filename,
            pdf_bytes=usable_bytes,
            mode=mode,
            lang=lang,
            preprocess=preprocess,
            binarize=binarize,
            handwriting=handwriting,
            online=online,
            online_key=online_key,
            online_model=online_model,
            force_ocr=force_ocr,
            remove_headers=remove_headers,
            total_pages=total_pages,
            pages=[PageResult(page=i + 1) for i in range(total_pages)],
            status="pending",
        )

        with self._lock:
            self._evict_locked(time.monotonic())  # reclaim stale jobs first
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        now = time.monotonic()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                # Touch: keeps actively-used jobs (polling, image render, SSE
                # reconnect) from aging out under the TTL.
                job.last_access = now
            # Throttled TTL sweep so idle sessions still reclaim memory even
            # with no new uploads, without sweeping on every single get().
            if now - self._last_sweep >= _SWEEP_MIN_INTERVAL:
                self._evict_locked(now)
            return job

    @staticmethod
    def _broadcast(job: Job, event: dict) -> None:
        """Fan an event out to every live SSE subscriber's queue.

        Runs on the event loop; queues are unbounded so put_nowait never
        blocks. Each SSE connection drains its own queue independently.
        """
        for q in list(job.subscribers):
            try:
                q.put_nowait(event)
            except Exception:
                pass

    async def run(self, job: Job) -> None:
        """Process every page of the job sequentially.

        Each page's extraction runs in the single-worker executor. The cancel
        flag is checked between pages. Progress and terminal events are
        broadcast to all subscriber queues for SSE consumption. The whole body
        is guarded so the job always reaches a terminal state (no hung SSE).
        """
        loop = asyncio.get_running_loop()
        job.status = "processing"

        try:
            # Open one document for the lifetime of the run. Do it ON the single
            # OCR worker thread (not the event loop) so that every fitz call in
            # the app — open, load_page, get_pixmap, close — happens on that one
            # thread. PyMuPDF/MuPDF is not thread-safe even across separate
            # Document objects, and concurrent multi-file uploads would otherwise
            # let this run() open a doc on the event loop while another job
            # renders on the worker thread.
            try:
                doc = await loop.run_in_executor(
                    _EXECUTOR,
                    lambda: fitz.open(stream=job.pdf_bytes, filetype="pdf"),
                )
            except Exception as exc:
                job.status = "error"
                job.error = f"corrupt: {exc}"
                self._broadcast(job, {"type": "error", "error": job.error})
                return

            try:
                for idx in range(job.total_pages):
                    if job.cancel_event.is_set():
                        job.status = "cancelled"
                        self._broadcast(job, {"type": "cancelled"})
                        return

                    page_result = job.pages[idx]

                    def _work(page_index: int = idx) -> dict:
                        page = doc.load_page(page_index)
                        return extract_page(
                            page,
                            mode=job.mode,
                            lang=job.lang,
                            preprocess=job.preprocess,
                            binarize=job.binarize,
                            handwriting=job.handwriting,
                            online=job.online,
                            online_key=job.online_key,
                            online_model=job.online_model,
                            force_ocr=job.force_ocr,
                        )

                    try:
                        result = await loop.run_in_executor(_EXECUTOR, _work)
                        page_result.source = result.get("source")
                        page_result.text = result.get("text", "") or ""
                        page_result.confidence = result.get("confidence")
                        page_result.lines = result.get("lines")
                        page_result.status = "done"
                        page_result.error = None
                    except Exception as exc:  # per-page failure: keep going
                        page_result.source = None
                        page_result.text = ""
                        page_result.confidence = None
                        page_result.lines = None
                        page_result.status = "error"
                        page_result.error = str(exc)

                    job.processed_pages += 1

                    self._broadcast(
                        job,
                        {
                            "type": "progress",
                            "processed": job.processed_pages,
                            "total": job.total_pages,
                            "page": page_result.to_dict(),
                        },
                    )

                # Cross-page cleanup: strip running headers/footers + page
                # numbers now that every page's text is available. Best-effort;
                # never fail the job over it. Updated pages are re-broadcast so
                # the live UI (and reconnect snapshot/exports) show cleaned text.
                if job.remove_headers and not job.cancel_event.is_set():
                    try:
                        from pipeline.postprocess import (
                            strip_running_headers_footers,
                            realign_line_confidences,
                        )
                        originals = [p.text or "" for p in job.pages]
                        cleaned, kept_indices = strip_running_headers_footers(
                            originals, return_kept=True
                        )
                        for p, new_text, kept in zip(
                            job.pages, cleaned, kept_indices
                        ):
                            if p.status == "done" and new_text != (p.text or ""):
                                if p.lines:
                                    # An OCR page's per-line list is 1:1 with its
                                    # text lines, so map confidences to the EXACT
                                    # surviving lines by their original index
                                    # (correct even for identical-text duplicate
                                    # lines). Fall back to the text matcher only
                                    # if the indices don't line up with the
                                    # cleaned text (rare blank-detection edge).
                                    mapped = [
                                        p.lines[i]
                                        for i in kept
                                        if 0 <= i < len(p.lines)
                                    ]
                                    if [
                                        str(m.get("text", "")) for m in mapped
                                    ] == new_text.split("\n"):
                                        p.lines = mapped
                                    else:
                                        p.lines = realign_line_confidences(
                                            new_text, p.lines
                                        )
                                p.text = new_text
                                self._broadcast(
                                    job,
                                    {
                                        "type": "progress",
                                        "processed": job.processed_pages,
                                        "total": job.total_pages,
                                        "page": p.to_dict(),
                                    },
                                )
                    except Exception:
                        pass

                # Completed all pages (unless cancelled above).
                if job.cancel_event.is_set():
                    job.status = "cancelled"
                    self._broadcast(job, {"type": "cancelled"})
                else:
                    # If every page errored we still report done; the page
                    # records carry the per-page error state. A whole-job error
                    # is only for fatal failures handled above.
                    job.status = "done"
                    self._broadcast(job, {"type": "done"})
            finally:
                # Close on the worker thread too (same single-thread invariant),
                # and never let a close error overwrite the terminal status.
                try:
                    await loop.run_in_executor(_EXECUTOR, doc.close)
                except Exception:
                    pass
        except Exception as exc:  # never strand subscribers in a hung stream
            job.status = "error"
            job.error = str(exc)
            self._broadcast(job, {"type": "error", "error": job.error})

    def cancel(self, job_id: str) -> None:
        job = self.get(job_id)
        if job is not None:
            job.cancel_event.set()

    def delete(self, job_id: str) -> bool:
        """Drop a job from the registry and request cancellation.

        Removes the job from ``_jobs`` immediately (so it's no longer findable)
        and sets its cancel flag. The PDF bytes are reclaimed once any in-flight
        page finishes — cancellation is cooperative and checked between pages, so
        a long page already running (e.g. a 120 s online-OCR call) keeps the
        bytes alive until it returns. Idempotent — returns False if unknown.
        """
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is not None:
            job.cancel_event.set()
            return True
        return False

    async def events(self, job_id: str):
        """Async generator yielding SSE event dicts for a job.

        On connect, replays a snapshot of every already-completed page so a
        fresh OR reconnecting client catches up without losing progress. Then
        streams live events from its own private subscriber queue until a
        terminal {"type":"done"|"error"|"cancelled"} event. Duplicate page
        events (snapshot + live) are harmless: the frontend keys pages by
        number and re-renders idempotently.
        """
        job = self.get(job_id)
        if job is None:
            yield {"type": "error", "error": "not found"}
            return

        terminal_types = {"done", "error", "cancelled"}

        # Subscribe BEFORE snapshotting so no event slips through the gap.
        q: "asyncio.Queue" = asyncio.Queue()
        job.subscribers.append(q)
        try:
            # Snapshot: replay completed pages for catch-up / reconnect.
            for p in job.pages:
                if p.status in ("done", "error"):
                    yield {
                        "type": "progress",
                        "processed": job.processed_pages,
                        "total": job.total_pages,
                        "page": p.to_dict(),
                    }
            if job.status in terminal_types:
                yield {"type": job.status, "error": job.error}
                return

            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # If the job finished while we waited and our queue drained,
                    # synthesize the terminal event so the client closes.
                    if job.status in terminal_types and q.empty():
                        yield {"type": job.status, "error": job.error}
                        return
                    continue

                yield event
                if event.get("type") in terminal_types:
                    return
        finally:
            try:
                job.subscribers.remove(q)
            except ValueError:
                pass

    def render_page_png(self, job_id: str, page_no: int, dpi: int = 150) -> bytes:
        """Render a 1-based page number to PNG bytes on demand."""
        job = self.get(job_id)
        if job is None:
            raise KeyError("job not found")

        doc = fitz.open(stream=job.pdf_bytes, filetype="pdf")
        try:
            if page_no < 1 or page_no > doc.page_count:
                raise KeyError("page out of range")
            page = doc.load_page(page_no - 1)
            zoom = float(dpi) / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            return pix.tobytes("png")
        finally:
            doc.close()


manager = JobManager()
