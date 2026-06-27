"use strict";

/* ============================================================
   Local PDF Extractor — frontend logic (dependency-free)
   ============================================================ */

(function () {
  // ---------- State ----------
  const settings = { mode: "max", lang: "en", preprocess: true, binarize: false, handwriting: false, online: false, online_key: "", online_model: "", force_ocr: false, remove_headers: true };

  // jobId -> { jobId, filename, totalPages, status, pages: Map<pageNo, pageData>,
  //            el: {...}, source, es: EventSource }
  const jobs = new Map();
  let jobOrder = []; // preserve display order

  // OCR lines below this confidence are highlighted as "to check".
  const LOWCONF_THRESH = 0.85;

  // ---------- Element refs ----------
  const $ = (id) => document.getElementById(id);

  const modeToggle = $("modeToggle");
  const modeHint = $("modeHint");
  const langSelect = $("langSelect");
  const preprocessToggle = $("preprocessToggle");
  const binarizeToggle = $("binarizeToggle");
  const forceOcrToggle = $("forceOcrToggle");
  const removeHeadersToggle = $("removeHeadersToggle");
  const handwritingToggle = $("handwritingToggle");
  const onlineToggle = $("onlineToggle");
  const onlineRow = $("onlineRow");
  const onlineKey = $("onlineKey");
  const onlineCheck = $("onlineCheck");
  const onlineStatus = $("onlineStatus");
  const onlineModel = $("onlineModel");
  const dropzone = $("dropzone");
  const fileInput = $("fileInput");
  const toolbar = $("toolbar");
  const emptyState = $("emptyState");
  const filesContainer = $("filesContainer");
  const searchInput = $("searchInput");
  const searchCount = $("searchCount");
  const copyAllBtn = $("copyAllBtn");
  const toast = $("toast");

  const fileCardTpl = $("fileCardTemplate");
  const pageCardTpl = $("pageCardTemplate");

  // ---------- Settings UI ----------
  modeToggle.addEventListener("click", (e) => {
    const btn = e.target.closest(".seg");
    if (!btn) return;
    settings.mode = btn.dataset.mode;
    modeToggle.querySelectorAll(".seg").forEach((b) => {
      const active = b === btn;
      b.classList.toggle("active", active);
      b.setAttribute("aria-checked", active ? "true" : "false");
    });
    modeHint.textContent =
      settings.mode === "max"
        ? "Max accuracy is slower but renders at higher DPI and uses angle correction."
        : "Faster mode renders at lower DPI and skips angle correction — quicker, slightly less accurate.";
  });

  langSelect.addEventListener("change", () => {
    settings.lang = langSelect.value;
  });

  preprocessToggle.addEventListener("change", () => {
    settings.preprocess = preprocessToggle.checked;
    // Binarize only makes sense when preprocessing is on.
    binarizeToggle.disabled = !preprocessToggle.checked;
  });

  binarizeToggle.addEventListener("change", () => {
    settings.binarize = binarizeToggle.checked;
  });

  if (forceOcrToggle) {
    forceOcrToggle.addEventListener("change", () => {
      settings.force_ocr = forceOcrToggle.checked;
    });
  }

  if (removeHeadersToggle) {
    settings.remove_headers = removeHeadersToggle.checked;
    removeHeadersToggle.addEventListener("change", () => {
      settings.remove_headers = removeHeadersToggle.checked;
    });
  }

  handwritingToggle.addEventListener("change", () => {
    settings.handwriting = handwritingToggle.checked;
  });

  // ---------- Online vision OCR (Gemini, opt-in) ----------
  const ONLINE_KEY_LS = "pdfx_gemini_key";
  const ONLINE_MODEL_LS = "pdfx_gemini_model";

  if (onlineToggle) {
    // Restore a previously saved key/model (stored only in this browser).
    try {
      const savedKey = localStorage.getItem(ONLINE_KEY_LS) || "";
      if (savedKey) {
        onlineKey.value = savedKey;
        settings.online_key = savedKey;
      }
      const savedModel = localStorage.getItem(ONLINE_MODEL_LS) || "";
      if (savedModel) settings.online_model = savedModel;
    } catch (_) {}

    // Pre-populate the model dropdown with free-tier-friendly defaults so a
    // model is always selectable, even before "Check key" runs (clicking
    // "Check key" then refines the list to exactly what your key can use).
    populateModels(["gemini-2.5-flash", "gemini-2.5-flash-lite"], "gemini-2.5-flash");

    onlineToggle.addEventListener("change", () => {
      settings.online = onlineToggle.checked;
      onlineRow.hidden = !onlineToggle.checked;
    });

    onlineKey.addEventListener("input", () => {
      settings.online_key = onlineKey.value.trim();
      try {
        localStorage.setItem(ONLINE_KEY_LS, settings.online_key);
      } catch (_) {}
      setOnlineStatus("", "");
    });

    onlineModel.addEventListener("change", () => {
      settings.online_model = onlineModel.value;
      try {
        localStorage.setItem(ONLINE_MODEL_LS, settings.online_model);
      } catch (_) {}
    });

    onlineCheck.addEventListener("click", checkOnlineKey);
  }

  function setOnlineStatus(msg, kind) {
    if (!onlineStatus) return;
    onlineStatus.textContent = msg;
    onlineStatus.className = "online-status" + (kind ? " " + kind : "");
  }

  async function checkOnlineKey() {
    const key = (onlineKey.value || "").trim();
    if (!key) {
      setOnlineStatus("Enter a key first.", "bad");
      return;
    }
    onlineCheck.disabled = true;
    setOnlineStatus("Checking…", "");
    const form = new FormData();
    form.append("api_key", key);
    try {
      const res = await fetch("/api/online/validate", { method: "POST", body: form });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setOnlineStatus(data.detail ? shortErr(data.detail) : "Key check failed.", "bad");
        return;
      }
      const models = data.supported && data.supported.length ? data.supported : [];
      populateModels(models, data.recommended);
      setOnlineStatus("✓ Key works · " + (models.length || "?") + " model(s)", "good");
    } catch (e) {
      setOnlineStatus("Network error checking key.", "bad");
    } finally {
      onlineCheck.disabled = false;
    }
  }

  function populateModels(models, recommended) {
    if (!onlineModel) return;
    onlineModel.innerHTML = "";
    const list = models && models.length ? models : recommended ? [recommended] : [];
    list.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      onlineModel.appendChild(opt);
    });
    const saved = settings.online_model;
    const choose =
      saved && list.includes(saved)
        ? saved
        : recommended && list.includes(recommended)
        ? recommended
        : list[0] || "";
    onlineModel.value = choose;
    settings.online_model = choose;
    try {
      localStorage.setItem(ONLINE_MODEL_LS, choose);
    } catch (_) {}
  }

  function shortErr(detail) {
    const s = String(detail);
    return s.length > 90 ? s.slice(0, 90) + "…" : s;
  }

  // ---------- Suggest best settings (opt-in) ----------
  const suggestBtn = $("suggestBtn");
  const suggestInput = $("suggestInput");
  const suggestResult = $("suggestResult");
  const suggestApply = $("suggestApply");
  let lastRecommended = null;

  if (suggestBtn) {
    suggestBtn.addEventListener("click", () => suggestInput.click());
    suggestInput.addEventListener("change", () => {
      const f = suggestInput.files && suggestInput.files[0];
      suggestInput.value = "";
      if (f) analyzeFile(f);
    });
    suggestApply.addEventListener("click", () => {
      if (lastRecommended) applyRecommended(lastRecommended);
    });
  }

  async function analyzeFile(file) {
    suggestBtn.disabled = true;
    suggestBtn.textContent = "Analyzing 1–2 pages…";
    suggestResult.hidden = true;
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("lang", settings.lang);
    try {
      const res = await fetch("/api/analyze", { method: "POST", body: form });
      if (!res.ok) {
        showToast("Analysis failed.");
        return;
      }
      const data = await res.json();
      renderSuggestion(data);
    } catch (e) {
      showToast("Analysis failed (network).");
    } finally {
      suggestBtn.disabled = false;
      suggestBtn.textContent = "Auto-detect best settings";
    }
  }

  function renderSuggestion(data) {
    lastRecommended = data.recommended || null;
    const rat = suggestResult.querySelector('[data-role="suggestRationale"]');
    const ev = suggestResult.querySelector('[data-role="suggestEvidence"]');
    if (data.needs_ocr === false) {
      rat.textContent = data.rationale;
      ev.innerHTML = "";
      suggestApply.hidden = true; // nothing to apply; defaults are fine
    } else {
      rat.textContent = data.rationale || "";
      suggestApply.hidden = false;
      const rows = (data.evidence || []).map((r) => {
        const tag = r.recommended ? " ★" : "";
        const conf = r.mean_conf == null ? "–" : Math.round(r.mean_conf * 100) + "%";
        const row = document.createElement("div");
        row.className = "ev-row" + (r.recommended ? " best" : "");
        [r.name + tag, "score " + r.composite, "conf " + conf, "lines " + r.confident_lines]
          .forEach((t) => {
            const s = document.createElement("span");
            s.textContent = t;
            row.appendChild(s);
          });
        return row;
      });
      ev.innerHTML = "";
      rows.forEach((r) => ev.appendChild(r));
      suggestApply.classList.toggle("btn-primary", data.decision === "decisive");
    }
    suggestResult.hidden = false;
    showToast(
      data.decision === "decisive"
        ? "Found clearly better settings — review and Apply."
        : "Recommendation ready."
    );
  }

  function applyRecommended(rec) {
    if (rec.mode) {
      settings.mode = rec.mode;
      modeToggle.querySelectorAll(".seg").forEach((b) => {
        const active = b.dataset.mode === rec.mode;
        b.classList.toggle("active", active);
        b.setAttribute("aria-checked", active ? "true" : "false");
      });
    }
    if (rec.lang) {
      settings.lang = rec.lang;
      langSelect.value = rec.lang;
    }
    settings.preprocess = !!rec.preprocess;
    preprocessToggle.checked = settings.preprocess;
    settings.binarize = !!rec.binarize;
    binarizeToggle.checked = settings.binarize;
    binarizeToggle.disabled = !settings.preprocess;
    showToast("Applied recommended settings. Now drop your PDF to extract.");
  }

  // ---------- Drag & drop + picker ----------
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => {
    handleFiles(fileInput.files);
    fileInput.value = "";
  });

  ["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "dragend"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer && e.dataTransfer.files) handleFiles(e.dataTransfer.files);
  });

  // Prevent the browser from navigating when a file is dropped outside the zone.
  ["dragover", "drop"].forEach((ev) =>
    window.addEventListener(ev, (e) => {
      if (!dropzone.contains(e.target)) e.preventDefault();
    })
  );

  function handleFiles(fileList) {
    const files = Array.from(fileList).filter(
      (f) => f.type === "application/pdf" || /\.pdf$/i.test(f.name)
    );
    if (!files.length) {
      showToast("Please choose PDF files only.");
      return;
    }
    files.forEach(uploadFile);
  }

  // ---------- Upload ----------
  async function uploadFile(file) {
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("mode", settings.mode);
    form.append("lang", settings.lang);
    form.append("preprocess", settings.preprocess ? "true" : "false");
    form.append("binarize", settings.binarize ? "true" : "false");
    form.append("handwriting", settings.handwriting ? "true" : "false");
    form.append("online", settings.online ? "true" : "false");
    form.append("online_key", settings.online_key || "");
    form.append("online_model", settings.online_model || "");
    form.append("force_ocr", settings.force_ocr ? "true" : "false");
    form.append("remove_headers", settings.remove_headers ? "true" : "false");

    // Optimistic card while uploading.
    const placeholderId = "pending-" + Math.random().toString(36).slice(2);
    const job = createFileCard({
      jobId: placeholderId,
      filename: file.name,
      totalPages: 0,
      status: "pending",
    });
    setFileStatus(job, "pending", "Uploading…");

    let res;
    try {
      res = await fetch("/api/upload", { method: "POST", body: form });
    } catch (err) {
      failJob(job, "Network error during upload: " + (err && err.message ? err.message : err));
      return;
    }

    if (!res.ok) {
      let detail = "Upload failed (" + res.status + ")";
      try {
        const data = await res.json();
        if (data && data.detail) detail = humanizeError(data.detail);
      } catch (_) {}
      failJob(job, detail);
      return;
    }

    let data;
    try {
      data = await res.json();
    } catch (err) {
      failJob(job, "Invalid server response.");
      return;
    }

    // If the user removed this card while it was still uploading, don't
    // resurrect it: free the real server job (now that we know its id) and bail
    // out before re-adding state / opening a stream.
    if (job.removed) {
      const id = data && data.job_id;
      if (id) fetch("/api/jobs/" + encodeURIComponent(id), { method: "DELETE" }).catch(() => {});
      return;
    }

    // Re-key the job from placeholder -> real job_id.
    rekeyJob(job, placeholderId, data.job_id);
    job.jobId = data.job_id;
    job.filename = data.filename || file.name;
    job.totalPages = data.total_pages || 0;
    job.status = data.status || "processing";
    job.el.fileName.textContent = job.filename;
    setFileStatus(job, "processing", "Processing…");
    updateProgress(job, 0, job.totalPages);

    openStream(job);
  }

  function rekeyJob(job, oldId, newId) {
    jobs.delete(oldId);
    jobs.set(newId, job);
    const idx = jobOrder.indexOf(oldId);
    if (idx !== -1) jobOrder[idx] = newId;
  }

  // ---------- SSE stream ----------
  function openStream(job) {
    // Never (re)open a stream for a job the user removed or that's gone from
    // state — a pollOnce reopen could otherwise leak an untracked EventSource.
    if (job.removed || !jobs.has(job.jobId)) return;
    if (job.es) {
      job.es.close();
      job.es = null;
    }
    const es = new EventSource("/api/jobs/" + encodeURIComponent(job.jobId) + "/stream");
    job.es = es;

    es.onmessage = (evt) => {
      let payload;
      try {
        payload = JSON.parse(evt.data);
      } catch (_) {
        return;
      }
      handleEvent(job, payload);
    };

    es.onerror = () => {
      // Network/stream dropped. If the job isn't terminal, fall back to polling once.
      if (["done", "error", "cancelled"].includes(job.status)) {
        es.close();
        return;
      }
      es.close();
      pollOnce(job);
    };
  }

  function handleEvent(job, payload) {
    if (payload.type === "progress") {
      if (payload.page) applyPage(job, payload.page);
      updateProgress(
        job,
        typeof payload.processed === "number" ? payload.processed : countDone(job),
        typeof payload.total === "number" ? payload.total : job.totalPages
      );
    } else if (payload.type === "done") {
      setFileStatus(job, "done", "Done");
      closeStream(job);
      refreshExportState();
    } else if (payload.type === "error") {
      failJob(job, payload.error ? humanizeError(payload.error) : "Processing failed.");
      closeStream(job);
    } else if (payload.type === "cancelled") {
      setFileStatus(job, "cancelled", "Cancelled");
      closeStream(job);
    }
  }

  function closeStream(job) {
    if (job.es) {
      job.es.close();
      job.es = null;
    }
    job.el.cancelBtn.disabled = true;
  }

  // Fallback: fetch the full job state once if the stream errored mid-flight.
  async function pollOnce(job) {
    // Bail if the user removed this card (or it's no longer tracked) so we don't
    // resurrect a removed job's stream or render into a detached DOM subtree.
    if (job.removed || !jobs.has(job.jobId)) return;
    try {
      const res = await fetch("/api/jobs/" + encodeURIComponent(job.jobId));
      if (job.removed || !jobs.has(job.jobId)) return;
      if (!res.ok) {
        if (!["done", "error", "cancelled"].includes(job.status)) {
          failJob(job, "Lost connection to the job stream.");
        }
        return;
      }
      const data = await res.json();
      job.totalPages = data.total_pages || job.totalPages;
      (data.pages || []).forEach((p) => {
        if (p.status === "done" || p.status === "error") applyPage(job, p);
      });
      updateProgress(job, data.processed_pages || 0, job.totalPages);
      if (data.status === "done") setFileStatus(job, "done", "Done");
      else if (data.status === "error") failJob(job, data.error ? humanizeError(data.error) : "Processing failed.");
      else if (data.status === "cancelled") setFileStatus(job, "cancelled", "Cancelled");
      else {
        // still processing — reopen the stream
        openStream(job);
      }
      refreshExportState();
    } catch (err) {
      if (!["done", "error", "cancelled"].includes(job.status)) {
        failJob(job, "Lost connection to the job stream.");
      }
    }
  }

  // ---------- Page rendering ----------
  function applyPage(job, page) {
    const pageNo = page.page;
    let entry = job.pages.get(pageNo);
    if (!entry) {
      entry = buildPageCard(job, pageNo);
      job.pages.set(pageNo, entry);
      insertPageCardInOrder(job, entry);
    }
    entry.data = page;

    // badge
    const badge = entry.el.badge;
    badge.className = "badge";
    if (page.status === "error") {
      badge.classList.add("error");
      badge.textContent = "Error";
    } else if (page.source === "text") {
      badge.classList.add("text");
      badge.textContent = "Text layer";
    } else if (page.source === "handwriting") {
      badge.classList.add("handwriting");
      badge.textContent = "Handwriting";
      badge.title = "Recognised with the local handwriting model (TrOCR).";
    } else if (page.source === "online") {
      badge.classList.add("online");
      badge.textContent = "Gemini";
      badge.title = "Transcribed online by Google Gemini.";
    } else if (page.source === "ocr") {
      badge.classList.add("ocr");
      const conf = typeof page.confidence === "number" ? page.confidence : null;
      if (conf !== null) {
        const pct = Math.round(conf * 100);
        badge.textContent = "OCR · " + pct + "%";
        if (conf < LOWCONF_THRESH) {
          badge.classList.add("lowconf");
          badge.title = "Low OCR confidence — verify this page against the original image.";
        } else {
          badge.title = "Average OCR confidence " + pct + "%";
        }
      } else {
        badge.textContent = "OCR";
      }
    } else {
      badge.classList.add("pending");
      badge.textContent = "Pending";
    }

    // text
    const text = page.status === "error"
      ? (page.error ? "Error: " + page.error : "Extraction failed for this page.")
      : (page.text || "");
    // Preserve a user's manual edits even if the server re-broadcasts this page
    // (e.g. the header/footer cleanup pass at the end of a job). Also don't
    // clobber text while the editor is OPEN (unsaved) — the textarea holds the
    // user's in-progress value.
    if (!entry.edited && !entry.editing) entry.text = text;

    // "N to check": count low-confidence OCR lines (before any manual edit, and
    // not while the editor is open so the chip doesn't flicker back mid-edit).
    let lcCount = 0;
    if (!entry.edited && !entry.editing && page.status === "done" && Array.isArray(page.lines)) {
      lcCount = page.lines.filter(
        (l) => l && typeof l.confidence === "number" && l.confidence < LOWCONF_THRESH
      ).length;
    }
    if (entry.el.toCheck) {
      if (lcCount > 0) {
        entry.el.toCheck.hidden = false;
        entry.el.toCheck.textContent = "⚠ " + lcCount + " to check";
        entry.el.toCheck.title =
          lcCount + " line(s) the OCR is unsure about — highlighted below. Click Edit to fix them.";
      } else {
        entry.el.toCheck.hidden = true;
      }
    }

    const q = searchInput.value.trim();
    if (q) {
      // Re-run the whole search so the match count includes this newly arrived
      // / re-broadcast page (renderPageBody alone wouldn't update searchCount).
      runSearch();
    } else {
      renderPageBody(entry, "");
    }
  }

  function buildPageCard(job, pageNo) {
    const frag = pageCardTpl.content.cloneNode(true);
    const root = frag.querySelector('[data-role="pageCard"]');
    const el = {
      pageNum: frag.querySelector('[data-role="pageNum"]'),
      badge: frag.querySelector('[data-role="badge"]'),
      toCheck: frag.querySelector('[data-role="toCheck"]'),
      pageText: frag.querySelector('[data-role="pageText"]'),
      pageEdit: frag.querySelector('[data-role="pageEdit"]'),
      editBtn: frag.querySelector('[data-role="editBtn"]'),
      toggleImg: frag.querySelector('[data-role="toggleImg"]'),
      copyPage: frag.querySelector('[data-role="copyPage"]'),
      imageWrap: frag.querySelector('[data-role="imageWrap"]'),
    };
    el.pageNum.textContent = "Page " + pageNo;

    const entry = { pageNo, root, el, text: "", data: null, imgLoaded: false, edited: false, editing: false };

    el.copyPage.addEventListener("click", () => {
      copyText(entry.text || "");
    });

    // Edit: swap the read-only text for a textarea; saving updates the page's
    // text so Copy and all exports use the corrected version.
    el.editBtn.addEventListener("click", () => {
      const editing = !el.pageEdit.hidden;
      if (!editing) {
        el.pageEdit.value = entry.text || "";
        el.pageEdit.hidden = false;
        el.pageText.hidden = true;
        el.editBtn.textContent = "Done";
        if (el.toCheck) el.toCheck.hidden = true;
        entry.editing = true;         // a re-broadcast must not overwrite text / reshow the chip mid-edit
        el.pageEdit.focus();
      } else {
        entry.text = el.pageEdit.value;
        entry.edited = true;          // stop low-confidence highlighting; keep edits on re-broadcast
        entry.editing = false;
        el.pageEdit.hidden = true;
        el.pageText.hidden = false;
        el.editBtn.textContent = "Edit";
        const q = searchInput.value.trim();
        if (q) runSearch(); else renderPageBody(entry, "");
        refreshExportState();
        showToast("Saved — Copy and exports now use your edited text.");
      }
    });

    el.toggleImg.addEventListener("click", () => {
      const wrap = el.imageWrap;
      const isHidden = wrap.hasAttribute("hidden");
      if (isHidden) {
        wrap.removeAttribute("hidden");
        el.toggleImg.textContent = "Hide original page";
        if (!entry.imgLoaded) {
          entry.imgLoaded = true;
          wrap.innerHTML = '<div class="img-loading">Rendering original page…</div>';
          const img = new Image();
          img.alt = "Original page " + pageNo;
          img.onload = () => {
            wrap.innerHTML = "";
            wrap.appendChild(img);
          };
          img.onerror = () => {
            wrap.innerHTML = '<div class="img-loading">Could not render this page.</div>';
            entry.imgLoaded = false;
          };
          img.src =
            "/api/jobs/" + encodeURIComponent(job.jobId) + "/pages/" + pageNo + "/image?dpi=150";
        }
      } else {
        wrap.setAttribute("hidden", "");
        el.toggleImg.textContent = "Show original page";
      }
    });

    return entry;
  }

  function insertPageCardInOrder(job, entry) {
    const container = job.el.pages;
    // Insert keeping ascending page order.
    let inserted = false;
    const children = container.children;
    for (let i = 0; i < children.length; i++) {
      const otherNo = Number(children[i].dataset.pageNo);
      if (otherNo > entry.pageNo) {
        container.insertBefore(entry.root, children[i]);
        inserted = true;
        break;
      }
    }
    entry.root.dataset.pageNo = String(entry.pageNo);
    if (!inserted) container.appendChild(entry.root);
  }

  // ---------- File card ----------
  function createFileCard(meta) {
    const frag = fileCardTpl.content.cloneNode(true);
    const el = {
      root: frag.querySelector(".file-card"),
      statusDot: frag.querySelector('[data-role="statusDot"]'),
      fileName: frag.querySelector('[data-role="fileName"]'),
      statusText: frag.querySelector('[data-role="statusText"]'),
      cancelBtn: frag.querySelector('[data-role="cancelBtn"]'),
      removeBtn: frag.querySelector('[data-role="removeBtn"]'),
      progressFill: frag.querySelector('[data-role="progressFill"]'),
      progressLabel: frag.querySelector('[data-role="progressLabel"]'),
      fileError: frag.querySelector('[data-role="fileError"]'),
      pages: frag.querySelector('[data-role="pages"]'),
      // Knowledge graph hooks (all optional; present only in newer markup).
      kgPanel: frag.querySelector('[data-role="kgPanel"]'),
      kgStatus: frag.querySelector('[data-role="kgStatus"]'),
      kgBuildBtn: frag.querySelector('[data-role="kgBuildBtn"]'),
      kgBuildNote: frag.querySelector('[data-role="kgBuildNote"]'),
      kgQueryWrap: frag.querySelector('[data-role="kgQueryWrap"]'),
      kgQueryInput: frag.querySelector('[data-role="kgQueryInput"]'),
      kgAskBtn: frag.querySelector('[data-role="kgAskBtn"]'),
      kgResults: frag.querySelector('[data-role="kgResults"]'),
      kgTools: frag.querySelector('[data-role="kgVizToggle"]'),
      kgVizToggle: frag.querySelector('[data-role="kgVizToggle"]'),
      kgExportBtn: frag.querySelector('[data-role="kgExportBtn"]'),
      kgRebuildBtn: frag.querySelector('[data-role="kgRebuildBtn"]'),
      kgVizWrap: frag.querySelector('[data-role="kgVizWrap"]'),
      kgCanvas: frag.querySelector('[data-role="kgCanvas"]'),
      kgVizNote: frag.querySelector('[data-role="kgVizNote"]'),
    };
    el.fileName.textContent = meta.filename;

    const job = {
      jobId: meta.jobId,
      filename: meta.filename,
      totalPages: meta.totalPages || 0,
      status: meta.status || "pending",
      pages: new Map(),
      es: null,
      el,
    };

    el.cancelBtn.addEventListener("click", () => cancelJob(job));
    if (el.removeBtn) el.removeBtn.addEventListener("click", () => removeJob(job));
    setupKnowledgeGraph(job);

    jobs.set(job.jobId, job);
    jobOrder.push(job.jobId);

    emptyState.hidden = true;
    toolbar.hidden = false;
    filesContainer.appendChild(el.root);
    return job;
  }

  // Remove a file card: stop its stream, free its memory on the server, and
  // drop it from the UI/state. Works at any stage (also cancels if in-flight).
  function removeJob(job) {
    job.removed = true;  // upload continuation checks this to avoid resurrecting it
    closeStream(job);
    if (job.jobId && String(job.jobId).indexOf("pending-") !== 0) {
      // Tell the server to drop the job (frees its in-memory PDF bytes).
      fetch("/api/jobs/" + encodeURIComponent(job.jobId), { method: "DELETE" }).catch(() => {});
    }
    if (job.el && job.el.root && job.el.root.parentNode) {
      job.el.root.parentNode.removeChild(job.el.root);
    }
    jobs.delete(job.jobId);
    const idx = jobOrder.indexOf(job.jobId);
    if (idx !== -1) jobOrder.splice(idx, 1);
    if (jobs.size === 0) {
      emptyState.hidden = false;
      toolbar.hidden = true;
    }
    refreshExportState();
  }

  function setFileStatus(job, status, label) {
    job.status = status;
    job.el.statusText.textContent = label;
    job.el.statusDot.className = "status-dot " + status;
    if (status === "done" || status === "error" || status === "cancelled") {
      job.el.cancelBtn.disabled = true;
    }
    // Reveal the knowledge-graph panel once extraction has finished (there is
    // now text to build from). Only on a successful run.
    if (status === "done" && job.el && job.el.kgPanel) {
      job.el.kgPanel.hidden = false;
    }
  }

  function updateProgress(job, processed, total) {
    total = total || job.totalPages || 0;
    const pct = total > 0 ? Math.min(100, Math.round((processed / total) * 100)) : 0;
    job.el.progressFill.style.width = pct + "%";
    job.el.progressLabel.textContent = processed + " / " + total;
  }

  function failJob(job, message) {
    setFileStatus(job, "error", "Error");
    job.el.fileError.hidden = false;
    job.el.fileError.textContent = message;
    closeStream(job);
  }

  function countDone(job) {
    let n = 0;
    job.pages.forEach((p) => {
      if (p.data && (p.data.status === "done" || p.data.status === "error")) n++;
    });
    return n;
  }

  async function cancelJob(job) {
    if (["done", "error", "cancelled"].includes(job.status)) return;
    job.el.cancelBtn.disabled = true;
    setFileStatus(job, "processing", "Cancelling…");
    try {
      await fetch("/api/jobs/" + encodeURIComponent(job.jobId) + "/cancel", { method: "POST" });
    } catch (_) {
      showToast("Could not reach server to cancel.");
    }
  }

  // ---------- Search & highlight ----------
  let searchTimer = null;
  searchInput.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(runSearch, 120);
  });

  function runSearch() {
    const q = searchInput.value.trim();
    let total = 0;
    jobs.forEach((job) => {
      job.pages.forEach((entry) => {
        total += applySearchToEntry(entry, q);
      });
    });
    if (q) {
      searchCount.hidden = false;
      searchCount.textContent = total + (total === 1 ? " match" : " matches");
    } else {
      searchCount.hidden = true;
    }
  }

  // Render a page's text into its <pre>, applying BOTH low-confidence line
  // highlighting (OCR pages, before the user edits) AND search highlighting.
  // Returns the number of search matches.
  function renderPageBody(entry, q) {
    const target = entry.el.pageText;
    const text = entry.text || "";
    // Keep the element truly :empty for blank pages so the CSS
    // "(no text on this page)" placeholder shows (an empty text node would
    // defeat :empty).
    if (!text) {
      target.textContent = "";
      return 0;
    }
    const ql = (q || "").toLowerCase();
    const lines =
      !entry.edited && entry.data && entry.data.status === "done" &&
      Array.isArray(entry.data.lines)
        ? entry.data.lines
        : null;
    const textLines = text.split("\n");
    const frag = document.createDocumentFragment();
    let count = 0;

    textLines.forEach((ln, i) => {
      const lc =
        lines && lines[i] && typeof lines[i].confidence === "number" &&
        lines[i].confidence < LOWCONF_THRESH;
      let wrap = null;
      let sink = frag;
      if (lc) {
        wrap = document.createElement("span");
        wrap.className = "lc-line";
        wrap.title =
          "Low confidence (" + Math.round(lines[i].confidence * 100) +
          "%) — double-check this line";
        sink = wrap;
      }
      const lower = ln.toLowerCase();
      if (ql && lower.length === ln.length && lower.includes(ql)) {
        // Common case: lowercase preserves length, so indices in `lower` align
        // 1:1 with `ln`. Use ql.length consistently for slice + advance.
        let idx = 0;
        while (true) {
          const found = lower.indexOf(ql, idx);
          if (found === -1) {
            sink.appendChild(document.createTextNode(ln.slice(idx)));
            break;
          }
          if (found > idx) sink.appendChild(document.createTextNode(ln.slice(idx, found)));
          const mark = document.createElement("mark");
          mark.textContent = ln.slice(found, found + ql.length);
          sink.appendChild(mark);
          count++;
          idx = found + ql.length;
        }
      } else if (ql && lower.includes(ql)) {
        // Rare: a character whose lowercase changes length (e.g. Turkish 'İ')
        // makes `lower` indices misalign with `ln`. Count the matches but render
        // the line plain rather than wrapping the wrong characters in <mark>.
        let from = 0, f;
        while ((f = lower.indexOf(ql, from)) !== -1) { count++; from = f + ql.length; }
        sink.appendChild(document.createTextNode(ln));
      } else {
        sink.appendChild(document.createTextNode(ln));
      }
      if (wrap) frag.appendChild(wrap);
      if (i < textLines.length - 1) frag.appendChild(document.createTextNode("\n"));
    });

    target.textContent = "";
    target.appendChild(frag);
    return count;
  }

  function applySearchToEntry(entry, q) {
    return renderPageBody(entry, q);
  }

  // ---------- Copy ----------
  copyAllBtn.addEventListener("click", () => {
    const parts = [];
    orderedJobs().forEach((job) => {
      parts.push("===== " + job.filename + " =====");
      orderedPages(job).forEach((entry) => {
        parts.push("--- Page " + entry.pageNo + " ---");
        parts.push(entry.text || "");
      });
      parts.push("");
    });
    copyText(parts.join("\n"));
  });

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(
        () => showToast("Copied to clipboard"),
        () => fallbackCopy(text)
      );
    } else {
      fallbackCopy(text);
    }
  }

  function fallbackCopy(text) {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand("copy");
      showToast("Copied to clipboard");
    } catch (_) {
      showToast("Copy failed");
    }
    document.body.removeChild(ta);
  }

  // ---------- Export ----------
  document.querySelectorAll("[data-export]").forEach((btn) => {
    btn.addEventListener("click", () => exportAll(btn.dataset.export));
  });

  function exportAll(kind) {
    const list = orderedJobs();
    if (!list.length) {
      showToast("Nothing to export yet.");
      return;
    }
    if (kind === "docx") {
      return exportDocx(list);
    }
    if (kind === "json") {
      const data = list.map((job) => ({
        filename: job.filename,
        job_id: String(job.jobId),
        status: job.status,
        total_pages: job.totalPages,
        pages: orderedPages(job).map((entry) => ({
          page: entry.pageNo,
          source: entry.data ? entry.data.source : null,
          status: entry.data ? entry.data.status : "pending",
          confidence: entry.data ? entry.data.confidence : null,
          text: entry.text || "",
        })),
      }));
      download("extraction.json", JSON.stringify(data, null, 2), "application/json");
      return;
    }

    if (kind === "md") {
      const parts = [];
      list.forEach((job) => {
        parts.push("# " + job.filename + "\n");
        orderedPages(job).forEach((entry) => {
          parts.push("## Page " + entry.pageNo + "\n");
          parts.push((entry.text || "_(no text)_") + "\n");
        });
      });
      download(baseName(list) + ".md", parts.join("\n"), "text/markdown");
      return;
    }

    // txt
    const parts = [];
    list.forEach((job) => {
      parts.push("===== " + job.filename + " =====\n");
      orderedPages(job).forEach((entry) => {
        parts.push("--- Page " + entry.pageNo + " ---");
        parts.push((entry.text || "") + "\n");
      });
    });
    download(baseName(list) + ".txt", parts.join("\n"), "text/plain");
  }

  async function exportDocx(list) {
    const documents = list.map((job) => ({
      filename: job.filename,
      pages: orderedPages(job).map((entry) => ({
        page: entry.pageNo,
        source: entry.data ? entry.data.source : null,
        confidence: entry.data ? entry.data.confidence : null,
        text: entry.text || "",
      })),
    }));
    try {
      const res = await fetch("/api/export/docx", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ documents }),
      });
      if (!res.ok) {
        showToast("Word export failed.");
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = (list.length === 1 ? baseName(list) : "extraction") + ".docx";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      showToast("Word document downloaded");
    } catch (e) {
      showToast("Word export failed.");
    }
  }

  function baseName(list) {
    if (list.length === 1) {
      return list[0].filename.replace(/\.pdf$/i, "") || "extraction";
    }
    return "extraction";
  }

  function download(filename, content, mime) {
    const blob = new Blob([content], { type: mime + ";charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  // ---------- Ordering helpers ----------
  function orderedJobs() {
    const out = [];
    jobOrder.forEach((id) => {
      const j = jobs.get(id);
      if (j) out.push(j);
    });
    return out;
  }

  function orderedPages(job) {
    return Array.from(job.pages.values()).sort((a, b) => a.pageNo - b.pageNo);
  }

  function refreshExportState() {
    // Export/copy always available once there is at least one job; nothing to gate.
  }

  // ---------- Knowledge graph (opt-in, online) ----------
  // Colors per entity type; chosen to read on both light & dark themes. Used
  // for the result-card dots/chips AND the canvas nodes so the two views match.
  const KG_TYPE_COLORS = {
    PERSON: "#7c3aed", ORG: "#0ea5b7", PLACE: "#0e9f6e", DATE: "#c2790f",
    ID: "#e0455c", AMOUNT: "#2563eb", CONCEPT: "#9333ea", EVENT: "#db2777",
    OTHER: "#8a8a99",
  };
  function kgTypeColor(t) {
    return KG_TYPE_COLORS[String(t || "OTHER").toUpperCase()] || KG_TYPE_COLORS.OTHER;
  }

  function setupKnowledgeGraph(job) {
    const el = job.el;
    if (!el || !el.kgPanel) return; // older markup without the KG panel
    job.kgBuilt = false;
    job.kgViz = null;
    job.kgLaidOut = false;

    if (el.kgBuildBtn) el.kgBuildBtn.addEventListener("click", () => buildKG(job));
    if (el.kgRebuildBtn) el.kgRebuildBtn.addEventListener("click", () => buildKG(job));
    if (el.kgAskBtn) el.kgAskBtn.addEventListener("click", () => askKG(job));
    if (el.kgQueryInput) {
      el.kgQueryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); askKG(job); }
      });
    }
    if (el.kgExportBtn) el.kgExportBtn.addEventListener("click", () => exportKG(job));
    if (el.kgVizToggle) el.kgVizToggle.addEventListener("click", () => toggleKGViz(job));
  }

  function setKgStatus(job, msg, kind) {
    const s = job.el.kgStatus;
    if (!s) return;
    s.textContent = msg || "";
    s.className = "kg-summary-status" + (kind ? " " + kind : "");
  }

  async function buildKG(job) {
    const el = job.el;
    el.kgPanel.open = true;
    if (el.kgBuildBtn) el.kgBuildBtn.disabled = true;
    if (el.kgRebuildBtn) el.kgRebuildBtn.disabled = true;
    setKgStatus(job, "Building…", "busy");
    el.kgBuildNote.textContent = "Reading entities & facts with Gemini — this can take a moment on a long document…";

    const form = new FormData();
    form.append("job_id", job.jobId);
    form.append("online_key", settings.online_key || "");
    form.append("online_model", settings.online_model || "");
    try {
      const res = await fetch("/api/graph/build", { method: "POST", body: form });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        if (el.kgBuildBtn) { el.kgBuildBtn.disabled = false; el.kgBuildBtn.hidden = false; }
        if (el.kgRebuildBtn) el.kgRebuildBtn.disabled = false;
        setKgStatus(job, "Failed", "bad");
        el.kgBuildNote.textContent = data.detail ? humanizeError(data.detail) : "Build failed.";
        return;
      }
      job.kgBuilt = true;
      job.kgViz = null;
      job.kgLaidOut = false;
      if (el.kgRebuildBtn) el.kgRebuildBtn.disabled = false;
      // Hide any previously open viz so it re-fetches the new graph next open.
      if (el.kgVizWrap) el.kgVizWrap.hidden = true;
      if (el.kgVizToggle) el.kgVizToggle.textContent = "Show graph view";

      if (!data.nodes) {
        setKgStatus(job, "No entities found", "");
        el.kgBuildNote.textContent =
          "Gemini found no extractable entities/facts in this document's text.";
        if (el.kgBuildBtn) { el.kgBuildBtn.disabled = false; el.kgBuildBtn.hidden = false; el.kgBuildBtn.textContent = "Try again"; }
        el.kgQueryWrap.hidden = true;
        return;
      }
      setKgStatus(job, data.nodes + " entities · " + data.edges + " facts", "good");
      let note = data.has_vectors
        ? "Graph ready — ask a question below (semantic + graph search)."
        : "Graph ready — ask a question below (keyword graph search).";
      if (data.pages_failed) {
        note += " " + data.pages_failed + " page(s) were skipped (rate limit or block).";
      }
      if (data.embed_error) {
        note += " Semantic search unavailable (" + shortErr(data.embed_error) +
          ") — using keyword search.";
      }
      el.kgBuildNote.textContent = note;
      if (el.kgBuildBtn) el.kgBuildBtn.hidden = true;
      el.kgQueryWrap.hidden = false;
      showToast("Knowledge graph: " + data.nodes + " entities, " + data.edges + " facts");
      if (el.kgQueryInput) el.kgQueryInput.focus();
    } catch (e) {
      if (el.kgBuildBtn) { el.kgBuildBtn.disabled = false; el.kgBuildBtn.hidden = false; }
      if (el.kgRebuildBtn) el.kgRebuildBtn.disabled = false;
      setKgStatus(job, "Failed", "bad");
      el.kgBuildNote.textContent = "Network error during build.";
    }
  }

  async function askKG(job) {
    const el = job.el;
    const q = (el.kgQueryInput.value || "").trim();
    if (!q) return;
    el.kgAskBtn.disabled = true;
    el.kgResults.innerHTML = "";
    const loading = document.createElement("p");
    loading.className = "kg-empty";
    loading.textContent = "Searching…";
    el.kgResults.appendChild(loading);

    const form = new FormData();
    form.append("job_id", job.jobId);
    form.append("query", q);
    form.append("online_key", settings.online_key || "");
    try {
      const res = await fetch("/api/graph/query", { method: "POST", body: form });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        renderKgMessage(job, data.detail ? humanizeError(data.detail) : "Query failed.");
        return;
      }
      renderKGResults(job, data);
    } catch (e) {
      renderKgMessage(job, "Network error during search.");
    } finally {
      el.kgAskBtn.disabled = false;
    }
  }

  function renderKgMessage(job, msg) {
    const c = job.el.kgResults;
    c.innerHTML = "";
    const p = document.createElement("p");
    p.className = "kg-empty";
    p.textContent = msg;
    c.appendChild(p);
  }

  // Build a "p.1, 2" provenance chip element.
  function kgPageTag(pages) {
    const tag = document.createElement("span");
    tag.className = "kg-pages";
    const list = (pages || []).filter((p) => p != null);
    tag.textContent = list.length
      ? (list.length === 1 ? "p." + list[0] : "p." + list.join(", "))
      : "";
    if (!list.length) tag.hidden = true;
    return tag;
  }

  // Render one triple as "subject → predicate → object" with a page tag.
  function kgTripleRow(t) {
    const row = document.createElement("div");
    row.className = "kg-triple";
    const s = document.createElement("span"); s.className = "kg-subj"; s.textContent = t.subject;
    const p = document.createElement("span"); p.className = "kg-pred"; p.textContent = t.predicate;
    const o = document.createElement("span"); o.className = "kg-obj"; o.textContent = t.object;
    const a1 = document.createElement("span"); a1.className = "kg-arrow"; a1.textContent = "→";
    const a2 = document.createElement("span"); a2.className = "kg-arrow"; a2.textContent = "→";
    row.appendChild(s); row.appendChild(a1); row.appendChild(p); row.appendChild(a2); row.appendChild(o);
    if (t.page != null) {
      const pg = document.createElement("span");
      pg.className = "kg-pages kg-triple-page";
      pg.textContent = "p." + t.page;
      row.appendChild(pg);
    }
    return row;
  }

  function renderKGResults(job, data) {
    const c = job.el.kgResults;
    c.innerHTML = "";

    const meta = document.createElement("div");
    meta.className = "kg-resultmeta";
    const modeTxt = data.mode === "semantic" ? "semantic + graph" : "graph (lexical)";
    meta.textContent = (data.answers || []).length
      ? "Top matches · " + modeTxt + " search"
      : "";
    if (meta.textContent) c.appendChild(meta);

    if (!data.answers || !data.answers.length) {
      renderKgMessage(job, "No matching entities found. Try different words.");
      // keep the meta line removed for a clean empty state
      return;
    }

    data.answers.forEach((a) => {
      const card = document.createElement("div");
      card.className = "kg-answer";

      const head = document.createElement("div");
      head.className = "kg-answer-head";
      const dot = document.createElement("span");
      dot.className = "kg-type-dot";
      dot.style.background = kgTypeColor(a.type);
      const name = document.createElement("span");
      name.className = "kg-entity";
      name.textContent = a.entity;
      const chip = document.createElement("span");
      chip.className = "kg-type-chip";
      chip.textContent = (a.type || "OTHER").toLowerCase();
      chip.style.color = kgTypeColor(a.type);
      head.appendChild(dot);
      head.appendChild(name);
      head.appendChild(chip);
      head.appendChild(kgPageTag(a.pages));
      card.appendChild(head);

      // Supporting facts (the entity's own triples) — the explainable evidence.
      const facts = (a.facts && a.facts.length) ? a.facts : a.path;
      if (facts && facts.length) {
        const fwrap = document.createElement("div");
        fwrap.className = "kg-facts";
        facts.slice(0, 6).forEach((t) => fwrap.appendChild(kgTripleRow(t)));
        card.appendChild(fwrap);
      }

      // If reached via traversal, show the connecting chain from the query seed.
      if (a.hops > 0 && a.path && a.path.length) {
        const link = document.createElement("div");
        link.className = "kg-linkpath";
        const lbl = document.createElement("span");
        lbl.className = "kg-link-label";
        lbl.textContent = "linked to your query via:";
        link.appendChild(lbl);
        a.path.forEach((t) => link.appendChild(kgTripleRow(t)));
        card.appendChild(link);
      }

      c.appendChild(card);
    });
  }

  // ----- Graph visualization (vanilla canvas; no external libs) -----
  async function toggleKGViz(job) {
    const el = job.el;
    if (!el.kgVizWrap) return;
    if (!el.kgVizWrap.hidden) {
      el.kgVizWrap.hidden = true;
      el.kgVizToggle.textContent = "Show graph view";
      return;
    }
    el.kgVizWrap.hidden = false;
    el.kgVizToggle.textContent = "Hide graph view";
    if (!job.kgViz) {
      el.kgVizNote.textContent = "Loading graph…";
      try {
        const res = await fetch("/api/graph/" + encodeURIComponent(job.jobId));
        const data = await res.json();
        if (!res.ok || !data.nodes) { el.kgVizNote.textContent = "No graph to show."; return; }
        job.kgViz = data;
        job.kgLaidOut = false;
      } catch (e) {
        el.kgVizNote.textContent = "Could not load the graph.";
        return;
      }
    }
    drawKGViz(job);
  }

  function layoutKG(viz) {
    // Force-directed layout (Fruchterman–Reingold-ish) in a virtual unit box,
    // deterministic init so the picture is stable across redraws.
    const nodes = viz.nodes.map((n, i) => {
      const ang = (2 * Math.PI * i) / Math.max(1, viz.nodes.length);
      return { id: n.id, x: 0.5 + 0.35 * Math.cos(ang), y: 0.5 + 0.35 * Math.sin(ang),
               vx: 0, vy: 0, deg: n.degree || 0, name: n.name, type: n.type };
    });
    const index = new Map(nodes.map((n) => [n.id, n]));
    const edges = viz.edges
      .map((e) => [index.get(e.source), index.get(e.target)])
      .filter((p) => p[0] && p[1]);

    const n = nodes.length || 1;
    const k = 0.9 / Math.sqrt(n);   // ideal edge length
    let temp = 0.10;
    const ITER = 220;
    for (let it = 0; it < ITER; it++) {
      // repulsion (O(n^2); fine for the capped node count)
      for (let i = 0; i < nodes.length; i++) {
        let fx = 0, fy = 0;
        const a = nodes[i];
        for (let j = 0; j < nodes.length; j++) {
          if (i === j) continue;
          const b = nodes[j];
          let dx = a.x - b.x, dy = a.y - b.y;
          let d2 = dx * dx + dy * dy;
          if (d2 < 1e-6) { dx = (i - j) * 1e-3; dy = 1e-3; d2 = 1e-6; }
          const f = (k * k) / d2;
          fx += dx * f; fy += dy * f;
        }
        // gravity toward center keeps disconnected nodes on-canvas
        fx += (0.5 - a.x) * 0.02;
        fy += (0.5 - a.y) * 0.02;
        a.vx = fx; a.vy = fy;
      }
      // attraction along edges
      edges.forEach(([a, b]) => {
        let dx = a.x - b.x, dy = a.y - b.y;
        const d = Math.sqrt(dx * dx + dy * dy) || 1e-4;
        const f = (d * d) / k;
        const ux = (dx / d) * f, uy = (dy / d) * f;
        a.vx -= ux; a.vy -= uy;
        b.vx += ux; b.vy += uy;
      });
      // integrate with cooling + clamp into the box
      nodes.forEach((a) => {
        const sp = Math.sqrt(a.vx * a.vx + a.vy * a.vy) || 1e-6;
        const step = Math.min(sp, temp) / sp;
        a.x = Math.max(0.04, Math.min(0.96, a.x + a.vx * step));
        a.y = Math.max(0.05, Math.min(0.95, a.y + a.vy * step));
      });
      temp = Math.max(0.008, temp * 0.985);
    }
    return { nodes, edges };
  }

  function drawKGViz(job) {
    const el = job.el;
    const canvas = el.kgCanvas;
    if (!canvas) return;
    if (!job.kgLaidOut) {
      job.kgLayout = layoutKG(job.kgViz);
      job.kgLaidOut = true;
    }
    const layout = job.kgLayout;
    const cssW = Math.max(280, canvas.clientWidth || canvas.parentElement.clientWidth || 640);
    const cssH = 400;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    canvas.style.height = cssH + "px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const cs = getComputedStyle(canvas);
    const textColor = cs.getPropertyValue("--text").trim() || "#222";
    const edgeColor = cs.getPropertyValue("--border-strong").trim() || "#ccc";
    const pad = 26;
    const X = (nx) => pad + nx * (cssW - 2 * pad);
    const Y = (ny) => pad + ny * (cssH - 2 * pad);

    // edges
    ctx.lineWidth = 1;
    ctx.strokeStyle = edgeColor;
    ctx.globalAlpha = 0.55;
    layout.edges.forEach(([a, b]) => {
      ctx.beginPath();
      ctx.moveTo(X(a.x), Y(a.y));
      ctx.lineTo(X(b.x), Y(b.y));
      ctx.stroke();
    });
    ctx.globalAlpha = 1;

    // label only the most-connected nodes to avoid clutter
    const labelled = new Set(
      layout.nodes.slice().sort((a, b) => b.deg - a.deg).slice(0, 26).map((n) => n.id)
    );
    ctx.font = "11px " + (cs.getPropertyValue("--sans").trim() || "sans-serif");
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    layout.nodes.forEach((nd) => {
      const r = 4 + Math.min(7, nd.deg * 1.4);
      ctx.beginPath();
      ctx.arc(X(nd.x), Y(nd.y), r, 0, 2 * Math.PI);
      ctx.fillStyle = kgTypeColor(nd.type);
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "rgba(255,255,255,0.55)";
      ctx.stroke();
      if (labelled.has(nd.id)) {
        const label = nd.name.length > 22 ? nd.name.slice(0, 21) + "…" : nd.name;
        ctx.fillStyle = textColor;
        ctx.fillText(label, X(nd.x), Y(nd.y) + r + 2);
      }
    });

    const more = job.kgViz.truncated
      ? " · showing top " + job.kgViz.nodes.length + " of " + job.kgViz.total_nodes
      : "";
    el.kgVizNote.textContent =
      job.kgViz.total_nodes + " entities · " + job.kgViz.total_edges + " facts" + more +
      " · labels on the most-connected entities";
  }

  async function exportKG(job) {
    try {
      const res = await fetch("/api/graph/" + encodeURIComponent(job.jobId) + "?full=1");
      if (!res.ok) { showToast("Could not export the graph."); return; }
      const data = await res.json();
      const payload = {
        filename: job.filename,
        node_count: data.node_count,
        triple_count: data.triple_count,
        embed_model: data.embed_model,
        entities: data.entities,
        triples: data.triples,
      };
      download(baseName([job]) + ".kg.json", JSON.stringify(payload, null, 2), "application/json");
      showToast("Knowledge graph exported");
    } catch (e) {
      showToast("Could not export the graph.");
    }
  }

  // ---------- Misc ----------
  function humanizeError(detail) {
    const d = String(detail).toLowerCase();
    if (d.includes("encrypt") || d.includes("password") || d.includes("needs_pass")) {
      return "This PDF is encrypted / password-protected and cannot be processed.";
    }
    if (d.includes("corrupt") || d.includes("cannot open") || d.includes("damaged")) {
      return "This file appears to be corrupt or is not a valid PDF.";
    }
    return String(detail);
  }

  let toastTimer = null;
  function showToast(msg) {
    toast.textContent = msg;
    toast.hidden = false;
    // force reflow so the transition runs
    void toast.offsetWidth;
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      toast.classList.remove("show");
      setTimeout(() => {
        toast.hidden = true;
      }, 220);
    }, 1800);
  }
})();
