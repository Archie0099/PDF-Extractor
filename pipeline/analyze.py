"""Per-document settings recommender.

Samples the first 1-2 OCR-eligible pages, sweeps a small grid of already-supported
configs, scores each by a COMPOSITE signal (mean confidence AND confident-line
coverage AND text mass AND cross-config agreement), and recommends settings.

Why not 'pick highest mean confidence': confidence-based auto-select was
benchmarked and rejected — binarize can score HIGHER mean confidence on garbage
while silently dropping faint text. The composite + coverage + a margin-handicap
toward the validated grayscale default guard against that.

This module only MEASURES already-supported, already-benchmarked configs and
RECOMMENDS one with evidence. It never changes the extraction pipeline, uses the
existing engine (det_limit_side_len stays 1536), and therefore cannot regress the
CER benchmark.
"""

from __future__ import annotations

import re
import time

import fitz  # type: ignore

from pipeline.extractor import render_page_image, DPI_MAX, DPI_FAST
from pipeline.preprocess import preprocess_image
from pipeline.ocr_engine import engine
from pipeline.textlayer import has_text_layer

# How many OCR-eligible pages to actually sample. Keep tiny: this runs PaddleOCR
# several times per page on the CPU, so 1-2 pages is the speed/robustness sweet
# spot. The first eligible page is usually representative of document "type".
MAX_SAMPLE_PAGES = 2

# Lines at/above this recognition confidence count as "real" content. Used for
# the coverage and text-mass signals (the part confidence-alone ignored).
CONFIDENT = 0.80

# An alternative config must beat the validated grayscale default by at least
# this much COMPOSITE margin to be recommended. The grayscale+tuned pipeline is
# the benchmarked baseline; don't flip off it on noise.
DECISIVE_MARGIN = 0.08
MARGINAL_MARGIN = 0.03

_WORD_RE = re.compile(r"[0-9a-zऀ-ॿ]+")  # latin digits/letters + devanagari


def _tokens(text: str) -> set:
    return set(_WORD_RE.findall((text or "").lower()))


def _config_grid(lang: str) -> list:
    """The SMALL grid we sweep. All are already-supported, already-benchmarked
    knobs — no new pipeline behavior, no upscaling, no fused ensemble.

    Order matters only for tie-stability; 'baseline' marks the validated default.
    """
    return [
        # name, mode, preprocess, binarize, baseline?
        {"name": "max_gray",     "mode": "max",  "preprocess": True,  "binarize": False, "baseline": True},
        {"name": "max_gray_raw", "mode": "max",  "preprocess": False, "binarize": False, "baseline": False},
        {"name": "max_binarize", "mode": "max",  "preprocess": True,  "binarize": True,  "baseline": False},
        {"name": "fast_gray",    "mode": "fast", "preprocess": True,  "binarize": False, "baseline": False},
    ]


def _ocr_one(img_bgr, *, lang: str, use_angle_cls: bool):
    """Run the SAME engine the real pipeline uses and return rich per-line stats.

    Returns (text, mean_conf, n_confident_lines, confident_char_mass, confident_text).
    """
    lines = engine.ocr_lines(img_bgr, lang=lang, use_angle_cls=use_angle_cls)
    if not lines:
        return "", None, 0, 0, ""
    text = "\n".join(ln["text"] for ln in lines)
    mean_conf = sum(ln["confidence"] for ln in lines) / len(lines)
    conf_lines = [ln for ln in lines if ln["confidence"] >= CONFIDENT]
    n_conf = len(conf_lines)
    mass = sum(len(ln["text"].strip()) for ln in conf_lines)
    conf_text = " ".join(ln["text"] for ln in conf_lines)
    return text, float(mean_conf), n_conf, mass, conf_text


def _score_page(results: dict) -> dict:
    """Given {config_name: per-config raw stats} for ONE page, compute a
    normalized composite score per config.

    Signals (all normalized to the best config ON THIS PAGE so absolute scale
    doesn't matter), then weighted:
      conf      0.30  mean line confidence
      coverage  0.30  # confident lines (catches binarize dropping faint text)
      mass      0.20  confident char count (don't reward tiny fragments)
      agree     0.20  token overlap vs the union of all configs' confident text
                      (a config that hallucinates differently is down-weighted)
    """
    names = list(results.keys())
    confs = {n: (results[n]["mean_conf"] or 0.0) for n in names}
    covs = {n: results[n]["n_conf"] for n in names}
    mass = {n: results[n]["mass"] for n in names}

    # cross-config agreement: the fraction of THIS config's confident tokens
    # that are corroborated by at least one OTHER config. (Comparing against the
    # union of ALL configs — including this one — is mathematically inert: it
    # reduces to |toks[n]| / |union|, which actually REWARDS a config that
    # hallucinates many unique tokens. We want the opposite: down-weight a config
    # whose confident output nobody else agrees with.)
    toks = {n: _tokens(results[n]["conf_text"]) for n in names}
    agree = {}
    for n in names:
        others = set()
        for m in names:
            if m != n:
                others |= toks[m]
        agree[n] = (len(toks[n] & others) / len(toks[n])) if toks[n] else 0.0

    def _norm(d):
        hi = max(d.values()) if d else 0
        if not hi:
            return {k: 0.0 for k in d}
        return {k: v / hi for k, v in d.items()}

    nconf, ncov, nmass = _norm(confs), _norm(covs), _norm(mass)
    out = {}
    for n in names:
        out[n] = (0.30 * nconf[n] + 0.30 * ncov[n] +
                  0.20 * nmass[n] + 0.20 * agree[n])
    return out


def suggest_settings(pdf_bytes: bytes, *, lang: str = "en",
                     max_pages: int = MAX_SAMPLE_PAGES) -> dict:
    """Analyze a PDF and recommend extraction settings for THIS document.

    Fast + opt-in: only renders/OCRs up to ``max_pages`` OCR-eligible pages.
    Born-digital pages short-circuit to a 'no OCR needed' recommendation.

    Returns a JSON-serializable dict. Never raises on a bad page; on total
    failure returns a safe 'use defaults' recommendation.
    """
    t0 = time.time()
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        return _fallback(f"Could not open PDF ({exc}); using defaults.", lang)

    try:
        if doc.needs_pass:
            return _fallback("PDF is encrypted; using defaults.", lang)
        total = doc.page_count
        if total <= 0:
            return _fallback("PDF has no pages; using defaults.", lang)

        # --- Pass 1: classify pages cheaply (text-layer vs needs-OCR). -------
        # Cap the scan to a small leading window so a 500-page born-digital PDF
        # (which never accumulates OCR pages) isn't fully scanned — the first
        # few pages define the document's type for "recommend settings".
        text_pages, ocr_page_indices = 0, []
        scan_cap = max_pages * 4
        for i in range(total):
            page = doc.load_page(i)
            if has_text_layer(page):
                text_pages += 1
            else:
                ocr_page_indices.append(i)
            if len(ocr_page_indices) >= max_pages or i >= min(total - 1, scan_cap):
                break

        # --- Born-digital short-circuit: no OCR needed. ----------------------
        if not ocr_page_indices:
            return {
                "ok": True,
                "needs_ocr": False,
                "decision": "decisive",
                "rationale": "This PDF already has a real text layer on the sampled pages — "
                             "it is extracted exactly, no OCR required.",
                "total_pages": total,
                "sampled_pages": [],
                "text_layer_pages": text_pages,
                "recommended": {
                    "mode": "max", "lang": lang,
                    "preprocess": True, "binarize": False, "handwriting": False,
                },
                "evidence": [],
                "elapsed_sec": round(time.time() - t0, 2),
            }

        # --- Pass 2: sweep the small grid on up to max_pages OCR pages. ------
        sample = ocr_page_indices[:max_pages]
        if not sample:  # guards a direct max_pages<=0 call -> no ZeroDivision
            return _fallback("No OCR-eligible pages sampled; using defaults.", lang)
        grid = _config_grid(lang)
        per_config_totals = {g["name"]: 0.0 for g in grid}
        evidence_rows = {g["name"]: dict(g, mean_conf=[], n_conf=[], mass=[]) for g in grid}

        for pidx in sample:
            page = doc.load_page(pidx)
            renders = {}
            page_raw = {}
            for g in grid:
                dpi = DPI_MAX if g["mode"] == "max" else DPI_FAST
                if dpi not in renders:
                    renders[dpi] = render_page_image(page, dpi)
                img = renders[dpi]
                if g["preprocess"]:
                    img = preprocess_image(img, mode=g["mode"], binarize=g["binarize"])
                use_cls = (g["mode"] == "max")
                text, mconf, ncf, mass, ctext = _ocr_one(
                    img, lang=lang, use_angle_cls=use_cls
                )
                page_raw[g["name"]] = {
                    "mean_conf": mconf, "n_conf": ncf, "mass": mass, "conf_text": ctext,
                }
                evidence_rows[g["name"]]["mean_conf"].append(mconf)
                evidence_rows[g["name"]]["n_conf"].append(ncf)
                evidence_rows[g["name"]]["mass"].append(mass)

            page_scores = _score_page(page_raw)
            for name, s in page_scores.items():
                per_config_totals[name] += s

        # Average composite across sampled pages.
        n = len(sample)
        composite = {k: v / n for k, v in per_config_totals.items()}

        baseline_name = next(g["name"] for g in grid if g["baseline"])
        base_score = composite[baseline_name]
        alt_name = max((g["name"] for g in grid if not g["baseline"]),
                       key=lambda k: composite[k])
        alt_score = composite[alt_name]
        margin = alt_score - base_score

        if margin >= DECISIVE_MARGIN:
            winner, decision = alt_name, "decisive"
        elif margin >= MARGINAL_MARGIN:
            winner, decision = alt_name, "marginal"
        else:
            winner, decision = baseline_name, "inconclusive"

        win_cfg = next(g for g in grid if g["name"] == winner)
        recommended = {
            "mode": win_cfg["mode"], "lang": lang,
            "preprocess": win_cfg["preprocess"],
            "binarize": win_cfg["binarize"], "handwriting": False,
        }

        # Build a readable evidence table.
        def _avg(xs):
            xs = [x for x in xs if x is not None]
            return round(sum(xs) / len(xs), 3) if xs else None
        evidence = []
        for g in grid:
            r = evidence_rows[g["name"]]
            evidence.append({
                "name": g["name"],
                "mode": g["mode"], "preprocess": g["preprocess"], "binarize": g["binarize"],
                "baseline": g["baseline"],
                "composite": round(composite[g["name"]], 3),
                "mean_conf": _avg(r["mean_conf"]),
                "confident_lines": round(sum(r["n_conf"]) / max(1, len(r["n_conf"])), 1),
                "recommended": g["name"] == winner,
            })
        evidence.sort(key=lambda e: e["composite"], reverse=True)

        rationale = _rationale(decision, winner, baseline_name, margin, win_cfg)
        return {
            "ok": True,
            "needs_ocr": True,
            "decision": decision,
            "rationale": rationale,
            "total_pages": total,
            "sampled_pages": [p + 1 for p in sample],
            "text_layer_pages": text_pages,
            "recommended": recommended,
            "evidence": evidence,
            "elapsed_sec": round(time.time() - t0, 2),
        }
    except Exception as exc:  # never let analysis crash the request
        return _fallback(f"Analysis failed ({exc}); using defaults.", lang)
    finally:
        try:
            doc.close()
        except Exception:
            pass


def _rationale(decision, winner, baseline_name, margin, cfg) -> str:
    if decision == "inconclusive":
        return ("No alternative clearly beat the validated default on the sampled "
                "page(s); recommending the default (grayscale, max accuracy).")
    knob = []
    if cfg["binarize"]:
        knob.append("binarize ON")
    if not cfg["preprocess"]:
        knob.append("preprocessing OFF")
    if cfg["mode"] == "fast":
        knob.append("Faster mode")
    desc = ", ".join(knob) if knob else "the default settings"
    strength = "clearly" if decision == "decisive" else "slightly"
    return (f"On the sampled page(s), {desc} {strength} outscored the default "
            f"(composite +{margin:.2f}: higher confident-line coverage/agreement, "
            f"not just mean confidence). Verify against the original before trusting it.")


def _fallback(msg: str, lang: str = "en") -> dict:
    return {
        "ok": True, "needs_ocr": True, "decision": "inconclusive",
        "rationale": msg, "total_pages": None, "sampled_pages": [],
        "text_layer_pages": None,
        "recommended": {"mode": "max", "lang": lang,
                        "preprocess": True, "binarize": False, "handwriting": False},
        "evidence": [],
    }
