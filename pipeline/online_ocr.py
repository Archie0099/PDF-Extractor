"""Optional ONLINE OCR via Google Gemini (handwriting + printed transcription).

This is an OPT-IN, accuracy-first fallback for pages the local engines
(PaddleOCR / TrOCR) read poorly. It sends a rendered PAGE IMAGE to Google's
Gemini API and returns a verbatim transcription.

PRIVACY: enabling this uploads page images to Google's servers — the bytes
leave this machine. It is OFF by default and only runs when the caller passes
a Gemini API key. Do not enable it for confidential documents you cannot send
to a third party.

SETUP: get a free API key from https://aistudio.google.com/ (Google AI Studio)
and pass it as ``api_key`` (or wire it to a ``GEMINI_API_KEY`` env var in the
caller). No extra pip dependency is needed: this module talks to the REST API
with the Python standard library only (``urllib.request`` + ``json`` +
``base64``). ``cv2`` is used to PNG-encode the OpenCV image.

Importing this module performs NO network call and has no import-time side
effects.
"""

import base64
import json
import urllib.error
import urllib.request

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Verified Gemini REST contract.
# ---------------------------------------------------------------------------
# Generation endpoint (non-streaming). The canonical reference path is
# /v1beta/{model=models/*}:generateContent; the model name on the wire is
# "models/<id>". {model} below is the BARE model id (e.g. "gemini-2.5-pro").
_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_GENERATE_URL = _API_BASE + "/{model}:generateContent"

# Auth header preferred over the legacy ?key= query param (keeps the key out of
# URLs/logs). Header name is literally "x-goog-api-key".
_API_KEY_HEADER = "x-goog-api-key"

# Strong handwriting-capable models, FREE-TIER-FRIENDLY first. As of 2026 the
# Pro models are NOT available on the API free tier (free-tier limit 0), so the
# default auto-pick must be a Flash model — Flash is already excellent at
# handwriting (benchmarked ~95% char-acc on a real cursive page). Users with
# paid billing can still select a Pro model from the dropdown.
SUPPORTED_MODELS = (
    "gemini-2.5-flash",        # free-tier workhorse; strong handwriting
    "gemini-3.5-flash",        # newer flash if the key exposes it
    "gemini-2.5-flash-lite",   # free-tier, lighter/faster, slightly weaker
    "gemini-3.1-flash-lite",
    "gemini-2.5-pro",          # better accuracy but needs PAID billing (not free tier)
    "gemini-3.1-pro-preview",
)
DEFAULT_MODEL = SUPPORTED_MODELS[0]

# Verbatim-transcription instruction. The online path runs for ANY page that
# needs OCR (handwritten OR printed), so this must NOT exclude printed text —
# it covers both. Still ignores pre-printed page furniture (Date/Page stamps,
# margin rules) which would otherwise leak into the output.
DEFAULT_PROMPT = (
    "Transcribe all text in this image verbatim — both printed and handwritten. "
    "Preserve line breaks and spelling exactly; do not correct, summarize, or "
    "add commentary. Ignore pre-printed page furniture such as 'Date'/'Page' "
    "header labels, margin rules and ruled lines. Output only the transcription."
)

# Inline image payload cap. The TOTAL request (image + prompt, base64-expanded)
# must stay under 20 MB; above that the Files API is required. We guard on the
# base64 string length and surface an actionable error well before the wire.
_MAX_INLINE_B64_BYTES = 20 * 1024 * 1024

# Cap the longest image side sent to Gemini. Gemini internally tiles images
# (~768 px tiles), so a huge 300-DPI render only inflates the upload with no
# accuracy gain — and a multi-MB PNG over a slow connection causes "write
# operation timed out" errors. Downscaling + JPEG keeps requests small and fast.
_MAX_IMAGE_SIDE = 2048
_JPEG_QUALITY = 90


def is_configured(api_key) -> bool:
    """True if a usable (non-empty) Gemini API key has been supplied.

    Does NOT validate the key against the network — it only checks presence,
    so calling it is free and side-effect-free.
    """
    return bool(api_key and str(api_key).strip())


def _clean_key(api_key) -> str:
    """Return the trimmed key, or raise a clean, KEY-FREE RuntimeError.

    A key with an interior newline (pasted wrapped across two lines) or a
    non-ASCII/control character would otherwise make http.client raise a raw
    ValueError — whose message INCLUDES the full key value — or a
    UnicodeEncodeError, neither of which subclasses HTTPError/URLError, so they
    escape the documented "only ever raise a single-line RuntimeError" contract
    and (for the ValueError case) echo the key into the per-page error text that
    jobs.py broadcasts/stores. Validate up front and never put the key in the
    message.
    """
    key = str(api_key).strip()
    if not key.isascii() or any(ord(c) < 0x20 or ord(c) == 0x7F for c in key):
        raise RuntimeError(
            "The Gemini API key contains invalid characters (control or "
            "non-ASCII). Re-copy it from https://aistudio.google.com/."
        )
    return key


def _resolve_model(model) -> str:
    """Return a clean bare model id, defaulting when unset.

    Accepts either a bare id ("gemini-2.5-pro") or the "models/<id>" form and
    normalizes to the bare id used in the URL placeholder.
    """
    name = (model or DEFAULT_MODEL).strip()
    if name.startswith("models/"):
        name = name[len("models/"):]
    return name or DEFAULT_MODEL


def _encode_image_b64(img_bgr: np.ndarray):
    """Downscale a huge image, JPEG-encode it, return ``(base64_str, mime_type)``.

    Caps the longest side at ``_MAX_IMAGE_SIDE`` and uses JPEG (much smaller than
    PNG for scans/photos) so the request uploads quickly even on slow links.
    """
    if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
        raise RuntimeError(
            "Online OCR received an empty image; nothing to transcribe."
        )
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest > _MAX_IMAGE_SIDE:
        scale = _MAX_IMAGE_SIDE / float(longest)
        img_bgr = cv2.resize(
            img_bgr,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(
        ".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError(
            "Online OCR could not encode the page image (cv2.imencode failed). "
            "Check that the image is a valid HxWx3 uint8 array."
        )
    # Standard base64 of the raw JPEG file bytes — NOT a data: URI.
    return base64.standard_b64encode(buf.tobytes()).decode("ascii"), "image/jpeg"


def _build_request_body(image_b64: str, mime_type: str, prompt: str) -> dict:
    """Assemble the generateContent request body per the verified contract."""
    return {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_b64,
                        }
                    },
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "text/plain",
        },
    }


def _extract_text(payload: dict) -> str:
    """Pull the transcription out of a generateContent response.

    Concatenates the ``text`` of every part in ``candidates[0].content.parts``
    (text output can be split across parts). Raises an actionable RuntimeError
    when the response carries no usable text — e.g. a prompt-level block
    (``promptFeedback.blockReason``) or a non-STOP ``finishReason`` such as
    SAFETY / MAX_TOKENS / RECITATION.
    """
    # Prompt blocked before any candidate was produced.
    feedback = payload.get("promptFeedback") or {}
    block_reason = feedback.get("blockReason")
    candidates = payload.get("candidates") or []
    if not candidates:
        if block_reason:
            raise RuntimeError(
                "Gemini blocked the request (promptFeedback.blockReason="
                "{}). The page image or prompt tripped a safety filter; "
                "online OCR is unavailable for this page.".format(block_reason)
            )
        raise RuntimeError(
            "Gemini returned no candidates and no text. The response was "
            "empty; try again or switch models."
        )

    candidate = candidates[0] or {}
    finish_reason = candidate.get("finishReason")
    content = candidate.get("content") or {}
    parts = content.get("parts") or []
    # ``part.get("text", "")`` only defaults when the key is ABSENT; an explicit
    # ``{"text": null}`` would yield None and break the join. Coerce defensively.
    text = "".join(
        str(part.get("text") or "") for part in parts if isinstance(part, dict)
    )

    if not text.strip():
        if finish_reason and finish_reason != "STOP":
            raise RuntimeError(
                "Gemini produced no text (finishReason={}). This usually "
                "means the output was cut off (MAX_TOKENS) or filtered "
                "(SAFETY/RECITATION); try a different page or model.".format(
                    finish_reason
                )
            )
        raise RuntimeError(
            "Gemini returned an empty transcription for this page."
        )

    # On a clean STOP we return the text as-is; if truncated but non-empty we
    # still return what we have (a partial transcription beats failing hard).
    return text


def _friendly_http_error(err: urllib.error.HTTPError) -> RuntimeError:
    """Translate an HTTPError into an actionable RuntimeError.

    Reads the JSON ``error`` body (code/message/status) to surface the API's
    own message and maps the common status codes to plain guidance.
    """
    code = err.code
    api_message = ""
    api_status = ""
    try:
        raw = err.read()
        if raw:
            body = json.loads(raw.decode("utf-8", "replace"))
            error_obj = body.get("error") or {}
            api_message = str(error_obj.get("message") or "").strip()
            api_status = str(error_obj.get("status") or "").strip()
    except Exception:
        # Body wasn't JSON or couldn't be read; fall back to the status code.
        pass

    detail = api_message or err.reason or "no detail provided"

    if code in (401, 403):
        return RuntimeError(
            "Gemini rejected the API key (HTTP {} {}): {}. Check that "
            "GEMINI_API_KEY is a valid, enabled key from "
            "https://aistudio.google.com/ and has access to this model."
            .format(code, api_status or "PERMISSION_DENIED", detail)
        )
    if code == 429:
        return RuntimeError(
            "Gemini rate limit / quota exceeded (HTTP 429 {}): {}. Slow down "
            "requests, wait and retry, switch to a lighter model, or request "
            "a quota increase in Google AI Studio."
            .format(api_status or "RESOURCE_EXHAUSTED", detail)
        )
    if code == 400:
        return RuntimeError(
            "Gemini rejected the request (HTTP 400 {}): {}. This is usually a "
            "malformed body, an unsupported model id, or (FAILED_PRECONDITION) "
            "billing not enabled for your region."
            .format(api_status or "INVALID_ARGUMENT", detail)
        )
    if code == 404:
        return RuntimeError(
            "Gemini model or resource not found (HTTP 404 {}): {}. Check the "
            "model id ({}).".format(
                api_status or "NOT_FOUND", detail, ", ".join(SUPPORTED_MODELS)
            )
        )
    if code in (500, 503, 504):
        return RuntimeError(
            "Gemini server error (HTTP {} {}): {}. The service is overloaded "
            "or timed out; retry in a moment or switch to another model."
            .format(code, api_status or "UNAVAILABLE", detail)
        )
    return RuntimeError(
        "Gemini request failed (HTTP {} {}): {}.".format(
            code, api_status or "ERROR", detail
        )
    )


def list_models(api_key, *, timeout=30) -> list:
    """Validate the key and return available bare model ids (no network on import).

    Calls ``GET /v1beta/models`` and returns the bare ids (``models/`` prefix
    stripped) of models that support ``generateContent``. Doubles as a cheap
    key-validation call for the UI. Raises an actionable RuntimeError on a bad
    key / network failure, mirroring ``transcribe_image_bgr``.
    """
    if not is_configured(api_key):
        raise RuntimeError(
            "Online OCR needs a Gemini API key but none was provided. Get a "
            "free key from https://aistudio.google.com/."
        )
    api_key = _clean_key(api_key)
    request = urllib.request.Request(
        _API_BASE,  # GET https://generativelanguage.googleapis.com/v1beta/models
        method="GET",
        headers={_API_KEY_HEADER: api_key},
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
    except (ValueError, AttributeError) as err:
        raise RuntimeError(
            "Gemini returned a non-JSON model list ({}).".format(err)
        ) from None
    if not isinstance(payload, dict):
        raise RuntimeError(
            "Gemini returned an unexpected model-list response (not a JSON object)."
        )

    out = []
    for entry in payload.get("models") or []:
        if not isinstance(entry, dict):
            continue
        methods = entry.get("supportedGenerationMethods") or []
        if methods and "generateContent" not in methods:
            continue
        name = str(entry.get("name") or "")
        if name.startswith("models/"):
            name = name[len("models/"):]
        if name:
            out.append(name)
    return out


def best_available_model(available) -> str:
    """Pick the strongest handwriting model we support that the key can access.

    Falls back to :data:`DEFAULT_MODEL` if none of ``available`` is recognized.
    """
    avail = set(available or ())
    for m in SUPPORTED_MODELS:
        if m in avail:
            return m
    return DEFAULT_MODEL


def transcribe_image_bgr(
    img_bgr: np.ndarray,
    *,
    api_key,
    model=None,
    prompt=None,
    timeout=120,
) -> str:
    """Transcribe an OpenCV BGR page image with Gemini and return the text.

    Parameters
    ----------
    img_bgr : np.ndarray
        An OpenCV ``HxWx3`` BGR ``uint8`` image (a rendered PDF page).
    api_key : str
        A Gemini API key from https://aistudio.google.com/. Required.
    model : str, optional
        A model id from :data:`SUPPORTED_MODELS` (bare id or "models/<id>").
        Defaults to :data:`DEFAULT_MODEL`.
    prompt : str, optional
        Override the verbatim-transcription instruction
        (:data:`DEFAULT_PROMPT`).
    timeout : float, optional
        Per-request socket timeout in seconds (default 120).

    Returns
    -------
    str
        The verbatim transcription (line breaks preserved).

    Raises
    ------
    RuntimeError
        With an actionable, single-line message (never a raw stack trace) for
        a missing/empty key, an invalid key, a rate limit, a network/timeout
        failure, or an empty/safety-blocked response.
    """
    if not is_configured(api_key):
        raise RuntimeError(
            "Online OCR needs a Gemini API key but none was provided. Get a "
            "free key from https://aistudio.google.com/ and pass it as the "
            "api_key argument (or set GEMINI_API_KEY)."
        )

    api_key = _clean_key(api_key)
    model_id = _resolve_model(model)
    prompt_text = prompt if (prompt and prompt.strip()) else DEFAULT_PROMPT

    image_b64, mime_type = _encode_image_b64(img_bgr)
    if len(image_b64) > _MAX_INLINE_B64_BYTES:
        raise RuntimeError(
            "Page image is too large for an inline Gemini request "
            "(~{:.1f} MB base64; the 20 MB inline limit applies to the whole "
            "request). Render the page at a lower DPI, or use Gemini's Files "
            "API for payloads this big.".format(
                len(image_b64) / (1024 * 1024)
            )
        )

    body = _build_request_body(image_b64, mime_type, prompt_text)
    data = json.dumps(body).encode("utf-8")
    url = _GENERATE_URL.format(model=model_id)

    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            _API_KEY_HEADER: api_key,  # already cleaned/validated above
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as err:
        # Non-2xx: surface the API's own error message via a friendly mapping.
        raise _friendly_http_error(err) from None
    except urllib.error.URLError as err:
        # DNS failure, connection refused, TLS error, or a socket timeout
        # (which arrives wrapped as URLError(reason=timeout)).
        raise RuntimeError(
            "Could not reach the Gemini API ({}). Check your internet "
            "connection and that generativelanguage.googleapis.com is "
            "reachable; the request may also have timed out (timeout={}s)."
            .format(err.reason, timeout)
        ) from None
    except TimeoutError as err:  # bare socket timeout on some Python versions
        raise RuntimeError(
            "The Gemini request timed out after {}s. Try a smaller image, a "
            "lighter model, or a longer timeout.".format(timeout)
        ) from None

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

    return _extract_text(payload)
