"""Client for the Modal Whisper sidecar (``modal/whisper_service.py``).

Two call modes:

  * :func:`transcribe_sync` — blocks until the Modal GPU returns. Used
    by ``pipeline.youtube.transcripts`` tier-2 fallback and by the IG
    content-video-analysis path.

  * :func:`enqueue_transcribe_async_jobs` — fans out one
    ``transcribe_async`` ``background_jobs`` row per video URL. Used by
    the IG / YT scrape pipelines when ``WHISPER_ASYNC=1`` so transcription
    runs OFF the critical path; the last job in a group enqueues an
    ``audience_refresh`` job to re-score with full transcripts.

The Modal app + function names are configurable via env vars so a
side-by-side test deployment doesn't need code changes.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

APP_NAME = os.environ.get("WHISPER_MODAL_APP_NAME", "whisper-transcribe")
FUNCTION_NAME = os.environ.get("WHISPER_MODAL_FUNCTION_NAME", "transcribe")


def is_configured() -> bool:
    """Cheap pre-flight check: do we have credentials + the modal SDK?

    Modal accepts auth via ``MODAL_TOKEN_ID``/``MODAL_TOKEN_SECRET`` env
    vars (server-friendly) OR via ``~/.modal.toml`` (dev workstation).
    """
    has_env_token = bool(
        os.environ.get("MODAL_TOKEN_ID")
        and os.environ.get("MODAL_TOKEN_SECRET")
    )
    has_toml = os.path.exists(os.path.expanduser("~/.modal.toml"))
    if not (has_env_token or has_toml):
        return False
    try:
        import modal  # noqa: F401
    except ImportError:
        return False
    return True


def is_async_mode() -> bool:
    """Whether the pipelines should defer transcription to background jobs."""
    return os.environ.get("WHISPER_ASYNC", "").lower() in {"1", "true", "yes"}


# ── Synchronous path ────────────────────────────────────────────────────────


def _transcribe_openai(audio_url: str) -> dict | None:
    """Direct OpenAI Whisper API path: download the media, send to
    audio.transcriptions. Returns the same dict shape as the Modal path.
    Used when WHISPER_BACKEND=openai (Modal-free). whisper-1 caps at 25MB —
    reels/shorts are well under that.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        logger.warning("OPENAI_API_KEY not set; OpenAI whisper unavailable")
        return None
    import tempfile

    import requests
    from openai import OpenAI

    tmp_path = None
    try:
        resp = requests.get(audio_url, timeout=120)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(resp.content)
            tmp_path = f.name
        client = OpenAI(api_key=key)
        model = os.environ.get("OPENAI_WHISPER_MODEL", "whisper-1")
        with open(tmp_path, "rb") as audio:
            tr = client.audio.transcriptions.create(
                model=model, file=audio, response_format="verbose_json"
            )
        segs = [
            {"start": s.get("start"), "end": s.get("end"), "text": s.get("text")}
            if isinstance(s, dict)
            else {"start": s.start, "end": s.end, "text": s.text}
            for s in (getattr(tr, "segments", None) or [])
        ]
        return {
            "text": getattr(tr, "text", "") or "",
            "segments": segs,
            "language": getattr(tr, "language", None),
            "language_probability": None,
            "avg_confidence": None,
            "duration": getattr(tr, "duration", None),
        }
    except Exception as e:  # noqa: BLE001
        logger.warning(f"OpenAI whisper failed for {audio_url}: {e}")
        return None
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def transcribe_sync(audio_url: str) -> dict | None:
    """Blocking transcription. Routes to OpenAI Whisper when
    WHISPER_BACKEND=openai, else the Modal GPU sidecar. Returns the
    function's dict or None on failure.

    Dict keys: ``text``, ``segments`` (list of ``{start, end, text}``),
    ``language``, ``language_probability``, ``avg_confidence``,
    ``duration``.
    """
    if not audio_url:
        return None

    # OpenAI Whisper API path (Modal-free) — opt in via WHISPER_BACKEND=openai.
    if os.environ.get("WHISPER_BACKEND", "").lower() == "openai":
        return _transcribe_openai(audio_url)

    if not is_configured():
        logger.warning("Modal not configured; whisper unavailable")
        return None

    import modal  # imported lazily so the worker can run without modal in dev

    try:
        # Modal ≥1.x replaced Function.lookup with Function.from_name; fall back
        # to lookup for older SDKs.
        getter = getattr(modal.Function, "from_name", None) or modal.Function.lookup
        fn = getter(APP_NAME, FUNCTION_NAME)
        return fn.remote(audio_url=audio_url)
    except Exception as e:  # noqa: BLE001 — modal raises a variety of errors
        logger.warning(f"Modal whisper failed for {audio_url}: {e}")
        return None


# ── Async path (enqueue background jobs) ────────────────────────────────────


def enqueue_transcribe_async_jobs(
    db,
    *,
    creator_id: str,
    brand_id: str | None,
    items: list[dict],
    source: str,
) -> int:
    """Fan out one ``transcribe_async`` job per item; return count enqueued.

    ``items`` is a list of dicts with at minimum ``video_url`` and
    ``post_id`` (IG) or ``video_id`` (YT). ``source`` is ``"instagram"``
    or ``"youtube"`` — determines which DB table the transcript rows
    land in when the worker processes the job.

    All jobs in one fanout share a ``group_id`` so the handler can detect
    when the last sibling finishes and enqueue a single
    ``audience_refresh`` job.
    """
    if not items:
        return 0

    from datetime import datetime, timezone

    group_id = f"{source}:{creator_id}:{int(datetime.now(timezone.utc).timestamp())}"
    rows: list[dict] = []
    for item in items:
        url = item.get("video_url")
        if not url:
            continue
        rows.append(
            {
                "job_type": "transcribe_async",
                "brand_id": brand_id,
                "status": "queued",
                "payload": {
                    "audio_url": url,
                    "creator_id": creator_id,
                    "source": source,
                    "post_id": item.get("post_id") or item.get("video_id"),
                    "caption": (item.get("caption") or "")[:1000],
                    "length": item.get("length") or item.get("duration_seconds"),
                    "group_id": group_id,
                },
            }
        )

    if not rows:
        return 0

    db.table("background_jobs").insert(rows).execute()
    logger.info(
        "Enqueued %d transcribe_async jobs for creator %s (group=%s)",
        len(rows), creator_id, group_id,
    )
    return len(rows)
