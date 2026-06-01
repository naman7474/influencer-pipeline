"""Tiered transcript fetcher for YouTube videos.

Two tiers:

  Tier 1 — ``youtube-transcript-api`` (free, ~85% hit on public videos)
    Scrapes YouTube's internal transcript endpoint. No API key. Returns
    immediately on success.

  Tier 2 — Modal Whisper sidecar (paid, fills the tier-1 gap)
    Calls into the ``whisper-transcribe`` Modal app via
    ``pipeline.whisper_client.transcribe_sync``. The Modal service
    downloads the video URL, runs faster-whisper on a GPU container,
    and returns text + segments. No yt-dlp, no local Whisper backends,
    no SSH-tunnelled pod.

Async mode (``WHISPER_ASYNC=1``) bypasses tier 2 inline and lets the
pipeline enqueue a ``transcribe_async`` job instead. That keeps
transcription off the critical path; an ``audience_refresh`` job
re-runs LLM evaluation once the async transcripts land.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def fetch_transcript(
    video_id: str,
    video_url: str,
    openai_key: Optional[str] = None,  # noqa: ARG001 — legacy kept for callers
    duration_seconds: Optional[int] = None,  # noqa: ARG001 — currently unused
) -> Optional[dict]:
    """Tier 1 first, tier 2 (Modal Whisper) fallback.

    Returns ``{video_id, text, source, segments?}`` on success or None
    on full miss. ``source`` is one of ``youtube_transcript_api`` |
    ``whisper_modal``. Callers use it for cost telemetry.
    """
    if not video_id:
        return None

    t1 = _try_transcript_api(video_id)
    if t1 is not None:
        return t1

    # ── Tier 2: Modal Whisper (sync) ─────────────────────────────
    # When the async transcription mode is on, the pipeline skips
    # inline tier-2 entirely — it enqueues background jobs after the
    # CIP is stored. That decision lives in pipeline.pipeline, not here.
    from pipeline.whisper_client import is_configured, transcribe_sync

    if not video_url or not is_configured():
        return None

    result = transcribe_sync(video_url)
    if not result or not result.get("text"):
        return None
    return {
        "video_id": video_id,
        "source": "whisper_modal",
        "text": result["text"],
        "segments": result.get("segments") or [],
        "language": result.get("language"),
        "avg_confidence": result.get("avg_confidence"),
    }


def _try_transcript_api(video_id: str) -> Optional[dict]:
    """Tier 1 — ``youtube-transcript-api``.

    Library is unofficial; swallow every exception class and fall through.
    Common exceptions: TranscriptsDisabled, NoTranscriptFound,
    VideoUnavailable, RequestBlocked / IpBlocked.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
    except ImportError:
        logger.debug("youtube-transcript-api not installed; skipping tier 1")
        return None

    segments: list[dict] = []
    try:
        # 1.x API (April 2025 breaking change): instance method `.fetch()`.
        if hasattr(YouTubeTranscriptApi, "fetch") or callable(
            getattr(YouTubeTranscriptApi(), "fetch", None)
        ):
            api = YouTubeTranscriptApi()
            fetched = api.fetch(video_id)
            segments = (
                fetched.to_raw_data()
                if hasattr(fetched, "to_raw_data")
                else [
                    {"text": s.text, "start": s.start, "duration": s.duration}
                    for s in fetched.snippets
                ]
            )
        else:
            # 0.x classmethod (deprecated)
            segments = YouTubeTranscriptApi.get_transcript(video_id)  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001 — library raises many types
        logger.debug(f"youtube-transcript-api failed for {video_id}: {e}")
        return None

    if not segments:
        return None

    text = " ".join((s.get("text") or "") for s in segments).strip()
    if not text:
        return None

    return {
        "video_id": video_id,
        "source": "youtube_transcript_api",
        "text": text,
        "segments": segments,
    }
