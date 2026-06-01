"""Instagram reel transcription — routes through the Modal Whisper sidecar.

Used by ``pipeline.handlers.handle_content_video_analysis`` to transcribe
a single submitted reel. The OpenAI-Whisper-API + local-Whisper paths
were retired in the pipeline rewrite; this module is now a thin wrapper
over ``pipeline.whisper_client.transcribe_sync``.

Public API (unchanged for back-compat):
  - :func:`transcribe_reels(reels, openai_api_key=None)` — returns list of
    dicts shaped like the legacy OpenAI Whisper output so downstream
    consumers (content_analyzer) keep working.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from pipeline.whisper_client import transcribe_sync

logger = logging.getLogger(__name__)


def transcribe_reels(
    reel_data: list[dict],
    openai_api_key: str | None = None,  # noqa: ARG001 — legacy compat
) -> list[dict]:
    """Transcribe a batch of reel videos via Modal Whisper.

    ``reel_data`` items must carry ``video_url``; ``post_id``, ``caption``
    and ``length`` are optional and copied through to the output rows.
    """
    out: list[dict] = []
    for reel in reel_data:
        video_url = reel.get("video_url")
        if not video_url:
            continue

        modal_result = transcribe_sync(video_url)
        if not modal_result:
            out.append(
                {
                    "post_id": reel.get("post_id"),
                    "error": "modal whisper unavailable or failed",
                    "transcript_text": None,
                }
            )
            continue

        segments = modal_result.get("segments") or []
        text = modal_result.get("text") or ""
        result: dict[str, Any] = {
            "post_id": reel.get("post_id"),
            "caption": reel.get("caption", ""),
            "reel_length_seconds": reel.get("length", 0),
            "transcript_text": text,
            "detected_language": modal_result.get("language"),
            "segments": segments,
            "avg_confidence": modal_result.get("avg_confidence", 0.0),
        }
        result["hook_text"] = _extract_hook(segments, threshold_seconds=3.0)
        music_flag, music_confidence = _classify_music(result)
        result["is_likely_music"] = music_flag
        result["music_detection_confidence"] = music_confidence

        out.append(result)
        logger.info(
            "Transcribed reel %s (%s, %d seg)",
            reel.get("post_id"),
            result.get("detected_language") or "?",
            len(segments),
        )

    return out


def _extract_hook(segments: list[dict], threshold_seconds: float = 3.0) -> str:
    """First N seconds of transcript = the hook."""
    hook_parts = []
    for seg in segments:
        if seg.get("start", 0) < threshold_seconds:
            hook_parts.append((seg.get("text") or "").strip())
    return " ".join(hook_parts).strip()


def _classify_music(transcript_result: dict) -> tuple[bool, float]:
    """Heuristic music/speech classifier. Returns ``(is_music, confidence)``.

    Confidence is the strength of the classification in [0, 1]:
      - 0.95+ : explicit music marker (🎶/♪/♫ in text)
      - 0.75  : low Whisper confidence + sparse words
      - 0.65  : short reel, sparse words, low confidence
      - 0.00  : none of the above triggered
    """
    text = transcript_result.get("transcript_text", "") or ""
    conf = transcript_result.get("avg_confidence", 0) or 0
    reel_length = transcript_result.get("reel_length_seconds", 0) or 0
    word_count = len(text.split())
    words_per_second = word_count / max(reel_length, 1)

    music_markers = ["🎶", "♪", "music", "outro", "♫"]
    if any(m in text.lower() for m in music_markers):
        return True, 0.95
    if conf < 0.50 and words_per_second < 1.5:
        return True, 0.75
    if reel_length <= 10 and word_count <= 5 and conf < 0.60:
        return True, 0.65
    return False, 0.0
