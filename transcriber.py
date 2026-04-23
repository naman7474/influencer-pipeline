import math
import os
import tempfile
import logging

import requests
from openai import OpenAI

logger = logging.getLogger(__name__)


def transcribe_reels(
    reel_data: list[dict], openai_api_key: str
) -> list[dict]:
    """
    Download reel videos and transcribe with Whisper.

    Args:
        reel_data: List of dicts with 'video_url', 'post_id', 'caption', 'length'
                   (from extract_reel_metrics()["video_urls_for_whisper"])
        openai_api_key: OpenAI API key

    Returns:
        List of transcription results

    Cost: ~$0.006/min of audio via Whisper API
          5 reels x avg 30 sec = 2.5 min = ~$0.015
    """
    client = OpenAI(api_key=openai_api_key)
    results = []

    for reel in reel_data:
        video_url = reel.get("video_url")
        if not video_url:
            continue

        tmp_path = None
        try:
            # Step 1: Download the video to a temp file
            video_response = requests.get(video_url, timeout=60)
            video_response.raise_for_status()

            with tempfile.NamedTemporaryFile(
                suffix=".mp4", delete=False
            ) as tmp:
                tmp.write(video_response.content)
                tmp_path = tmp.name

            # Step 2: Send to Whisper
            with open(tmp_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )

            # Step 3: Parse the result
            result = {
                "post_id": reel.get("post_id"),
                "caption": reel.get("caption", ""),
                "reel_length_seconds": reel.get("length", 0),
                "transcript_text": transcription.text,
                "detected_language": transcription.language,
                "segments": [
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                    }
                    for seg in (transcription.segments or [])
                ],
                "avg_confidence": _avg_segment_confidence(
                    transcription.segments
                ),
            }

            # Step 4: Extract hook (first 3 seconds of transcript)
            result["hook_text"] = _extract_hook(
                result["segments"], threshold_seconds=3.0
            )

            # Step 5: Detect if transcript is likely background music
            music_flag, music_confidence = _classify_music(result)
            result["is_likely_music"] = music_flag
            result["music_detection_confidence"] = music_confidence

            results.append(result)
            logger.info(
                f"Transcribed reel {reel.get('post_id')} "
                f"({transcription.language})"
            )

        except Exception as e:
            logger.error(f"Transcription failed for {reel.get('post_id')}: {e}")
            results.append(
                {
                    "post_id": reel.get("post_id"),
                    "error": str(e),
                    "transcript_text": None,
                }
            )

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return results


def _extract_hook(
    segments: list[dict], threshold_seconds: float = 3.0
) -> str:
    """Extract the first N seconds of transcript as the 'hook'."""
    hook_parts = []
    for seg in segments:
        if seg.get("start", 0) < threshold_seconds:
            hook_parts.append(seg.get("text", "").strip())
    return " ".join(hook_parts).strip()


def _is_likely_music(transcript_result: dict) -> bool:
    """Legacy wrapper — keeps the boolean API callers still use."""
    flag, _ = _classify_music(transcript_result)
    return flag


def _classify_music(transcript_result: dict) -> tuple[bool, float]:
    """
    Heuristic music/speech classifier. Returns (is_music, confidence).

    Confidence is the strength of the classification in [0, 1]:
      - 0.95+ : explicit music marker (🎶/♪/♫ in text) — near certain
      - 0.75  : low Whisper confidence + sparse words — likely music
      - 0.65  : short reel, sparse words, low confidence
      - 0.00  : none of the above triggered
    Downstream scorers can weight transcripts by this uncertainty
    instead of treating the boolean as ground truth.
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


def _avg_segment_confidence(segments) -> float:
    """
    Compute average confidence from Whisper segments.
    Higher confidence -> cleaner audio -> better production quality.

    Note: avg_logprob is negative; closer to 0 = more confident.
    We convert to a 0-1 scale where 1 = most confident.
    """
    if not segments:
        return 0.0

    logprobs = []
    for seg in segments:
        if hasattr(seg, "avg_logprob") and seg.avg_logprob is not None:
            logprobs.append(seg.avg_logprob)

    if not logprobs:
        return 0.0

    avg_logprob = sum(logprobs) / len(logprobs)
    return round(math.exp(avg_logprob), 3)
