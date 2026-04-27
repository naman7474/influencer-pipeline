"""Tiered transcript fetcher for YouTube videos.

Bright Data does NOT offer a dedicated captions dataset (only
Videos / Channels / Comments). YouTube's official `captions.download`
API requires OAuth as the channel owner. So our two-tier flow is:

  Tier 1 — youtube-transcript-api (free)
    Scrapes YouTube's internal transcript endpoint (same one the YT UI
    fetches for the panel). Works for any public video with captions
    (manual or auto), no API key required. Breaks when YT blocks the
    calling IP — common on cloud datacenters. The library accepts a
    `proxy_config` if you need to route through residential proxies;
    we leave that off by default and rely on Tier 2 when blocked.

  Tier 2 — Whisper on yt-dlp audio (~$0.01/video)
    Last resort. Download audio via yt-dlp, send to OpenAI Whisper.
    Handles videos with no captions at all and videos where Tier 1
    is rate-limited. Costs more but works without external auth.

The legacy Bright Data captions tier was removed once we confirmed no
such dataset exists in the BD catalog.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from pipeline.brightdata_client import BrightdataClient

logger = logging.getLogger(__name__)


# Long-video handling thresholds.
# Videos longer than LONG_VIDEO_THRESHOLD_S get sliced into three windows
# (first / middle / last) to stay under Whisper's 25 MB hard cap and give
# the LLM a representative cross-section instead of cutting off at minute 25.
LONG_VIDEO_THRESHOLD_S = 300  # 5 min — under this, download in full
SAMPLE_WINDOW_S = 150         # 2.5 min per slice (×3 = 7.5 min audio total)
YT_DLP_TIMEOUT_S = 90         # was 180; bursty hangs cost more than retries


def fetch_transcript(
    video_id: str,
    video_url: str,
    bd_client: Optional[BrightdataClient] = None,  # noqa: ARG001 — kept for back-compat
    openai_key: Optional[str] = None,
    duration_seconds: Optional[int] = None,
) -> Optional[dict]:
    """Tiered transcript fetcher.

    Returns {video_id, text, source, segments?} on success, None on full
    miss. `source` is one of: 'youtube_transcript_api', 'whisper',
    'whisper_sampled'. Callers use `source` for cost telemetry.

    `bd_client` is accepted but unused — Bright Data has no captions
    dataset. Argument retained so existing call sites don't break.

    `duration_seconds` is optional but recommended — when set, long videos
    fall into the 3-segment sampler instead of attempting a full download.
    """
    if not video_id:
        return None

    # ── Tier 1: youtube-transcript-api (free) ────────────────────
    t1 = _try_transcript_api(video_id)
    if t1 is not None:
        return t1

    # ── Tier 2: Whisper on yt-dlp audio (~$0.01) ─────────────────
    if openai_key and video_url:
        t2 = _try_whisper(
            video_id, video_url, openai_key, duration_seconds=duration_seconds
        )
        if t2 is not None:
            return t2

    return None


# ─────────────────────────────────────────────────────────────
# Tier implementations
# ─────────────────────────────────────────────────────────────


def _try_transcript_api(video_id: str) -> Optional[dict]:
    """Tier 1 — youtube-transcript-api.

    Library is unofficial; swallow every exception class and fall through.
    The exceptions we expect: TranscriptsDisabled, NoTranscriptFound,
    VideoUnavailable, RequestBlocked / IpBlocked (rate-limit).

    The library shipped a breaking API change in 1.0 (April 2025): the
    classmethod `YouTubeTranscriptApi.get_transcript(video_id)` was
    replaced by an instance method `YouTubeTranscriptApi().fetch(video_id)`
    that returns a `FetchedTranscript` object exposing `.snippets`.
    We support 1.x first; if that attribute doesn't exist we fall back
    to the legacy classmethod so older installs still work.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
    except ImportError:
        logger.debug("youtube-transcript-api not installed; skipping tier 1")
        return None

    segments: list[dict] = []
    try:
        # 1.x API
        if hasattr(YouTubeTranscriptApi, "fetch") or callable(
            getattr(YouTubeTranscriptApi(), "fetch", None)
        ):
            api = YouTubeTranscriptApi()
            fetched = api.fetch(video_id)
            # FetchedTranscript exposes .snippets (list of objects with
            # .text/.start/.duration) and .to_raw_data() for the legacy
            # list-of-dicts shape downstream consumers may want.
            raw = (
                fetched.to_raw_data()
                if hasattr(fetched, "to_raw_data")
                else [
                    {"text": s.text, "start": s.start, "duration": s.duration}
                    for s in fetched.snippets
                ]
            )
            segments = raw
        else:
            # 0.x API (deprecated, kept for back-compat)
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


def _try_whisper(
    video_id: str,
    video_url: str,
    openai_key: str,
    duration_seconds: Optional[int] = None,
) -> Optional[dict]:
    """Tier 3 — yt-dlp audio download + OpenAI Whisper.

    For videos over `LONG_VIDEO_THRESHOLD_S`, downloads three sampled
    segments (first / middle / last `SAMPLE_WINDOW_S` seconds each) so we
    stay under Whisper's 25 MB cap without losing a fair representation
    of the video. Short videos go through the full-download path.
    """
    is_sampled = bool(
        duration_seconds and duration_seconds > LONG_VIDEO_THRESHOLD_S
    )
    audio_path = _download_audio_yt_dlp(
        video_url, duration_seconds=duration_seconds
    )
    if not audio_path:
        return None
    try:
        text = _whisper_transcribe(audio_path, openai_key)
    finally:
        try:
            Path(audio_path).unlink(missing_ok=True)
        except OSError:
            pass

    if not text:
        return None

    return {
        "video_id": video_id,
        "source": "whisper_sampled" if is_sampled else "whisper",
        "text": text,
    }


def _download_audio_yt_dlp(
    video_url: str, duration_seconds: Optional[int] = None
) -> Optional[str]:
    """Invoke yt-dlp via subprocess; returns filepath or None on failure.

    For long videos (> ``LONG_VIDEO_THRESHOLD_S``), passes three
    ``--download-sections`` flags so yt-dlp splices a single audio file
    containing the first / middle / last ``SAMPLE_WINDOW_S`` seconds of
    the source. Short videos download in full.
    """
    try:
        import yt_dlp  # noqa: F401 — just to confirm installed
    except ImportError:
        logger.debug("yt-dlp not installed; skipping tier 3")
        return None

    tmpdir = tempfile.mkdtemp(prefix="yt_transcript_")
    out_template = str(Path(tmpdir) / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f",
        "bestaudio/best",
        "-x",  # extract audio
        "--audio-format",
        "m4a",
        "--audio-quality",
        "5",  # lower quality = smaller file, fine for transcription
        "--no-warnings",
        "--quiet",
        "-o",
        out_template,
    ]

    if duration_seconds and duration_seconds > LONG_VIDEO_THRESHOLD_S:
        d = int(duration_seconds)
        w = SAMPLE_WINDOW_S
        # First w sec, w sec centred on the midpoint, last w sec.
        first = (0, w)
        mid_start = max(d // 2 - w // 2, w)
        middle = (mid_start, mid_start + w)
        last = (max(d - w, w), d)
        for start, end in (first, middle, last):
            cmd += ["--download-sections", f"*{start}-{end}"]
        logger.info(
            "yt-dlp sampling 3×%ss segments (video duration %ss): %s",
            w,
            d,
            ", ".join(f"{s}-{e}" for s, e in (first, middle, last)),
        )

    cmd.append(video_url)

    try:
        subprocess.run(
            cmd, check=True, timeout=YT_DLP_TIMEOUT_S, capture_output=True
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"yt-dlp audio download failed for {video_url}: {e}")
        return None

    # yt-dlp writes the file with the video id as the basename.
    for candidate in Path(tmpdir).iterdir():
        if candidate.is_file() and candidate.stat().st_size > 0:
            return str(candidate)
    return None


def _whisper_transcribe(audio_path: str, openai_key: str) -> Optional[str]:
    """Run the audio file through OpenAI's Whisper. Returns text or None."""
    try:
        from openai import OpenAI
    except ImportError:
        logger.debug("openai library not installed; cannot run Whisper")
        return None

    try:
        client = OpenAI(api_key=openai_key)
        with open(audio_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
            )
        return (resp.text or "").strip() or None
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Whisper transcription failed for {audio_path}: {e}")
        return None
