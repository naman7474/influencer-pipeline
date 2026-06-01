"""YouTube video discovery — YouTube Data API v3 only.

Uses the YouTube Data API as the sole video discovery source:
  - ~3 quota units per channel (channels.list + playlistItems.list + videos.list)
  - Canonical stats (views/likes/comments fresh from the source)
  - Free within the 10K/day default quota (~3300 channels/day on one key)

Parallel to pipeline/scraper_reels.py + scraper_posts.py combined.
"""

from pipeline.youtube.youtube_api import YouTubeAPIClient


def scrape_videos_discovery(
    channel_url: str,  # noqa: ARG001 — kept in signature for symmetry; api path
                       # works off channel_id, not url
    num_videos: int = 20,
    *,
    yt_api: YouTubeAPIClient,
    channel_id: str,
) -> list[dict]:
    """Discover recent videos for a channel via the YouTube Data API.

    Returns records normalized to the same keys ``extract_video_metrics``
    expects (url, views, likes, num_comments, length, published_at, etc.).
    """
    if not yt_api or not yt_api.available or not channel_id:
        raise RuntimeError(
            "scrape_videos_discovery requires a configured YouTubeAPIClient "
            "and a channel_id."
        )
    return _discover_via_api(yt_api, channel_id, num_videos)


def _discover_via_api(
    yt_api: YouTubeAPIClient, channel_id: str, num_videos: int
) -> list[dict]:
    """Discovery via YouTube Data API. Normalizes the API output to the
    Bright Data field names `extract_video_metrics` already expects."""
    raw = yt_api.list_channel_uploads(channel_id, limit=num_videos)
    return [_api_record_to_bd_shape(r) for r in raw]


def _api_record_to_bd_shape(rec: dict) -> dict:
    """Translate one `fetch_video_stats`-shaped record into the keys
    `extract_video_metrics` expects from Bright Data.
    """
    video_id = rec.get("video_id") or rec.get("id")
    url = (
        f"https://www.youtube.com/watch?v={video_id}"
        if video_id
        else None
    )
    return {
        # identity + url
        "video_id": video_id,
        "url": url,
        "title": rec.get("title"),
        "description": rec.get("description"),
        "tags": rec.get("tags") or [],
        "category_id": rec.get("category_id"),
        # live / shorts detection — the extractor's `_is_short` looks at
        # /shorts/ in the url; the API doesn't mark shorts explicitly, so
        # we also pass duration for the fallback-threshold check.
        "is_live": rec.get("live_broadcast_content") == "live",
        "is_livestream": rec.get("live_broadcast_content") == "live",
        "length": _iso8601_duration_to_seconds(rec.get("duration_iso8601")),
        "duration_seconds": _iso8601_duration_to_seconds(
            rec.get("duration_iso8601")
        ),
        # stats
        "views": rec.get("view_count") or 0,
        "view_count": rec.get("view_count") or 0,
        "likes": rec.get("like_count") or 0,
        "like_count": rec.get("like_count") or 0,
        "num_comments": rec.get("comment_count") or 0,
        "comment_count": rec.get("comment_count") or 0,
        # metadata — synthesize thumbnail from the public ytimg CDN. No
        # API call needed: every public video has hqdefault.jpg at this
        # URL pattern. maxresdefault.jpg only exists for HD uploads, so
        # hqdefault is the safer choice for a unified fallback.
        "thumbnail": (
            f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg" if video_id else None
        ),
        "thumbnail_url": (
            f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg" if video_id else None
        ),
        "date_posted": rec.get("published_at"),
        "published_at": rec.get("published_at"),
        # captions: the API exposes contentDetails.caption which
        # `fetch_video_stats` maps to `has_captions`. No inline transcript
        # text from the API — that's the transcripts.py tier-1/2 job.
        "has_captions": rec.get("has_captions", False),
        "captions_available": rec.get("has_captions", False),
        # empty top_comments so downstream doesn't crash; real comments
        # arrive via `list_comment_threads` on the videos we actually
        # care about.
        "top_comments": [],
    }


def _iso8601_duration_to_seconds(iso: str | None) -> float:
    """Parse `PT1H23M45S` to seconds. Returns 0 on invalid input.

    YouTube API returns durations in ISO 8601. Full parsing via isodate
    would be more correct but adds a dep — the subset YT emits (PT + H/M/S)
    is parseable with a tiny regex.
    """
    if not iso or not isinstance(iso, str):
        return 0
    import re

    m = re.fullmatch(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso.strip()
    )
    if not m:
        return 0
    h = int(m.group(1) or 0)
    mi = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    return float(h * 3600 + mi * 60 + s)


def select_top_videos(
    raw_videos: list[dict], top_n: int = 10, include_shorts: bool = True
) -> list[dict]:
    """Pick top N videos by engagement for transcript analysis.

    YT engagement = views + (likes*10) + (comments*20). Likes and comments are
    scarcer than views, so they're weighted up to surface genuinely engaging
    content, not just viral-but-shallow hits.
    """
    videos = [v for v in raw_videos if v.get("url") or v.get("video_url")]
    if not include_shorts:
        videos = [v for v in videos if not _is_short(v)]

    def score(v: dict) -> int:
        views = _to_int(v.get("views") or v.get("view_count"))
        likes = _to_int(v.get("likes") or v.get("like_count"))
        comments = _to_int(v.get("num_comments") or v.get("comment_count"))
        return views + likes * 10 + comments * 20

    videos.sort(key=score, reverse=True)
    return videos[:top_n]


def _to_int(val, default: int = 0) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        try:
            return int(float(val))
        except (TypeError, ValueError):
            return default


def _to_num(val, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _is_short(video: dict) -> bool:
    """A video is a Short when URL contains /shorts/ or duration <= 60s."""
    url = (video.get("url") or video.get("video_url") or "").lower()
    if "/shorts/" in url:
        return True
    duration = _to_num(video.get("length") or video.get("duration_seconds"))
    return 0 < duration <= 60


def extract_video_metrics(raw_video: dict) -> dict:
    """Normalize one Bright Data YT video record into `youtube_videos` shape."""
    views = _to_int(raw_video.get("views") or raw_video.get("view_count"))
    likes = _to_int(raw_video.get("likes") or raw_video.get("like_count"))
    comments = _to_int(
        raw_video.get("num_comments") or raw_video.get("comment_count")
    )
    duration = _to_num(
        raw_video.get("length") or raw_video.get("duration_seconds")
    )

    url = raw_video.get("url") or raw_video.get("video_url") or ""
    video_id = (
        raw_video.get("video_id")
        or raw_video.get("id")
        or _extract_video_id_from_url(url)
    )

    # Captions: Bright Data sometimes returns "transcript" inline on the video
    # record, sometimes a boolean "has_captions" / "captions_available" flag.
    transcript_inline = raw_video.get("transcript") or raw_video.get("captions")
    has_captions = bool(
        raw_video.get("has_captions")
        or raw_video.get("captions_available")
        or transcript_inline
    )

    return {
        "video_id": video_id,
        "url": url,
        "title": raw_video.get("title") or raw_video.get("name"),
        "description": raw_video.get("description"),
        "tags": raw_video.get("tags") or raw_video.get("keywords") or [],
        "category_id": _to_int(
            raw_video.get("category_id") or raw_video.get("categoryId"), default=0
        )
        or None,
        "is_short": _is_short(raw_video),
        "is_livestream": bool(raw_video.get("is_live") or raw_video.get("is_livestream")),
        "duration_seconds": duration,
        "view_count": views,
        "like_count": likes,
        "comment_count": comments,
        "thumbnail_url": raw_video.get("thumbnail") or raw_video.get("thumbnail_url"),
        "has_captions": has_captions,
        "caption_source": "youtube_auto" if has_captions else None,
        "transcript_inline": transcript_inline,  # consumed by transcriber stage
        "published_at": raw_video.get("date_posted") or raw_video.get("published_at"),
        # Top comments used by audience intelligence (mirrors reels flow)
        "top_comments": raw_video.get("top_comments") or [],
    }


def aggregate_channel_metrics(videos: list[dict]) -> dict:
    """Compute Tier B channel-level metrics across a sample of recent videos.

    Analogous to `extract_reel_metrics` on the IG side. Produces the
    YT-specific inputs the scorer in confidence.py needs:

      - avg_views_per_sub: views / subs; a watch-time proxy.
      - watch_through_proxy: likes/views, weak retention signal.
      - upload_cadence_days: avg gap between published_at timestamps.
      - content_mix: short vs long vs live counts.
    """
    if not videos:
        return {}

    likes_per_view = []
    view_counts = []
    shorts = 0
    longs = 0
    lives = 0
    lengths = []
    all_top_comments = []
    publish_ts = []

    for v in videos:
        views = _to_int(v.get("view_count") or v.get("views"))
        likes = _to_int(v.get("like_count") or v.get("likes"))
        if views > 0:
            likes_per_view.append(likes / views)
            view_counts.append(views)
        length = _to_num(v.get("duration_seconds") or v.get("length"))
        if length > 0:
            lengths.append(length)
        if v.get("is_short"):
            shorts += 1
        elif v.get("is_livestream"):
            lives += 1
        else:
            longs += 1
        for comment in v.get("top_comments") or []:
            all_top_comments.append(
                {
                    "user": comment.get("user") or comment.get("author"),
                    "text": comment.get("text") or comment.get("comment"),
                    "date": comment.get("date") or comment.get("published_at"),
                    "likes": comment.get("likes") or comment.get("like_count"),
                    "source_video_id": v.get("video_id"),
                }
            )
        if v.get("published_at"):
            publish_ts.append(v["published_at"])

    avg_views = sum(view_counts) / max(len(view_counts), 1)

    # upload cadence: avg gap between sorted timestamps.
    upload_cadence_days = None
    if len(publish_ts) >= 2:
        from datetime import datetime

        def _parse(ts):
            try:
                return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except ValueError:
                return None

        parsed = sorted(filter(None, (_parse(t) for t in publish_ts)))
        if len(parsed) >= 2:
            gaps = [
                (parsed[i + 1] - parsed[i]).total_seconds() / 86400
                for i in range(len(parsed) - 1)
            ]
            upload_cadence_days = round(sum(gaps) / len(gaps), 1)

    return {
        "avg_view_count": round(avg_views, 1),
        "watch_through_proxy": round(
            sum(likes_per_view) / max(len(likes_per_view), 1), 3
        ),
        "avg_video_length_seconds": round(sum(lengths) / max(len(lengths), 1), 1),
        "upload_cadence_days": upload_cadence_days,
        "content_mix": {
            "youtube_short": shorts,
            "youtube_long": longs,
            "youtube_live": lives,
        },
        "top_comments_from_videos": all_top_comments,
    }


def _extract_video_id_from_url(url: str) -> str | None:
    """Pull `abc123XYZ00` out of the common URL shapes.

    https://www.youtube.com/watch?v=<id>
    https://youtu.be/<id>
    https://www.youtube.com/shorts/<id>
    """
    if not url:
        return None
    if "v=" in url:
        tail = url.split("v=", 1)[1]
        return tail.split("&", 1)[0]
    if "youtu.be/" in url:
        tail = url.split("youtu.be/", 1)[1]
        return tail.split("?", 1)[0].rstrip("/")
    if "/shorts/" in url:
        tail = url.split("/shorts/", 1)[1]
        return tail.split("?", 1)[0].rstrip("/")
    return None
