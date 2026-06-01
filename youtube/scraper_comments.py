"""YouTube comments fetch — YouTube Data API v3 only.

Uses YouTube Data API's `commentThreads.list` as the sole source:
  - 1 quota unit per video (one call each, up to 100 threads per call)
  - `order=relevance` returns the top comments by default
  - No-OAuth read is allowed on public videos
"""

from datetime import datetime

from pipeline.youtube.youtube_api import YouTubeAPIClient


def scrape_comments(
    video_urls: list[str],
    *,
    yt_api: YouTubeAPIClient,
) -> list[dict]:
    """Fetch top comment threads for a batch of video URLs via the YT API."""
    if not yt_api or not yt_api.available:
        raise RuntimeError(
            "scrape_comments requires a configured YouTubeAPIClient; "
            "BrightData fallback was removed in the pipeline rewrite."
        )
    return _fetch_via_api(yt_api, video_urls)


def _fetch_via_api(
    yt_api: YouTubeAPIClient, video_urls: list[str]
) -> list[dict]:
    """Loop over video URLs calling `list_comment_threads` for each."""
    out: list[dict] = []
    for url in video_urls:
        vid = _video_id_from_url(url)
        if not vid:
            continue
        threads = yt_api.list_comment_threads(vid, order="relevance", max_results=20)
        # extract_comment_metrics expects Bright Data's field names; the
        # YouTubeAPIClient has already normalized to the same shape
        # (`author`, `author_channel_id`, `text`, `date`, `replies`).
        # Tag each thread with its source video URL so we can group comments
        # per-video for the per-post analysis path.
        for t in threads:
            t["_video_url"] = url
        out.extend(threads)
    return out


def _video_id_from_url(url: str) -> str | None:
    """Pull the 11-char video id out of any of the URL shapes YT uses."""
    if not url:
        return None
    if "v=" in url:
        return url.split("v=", 1)[1].split("&", 1)[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/", 1)[1].split("?", 1)[0].rstrip("/")
    if "/shorts/" in url:
        return url.split("/shorts/", 1)[1].split("?", 1)[0].rstrip("/")
    return None


def select_top_videos_for_comments(
    videos: list[dict], top_n: int = 5
) -> list[str]:
    """Pick top N by comment count for comment scraping (mirrors IG logic)."""
    commented = [
        v
        for v in videos
        if (v.get("comment_count") or v.get("num_comments") or 0) > 0
        and (v.get("url") or v.get("video_url"))
    ]
    commented.sort(
        key=lambda v: v.get("comment_count") or v.get("num_comments") or 0,
        reverse=True,
    )
    return [v.get("url") or v.get("video_url") for v in commented[:top_n]]


def extract_comment_metrics(
    comments: list[dict], creator_channel_id: str | None, creator_handle: str
) -> dict:
    """Compute the same Tier B/D metrics shape as IG scraper_comments.

    Creator reply detection uses `channel_id` match first (canonical), falling
    back to handle match if channel_id is absent in the scraped payload.
    """
    if not comments:
        return {}

    commenter_ids: list[str] = []  # YT channel ids are the identity here
    commenter_handles: list[str] = []
    comment_texts: list[str] = []
    comment_timestamps: list[datetime] = []
    creator_replies = 0
    # Per-video grouping for the per-post analysis path (keyed by video URL,
    # matching how build_items joins comments to items). Mirrors the IG path.
    comments_by_post: dict[str, list[dict]] = {}

    for comment in comments:
        author_id = comment.get("author_channel_id") or comment.get("channel_id")
        author_handle = (
            comment.get("author") or comment.get("author_name") or ""
        ).lstrip("@")
        text = comment.get("text") or comment.get("comment") or ""
        date_str = (
            comment.get("date")
            or comment.get("published_at")
            or comment.get("comment_date")
        )

        vurl = (comment.get("_video_url") or "").rstrip("/")
        if vurl:
            comments_by_post.setdefault(vurl, []).append(
                {"user": author_handle, "text": text, "timestamp": date_str}
            )

        commenter_ids.append(author_id or author_handle)
        commenter_handles.append(author_handle)
        comment_texts.append(text)

        if date_str:
            try:
                dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
                comment_timestamps.append(dt)
            except (ValueError, TypeError):
                pass

        # Replies: YT threads nest replies under the top-level comment.
        for reply in comment.get("replies") or []:
            r_id = reply.get("author_channel_id") or reply.get("channel_id")
            r_handle = (reply.get("author") or "").lstrip("@")
            if creator_channel_id and r_id and r_id == creator_channel_id:
                creator_replies += 1
            elif creator_handle and r_handle.lower() == creator_handle.lower():
                creator_replies += 1

    unique_commenters = list(set(commenter_ids))
    hour_distribution = _cluster_comment_hours(comment_timestamps)

    return {
        "creator_reply_count": creator_replies,
        "creator_reply_rate": round(creator_replies / max(len(comments), 1), 3),
        "unique_commenters": unique_commenters,
        "unique_commenter_count": len(unique_commenters),
        "_comment_texts": comment_texts,
        "_commenter_handles": commenter_handles,
        "_comment_timestamps": [dt.isoformat() for dt in comment_timestamps],
        "_comments_by_post": comments_by_post,
        "comment_hour_distribution_utc": hour_distribution,
    }


def _cluster_comment_hours(timestamps: list[datetime]) -> dict:
    if not timestamps:
        return {}
    hour_counts: dict[int, int] = {}
    for dt in timestamps:
        hour_counts[dt.hour] = hour_counts.get(dt.hour, 0) + 1
    total = len(timestamps)
    return {
        str(h): round(c / total, 3) for h, c in sorted(hour_counts.items())
    }
