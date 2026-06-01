"""Instagram reels scrape — Apify-backed."""
from pipeline import apify_instagram_bundle


def scrape_reels_discovery(
    profile_url: str,
    num_reels: int = 10,
) -> list[dict]:
    """Discover reels for a creator via the cached Apify bundle."""
    username = _url_to_username(profile_url)
    bundle = apify_instagram_bundle.fetch(username, num_reels=num_reels)
    return bundle["reels"]


def _url_to_username(url: str) -> str:
    cleaned = url.rstrip("/")
    return cleaned.rsplit("/", 1)[-1].lstrip("@")


def select_top_reels(raw_reels: list[dict], top_n: int = 5) -> list[dict]:
    """From discovered reels, pick the top N by engagement for transcription."""
    reels = [r for r in raw_reels if r.get("url")]

    reels.sort(
        key=lambda r: (r.get("likes", 0) or 0)
        + (r.get("num_comments", 0) or 0),
        reverse=True,
    )

    return reels[:top_n]


def _to_num(val, default=0):
    """Coerce a value to float — upstream sometimes returns strings."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def extract_reel_metrics(reels: list[dict]) -> dict:
    """Compute Tier B reel-specific metrics."""
    if not reels:
        return {}

    views_to_likes = []
    rewatch_rates = []
    lengths = []
    video_urls = []
    all_top_comments = []

    for reel in reels:
        views = _to_num(reel.get("views") or reel.get("video_view_count"))
        plays = _to_num(reel.get("video_play_count"))
        likes = _to_num(reel.get("likes"))
        length = _to_num(reel.get("length"))

        if views > 0:
            views_to_likes.append(likes / views)

        if views > 0:
            rewatch_rates.append(plays / views)

        if length > 0:
            lengths.append(length)

        video_url = reel.get("video_url")
        if video_url:
            video_urls.append(
                {
                    "post_id": reel.get("post_id"),
                    "video_url": video_url,
                    "caption": reel.get("description", ""),
                    "length": length,
                }
            )

        top_comments = reel.get("top_comments") or []
        for comment in top_comments:
            all_top_comments.append(
                {
                    "user": comment.get("comment_user")
                    or comment.get("user_commenting"),
                    "text": comment.get("comment") or comment.get("text"),
                    "date": comment.get("comment_date")
                    or comment.get("date_of_comment"),
                    "likes": comment.get("likes_number")
                    or comment.get("likes"),
                    "source_post_id": reel.get("post_id"),
                }
            )

    return {
        "avg_views_to_likes_ratio": round(
            sum(views_to_likes) / max(len(views_to_likes), 1), 4
        ),
        "avg_rewatch_rate": round(
            sum(rewatch_rates) / max(len(rewatch_rates), 1), 3
        ),
        "avg_reel_length_seconds": round(
            sum(lengths) / max(len(lengths), 1), 1
        ),
        "video_urls_for_whisper": video_urls,
        "top_comments_from_reels": all_top_comments,
    }
