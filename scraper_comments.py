"""Instagram comments scrape — Apify-backed."""
import logging
from datetime import datetime

from pipeline import apify_instagram_bundle

logger = logging.getLogger(__name__)


def scrape_comments(post_urls: list[str]) -> list[dict]:
    """Scrape comments for a batch of post URLs via the cached Apify bundle.

    The bundle is keyed by username, so the comments come pre-scraped
    when ``apify_instagram_bundle.fetch()`` ran for this creator. Here
    we filter the cache down to the requested post URLs (or fall back
    to the full set if filtering removes everything).
    """
    if not post_urls:
        return []
    username = apify_instagram_bundle.any_cached_username()
    if not username:
        logger.warning("No cached bundle; pipeline must scrape profile first.")
        return []
    bundle = apify_instagram_bundle.get_cached(username) or {}
    all_comments = bundle.get("comments") or []
    wanted = {u.rstrip("/") for u in post_urls}
    filtered = [
        c for c in all_comments
        if (c.get("post_url") or "").rstrip("/") in wanted
    ]
    return filtered or all_comments


def select_top_posts_for_comments(
    posts: list[dict], top_n: int = 5
) -> list[str]:
    """
    Pick the top N posts by comment count for comment scraping.

    We want the most-commented posts because:
    1. More data for language/geo inference
    2. Higher signal-to-noise ratio
    3. Better representation of engaged audience
    """
    commented_posts = [
        p
        for p in posts
        if (p.get("num_comments") or 0) > 0 and p.get("url")
    ]
    commented_posts.sort(
        key=lambda p: p.get("num_comments", 0), reverse=True
    )
    return [p["url"] for p in commented_posts[:top_n]]


def extract_comment_metrics(
    comments: list[dict], creator_handle: str
) -> dict:
    """Compute Tier B and Tier D metrics from comments."""
    if not comments:
        return {}

    commenter_handles = []
    comment_texts = []
    comment_timestamps = []
    creator_replies = 0
    # Per-post grouping for the per-video analysis path (LLM_PER_POST). Keyed by
    # normalised post_url so the orchestrator can give each post its own
    # comments for comment-classification + audience-intent. The flattened
    # fields below are preserved unchanged for the legacy creator-level path.
    comments_by_post: dict[str, list[dict]] = {}

    for comment in comments:
        user = comment.get("comment_user") or comment.get("user_commenting", "")
        text = comment.get("comment") or comment.get("comments", "")
        date_str = comment.get("comment_date") or comment.get("date_of_comment")

        commenter_handles.append(user)
        comment_texts.append(text)

        iso_ts = None
        if date_str:
            try:
                dt = datetime.fromisoformat(
                    str(date_str).replace("Z", "+00:00")
                )
                comment_timestamps.append(dt)
                iso_ts = dt.isoformat()
            except (ValueError, TypeError):
                pass

        post_url = (
            comment.get("source_post_url")
            or comment.get("post_url")
            or comment.get("postUrl")
            or comment.get("input_url")
            or comment.get("inputUrl")
            or ""
        ).rstrip("/")
        if post_url:
            comments_by_post.setdefault(post_url, []).append(
                {"user": user, "text": text, "timestamp": iso_ts}
            )

        replies = comment.get("replies") or []
        for reply in replies:
            reply_user = (
                reply.get("comment_user")
                or reply.get("user_commenting", "")
            )
            if reply_user.lower() == creator_handle.lower():
                creator_replies += 1

    unique_commenters = list(set(commenter_handles))
    hour_distribution = _cluster_comment_hours(comment_timestamps)

    return {
        "creator_reply_count": creator_replies,
        "creator_reply_rate": round(
            creator_replies / max(len(comments), 1), 3
        ),
        "unique_commenters": unique_commenters,
        "unique_commenter_count": len(unique_commenters),
        "_comment_texts": comment_texts,
        "_commenter_handles": commenter_handles,
        "_comment_timestamps": [dt.isoformat() for dt in comment_timestamps],
        "_comments_by_post": comments_by_post,
        "comment_hour_distribution_utc": hour_distribution,
    }


def _cluster_comment_hours(timestamps: list[datetime]) -> dict:
    """Cluster comment timestamps by UTC hour."""
    if not timestamps:
        return {}

    hour_counts = {}
    for dt in timestamps:
        h = dt.hour
        hour_counts[h] = hour_counts.get(h, 0) + 1

    total = len(timestamps)
    return {
        str(h): round(c / total, 3) for h, c in sorted(hour_counts.items())
    }
