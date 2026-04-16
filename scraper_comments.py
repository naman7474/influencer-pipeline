from datetime import datetime

from pipeline.brightdata_client import BrightdataClient

DATASET_COMMENTS = "gd_ltppn085pokosxh13"


def scrape_comments(
    client: BrightdataClient, post_urls: list[str]
) -> list[dict]:
    """
    Scrape recent comments from posts.

    Each URL returns up to 10 most recent comments.
    For 5 posts, that's ~50 comment records.

    Cost: ~10 records per post URL = $0.075 for 5 posts at $1.50/1K
    """
    payload = [{"url": url} for url in post_urls]
    return client.trigger_and_wait(DATASET_COMMENTS, payload)


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

    for comment in comments:
        user = comment.get("comment_user") or comment.get("user_commenting", "")
        text = comment.get("comment") or comment.get("comments", "")
        date_str = comment.get("comment_date") or comment.get("date_of_comment")

        commenter_handles.append(user)
        comment_texts.append(text)

        if date_str:
            try:
                dt = datetime.fromisoformat(
                    str(date_str).replace("Z", "+00:00")
                )
                comment_timestamps.append(dt)
            except (ValueError, TypeError):
                pass

        # Check replies for creator engagement
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
        "comment_hour_distribution_utc": hour_distribution,
    }


def _cluster_comment_hours(timestamps: list[datetime]) -> dict:
    """
    Cluster comment timestamps by UTC hour.

    For Indian audiences:
    - IST = UTC+5:30
    - Peak evening hours in India (7-11 PM IST) = 13:30-17:30 UTC
    - If most comments cluster in UTC 13-18, strong India signal
    """
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
