"""Instagram posts scrape — Apify-backed."""
import logging
import os
import re
from collections import Counter
from datetime import datetime

from pipeline import apify_instagram_bundle
from pipeline.apify_client import make_default_client

logger = logging.getLogger(__name__)

# Used by content video analysis (handlers.handle_content_video_analysis)
# to grab a single post by URL. The bundle fetches by username; this
# direct actor call is the right tool for one-off URL lookups.
APIFY_ACTOR_POST = os.environ.get(
    "APIFY_ACTOR_IG_POST", "apify/instagram-post-scraper"
)


def scrape_single_post(post_url: str) -> dict | None:
    """Scrape a single Instagram post/reel by URL via Apify.

    Used by the content-submission analyzer to fetch video_url + metadata
    for a creator-submitted reel. Returns a BD-shaped dict (the actor
    output is translated to keep downstream consumers unchanged) or None
    if Apify returned nothing.
    """
    client = make_default_client()
    items = client.trigger_and_wait(
        APIFY_ACTOR_POST, {"directUrls": [post_url], "resultsLimit": 1}
    )
    if not items:
        logger.warning("No data returned for post: %s", post_url)
        return None
    return _translate_apify_post(items[0])


def _translate_apify_post(item: dict) -> dict:
    """Translate one apify/instagram-post-scraper item into the BD-shaped
    fields downstream consumers (extract_post_metrics, content_analyzer)
    expect.
    """
    return {
        "post_id": item.get("id") or item.get("shortCode"),
        "url": item.get("url"),
        "description": item.get("caption") or "",
        "likes": item.get("likesCount") or 0,
        "num_comments": item.get("commentsCount") or 0,
        "views": item.get("videoViewCount") or item.get("videoPlayCount") or 0,
        "video_view_count": item.get("videoViewCount") or 0,
        "video_play_count": item.get("videoPlayCount") or 0,
        "length": item.get("videoDuration") or 0,
        "video_url": item.get("videoUrl"),
        "content_type": "Video" if item.get("type") == "Video" else item.get("type"),
        "date_posted": item.get("timestamp"),
        "hashtags": item.get("hashtags") or [],
        "tagged_users": item.get("taggedUsers") or [],
        "is_paid_partnership": item.get("isSponsored") or False,
    }


def scrape_posts_discovery(
    profile_url: str,
    num_posts: int = 5,
    days_back: int = 90,  # noqa: ARG001 — kept for signature stability; the
                          # bundle applies its own recency window
) -> list[dict]:
    """Discover recent posts from a creator's profile via the Apify bundle."""
    username = _url_to_username(profile_url)
    bundle = apify_instagram_bundle.fetch(username, num_posts=num_posts)
    return bundle["posts"]


def _url_to_username(url: str) -> str:
    cleaned = url.rstrip("/")
    return cleaned.rsplit("/", 1)[-1].lstrip("@")


def _normalize_content_type(raw_type: str) -> str:
    """Normalize raw content types to match DB enum values."""
    mapping = {
        "Reel": "Video",
        "reel": "Video",
        "Sidecar": "Carousel",
        "sidecar": "Carousel",
        "GraphVideo": "Video",
        "GraphImage": "Image",
        "GraphSidecar": "Carousel",
    }
    return mapping.get(raw_type, raw_type)


def _median(values: list[float]) -> float:
    """Compute proper median — averages two middle elements for even-length lists."""
    if not values:
        return 0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2
    return s[mid]


def extract_post_metrics(
    posts: list[dict], follower_count: int, handle: str = ""
) -> dict:
    """
    Compute all Tier A metrics from the posts array.

    Args:
        posts: Raw post data (BD-shaped, also what the Apify bundle emits)
        follower_count: From the profile scrape (denominator for engagement rate)
        handle: Creator's Instagram handle (to filter from brand mentions)
    """
    if not posts:
        return {}

    if follower_count is not None and follower_count < 100:
        return {
            "data_quality_flag": "low_followers",
            "avg_engagement_rate": None,
            "median_engagement_rate": None,
            "posts_per_week": 0,
            "posting_consistency_stddev_days": None,
            "engagement_trend": "insufficient_data",
            "_captions": [
                p.get("description", "") for p in posts if p.get("description")
            ],
        }

    engagement_rates = []
    likes_comments_ratios = []
    engagement_by_type = {"Video": [], "Carousel": [], "Image": []}

    sponsored_posts = []
    organic_posts = []
    brand_mentions = set()
    all_hashtags = []
    post_datetimes = []

    for post in posts:
        likes = post.get("likes", 0) or 0
        comments = post.get("num_comments", 0) or 0
        ct = _normalize_content_type(post.get("content_type", "Image"))

        if follower_count > 0:
            er = (likes + comments) / follower_count
            engagement_rates.append(er)

            if ct in engagement_by_type:
                engagement_by_type[ct].append(er)

        if comments > 0:
            likes_comments_ratios.append(likes / comments)

        is_sponsored = _is_sponsored_post(post)
        if is_sponsored:
            sponsored_posts.append(post)
        else:
            organic_posts.append(post)

        mentions = _extract_brand_mentions(post)
        if handle:
            mentions.discard(handle)
            mentions.discard(handle.lower())
        brand_mentions.update(mentions)

        if post.get("hashtags"):
            all_hashtags.extend(post["hashtags"])

        if post.get("date_posted"):
            try:
                dt = datetime.fromisoformat(
                    post["date_posted"].replace("Z", "+00:00")
                )
                post_datetimes.append(dt)
            except (ValueError, TypeError):
                pass

    post_datetimes.sort()

    posting_gaps_days = []
    for i in range(1, len(post_datetimes)):
        gap = (post_datetimes[i] - post_datetimes[i - 1]).total_seconds() / 86400
        posting_gaps_days.append(gap)

    type_counts = {}
    for post in posts:
        ct = _normalize_content_type(post.get("content_type", "Image"))
        type_counts[ct] = type_counts.get(ct, 0) + 1
    total_posts = len(posts)
    content_mix = {k: round(v / total_posts, 3) for k, v in type_counts.items()}

    avg_sponsored_er = 0
    avg_organic_er = 0
    if sponsored_posts and follower_count > 0:
        avg_sponsored_er = sum(
            (p.get("likes", 0) + p.get("num_comments", 0)) / follower_count
            for p in sponsored_posts
        ) / len(sponsored_posts)
    if organic_posts and follower_count > 0:
        avg_organic_er = sum(
            (p.get("likes", 0) + p.get("num_comments", 0)) / follower_count
            for p in organic_posts
        ) / len(organic_posts)

    engagement_trend = _compute_engagement_trend(posts, follower_count)

    posting_hours = [dt.hour for dt in post_datetimes]
    peak_hours = _find_peak_hours(posting_hours)

    if len(post_datetimes) >= 2:
        span_days = (
            post_datetimes[-1] - post_datetimes[0]
        ).total_seconds() / 86400
        posts_per_week = round(
            len(post_datetimes) / max(span_days, 1) * 7, 1
        )
    else:
        posts_per_week = 0

    consistency_stddev = 0
    if len(posting_gaps_days) >= 2:
        mean_gap = sum(posting_gaps_days) / len(posting_gaps_days)
        variance = sum((g - mean_gap) ** 2 for g in posting_gaps_days) / (
            len(posting_gaps_days) - 1
        )
        consistency_stddev = round(variance**0.5, 2)

    return {
        "avg_engagement_rate": round(
            sum(engagement_rates) / max(len(engagement_rates), 1), 5
        ),
        "median_engagement_rate": round(_median(engagement_rates), 5),
        "avg_likes_to_comments_ratio": (
            round(
                sum(likes_comments_ratios) / len(likes_comments_ratios), 1
            )
            if likes_comments_ratios
            else None
        ),
        "engagement_by_content_type": {
            k: round(sum(v) / max(len(v), 1), 5)
            for k, v in engagement_by_type.items()
            if v
        },
        "engagement_trend": engagement_trend,
        "posts_per_week": posts_per_week,
        "posting_consistency_stddev_days": consistency_stddev,
        "content_mix": content_mix,
        "peak_posting_hours": peak_hours,
        "sponsored_post_rate": round(
            len(sponsored_posts) / max(total_posts, 1), 3
        ),
        "sponsored_vs_organic_er_delta": (
            round(avg_sponsored_er - avg_organic_er, 5)
            if sponsored_posts
            else None
        ),
        "brand_mentions": list(brand_mentions),
        "brand_mentions_count": len(brand_mentions),
        "top_hashtags": _top_n_items(all_hashtags, 20),
        "total_unique_hashtags": len(set(all_hashtags)),
        "_post_urls": [p.get("url") for p in posts if p.get("url")],
        "_reel_urls": [
            p.get("url")
            for p in posts
            if _normalize_content_type(p.get("content_type", "")) == "Video"
            and p.get("url")
        ],
        "_captions": [p.get("description", "") for p in posts],
        "_post_datetimes": [dt.isoformat() for dt in post_datetimes],
    }


def _is_sponsored_post(post: dict) -> bool:
    """Detect sponsored/paid content from multiple signals."""
    if post.get("is_paid_partnership"):
        return True

    caption = (post.get("description") or "").lower()
    hashtags = [h.lower() for h in (post.get("hashtags") or [])]

    sponsored_tags = {
        "#ad",
        "#sponsored",
        "#paidpartnership",
        "#collab",
        "#gifted",
        "#brandpartner",
        "#partner",
    }
    if any(tag in sponsored_tags for tag in hashtags):
        return True

    sponsored_patterns = [
        r"\bpaid partnership\b",
        r"\b#ad\b",
        r"\bsponsored by\b",
        r"\bin collaboration with\b",
        r"\bpartnered with\b",
    ]
    if any(re.search(pat, caption) for pat in sponsored_patterns):
        return True

    return False


def _extract_brand_mentions(post: dict) -> set:
    """Extract @brand handles from caption and tagged users."""
    mentions = set()

    tagged = post.get("tagged_users") or []
    for user in tagged:
        if isinstance(user, dict):
            mentions.add(user.get("username", ""))
        elif isinstance(user, str):
            mentions.add(user)

    coauthors = post.get("coauthor_producers") or []
    for author in coauthors:
        if isinstance(author, dict):
            mentions.add(author.get("username", ""))
        elif isinstance(author, str):
            mentions.add(author)

    caption = post.get("description") or ""
    at_mentions = re.findall(r"@([a-zA-Z0-9_.]+)", caption)
    mentions.update(at_mentions)

    mentions.discard("")
    return mentions


def _compute_engagement_trend(posts: list[dict], follower_count: int) -> str:
    """Simple linear regression on engagement over time -> trend label."""
    if len(posts) < 4 or follower_count == 0:
        return "insufficient_data"

    dated_ers = []
    for post in posts:
        try:
            dt = datetime.fromisoformat(
                post["date_posted"].replace("Z", "+00:00")
            )
            er = (post.get("likes", 0) + post.get("num_comments", 0)) / follower_count
            dated_ers.append((dt.timestamp(), er))
        except (ValueError, TypeError, KeyError):
            continue

    if len(dated_ers) < 4:
        return "insufficient_data"

    dated_ers.sort(key=lambda x: x[0])

    n = len(dated_ers)
    x_vals = [x[0] for x in dated_ers]
    y_vals = [x[1] for x in dated_ers]

    x_min, x_max = min(x_vals), max(x_vals)
    if x_max == x_min:
        return "stable"
    x_norm = [(x - x_min) / (x_max - x_min) for x in x_vals]

    x_mean = sum(x_norm) / n
    y_mean = sum(y_vals) / n
    numerator = sum(
        (x_norm[i] - x_mean) * (y_vals[i] - y_mean) for i in range(n)
    )
    denominator = sum((x_norm[i] - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return "stable"

    slope = numerator / denominator
    relative_slope = slope / max(y_mean, 0.0001)

    if relative_slope > 0.15:
        return "growing"
    elif relative_slope < -0.15:
        return "declining"
    else:
        return "stable"


def _find_peak_hours(hours: list[int], top_n: int = 3) -> list[int]:
    """Find the most common posting hours."""
    if not hours:
        return []
    counts = Counter(hours)
    return [h for h, _ in counts.most_common(top_n)]


def _top_n_items(items: list, n: int = 20) -> list[dict]:
    """Return top N items by frequency."""
    counts = Counter(items)
    return [{"tag": tag, "count": count} for tag, count in counts.most_common(n)]
