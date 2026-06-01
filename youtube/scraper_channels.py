"""YouTube channel scrape — YouTube Data API v3 only.

Parallel to pipeline/scraper_profiles.py on the Instagram side. Pulls
channel stats and metadata via the YT Data API and synthesises
``external_links`` by regexing the channel description text.
"""

import logging
import re

from pipeline.contact_extract import (
    extract_email_from_text,
    extract_phone_from_text,
)

logger = logging.getLogger(__name__)

# URL regexes used to synthesise external_links from a channel's
# description text — duplicated here so this module doesn't have to
# import stitching (which imports back).
_IG_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?instagram\.com/([A-Za-z0-9_.]+)", re.I
)
_TT_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?tiktok\.com/@([A-Za-z0-9_.]+)", re.I
)
_TWITTER_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:twitter\.com|x\.com)/([A-Za-z0-9_]+)",
    re.I,
)
_GENERIC_URL_RE = re.compile(r"https?://[^\s\)<>\"]+", re.I)


LOW_SUBSCRIBER_CUTOFF = 100


def scrape_channels(
    channel_urls: list[str],
    *,
    yt_api,
    channel_ids: list[str] | None = None,
) -> list[dict]:
    """Pull channel records via the YouTube Data API.

    ``channel_ids`` is parallel to ``channel_urls`` (same length / order).
    When omitted, the caller resolved them already and we only have URLs
    — that path is unreachable in the current orchestrator (which always
    resolves channel ids first), but kept here as a guard.
    """
    if not yt_api or not yt_api.available:
        raise RuntimeError(
            "scrape_channels needs a configured YouTubeAPIClient; "
            "BrightData fallback was removed in the pipeline rewrite."
        )

    if channel_ids is None:
        raise RuntimeError(
            "scrape_channels requires channel_ids (resolve via "
            "pipeline.youtube.handle_resolver first)."
        )

    # `fetch_channel_stats` batches up to 50 ids internally.
    stats_by_id = yt_api.fetch_channel_stats(channel_ids)

    records: list[dict] = []
    for url, channel_id in zip(channel_urls, channel_ids):
        stats = stats_by_id.get(channel_id) or {}
        description = stats.get("description") or ""
        records.append(
            {
                "url": url,
                "channel_id": channel_id,
                "channel_name": stats.get("title"),
                "custom_url": stats.get("custom_url"),
                "description": description,
                "profile_image": stats.get("avatar_url"),
                "subscriber_count": stats.get("subscriber_count"),
                "view_count": stats.get("view_count"),
                "video_count": stats.get("video_count"),
                "country": stats.get("country"),
                "topic_categories": stats.get("topic_categories") or [],
                "external_links": _extract_links_from_text(description),
                "created_date": stats.get("published_at"),
                "_data_provenance": "youtube_data_api_v3",
            }
        )
    return records


def _extract_links_from_text(text: str | None) -> list[dict]:
    """Pull URLs the creator pasted into their description."""
    if not text:
        return []
    out: list[dict] = []
    seen: set[str] = set()

    def add(label: str, url: str) -> None:
        u = url.rstrip(".,;)")
        if u in seen:
            return
        seen.add(u)
        out.append({"label": label, "url": u})

    for m in _IG_URL_RE.finditer(text):
        add("instagram", m.group(0))
    for m in _TT_URL_RE.finditer(text):
        add("tiktok", m.group(0))
    for m in _TWITTER_URL_RE.finditer(text):
        add("twitter", m.group(0))
    for m in _GENERIC_URL_RE.finditer(text):
        add("link", m.group(0))
    return out


def classify_creator_tier(subscribers: int) -> str:
    """Same tier cutoffs as IG — subscribers read as `followers` on the YT side."""
    if subscribers < 10_000:
        return "nano"
    if subscribers < 50_000:
        return "micro"
    if subscribers < 500_000:
        return "mid"
    if subscribers < 1_000_000:
        return "macro"
    return "mega"


def _to_int(val, default: int = 0) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def extract_channel_metrics(raw_channel: dict) -> dict:
    """Normalize a YT channel record into creator_social_profiles shape."""
    subs = _to_int(
        raw_channel.get("subscribers")
        or raw_channel.get("subscriber_count")
        or raw_channel.get("followers")
    )
    videos_count = _to_int(
        raw_channel.get("videos_count")
        or raw_channel.get("video_count")
        or raw_channel.get("posts_count")
    )
    total_views = _to_int(
        raw_channel.get("views")
        or raw_channel.get("view_count")
        or raw_channel.get("total_views")
    )

    data_quality_flags: list[str] = []
    if subs < LOW_SUBSCRIBER_CUTOFF:
        data_quality_flags.append("low_subscribers")

    raw_handle = (
        raw_channel.get("handle")
        or raw_channel.get("custom_url")
        or raw_channel.get("channel_name")
    )
    handle = (raw_handle or "").strip().lstrip("@").strip() or None

    channel_id = (
        raw_channel.get("channel_id")
        or raw_channel.get("id")
        or raw_channel.get("ucid")
    )

    raw_links = (
        raw_channel.get("Links")
        or raw_channel.get("links")
        or raw_channel.get("external_links")
        or []
    )
    external_links: list[dict | str] = []
    if isinstance(raw_links, dict):
        external_links = [
            {"label": k, "url": v} for k, v in raw_links.items() if v
        ]
    elif isinstance(raw_links, list):
        for item in raw_links:
            if isinstance(item, str) and item:
                external_links.append({"label": "link", "url": item})
            elif isinstance(item, dict):
                external_links.append(item)

    details = raw_channel.get("Details") or {}
    country = (
        (details.get("location") if isinstance(details, dict) else None)
        or raw_channel.get("country")
        or raw_channel.get("country_code")
    )

    bio = (
        raw_channel.get("Description")
        or raw_channel.get("description")
        or raw_channel.get("about")
    )

    # YouTube's About-page email is gated behind a CAPTCHA reveal that
    # the Data API can't fetch. Email always comes from bio extraction.
    email = extract_email_from_text(bio)
    phone = extract_phone_from_text(bio)

    return {
        "handle": handle,
        "platform_user_id": channel_id,
        "profile_url": raw_channel.get("url")
        or (f"https://www.youtube.com/channel/{channel_id}" if channel_id else None),
        "display_name": raw_channel.get("name") or raw_channel.get("title"),
        "avatar_url": raw_channel.get("profile_image")
        or raw_channel.get("thumbnail")
        or raw_channel.get("avatar"),
        "bio": bio,
        "category": raw_channel.get("category") or raw_channel.get("topic"),
        "country": country,
        "is_verified": bool(raw_channel.get("verified", False)),
        "is_business": False,
        "followers_or_subs": subs,
        "posts_or_videos_count": videos_count,
        "total_views": total_views,
        "channel_created_at": raw_channel.get("created_date")
        or raw_channel.get("published_at"),
        "email": email,
        "phone": phone,
        "tier": classify_creator_tier(subs),
        "external_links": external_links,
        "data_quality_flags": data_quality_flags,
    }
