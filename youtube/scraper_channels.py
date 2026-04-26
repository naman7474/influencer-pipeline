"""Bright Data YouTube channel scraper.

Parallel to pipeline/scraper_profiles.py on the Instagram side. Triggers the
Bright Data YouTube-channel dataset, waits for completion, and returns raw
records. `extract_channel_metrics` normalizes the record into the shape we
write to `creator_social_profiles` (platform='youtube').
"""

import os

from pipeline.brightdata_client import BrightdataClient
from pipeline.contact_extract import (
    extract_email_from_text,
    extract_phone_from_text,
)

# Bright Data dataset id for "YouTube - Channels". Override via env so ops can
# rotate the dataset without a code change; the literal is a sensible default
# matching the account's current YT dataset.
DATASET_YT_CHANNELS = os.environ.get(
    "BRIGHTDATA_DATASET_YT_CHANNELS", "gd_lk538t2k2p1k3oos71"
)

LOW_SUBSCRIBER_CUTOFF = 100


def scrape_channels(
    client: BrightdataClient, channel_urls: list[str]
) -> list[dict]:
    """Scrape YouTube channel profile data for a batch of channel URLs.

    Args:
        client: Initialized BrightdataClient.
        channel_urls: Canonical channel URLs — any of:
            https://www.youtube.com/@handle
            https://www.youtube.com/channel/UCxxxxxxxxxxxxxxxxxxxxxx
            https://www.youtube.com/c/customname
    """
    payload = [{"url": url} for url in channel_urls]
    return client.trigger_and_wait(DATASET_YT_CHANNELS, payload)


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
    """Normalize a Bright Data YT channel record into creator_social_profiles shape.

    Bright Data field names vary slightly across dataset versions. We probe
    a few aliases rather than hard-failing — anything we can't find comes back
    as None and the downstream write ignores nulls.
    """
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

    # Handle normalization: strip whitespace and a leading '@' so it's
    # consistent with IG handles (Bright Data sometimes returns "  @x  ").
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

    # Bright Data uses capitalized keys for some fields (`Links`, `Description`,
    # `Details`) — probe both casings to tolerate dataset version drift.
    raw_links = (
        raw_channel.get("Links")
        or raw_channel.get("links")
        or raw_channel.get("external_links")
        or []
    )
    # Three observed shapes: list of URL strings (current dataset), list of
    # {label,url} dicts (older), or a dict {label: url}. Normalize to the
    # list-of-{label,url} dict shape that stitching.extract_handles_from_links
    # already accepts.
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

    # Country lives in `Details.location` (BD) — fall back to flat keys if the
    # dataset shape changes.
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
    # neither the Data API nor BD can fetch. So `email` always comes from
    # bio extraction — `business@MKBHD.com` and similar are common.
    email = extract_email_from_text(bio)
    phone = extract_phone_from_text(bio)

    return {
        # Identity
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
        "is_business": False,  # YT has no direct analogue; leave false
        # Metrics
        "followers_or_subs": subs,
        "posts_or_videos_count": videos_count,
        "total_views": total_views,
        # Channel-age signal — fed into the YT-aware professionalism scorer.
        "channel_created_at": raw_channel.get("created_date")
        or raw_channel.get("published_at"),
        # Contact info extracted from bio (YT has no dedicated email field)
        "email": email,
        "phone": phone,
        # Computed
        "tier": classify_creator_tier(subs),
        # Cross-platform stitching signal
        "external_links": external_links,
        # Data quality signalling
        "data_quality_flags": data_quality_flags,
    }
