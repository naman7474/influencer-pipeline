"""Instagram profile scrape — Apify-backed."""
from pipeline import apify_instagram_bundle
from pipeline.contact_extract import (
    extract_email_from_text,
    extract_phone_from_text,
)


def scrape_profiles(profile_urls: list[str]) -> list[dict]:
    """Scrape Instagram profiles via the cached Apify bundle.

    Returns BD-shaped records — the downstream extractors haven't been
    rewritten to consume Apify's native shape yet.
    """
    out: list[dict] = []
    for url in profile_urls:
        username = _url_to_username(url)
        bundle = apify_instagram_bundle.fetch(username)
        if bundle["profile"]:
            out.append(bundle["profile"])
    return out


def _url_to_username(url: str) -> str:
    cleaned = url.rstrip("/")
    return cleaned.rsplit("/", 1)[-1].lstrip("@")


LOW_FOLLOWER_CUTOFF = 100


def extract_profile_metrics(raw_profile: dict) -> dict:
    """
    Extract and compute Tier A profile-level metrics from the raw profile record.

    Emits a `data_quality_flags` list that downstream scorers consult:
      - "low_followers": < 100 followers, ER math is noise; we still
        extract but downstream will skip the ER sub-score.
    """
    followers = raw_profile.get("followers", 0) or 0
    following = raw_profile.get("following", 0) or 0
    posts_count = raw_profile.get("posts_count", 0) or 0

    data_quality_flags: list[str] = []
    if followers < LOW_FOLLOWER_CUTOFF:
        data_quality_flags.append("low_followers")

    bio = raw_profile.get("biography") or ""
    email = raw_profile.get("contact_email") or extract_email_from_text(bio)
    phone = (
        raw_profile.get("contact_phone_number")
        or extract_phone_from_text(bio)
    )

    return {
        "handle": raw_profile.get("account"),
        "instagram_id": raw_profile.get("id"),
        "fbid": raw_profile.get("fbid"),
        "display_name": raw_profile.get("profile_name"),
        "avatar_url": raw_profile.get("profile_image_link"),
        "bio": raw_profile.get("biography"),
        "external_url": raw_profile.get("external_url"),
        "city": raw_profile.get("city"),
        "country": raw_profile.get("country"),
        "category": raw_profile.get("category") or raw_profile.get("category_name"),
        "followers": followers,
        "following": following,
        "posts_count": posts_count,
        "is_business": raw_profile.get("is_business_account", False),
        "is_professional": raw_profile.get("is_professional_account", False),
        "is_verified": raw_profile.get("is_verified", False),
        "follower_following_ratio": round(followers / max(following, 1), 1),
        "posts_to_follower_efficiency": round(
            followers / max(posts_count, 1), 1
        ),
        "brightdata_avg_engagement": raw_profile.get("avg_engagement"),
        "tier": classify_creator_tier(followers),
        "bio_hashtags": raw_profile.get("bio_hashtags", []),
        "post_hashtags": raw_profile.get("post_hashtags", []),
        "email": email,
        "phone": phone,
        "data_quality_flags": data_quality_flags,
    }


def classify_creator_tier(followers: int) -> str:
    """Standard influencer tier classification."""
    if followers < 10_000:
        return "nano"
    elif followers < 50_000:
        return "micro"
    elif followers < 500_000:
        return "mid"
    elif followers < 1_000_000:
        return "macro"
    else:
        return "mega"
