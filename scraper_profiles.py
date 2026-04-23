from pipeline.brightdata_client import BrightdataClient

DATASET_PROFILES = "gd_l1vikfch901nx3by4"


def scrape_profiles(
    client: BrightdataClient, profile_urls: list[str]
) -> list[dict]:
    """
    Scrape Instagram profiles.

    Args:
        client: Initialized BrightdataClient
        profile_urls: List of Instagram profile URLs
                      e.g. ["https://www.instagram.com/username/"]

    Returns:
        List of profile data dicts

    Cost: 1 record per profile = $0.0015/profile at $1.50/1K
    """
    payload = [{"url": url} for url in profile_urls]
    return client.trigger_and_wait(DATASET_PROFILES, payload)


LOW_FOLLOWER_CUTOFF = 100


def extract_profile_metrics(raw_profile: dict) -> dict:
    """
    Extract and compute Tier A profile-level metrics from raw Brightdata response.

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

    return {
        # --- Identity ---
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
        # --- Account Signals ---
        "followers": followers,
        "following": following,
        "posts_count": posts_count,
        "is_business": raw_profile.get("is_business_account", False),
        "is_professional": raw_profile.get("is_professional_account", False),
        "is_verified": raw_profile.get("is_verified", False),
        # --- Computed Metrics (Tier A) ---
        "follower_following_ratio": round(followers / max(following, 1), 1),
        "posts_to_follower_efficiency": round(
            followers / max(posts_count, 1), 1
        ),
        "brightdata_avg_engagement": raw_profile.get("avg_engagement"),
        # --- Creator Tier Classification ---
        "tier": classify_creator_tier(followers),
        # --- Raw hashtags for niche classification ---
        "bio_hashtags": raw_profile.get("bio_hashtags", []),
        "post_hashtags": raw_profile.get("post_hashtags", []),
        # --- Contact Info ---
        "email": raw_profile.get("contact_email"),
        "phone": raw_profile.get("contact_phone_number"),
        # --- Data quality signalling ---
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
