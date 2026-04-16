"""
Database layer — stores CIP results into Supabase/Postgres.

Uses the Supabase Python client to upsert scraped data
and computed intelligence into the schema defined in db-schema.md.
"""

import logging
import socket
from datetime import datetime

from supabase import create_client, Client

from pipeline.scraper_posts import _normalize_content_type

logger = logging.getLogger(__name__)


def init_supabase(url: str, service_key: str) -> Client:
    """Initialize Supabase client with service_role key (bypasses RLS)."""
    return create_client(url, service_key)


def upsert_creator(db: Client, profile: dict) -> str:
    """
    Upsert creator profile into the creators table.
    Returns the creator UUID.
    """
    # Handle external_url — DB schema is text (single string)
    # BrightData sometimes returns a list
    ext_url = profile.get("external_url")
    if isinstance(ext_url, list):
        ext_url = ext_url[0] if ext_url else None
    elif ext_url == "":
        ext_url = None

    row = {
        "handle": profile["handle"],
        "instagram_id": profile.get("instagram_id"),
        "fbid": profile.get("fbid"),
        "display_name": profile.get("display_name"),
        "biography": profile.get("bio"),
        "external_url": ext_url,
        "avatar_url": profile.get("avatar_url"),
        "category": profile.get("category"),
        "city": profile.get("city"),
        "country": profile.get("country"),
        "is_business": profile.get("is_business", False),
        "is_professional": profile.get("is_professional", False),
        "is_verified": profile.get("is_verified", False),
        "followers": profile.get("followers", 0),
        "following": profile.get("following", 0),
        "posts_count": profile.get("posts_count", 0),
        "tier": profile.get("tier", "nano"),
        "follower_following_ratio": profile.get("follower_following_ratio"),
        "posts_to_follower_efficiency": profile.get(
            "posts_to_follower_efficiency"
        ),
        "contact_email": profile.get("email"),
        "contact_phone": profile.get("phone"),
        "brightdata_avg_engagement": profile.get("brightdata_avg_engagement"),
        "bio_hashtags": profile.get("bio_hashtags") or [],
        "post_hashtags": profile.get("post_hashtags") or [],
        "last_scraped_at": datetime.now().isoformat(),
    }

    result = (
        db.table("creators")
        .upsert(row, on_conflict="handle")
        .execute()
    )
    creator_id = result.data[0]["id"]
    logger.info(f"Upserted creator @{profile['handle']} -> {creator_id}")
    return creator_id


def upsert_posts(
    db: Client, creator_id: str, raw_posts: list[dict]
) -> list[str]:
    """Insert posts into the posts table. Returns list of post UUIDs."""
    if not raw_posts:
        return []

    post_ids = []
    for post in raw_posts:
        post_id = post.get("post_id") or post.get("id") or ""
        url = post.get("url") or ""
        if not post_id and not url:
            continue

        row = {
            "creator_id": creator_id,
            "post_id": post_id,
            "url": url,
            "description": post.get("description"),
            "hashtags": post.get("hashtags") or [],
            "content_type": _normalize_content_type(
                post.get("content_type", "Image")
            ),
            "likes": int(post.get("likes") or 0),
            "num_comments": int(post.get("num_comments") or 0),
            "video_view_count": _to_int(post.get("video_view_count")),
            "video_play_count": _to_int(post.get("video_play_count")),
            "is_paid_partnership": post.get("is_paid_partnership", False),
            "partnership_details": post.get("partnership_details"),
            "coauthor_producers": post.get("coauthor_producers") or [],
            "tagged_users": post.get("tagged_users") or [],
            "photos": post.get("photos") or [],
            "videos": post.get("videos") or [],
            "thumbnail_url": post.get("thumbnail"),
            "display_url": post.get("display_url"),
            "date_posted": post.get("date_posted"),
        }

        try:
            result = (
                db.table("posts")
                .upsert(row, on_conflict="creator_id,post_id")
                .execute()
            )
            if result.data:
                post_ids.append(result.data[0]["id"])
        except Exception as e:
            logger.warning(f"Failed to upsert post {post_id}: {e}")

    logger.info(f"Upserted {len(post_ids)} posts for creator {creator_id}")
    return post_ids


def insert_creator_scores(
    db: Client,
    creator_id: str,
    scores: dict,
    post_metrics: dict,
    reel_metrics: dict,
    comment_metrics: dict,
) -> None:
    """Insert computed CPI scores + all Tier A/B metrics."""
    row = {
        "creator_id": creator_id,
        # CPI sub-scores
        "engagement_quality": scores.get("engagement_quality", 0),
        "content_quality": scores.get("content_quality", 0),
        "audience_authenticity": scores.get("audience_authenticity", 0),
        "growth_trajectory": scores.get("growth_trajectory", 0),
        "professionalism": scores.get("professionalism", 0),
        "cpi": scores.get("cpi", 0),
        # Post metrics (Tier A)
        "avg_engagement_rate": post_metrics.get("avg_engagement_rate"),
        "median_engagement_rate": post_metrics.get("median_engagement_rate"),
        "avg_likes_to_comments_ratio": post_metrics.get(
            "avg_likes_to_comments_ratio"
        ),
        "engagement_trend": post_metrics.get(
            "engagement_trend", "insufficient_data"
        ),
        "engagement_by_content_type": post_metrics.get(
            "engagement_by_content_type", {}
        ),
        "posts_per_week": post_metrics.get("posts_per_week"),
        "posting_consistency_stddev": post_metrics.get(
            "posting_consistency_stddev_days"
        ),
        "content_mix": post_metrics.get("content_mix", {}),
        "peak_posting_hours": post_metrics.get("peak_posting_hours", []),
        "sponsored_post_rate": post_metrics.get("sponsored_post_rate"),
        "sponsored_vs_organic_delta": post_metrics.get(
            "sponsored_vs_organic_er_delta"
        ),
        "brand_mentions_count": post_metrics.get("brand_mentions_count", 0),
        "brand_mentions": post_metrics.get("brand_mentions", []),
        "top_hashtags": post_metrics.get("top_hashtags", []),
        # Reel metrics (Tier B)
        "avg_views_to_likes_ratio": reel_metrics.get(
            "avg_views_to_likes_ratio"
        ),
        "avg_rewatch_rate": reel_metrics.get("avg_rewatch_rate"),
        "avg_reel_length_seconds": reel_metrics.get(
            "avg_reel_length_seconds"
        ),
        # Comment metrics (Tier B)
        "creator_reply_rate": comment_metrics.get("creator_reply_rate"),
        "unique_commenter_count": comment_metrics.get(
            "unique_commenter_count", 0
        ),
        "comment_hour_distribution": comment_metrics.get(
            "comment_hour_distribution_utc", {}
        ),
    }

    db.table("creator_scores").insert(row).execute()
    logger.info(
        f"Inserted scores for {creator_id} — CPI: {scores.get('cpi')}"
    )


def insert_caption_intelligence(
    db: Client, creator_id: str, intel: dict
) -> None:
    """Insert Gemini caption analysis results."""
    niche = intel.get("niche_classification", {})
    tone = intel.get("tone_profile", {})
    lang = intel.get("language_analysis", {})
    cta = intel.get("cta_patterns", {})
    brands = intel.get("brand_mentions", {})
    themes = intel.get("content_themes", {})
    auth = intel.get("authenticity_signals", {})

    row = {
        "creator_id": creator_id,
        "primary_niche": niche.get("primary_niche"),
        "secondary_niche": niche.get("secondary_niche"),
        "niche_confidence": niche.get("confidence"),
        "primary_tone": tone.get("primary_tone"),
        "secondary_tone": tone.get("secondary_tone"),
        "formality_score": tone.get("formality_score"),
        "humor_score": tone.get("humor_score"),
        "authenticity_feel": tone.get("authenticity_feel"),
        "primary_language": lang.get("primary_language"),
        "language_mix": lang.get("language_mix_percentages", {}),
        "uses_transliteration": lang.get("uses_transliteration", False),
        "script_types": lang.get("script_types", []),
        "dominant_cta_style": cta.get("dominant_cta_style", "none"),
        "cta_frequency": cta.get("cta_frequency"),
        "is_conversion_oriented": cta.get("conversion_oriented", False),
        "organic_brand_mentions": brands.get("organic_brand_mentions", []),
        "paid_brand_mentions": brands.get("paid_brand_mentions", []),
        "brand_categories": brands.get("brand_categories", []),
        "recurring_topics": themes.get("recurring_topics", []),
        "content_pillars": themes.get("content_pillars", []),
        "personal_storytelling_freq": auth.get(
            "personal_storytelling_frequency"
        ),
        "vulnerability_openness": auth.get("vulnerability_openness"),
        "engagement_bait_score": auth.get("engagement_bait_score"),
        "raw_llm_response": intel,
        "posts_analyzed": intel.get("_captions_analyzed", 0),
    }

    db.table("caption_intelligence").insert(row).execute()
    logger.info(f"Inserted caption intelligence for {creator_id}")


def insert_transcript_intelligence(
    db: Client, creator_id: str, intel: dict
) -> None:
    """Insert Gemini transcript analysis results."""
    speaking = intel.get("speaking_language", {})
    hooks = intel.get("hook_analysis", {})
    depth = intel.get("content_depth", {})
    audio = intel.get("audio_production", {})
    regional = intel.get("regional_signals", {})

    row = {
        "creator_id": creator_id,
        # Speaking language
        "primary_spoken_language": speaking.get("primary_spoken_language"),
        "languages_spoken": speaking.get("languages_spoken", []),
        "caption_vs_spoken_mismatch": speaking.get(
            "caption_vs_spoken_mismatch", False
        ),
        # Hook analysis
        "avg_hook_quality": hooks.get("avg_hook_quality"),
        "dominant_hook_style": hooks.get("dominant_hook_style"),
        "hook_details": hooks.get("hooks", []),
        # Brand mention analysis
        "brand_mention_analysis": intel.get("brand_mention_analysis", []),
        # Content depth
        "avg_word_count": depth.get("avg_word_count_per_reel", 0),
        "vocabulary_complexity": depth.get("vocabulary_complexity"),
        "educational_density": depth.get("educational_density"),
        "storytelling_score": depth.get("storytelling_score"),
        "filler_word_frequency": depth.get("filler_word_frequency"),
        # Audio production
        "audio_quality_rating": audio.get(
            "overall_quality_assessment", "casual"
        ),
        "uses_background_music": audio.get("uses_background_music", False),
        "voiceover_vs_oncamera": audio.get("voiceover_vs_oncamera"),
        "pacing": audio.get("pacing"),
        # Regional signals
        "cultural_references": regional.get("cultural_references", []),
        "local_places_mentioned": regional.get("local_places_mentioned", []),
        "regional_language_phrases": regional.get(
            "regional_language_phrases", []
        ),
        "estimated_region": regional.get("estimated_region"),
        # Meta
        "raw_llm_response": intel,
        "reels_analyzed": len(hooks.get("hooks", [])),
    }

    db.table("transcript_intelligence").insert(row).execute()
    logger.info(f"Inserted transcript intelligence for {creator_id}")


def insert_audience_intelligence(
    db: Client, creator_id: str, intel: dict
) -> None:
    """Insert Gemini comment/audience analysis results."""
    lang_dist = intel.get("audience_language_distribution", {})
    geo = intel.get("audience_geography_inference", {})
    auth = intel.get("audience_authenticity", {})
    sent = intel.get("audience_sentiment", {})
    demo = intel.get("audience_demographics_inference", {})
    eng = intel.get("engagement_quality", {})

    row = {
        "creator_id": creator_id,
        "audience_languages": lang_dist.get("languages", {}),
        "primary_audience_language": lang_dist.get(
            "primary_audience_language"
        ),
        "is_multilingual_audience": lang_dist.get(
            "multilingual_audience", False
        ),
        "geo_regions": geo.get("regions", []),
        "domestic_percentage": geo.get(
            "domestic_vs_international_split", {}
        ).get("domestic_percentage"),
        "primary_country": geo.get(
            "domestic_vs_international_split", {}
        ).get("primary_country"),
        "authenticity_score": auth.get("authenticity_score"),
        "emoji_only_percentage": auth.get("emoji_only_percentage"),
        "generic_comment_percentage": auth.get("generic_comment_percentage"),
        "substantive_comment_percentage": auth.get(
            "substantive_comment_percentage"
        ),
        "suspicious_patterns": auth.get("suspicious_patterns", []),
        "overall_sentiment": sent.get("overall_sentiment"),
        "sentiment_score": sent.get("sentiment_score"),
        "positive_themes": sent.get("common_positive_themes", []),
        "negative_themes": sent.get("common_negative_themes", []),
        "estimated_age_group": demo.get("estimated_age_group"),
        "estimated_gender_skew": demo.get("estimated_gender_skew"),
        "interest_signals": demo.get("interest_signals", []),
        "engagement_quality_score": eng.get("quality_score"),
        "conversation_depth": eng.get("conversation_depth"),
        "community_strength": eng.get("community_feel"),
        "raw_llm_response": intel,
    }

    db.table("audience_intelligence").insert(row).execute()
    logger.info(f"Inserted audience intelligence for {creator_id}")


def store_full_cip(db: Client, cip: dict) -> str:
    """
    Store a complete CIP into the database.
    This is the main entry point — called BEFORE _clean_internal_fields
    so we have access to _raw_posts and other internal data.

    Returns the creator UUID.
    """
    profile = cip.get("profile", {})
    creator_id = upsert_creator(db, profile)

    # Store raw posts
    raw_posts = cip.get("_raw_posts", [])
    if raw_posts:
        upsert_posts(db, creator_id, raw_posts)

    # Scores (with reel + comment metrics)
    if cip.get("scores"):
        insert_creator_scores(
            db,
            creator_id,
            cip["scores"],
            post_metrics=cip.get("posts", {}),
            reel_metrics=cip.get("reels", {}),
            comment_metrics=cip.get("comments", {}),
        )

    # Caption intelligence
    if cip.get("caption_intelligence"):
        insert_caption_intelligence(
            db, creator_id, cip["caption_intelligence"]
        )

    # Transcript intelligence
    if cip.get("transcript_intelligence"):
        insert_transcript_intelligence(
            db, creator_id, cip["transcript_intelligence"]
        )

    # Audience intelligence
    if cip.get("audience_intelligence"):
        insert_audience_intelligence(
            db, creator_id, cip["audience_intelligence"]
        )

    logger.info(f"Stored full CIP for @{profile.get('handle')} -> {creator_id}")
    return creator_id


def get_brand(db: Client, brand_id: str) -> dict | None:
    """Fetch a single brand row."""
    result = (
        db.table("brands")
        .select("*")
        .eq("id", brand_id)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


def get_all_brands(db: Client) -> list[dict]:
    """Fetch all brands for batch matching or sync jobs."""
    result = db.table("brands").select("*").execute()
    return result.data or []


def upsert_brand_geo(db: Client, brand_id: str, geo_rows: list[dict]) -> None:
    """Replace brand geo rows with the latest computed Shopify output."""
    db.table("brand_shopify_geo").delete().eq("brand_id", brand_id).execute()

    if not geo_rows:
        logger.info(f"No geo rows to upsert for brand {brand_id}")
        return

    rows = [{**row, "brand_id": brand_id} for row in geo_rows]
    db.table("brand_shopify_geo").insert(rows).execute()
    logger.info(f"Upserted {len(rows)} geo rows for brand {brand_id}")


def upsert_brand_products(
    db: Client, brand_id: str, products: list[dict]
) -> None:
    """Replace Shopify product rows for a brand."""
    db.table("brand_products").delete().eq("brand_id", brand_id).execute()

    if not products:
        logger.info(f"No brand products to upsert for brand {brand_id}")
        return

    rows = [{**product, "brand_id": brand_id} for product in products]
    db.table("brand_products").upsert(
        rows,
        on_conflict="brand_id,shopify_product_id",
    ).execute()
    logger.info(f"Upserted {len(rows)} brand products for brand {brand_id}")


def update_brand_shopify_summary(
    db: Client, brand_id: str, summary: dict
) -> None:
    """Update summary Shopify fields stored on the brand row."""
    db.table("brands").update(summary).eq("id", brand_id).execute()


def update_brand_shopify_sync_state(
    db: Client,
    brand_id: str,
    *,
    status: str,
    error: str | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
) -> None:
    """Update the background sync status shown across the brand UI."""
    payload = {
        "shopify_sync_status": status,
        "shopify_sync_error": error,
        "shopify_sync_started_at": started_at,
        "shopify_sync_completed_at": completed_at,
    }
    db.table("brands").update(payload).eq("id", brand_id).execute()


def get_runnable_background_jobs(db: Client, limit: int = 5) -> list[dict]:
    """Fetch queued jobs that are ready to run."""
    result = (
        db.table("background_jobs")
        .select("*")
        .eq("status", "queued")
        .lte("available_at", datetime.utcnow().isoformat())
        .order("created_at")
        .limit(limit)
        .execute()
    )
    return result.data or []


def claim_background_job(
    db: Client, job_id: str, worker_id: str | None = None
) -> dict | None:
    """Transition a queued job to running for a specific worker."""
    now = datetime.utcnow().isoformat()
    owner = worker_id or socket.gethostname()
    result = (
        db.table("background_jobs")
        .update(
            {
                "status": "running",
                "locked_at": now,
                "started_at": now,
                "locked_by": owner,
                "attempt_count": 1,
                "updated_at": now,
                "last_error": None,
            }
        )
        .eq("id", job_id)
        .eq("status", "queued")
        .execute()
    )

    if result.data:
        return result.data[0]

    claimed_result = (
        db.table("background_jobs")
        .select("*")
        .eq("id", job_id)
        .eq("status", "running")
        .eq("locked_by", owner)
        .limit(1)
        .execute()
    )
    return claimed_result.data[0] if claimed_result.data else None


def complete_background_job(db: Client, job_id: str) -> None:
    """Mark a background job as completed."""
    now = datetime.utcnow().isoformat()
    db.table("background_jobs").update(
        {
            "status": "succeeded",
            "completed_at": now,
            "locked_at": None,
            "locked_by": None,
            "updated_at": now,
            "last_error": None,
        }
    ).eq("id", job_id).execute()


def fail_background_job(db: Client, job_id: str, error_message: str) -> None:
    """Mark a background job as failed."""
    now = datetime.utcnow().isoformat()
    db.table("background_jobs").update(
        {
            "status": "failed",
            "completed_at": now,
            "locked_at": None,
            "locked_by": None,
            "updated_at": now,
            "last_error": error_message[:1000],
        }
    ).eq("id", job_id).execute()


def get_brand_products(db: Client, brand_id: str) -> list[dict]:
    """Fetch Shopify products for a brand."""
    result = (
        db.table("brand_products")
        .select("*")
        .eq("brand_id", brand_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []


def get_brand_geo(db: Client, brand_id: str) -> list[dict]:
    """Fetch Shopify geo rows for a brand, ordered by gap score."""
    result = (
        db.table("brand_shopify_geo")
        .select("*")
        .eq("brand_id", brand_id)
        .order("gap_score", desc=True)
        .execute()
    )
    return result.data or []


def get_all_creators_for_matching(db: Client) -> list[dict]:
    """Fetch denormalized creator rows used by the brand matching engine."""
    result = db.table("mv_creator_leaderboard").select("*").execute()
    return result.data or []


def upsert_brand_matches(
    db: Client, brand_id: str, match_rows: list[dict]
) -> None:
    """Persist creator-brand match scores for a brand."""
    if not match_rows:
        logger.info(f"No brand matches to write for brand {brand_id}")
        return

    rows = [{**row, "brand_id": brand_id} for row in match_rows]
    db.table("creator_brand_matches").upsert(
        rows,
        on_conflict="creator_id,brand_id",
    ).execute()
    logger.info(f"Upserted {len(rows)} brand matches for brand {brand_id}")


def _to_int(val):
    """Safely convert to int."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None
