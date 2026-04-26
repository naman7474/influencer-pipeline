"""
Database layer — stores CIP results into Supabase/Postgres.

Uses the Supabase Python client to upsert scraped data
and computed intelligence into the schema defined in db-schema.md.
"""

import json
import logging
import os
import socket
from datetime import datetime

from supabase import create_client, Client

from pipeline.media_store import persist_avatar, persist_thumbnail
from pipeline.scraper_posts import _normalize_content_type

logger = logging.getLogger(__name__)

# Opt-in flag: route the pure-DB CIP writes through the transactional
# RPC defined in migration 036 instead of the per-table Python inserts.
USE_TX_RPC = os.environ.get("PIPELINE_USE_TX_RPC", "").lower() in (
    "1", "true", "yes",
)


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

    # Re-host avatar in Supabase Storage so the URL never expires
    raw_avatar = profile.get("avatar_url")
    persisted_avatar = persist_avatar(db, profile["handle"], raw_avatar) if raw_avatar else None

    row = {
        "handle": profile["handle"],
        "instagram_id": profile.get("instagram_id"),
        "fbid": profile.get("fbid"),
        "display_name": profile.get("display_name"),
        "biography": profile.get("bio"),
        "external_url": ext_url,
        "avatar_url": persisted_avatar or raw_avatar,
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

        # Re-host thumbnail in Supabase Storage so the URL never expires
        raw_thumb = post.get("thumbnail")
        persisted_thumb = persist_thumbnail(db, post_id, raw_thumb) if raw_thumb else None

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
            "thumbnail_url": persisted_thumb or raw_thumb,
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
    pipeline_version: str = "1.1",
) -> None:
    """Insert computed CPI scores + all Tier A/B metrics."""
    row = {
        "creator_id": creator_id,
        "pipeline_version": pipeline_version,
        # CPI sub-scores
        "engagement_quality": scores.get("engagement_quality", 0),
        "content_quality": scores.get("content_quality", 0),
        "audience_authenticity": scores.get("audience_authenticity", 0),
        "growth_trajectory": scores.get("growth_trajectory", 0),
        "professionalism": scores.get("professionalism", 0),
        "cpi": scores.get("cpi", 0),
        # Confidence envelope (W1)
        "confidence": scores.get("confidence", {}),
        "coverage_percentage": scores.get("coverage_percentage"),
        "confidence_tier": scores.get("confidence_tier", "low"),
        "missing_inputs": scores.get("missing_inputs", []),
        "llm_calls_succeeded": scores.get("llm_calls_succeeded", {}),
        "data_quality_flags": scores.get("data_quality_flags", []),
        # Fraud flags (W4)
        "fraud_flags": scores.get("_fraud_flags_full", []),
        "fraud_flag_codes": scores.get("fraud_flag_codes", []),
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

    db.table("creator_scores").upsert(
        row,
        on_conflict="creator_id,pipeline_version,computed_at_date",
    ).execute()
    logger.info(
        f"Upserted scores for {creator_id} — CPI: {scores.get('cpi')} "
        f"(tier={scores.get('confidence_tier')}, "
        f"coverage={scores.get('coverage_percentage')}%)"
    )


def _build_data_quality_envelope(
    intel: dict,
    *,
    expected_keys: list[str],
    sample_size: int | None,
    schema_version: str = "1.0",
) -> dict:
    """Build the data_quality envelope persisted on an intelligence row.

    W1 version: relies on top-level key presence to estimate coverage.
    W2 will replace this with Pydantic-validator-aware coverage.
    """
    if intel.get("_llm_failure"):
        return {
            "confidence": 0.0,
            "coverage_percentage": 0,
            "was_defaulted": True,
            "missing_fields": list(expected_keys),
            "sample_size": sample_size or 0,
            "schema_version": schema_version,
            "llm_failure": True,
            "error": intel.get("error", "llm_failure"),
        }

    present = [k for k in expected_keys if intel.get(k)]
    missing = [k for k in expected_keys if not intel.get(k)]
    coverage = (
        round(len(present) / len(expected_keys), 3)
        if expected_keys else 1.0
    )
    return {
        "confidence": coverage,
        "coverage_percentage": round(coverage * 100),
        "was_defaulted": len(missing) > 0,
        "missing_fields": missing,
        "sample_size": sample_size or 0,
        "schema_version": schema_version,
    }


def insert_caption_intelligence(
    db: Client,
    creator_id: str,
    intel: dict,
    platform: str = "instagram",
) -> None:
    """Insert Gemini caption analysis results.

    `platform` was added in migration 047 so a YT rescrape no longer
    overwrites the creator's IG analysis. Default stays 'instagram' so
    every existing IG call-site keeps working without argument changes.
    """
    niche = intel.get("niche_classification", {})
    tone = intel.get("tone_profile", {})
    lang = intel.get("language_analysis", {})
    cta = intel.get("cta_patterns", {})
    brands = intel.get("brand_mentions", {})
    themes = intel.get("content_themes", {})
    auth = intel.get("authenticity_signals", {})

    row = {
        "creator_id": creator_id,
        "platform": platform,
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
        "data_quality": _build_data_quality_envelope(
            intel,
            expected_keys=[
                "niche_classification", "tone_profile", "language_analysis",
                "cta_patterns", "brand_mentions", "content_themes",
                "authenticity_signals",
            ],
            sample_size=intel.get("_captions_analyzed"),
        ),
    }

    db.table("caption_intelligence").upsert(
        row,
        on_conflict="creator_id,analyzed_at_date",
    ).execute()
    logger.info(f"Upserted caption intelligence for {creator_id}")


def insert_transcript_intelligence(
    db: Client,
    creator_id: str,
    intel: dict,
    platform: str = "instagram",
) -> None:
    """Insert Gemini transcript analysis results.

    Platform-scoped — see insert_caption_intelligence.
    """
    speaking = intel.get("speaking_language", {})
    hooks = intel.get("hook_analysis", {})
    depth = intel.get("content_depth", {})
    audio = intel.get("audio_production", {})
    regional = intel.get("regional_signals", {})

    row = {
        "creator_id": creator_id,
        "platform": platform,
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
        "data_quality": _build_data_quality_envelope(
            intel,
            expected_keys=[
                "speaking_language", "hook_analysis", "content_depth",
                "audio_production", "regional_signals",
            ],
            sample_size=len(hooks.get("hooks", [])) or None,
        ),
    }

    db.table("transcript_intelligence").upsert(
        row,
        on_conflict="creator_id,analyzed_at_date",
    ).execute()
    logger.info(f"Upserted transcript intelligence for {creator_id}")


def insert_audience_intelligence(
    db: Client,
    creator_id: str,
    intel: dict,
    platform: str = "instagram",
) -> None:
    """Insert Gemini comment/audience analysis results.

    Platform-scoped — see insert_caption_intelligence.
    """
    lang_dist = intel.get("audience_language_distribution", {})
    geo = intel.get("audience_geography_inference", {})
    auth = intel.get("audience_authenticity", {})
    sent = intel.get("audience_sentiment", {})
    demo = intel.get("audience_demographics_inference", {})
    eng = intel.get("engagement_quality", {})

    row = {
        "creator_id": creator_id,
        "platform": platform,
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
        "data_quality": _build_data_quality_envelope(
            intel,
            expected_keys=[
                "audience_language_distribution",
                "audience_geography_inference",
                "audience_authenticity",
                "audience_sentiment",
                "audience_demographics_inference",
                "engagement_quality",
            ],
            sample_size=intel.get("_comments_analyzed"),
        ),
    }

    db.table("audience_intelligence").upsert(
        row,
        on_conflict="creator_id,analyzed_at_date",
    ).execute()
    logger.info(f"Upserted audience intelligence for {creator_id}")


def store_full_cip(
    db: Client,
    cip: dict,
    existing_creator_id: str | None = None,
) -> str:
    """
    Store a complete CIP into the database.
    This is the main entry point — called BEFORE _clean_internal_fields
    so we have access to _raw_posts and other internal data.

    When PIPELINE_USE_TX_RPC is set, the pure-DB writes run through
    the transactional RPC defined in migration 036. Media persistence
    (Supabase Storage) always runs on the Python side because it is
    not transactional with DB state.

    `existing_creator_id` (Phase 2.5 auto-stitch): when set, the caller
    has an existing canonical `creators.id` we should write onto (e.g.
    a YT scrape discovered this creator's IG handle in external_links
    and auto-enqueued an IG scrape bound to the same row). In that
    case we skip `upsert_creator` (which keys on handle and would
    create a new row) and instead route writes through
    `upsert_social_profile` against the existing id. Returns the same
    `existing_creator_id` the caller passed in.

    Returns the creator UUID.
    """
    profile = cip.get("profile", {})
    if USE_TX_RPC and not existing_creator_id:
        return _store_full_cip_via_rpc(db, cip)

    if existing_creator_id:
        creator_id = existing_creator_id
        # Write the IG profile onto the existing canonical creators row.
        upsert_social_profile(db, creator_id, "instagram", profile)
    else:
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
            pipeline_version=cip.get("pipeline_version", "1.1"),
        )

    # Caption intelligence (skip if LLM failed — nothing substantive to store)
    cap = cip.get("caption_intelligence")
    if cap and not (isinstance(cap, dict) and cap.get("_llm_failure")):
        insert_caption_intelligence(db, creator_id, cap)

    # Transcript intelligence
    tr = cip.get("transcript_intelligence")
    if tr and not (isinstance(tr, dict) and tr.get("_llm_failure")):
        insert_transcript_intelligence(db, creator_id, tr)

    # Audience intelligence
    aud = cip.get("audience_intelligence")
    if aud and not (isinstance(aud, dict) and aud.get("_llm_failure")):
        insert_audience_intelligence(db, creator_id, aud)

    logger.info(f"Stored full CIP for @{profile.get('handle')} -> {creator_id}")
    return creator_id


def _store_full_cip_via_rpc(db: Client, cip: dict) -> str:
    """Transactional write path gated on PIPELINE_USE_TX_RPC.

    Media persistence (Storage) runs here first, then the pure-DB
    upserts go through the 036 RPC in a single transaction. Falls
    back to the per-table path if the RPC is not deployed yet.
    """
    profile = dict(cip.get("profile") or {})

    raw_avatar = profile.get("avatar_url")
    if raw_avatar and profile.get("handle"):
        try:
            persisted = persist_avatar(db, profile["handle"], raw_avatar)
            if persisted:
                profile["avatar_url"] = persisted
        except Exception as e:
            logger.warning(f"persist_avatar failed, keeping raw URL: {e}")

    # Copy media-hydrated profile back into the CIP before RPC call.
    rpc_cip = dict(cip)
    rpc_cip["profile"] = profile
    # Strip internal-only fields; the RPC only reads declared keys but
    # sending _raw_posts needlessly bloats the payload.
    rpc_cip = {
        k: v for k, v in rpc_cip.items() if not k.startswith("_")
    }

    try:
        result = db.rpc(
            "store_creator_cip",
            {"p_cip": json.loads(json.dumps(rpc_cip, default=str))},
        ).execute()
    except Exception as e:
        logger.error(
            f"store_creator_cip RPC failed for @{profile.get('handle')}: {e}. "
            "Falling back to per-table upserts."
        )
        # Unset the flag for this call so the fallback runs the normal path.
        return _store_full_cip_legacy(db, cip)

    creator_id = (
        result.data if isinstance(result.data, str) else str(result.data)
    )
    # Posts still go through the Python path so thumbnails persist
    # through Storage — the RPC only owns creators + intelligence.
    raw_posts = cip.get("_raw_posts") or []
    if raw_posts:
        upsert_posts(db, creator_id, raw_posts)
    logger.info(
        f"Stored full CIP via RPC for @{profile.get('handle')} -> {creator_id}"
    )
    return creator_id


def _store_full_cip_legacy(db: Client, cip: dict) -> str:
    """Fallback path — mirrors store_full_cip without the RPC branch."""
    profile = cip.get("profile", {})
    creator_id = upsert_creator(db, profile)
    raw_posts = cip.get("_raw_posts") or []
    if raw_posts:
        upsert_posts(db, creator_id, raw_posts)
    if cip.get("scores"):
        insert_creator_scores(
            db, creator_id, cip["scores"],
            post_metrics=cip.get("posts", {}),
            reel_metrics=cip.get("reels", {}),
            comment_metrics=cip.get("comments", {}),
            pipeline_version=cip.get("pipeline_version", "1.1"),
        )
    if cip.get("caption_intelligence") and not cip["caption_intelligence"].get(
        "_llm_failure"
    ):
        insert_caption_intelligence(db, creator_id, cip["caption_intelligence"])
    if cip.get("transcript_intelligence") and not cip["transcript_intelligence"].get(
        "_llm_failure"
    ):
        insert_transcript_intelligence(
            db, creator_id, cip["transcript_intelligence"]
        )
    if cip.get("audience_intelligence") and not cip["audience_intelligence"].get(
        "_llm_failure"
    ):
        insert_audience_intelligence(db, creator_id, cip["audience_intelligence"])
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


# ── Multi-platform helpers (migration 043/044) ─────────────
# These back the YouTube path and, going forward, replace direct
# writes to creators.{handle, followers, instagram_id, ...}. The IG
# pipeline still writes those columns for one release as a shadow.


def upsert_social_profile(
    db: Client,
    creator_id: str,
    platform: str,
    profile: dict,
) -> str:
    """Upsert one row into creator_social_profiles.

    `profile` is the normalized dict emitted by
    youtube.scraper_channels.extract_channel_metrics or the IG-side
    extract_profile_metrics (wrapped with a platform key).
    """
    row = {
        "creator_id": creator_id,
        "platform": platform,
        "handle": profile["handle"],
        "platform_user_id": profile.get("platform_user_id")
        or profile.get("instagram_id"),
        "profile_url": profile.get("profile_url"),
        "display_name": profile.get("display_name"),
        "bio": profile.get("bio") or profile.get("biography"),
        "avatar_url": profile.get("avatar_url"),
        "category": profile.get("category"),
        "country": profile.get("country"),
        "is_verified": profile.get("is_verified", False),
        "is_business": profile.get("is_business", False),
        "followers_or_subs": profile.get("followers_or_subs")
        or profile.get("followers")
        or 0,
        "posts_or_videos_count": profile.get("posts_or_videos_count")
        or profile.get("posts_count")
        or 0,
        "avg_engagement": profile.get("avg_engagement")
        or profile.get("brightdata_avg_engagement"),
        "external_links": profile.get("external_links") or [],
        "last_synced_at": datetime.now().isoformat(),
    }

    result = (
        db.table("creator_social_profiles")
        .upsert(row, on_conflict="creator_id,platform")
        .execute()
    )
    csp_id = result.data[0]["id"]
    logger.info(
        f"Upserted {platform} social profile for creator {creator_id} -> {csp_id}"
    )
    return csp_id


def upsert_youtube_videos(
    db: Client, creator_id: str, videos: list[dict]
) -> list[str]:
    """Insert YouTube videos for a creator. Returns list of row UUIDs."""
    if not videos:
        return []

    rows = []
    for v in videos:
        video_id = v.get("video_id")
        if not video_id:
            continue
        rows.append(
            {
                "creator_id": creator_id,
                "video_id": video_id,
                "url": v.get("url"),
                "title": v.get("title"),
                "description": v.get("description"),
                "tags": v.get("tags") or [],
                "category_id": v.get("category_id"),
                "is_short": bool(v.get("is_short", False)),
                "is_livestream": bool(v.get("is_livestream", False)),
                "duration_seconds": v.get("duration_seconds"),
                "view_count": v.get("view_count") or 0,
                "like_count": v.get("like_count") or 0,
                "comment_count": v.get("comment_count") or 0,
                "thumbnail_url": v.get("thumbnail_url"),
                "has_captions": bool(v.get("has_captions", False)),
                "caption_source": v.get("caption_source"),
                "published_at": v.get("published_at"),
            }
        )

    if not rows:
        return []

    result = (
        db.table("youtube_videos")
        .upsert(rows, on_conflict="creator_id,video_id")
        .execute()
    )
    ids = [r["id"] for r in (result.data or [])]
    logger.info(f"Upserted {len(ids)} YouTube videos for creator {creator_id}")
    return ids


def upsert_creator_score_platform(
    db: Client, creator_id: str, platform: str, scores: dict
) -> None:
    """Write a per-platform creator_scores row.

    Separate from upsert_creator_scores (IG path) so the platform column
    is explicit. For IG, callers can keep using upsert_creator_scores and
    it will default platform='instagram' at the DB level.
    """
    row = {
        **scores,
        "creator_id": creator_id,
        "platform": platform,
        "computed_at": datetime.now().isoformat(),
    }
    db.table("creator_scores").insert(row).execute()
    logger.info(f"Wrote {platform} creator_scores row for {creator_id}")


def upsert_creator_platform_embedding(
    db: Client, creator_id: str, platform: str, embedding: list[float]
) -> None:
    """Write a per-platform content embedding (migration 046).

    Replaces the single-valued `creators.content_embedding` for scoring
    purposes. A YT rescrape no longer overwrites IG's DNA vector.
    Existing `creators.content_embedding` is kept as a shadow for one
    release — IG callers should populate both this table AND the shadow
    column until the shadow is dropped in the follow-up migration.
    """
    if not embedding:
        return
    db.table("creator_content_embeddings").upsert(
        {
            "creator_id": creator_id,
            "platform": platform,
            "embedding": embedding,
            "computed_at": datetime.now().isoformat(),
        },
        on_conflict="creator_id,platform",
    ).execute()
    logger.debug(
        f"Upserted {platform} content embedding for creator {creator_id}"
    )


def upsert_brand_platform_analysis(
    db: Client, brand_id: str, platform: str, analysis: dict
) -> None:
    """Upsert into brand_platform_analyses (generalizes brands.ig_*)."""
    row = {
        "brand_id": brand_id,
        "platform": platform,
        "handle": analysis.get("handle"),
        "profile_url": analysis.get("profile_url"),
        "analysis_status": analysis.get("analysis_status", "none"),
        "analysis_completed_at": analysis.get("analysis_completed_at"),
        "analysis_error": analysis.get("analysis_error"),
        "content_dna": analysis.get("content_dna"),
        "audience_profile": analysis.get("audience_profile"),
        "collaborators": analysis.get("collaborators") or [],
        "content_embedding": analysis.get("content_embedding"),
        "embedding_computed_at": analysis.get("embedding_computed_at"),
    }
    db.table("brand_platform_analyses").upsert(
        row, on_conflict="brand_id,platform"
    ).execute()
    logger.info(f"Upserted {platform} brand analysis for brand {brand_id}")


def _find_creator_by_platform_profile(
    db: Client, platform: str, platform_user_id: str | None, handle: str | None
) -> str | None:
    """Look up an existing creator via creator_social_profiles.

    Prefers platform_user_id (stable) over handle (can change). Returns
    creator_id or None. Used by the YT ingestion path so a single creator
    with both IG and YT profiles ends up on one row rather than two.
    """
    if platform_user_id:
        res = (
            db.table("creator_social_profiles")
            .select("creator_id")
            .eq("platform", platform)
            .eq("platform_user_id", platform_user_id)
            .limit(1)
            .execute()
        )
        if res.data:
            return res.data[0]["creator_id"]
    if handle:
        res = (
            db.table("creator_social_profiles")
            .select("creator_id")
            .eq("platform", platform)
            .eq("handle", handle)
            .limit(1)
            .execute()
        )
        if res.data:
            return res.data[0]["creator_id"]
    return None


def _create_youtube_creator_shell(db: Client, profile: dict) -> str:
    """Create a minimal `creators` row for a YouTube-first creator.

    The legacy `creators` table requires `handle`. For YT-primary creators
    we use the YT handle (strip leading @). A follow-up migration will let
    `creators.handle` go nullable — until then this keeps the NOT NULL
    constraint satisfied without colliding with IG handles (YT handles
    can legally collide with IG; we suffix `_yt` on collision).
    """
    handle_base = profile.get("handle") or profile.get("platform_user_id") or ""
    handle = handle_base
    # Disambiguate against an existing IG handle with the same string.
    existing = (
        db.table("creators").select("id").eq("handle", handle).limit(1).execute()
    )
    if existing.data:
        handle = f"{handle_base}_yt"

    row = {
        "handle": handle,
        "display_name": profile.get("display_name"),
        "biography": profile.get("bio"),
        "avatar_url": profile.get("avatar_url"),
        "category": profile.get("category"),
        "country": profile.get("country"),
        "is_verified": profile.get("is_verified", False),
        "followers": profile.get("followers_or_subs") or 0,
        "posts_count": profile.get("posts_or_videos_count") or 0,
        "tier": profile.get("tier", "nano"),
        # YT scraper extracts contact info from bio (no dedicated field
        # on YouTube). Persist on the canonical creators row so the
        # outreach flow finds it the same way it finds IG-side emails.
        "contact_email": profile.get("email"),
        "contact_phone": profile.get("phone"),
        "last_scraped_at": datetime.now().isoformat(),
    }
    result = db.table("creators").insert(row).execute()
    return result.data[0]["id"]


def store_youtube_cip(
    db: Client,
    cip: dict,
    existing_creator_id: str | None = None,
) -> str:
    """Persist a YouTube CIP (output of build_youtube_creator_intelligence_profile).

    Mirrors store_full_cip's responsibility but writes onto the
    platform-aware tables:
      - creator_social_profiles(platform='youtube')
      - youtube_videos
      - creator_scores(platform='youtube')
      - caption_intelligence / transcript_intelligence / audience_intelligence

    Phase 2.5 auto-stitch: when `existing_creator_id` is set, we skip the
    find-or-create step and attach the YT profile onto the caller-provided
    `creators.id` (e.g., IG scrape discovered the creator's YT handle in
    its external_url and fanned out a YT scrape bound to the same row).

    Returns the creator UUID.
    """
    profile = cip.get("profile") or {}
    resolved = cip.get("resolved") or {}

    # 1. Find or create the canonical `creators` row.
    if existing_creator_id:
        creator_id = existing_creator_id
    else:
        creator_id = _find_creator_by_platform_profile(
            db,
            platform="youtube",
            platform_user_id=resolved.get("channel_id")
            or profile.get("platform_user_id"),
            handle=profile.get("handle"),
        )
        if not creator_id:
            creator_id = _create_youtube_creator_shell(db, profile)

    # 1b. Backfill contact info on the canonical creators row when the
    # existing row has NULL for contact_email / contact_phone but our YT
    # scrape found one in the bio. Don't overwrite an existing IG-side
    # value — that one was explicit; ours is bio-extracted.
    yt_email = profile.get("email")
    yt_phone = profile.get("phone")
    if yt_email or yt_phone:
        try:
            current = (
                db.table("creators")
                .select("contact_email,contact_phone")
                .eq("id", creator_id)
                .single()
                .execute()
            )
            updates: dict = {}
            if yt_email and not (current.data or {}).get("contact_email"):
                updates["contact_email"] = yt_email
            if yt_phone and not (current.data or {}).get("contact_phone"):
                updates["contact_phone"] = yt_phone
            if updates:
                db.table("creators").update(updates).eq("id", creator_id).execute()
                logger.info(
                    f"Backfilled contact info on creator {creator_id}: "
                    f"{list(updates.keys())}"
                )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"contact backfill skipped for {creator_id}: {e}")

    # 2. Write / refresh the YT social profile row.
    upsert_social_profile(db, creator_id, "youtube", profile)

    # 3. Videos.
    upsert_youtube_videos(db, creator_id, cip.get("videos") or [])

    # 4. Scores — route YT-specific sub-metrics onto their columns.
    scores = cip.get("scores") or {}
    if scores:
        score_row = {
            "cpi": scores.get("cpi", 0),
            "engagement_quality": scores.get("engagement_quality", 0),
            "content_quality": scores.get("content_quality", 0),
            "audience_authenticity": scores.get("audience_authenticity", 0),
            "growth_trajectory": scores.get("growth_trajectory", 0),
            "professionalism": scores.get("professionalism", 0),
            "avg_views_per_sub": scores.get("avg_views_per_sub"),
            "watch_through_proxy": scores.get("watch_through_proxy"),
            "upload_cadence_days": scores.get("upload_cadence_days"),
            "pipeline_version": cip.get("pipeline_version", "1.1"),
            "confidence": scores.get("confidence"),
            "scoring_inputs": scores.get("scoring_inputs", {}),
        }
        upsert_creator_score_platform(db, creator_id, "youtube", score_row)

    # 5. Intelligence blocks — platform='youtube' so they don't overwrite
    # any IG analysis this creator already has (migration 047).
    cap = cip.get("caption_intelligence")
    if cap and not (isinstance(cap, dict) and cap.get("_llm_failure")):
        insert_caption_intelligence(db, creator_id, cap, platform="youtube")
    tr = cip.get("transcript_intelligence")
    if tr and not (isinstance(tr, dict) and tr.get("_llm_failure")):
        insert_transcript_intelligence(db, creator_id, tr, platform="youtube")
    ai = cip.get("audience_intelligence")
    if ai and not (isinstance(ai, dict) and ai.get("_llm_failure")):
        insert_audience_intelligence(db, creator_id, ai, platform="youtube")

    logger.info(
        f"Stored YouTube CIP for creator {creator_id} "
        f"(channel_id={resolved.get('channel_id')})"
    )
    return creator_id
