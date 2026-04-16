import copy
import logging
from datetime import datetime

from pipeline.brightdata_client import BrightdataClient
from pipeline.scraper_profiles import scrape_profiles, extract_profile_metrics
from pipeline.scraper_posts import (
    scrape_posts_discovery,
    extract_post_metrics,
)
from pipeline.scraper_reels import (
    scrape_reels_discovery,
    select_top_reels,
    extract_reel_metrics,
)
from pipeline.scraper_comments import (
    scrape_comments,
    select_top_posts_for_comments,
    extract_comment_metrics,
)
from pipeline.transcriber import transcribe_reels as transcribe_reels_whisper
from pipeline.llm_client import init_gemini
from pipeline.llm_captions import analyze_captions
from pipeline.llm_transcripts import analyze_transcripts
from pipeline.llm_comments import analyze_comments

logger = logging.getLogger(__name__)


def build_creator_intelligence_profile(
    profile_url: str,
    brightdata_token: str,
    gemini_api_key: str,
    openai_api_key: str,
    num_posts: int = 20,
    num_reels: int = 5,
    num_comment_posts: int = 5,
    days_back: int = 90,
) -> dict:
    """
    Full pipeline: Scrape -> Transcribe -> Analyse -> Compute -> Return CIP.

    Total Brightdata cost per creator: ~$0.11
      - Profile: 1 record = $0.0015
      - Posts Discovery (20): 20 records = $0.03
      - Reels Detail (5): 5 records = $0.0075
      - Comments (5 posts x 10): 50 records = $0.075

    Total Whisper cost per creator: ~$0.015
      - 5 reels x ~30s avg = 2.5 min at $0.006/min

    Total Gemini cost per creator: ~$0.002
      - 3 calls x ~3K tokens avg = ~9K tokens at $0.10/1M

    GRAND TOTAL: ~$0.13 per creator
    """
    logger.info(f"Starting CIP build for {profile_url}")

    bd_client = BrightdataClient(api_token=brightdata_token)
    gemini_client = init_gemini(gemini_api_key)

    cip = {
        "profile_url": profile_url,
        "pipeline_version": "1.0",
        "scraped_at": datetime.now().isoformat(),
        "data_provenance": "brightdata_scraper_api",
    }

    # ── STEP 1: Profile Scrape ──
    logger.info("  [Step 1/6] Scraping profile...")
    raw_profiles = scrape_profiles(bd_client, [profile_url])

    if not raw_profiles:
        return {
            "error": "Profile scrape returned no data",
            "profile_url": profile_url,
        }

    raw_profile = raw_profiles[0]
    profile_metrics = extract_profile_metrics(raw_profile)
    cip["profile"] = profile_metrics

    handle = profile_metrics["handle"]
    followers = profile_metrics["followers"]

    # ── STEP 2: Posts Discovery ──
    logger.info(f"  [Step 2/6] Discovering {num_posts} recent posts...")
    raw_posts = scrape_posts_discovery(
        bd_client,
        profile_url,
        num_posts=num_posts,
        days_back=days_back,
    )

    logger.info(
        f"  Posts Discovery returned {len(raw_posts)} posts."
    )

    post_metrics = extract_post_metrics(raw_posts, followers, handle=handle)
    cip["posts"] = post_metrics
    cip["_raw_posts"] = raw_posts  # needed by store_full_cip for DB insert

    # ── STEP 3a: Reels Discovery ──
    logger.info(f"  [Step 3a/6] Discovering {num_reels} reels from profile...")
    raw_reels = scrape_reels_discovery(bd_client, profile_url, num_reels=num_reels)

    # Filter out reels that don't belong to this creator
    creator_ig_id = profile_metrics.get("instagram_id")
    if raw_reels and creator_ig_id:
        own_reels = [
            r for r in raw_reels
            if str(r.get("post_id", "")).endswith(f"_{creator_ig_id}")
            or str(r.get("input_url", "")).rstrip("/") == profile_url.rstrip("/")
        ]
        if len(own_reels) < len(raw_reels):
            logger.info(
                f"  Filtered reels: {len(raw_reels)} -> {len(own_reels)} "
                f"(removed {len(raw_reels) - len(own_reels)} from other accounts)"
            )
        raw_reels = own_reels

    if raw_reels:
        top_reels = select_top_reels(raw_reels, top_n=num_reels)
        reel_metrics = extract_reel_metrics(top_reels)
        cip["reels"] = reel_metrics
    else:
        logger.info("  [Step 3a/6] No reels found, skipping...")
        cip["reels"] = {}

    # ── STEP 3b: Comments ──
    comment_urls = select_top_posts_for_comments(
        raw_posts, top_n=num_comment_posts
    )

    if comment_urls:
        logger.info(
            f"  [Step 3b/6] Scraping comments from {len(comment_urls)} posts..."
        )
        raw_comments = scrape_comments(bd_client, comment_urls)
        comment_metrics = extract_comment_metrics(raw_comments, handle)
        cip["comments"] = comment_metrics
    else:
        logger.info("  [Step 3b/6] No commented posts found, skipping...")
        cip["comments"] = {}

    # ── STEP 4: Whisper Transcription ──
    video_urls_for_whisper = cip.get("reels", {}).get(
        "video_urls_for_whisper", []
    )

    if video_urls_for_whisper:
        logger.info(
            f"  [Step 4/6] Transcribing {len(video_urls_for_whisper)} reels "
            "with Whisper..."
        )
        transcripts = transcribe_reels_whisper(
            video_urls_for_whisper, openai_api_key
        )
        cip["transcripts"] = transcripts
    else:
        logger.info("  [Step 4/6] No video URLs available, skipping Whisper...")
        cip["transcripts"] = []

    # ── STEP 5: Gemini LLM Analysis ──
    logger.info("  [Step 5/6] Running Gemini analysis...")

    # 5a: Caption Analysis
    captions = post_metrics.get("_captions", [])
    if captions:
        cip["caption_intelligence"] = analyze_captions(
            gemini_client,
            handle=handle,
            bio=profile_metrics.get("bio", ""),
            category=profile_metrics.get("category", ""),
            captions=captions,
        )

    # 5b: Transcript Analysis (filter out music-only transcripts)
    valid_transcripts = [
        t for t in cip.get("transcripts", [])
        if t.get("transcript_text") and not t.get("is_likely_music", False)
    ]
    music_count = sum(
        1 for t in cip.get("transcripts", []) if t.get("is_likely_music")
    )
    if music_count:
        logger.info(
            f"  Filtered {music_count} music-only transcript(s) from LLM analysis"
        )
    if valid_transcripts:
        cip["transcript_intelligence"] = analyze_transcripts(
            gemini_client,
            handle=handle,
            transcripts=valid_transcripts,
        )

    # 5c: Comment Analysis
    comment_texts = cip.get("comments", {}).get("_comment_texts", [])
    if comment_texts:
        cip["audience_intelligence"] = analyze_comments(
            gemini_client,
            handle=handle,
            comment_texts=comment_texts,
            comment_timestamps=cip.get("comments", {}).get(
                "_comment_timestamps"
            ),
            commenter_handles=cip.get("comments", {}).get(
                "_commenter_handles"
            ),
            comment_hour_distribution=cip.get("comments", {}).get(
                "comment_hour_distribution_utc"
            ),
            num_posts_with_comments=len(
                cip.get("comments", {}).get("_comment_texts", [])
            ),
        )

    # ── STEP 6: Compute Final Scores ──
    logger.info("  [Step 6/6] Computing Creator Performance Index...")
    cip["scores"] = compute_creator_scores(cip)

    logger.info(f"CIP complete for @{handle} — CPI: {cip['scores']['cpi']}")
    return cip


def _classify_audio_quality(transcripts: list[dict]) -> str:
    """Classify audio quality from Whisper confidence scores instead of LLM guessing."""
    confidences = [
        t.get("avg_confidence", 0)
        for t in transcripts
        if t.get("transcript_text") and not t.get("is_likely_music", False)
    ]
    if not confidences:
        return "casual"
    avg = sum(confidences) / len(confidences)
    if avg >= 0.85:
        return "professional"
    elif avg >= 0.65:
        return "semi_professional"
    elif avg >= 0.45:
        return "casual"
    else:
        return "raw"


def compute_creator_scores(cip: dict) -> dict:
    """
    Compute the Creator Performance Index (CPI) and sub-scores.

    CPI is a weighted composite:
    - Engagement Quality: 30%
    - Content Quality: 25%
    - Audience Authenticity: 20%
    - Growth Trajectory: 15%
    - Professionalism: 10%
    """
    scores = {}

    # --- Engagement Quality (0-100) ---
    # Tier-adjusted ER benchmarks: what counts as "excellent" varies by size
    tier = cip.get("profile", {}).get("tier", "micro")
    er_benchmarks = {
        "nano": 0.06,
        "micro": 0.04,
        "mid": 0.025,
        "macro": 0.015,
        "mega": 0.01,
    }
    er_benchmark = er_benchmarks.get(tier, 0.03)

    avg_er = cip.get("posts", {}).get("avg_engagement_rate", 0)
    reply_rate = cip.get("comments", {}).get("creator_reply_rate", 0)
    rewatch = cip.get("reels", {}).get("avg_rewatch_rate", 0)

    # LLM-derived engagement quality (has good variance 0.1-0.9)
    llm_eq = (
        cip.get("audience_intelligence", {})
        .get("engagement_quality", {})
        .get("quality_score")
    )
    # LLM conversation depth and community feel
    conv_depth = (
        cip.get("audience_intelligence", {})
        .get("engagement_quality", {})
        .get("conversation_depth", "shallow")
    )
    community = (
        cip.get("audience_intelligence", {})
        .get("engagement_quality", {})
        .get("community_feel", "weak")
    )

    er_score = min(avg_er / er_benchmark, 1.0) * 40
    reply_score = min(reply_rate / 0.5, 1.0) * 10
    # Cap rewatch at 5.0 to prevent runaway values, benchmark at 2.0
    rewatch_score = min(min(rewatch, 5.0) / 2.0, 1.0) * 10

    # LLM engagement quality: blend in if available (20 pts)
    if llm_eq is not None:
        llm_eq_score = llm_eq * 20
    else:
        llm_eq_score = 10  # neutral default

    # Community/depth signals (20 pts)
    depth_map = {"deep": 10, "moderate": 6, "shallow": 2}
    community_map = {"strong": 10, "moderate": 6, "weak": 2}
    depth_score = depth_map.get(conv_depth, 2)
    community_score = community_map.get(community, 2)

    scores["engagement_quality"] = round(
        er_score + reply_score + rewatch_score + llm_eq_score
        + depth_score + community_score,
        1,
    )

    # --- Content Quality (0-100) ---
    hook_quality = (
        cip.get("transcript_intelligence", {})
        .get("hook_analysis", {})
        .get("avg_hook_quality", 0.5)
    )
    consistency = cip.get("posts", {}).get(
        "posting_consistency_stddev_days", 10
    )
    posts_per_week = cip.get("posts", {}).get("posts_per_week", 0)

    # LLM content signals
    storytelling = (
        cip.get("transcript_intelligence", {})
        .get("content_depth", {})
        .get("storytelling_score", 0.5)
    )
    edu_density = (
        cip.get("transcript_intelligence", {})
        .get("content_depth", {})
        .get("educational_density", 0.3)
    )

    hook_score = hook_quality * 25
    consistency_score = max(0, (1 - min(consistency / 14, 1))) * 20
    frequency_score = min(posts_per_week / 5, 1.0) * 20
    storytelling_score = storytelling * 20
    edu_score = edu_density * 15
    scores["content_quality"] = round(
        hook_score + consistency_score + frequency_score
        + storytelling_score + edu_score,
        1,
    )

    # --- Audience Authenticity (0-100) ---
    auth_score_raw = (
        cip.get("audience_intelligence", {})
        .get("audience_authenticity", {})
        .get("authenticity_score", 0.5)
    )
    substantive_pct = (
        cip.get("audience_intelligence", {})
        .get("audience_authenticity", {})
        .get("substantive_comment_percentage", 0.1)
    )
    follower_ratio = cip.get("profile", {}).get(
        "follower_following_ratio", 1
    )
    ratio_signal = min(follower_ratio / 50, 1.0)

    scores["audience_authenticity"] = round(
        auth_score_raw * 40
        + substantive_pct * 30
        + ratio_signal * 30,
        1,
    )

    # --- Growth Trajectory (0-100) ---
    trend = cip.get("posts", {}).get("engagement_trend", "stable")
    trend_map = {
        "growing": 80,
        "stable": 50,
        "declining": 20,
        "insufficient_data": 50,
    }
    scores["growth_trajectory"] = trend_map.get(trend, 50)

    # --- Professionalism (0-100) ---
    is_business = cip.get("profile", {}).get("is_business", False)
    is_verified = cip.get("profile", {}).get("is_verified", False)
    has_email = bool(cip.get("profile", {}).get("email"))

    # Use Whisper confidence for audio quality instead of LLM guess
    audio_quality = _classify_audio_quality(
        cip.get("transcripts", [])
    )
    quality_map = {
        "professional": 30,
        "semi_professional": 20,
        "casual": 10,
        "raw": 5,
    }

    prof_score = (
        (25 if is_business else 0)
        + (20 if is_verified else 0)
        + (25 if has_email else 0)
        + quality_map.get(audio_quality, 10)
    )
    scores["professionalism"] = min(prof_score, 100)

    # --- COMPOSITE CPI ---
    scores["cpi"] = round(
        scores["engagement_quality"] * 0.30
        + scores["content_quality"] * 0.25
        + scores["audience_authenticity"] * 0.20
        + scores["growth_trajectory"] * 0.15
        + scores["professionalism"] * 0.10,
        1,
    )

    return scores


def clean_cip_for_export(cip: dict) -> dict:
    """Return a deep copy with internal _-prefixed fields removed."""
    cleaned = copy.deepcopy(cip)
    _strip_internal_fields(cleaned)
    return cleaned


def _strip_internal_fields(obj):
    """Recursively remove fields prefixed with _ (internal pipeline data)."""
    if isinstance(obj, dict):
        keys_to_remove = [k for k in obj if k.startswith("_")]
        for k in keys_to_remove:
            del obj[k]
        for v in obj.values():
            _strip_internal_fields(v)
    elif isinstance(obj, list):
        for item in obj:
            _strip_internal_fields(item)
