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
from pipeline.confidence import (
    CoverageTracker,
    CPI_WEIGHTS,
    pull as _pull,
)
from pipeline.fraud_flags import compute_fraud_flags, flag_codes

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "1.1"


def build_creator_intelligence_profile(
    profile_url: str,
    brightdata_token: str,
    gemini_api_key: str,
    openai_api_key: str,
    num_posts: int = 20,
    num_reels: int = 5,
    num_comment_posts: int = 5,
    days_back: int = 90,
    er_benchmarks: dict | None = None,
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
        "pipeline_version": PIPELINE_VERSION,
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
    llm_status = _summarize_llm_status(cip)
    cip["scores"] = compute_creator_scores(
        cip, llm_status=llm_status, er_benchmarks=er_benchmarks
    )

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


DEFAULT_ER_BENCHMARKS: dict[str, float] = {
    "nano": 0.06,
    "micro": 0.04,
    "mid": 0.025,
    "macro": 0.015,
    "mega": 0.01,
}


def _summarize_llm_status(cip: dict) -> dict:
    """Translate the CIP's LLM outputs into a success map.

    A dimension is counted as succeeded iff the corresponding intel
    block is present and does not carry the `_llm_failure` sentinel
    (emitted by the LLMFailure wrapper in W2).
    """
    def _ok(block):
        if not isinstance(block, dict):
            return bool(block)
        if block.get("_llm_failure"):
            return False
        return bool(block)

    return {
        "captions": _ok(cip.get("caption_intelligence")),
        "transcripts": _ok(cip.get("transcript_intelligence")),
        "comments": _ok(cip.get("audience_intelligence")),
    }


def compute_creator_scores(
    cip: dict,
    llm_status: dict | None = None,
    er_benchmarks: dict | None = None,
) -> dict:
    """
    Compute the Creator Performance Index (CPI) and sub-scores.

    CPI is a weighted composite:
    - Engagement Quality: 30%
    - Content Quality: 25%
    - Audience Authenticity: 20%
    - Growth Trajectory: 15%
    - Professionalism: 10%

    The math is unchanged from v1.0 — this version additionally
    records which inputs were present vs. defaulted so the returned
    scores carry a `confidence` envelope downstream consumers can
    gate on.
    """
    scores: dict = {}
    tracker = CoverageTracker()
    llm_status = llm_status or {}

    # Record LLM dimension success/failure up front so every
    # defaulted downstream key can be labelled with the real reason.
    for dim in ("captions", "transcripts", "comments"):
        if dim in llm_status:
            if llm_status[dim]:
                tracker.mark_llm_success(dim)
            else:
                err_msg = (
                    (cip.get(f"{_dim_to_cip_key(dim)}") or {}).get(
                        "error", "llm_failure"
                    )
                )
                tracker.mark_llm_failure(dim, str(err_msg))

    # Surface any upstream data-quality flags from the profile / posts
    # scrapers (set by W9). They do not affect math but ride along in
    # the confidence envelope.
    for flag in cip.get("profile", {}).get("data_quality_flags", []) or []:
        tracker.add_data_quality_flag(flag)
    posts_flag = cip.get("posts", {}).get("data_quality_flag")
    if posts_flag:
        tracker.add_data_quality_flag(posts_flag)

    # --- Engagement Quality (0-100) ---
    # Tier-adjusted ER benchmarks: what counts as "excellent" varies by size
    tier = cip.get("profile", {}).get("tier", "micro")
    benchmarks = er_benchmarks or DEFAULT_ER_BENCHMARKS
    er_benchmark = benchmarks.get(tier, 0.03)

    posts_flag = cip.get("posts", {}).get("data_quality_flag")
    avg_er = _pull(
        cip, ["posts", "avg_engagement_rate"],
        default=0, key="avg_er", tracker=tracker,
    )
    # Low-followers stub emits None — re-tag with the real reason.
    if avg_er is None or posts_flag == "low_followers":
        tracker.mark_default("avg_er", reason="low_followers")
        avg_er = 0
    reply_rate = _pull(
        cip, ["comments", "creator_reply_rate"],
        default=0, key="reply_rate", tracker=tracker,
    )
    rewatch = _pull(
        cip, ["reels", "avg_rewatch_rate"],
        default=0, key="rewatch", tracker=tracker,
    )

    # LLM-derived engagement quality — explicit None is the "missing"
    # sentinel the old code already handled.
    llm_eq = _pull(
        cip,
        ["audience_intelligence", "engagement_quality", "quality_score"],
        default=None, key="llm_engagement_quality", tracker=tracker,
    )
    conv_depth = _pull(
        cip,
        ["audience_intelligence", "engagement_quality", "conversation_depth"],
        default="shallow", key="conv_depth", tracker=tracker,
    )
    community = _pull(
        cip,
        ["audience_intelligence", "engagement_quality", "community_feel"],
        default="weak", key="community_feel", tracker=tracker,
    )

    # If the comments LLM dimension failed, re-tag every engagement-quality
    # LLM key with reason=llm_failure (pull() logs them as "missing").
    if llm_status.get("comments") is False:
        for k in ("llm_engagement_quality", "conv_depth", "community_feel"):
            if tracker.is_defaulted(k):
                tracker.mark_default(k, reason="llm_failure")

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
    hook_quality = _pull(
        cip,
        ["transcript_intelligence", "hook_analysis", "avg_hook_quality"],
        default=0.5, key="hook_quality", tracker=tracker,
    )
    consistency = _pull(
        cip, ["posts", "posting_consistency_stddev_days"],
        default=10, key="consistency_stddev", tracker=tracker,
    )
    posts_per_week = _pull(
        cip, ["posts", "posts_per_week"],
        default=0, key="posts_per_week", tracker=tracker,
    )
    storytelling = _pull(
        cip,
        ["transcript_intelligence", "content_depth", "storytelling_score"],
        default=0.5, key="storytelling", tracker=tracker,
    )
    edu_density = _pull(
        cip,
        ["transcript_intelligence", "content_depth", "educational_density"],
        default=0.3, key="educational_density", tracker=tracker,
    )
    if llm_status.get("transcripts") is False:
        for k in ("hook_quality", "storytelling", "educational_density"):
            if tracker.is_defaulted(k):
                tracker.mark_default(k, reason="llm_failure")

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
    auth_score_raw = _pull(
        cip,
        ["audience_intelligence", "audience_authenticity", "authenticity_score"],
        default=0.5, key="authenticity_score", tracker=tracker,
    )
    substantive_pct = _pull(
        cip,
        ["audience_intelligence", "audience_authenticity",
         "substantive_comment_percentage"],
        default=0.1, key="substantive_pct", tracker=tracker,
    )
    follower_ratio = _pull(
        cip, ["profile", "follower_following_ratio"],
        default=1, key="follower_ratio", tracker=tracker,
    )
    if llm_status.get("comments") is False:
        for k in ("authenticity_score", "substantive_pct"):
            if tracker.is_defaulted(k):
                tracker.mark_default(k, reason="llm_failure")
    ratio_signal = min(follower_ratio / 50, 1.0)

    scores["audience_authenticity"] = round(
        auth_score_raw * 40
        + substantive_pct * 30
        + ratio_signal * 30,
        1,
    )

    # --- Growth Trajectory (0-100) ---
    # Treat "insufficient_data" as a present-but-insufficient sentinel
    # so downstream can distinguish it from a real "stable".
    trend = _pull(
        cip, ["posts", "engagement_trend"],
        default="stable", key="engagement_trend", tracker=tracker,
        present_predicate=lambda v: v != "insufficient_data",
    )
    if trend == "insufficient_data":
        tracker.mark_default(
            "engagement_trend", reason="fewer_than_4_posts"
        )
    trend_map = {
        "growing": 80,
        "stable": 50,
        "declining": 20,
        "insufficient_data": 50,
    }
    scores["growth_trajectory"] = trend_map.get(trend, 50)

    # --- Professionalism (0-100) ---
    is_business = _pull(
        cip, ["profile", "is_business"],
        default=False, key="is_business", tracker=tracker,
    )
    is_verified = _pull(
        cip, ["profile", "is_verified"],
        default=False, key="is_verified", tracker=tracker,
    )
    email_value = _pull(
        cip, ["profile", "email"],
        default=None, key="has_email", tracker=tracker,
    )
    has_email = bool(email_value)

    # Use Whisper confidence for audio quality instead of LLM guess
    transcripts_list = cip.get("transcripts") or []
    audio_quality = _classify_audio_quality(transcripts_list)
    if transcripts_list:
        tracker.mark_present("audio_quality")
    else:
        tracker.mark_default("audio_quality", reason="no_transcripts")
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
        scores["engagement_quality"] * CPI_WEIGHTS["engagement_quality"]
        + scores["content_quality"] * CPI_WEIGHTS["content_quality"]
        + scores["audience_authenticity"] * CPI_WEIGHTS["audience_authenticity"]
        + scores["growth_trajectory"] * CPI_WEIGHTS["growth_trajectory"]
        + scores["professionalism"] * CPI_WEIGHTS["professionalism"],
        1,
    )

    # --- Fraud flags (sub-score weights untouched) ---
    fraud = compute_fraud_flags(cip)
    cip["fraud_flags"] = fraud
    scores["fraud_flag_codes"] = flag_codes(fraud)
    # Full payload ridealong for persistence; the engine reads codes only.
    scores["_fraud_flags_full"] = fraud

    envelope = tracker.to_dict(subscore_weights=CPI_WEIGHTS)
    scores["confidence"] = envelope
    scores["coverage_percentage"] = round(
        envelope["overall_coverage"] * 100, 2
    )
    scores["confidence_tier"] = envelope["tier"]
    scores["missing_inputs"] = envelope["missing_inputs"]
    scores["llm_calls_succeeded"] = envelope["llm_calls_succeeded"]
    scores["data_quality_flags"] = envelope["data_quality_flags"]

    return scores


def _dim_to_cip_key(dim: str) -> str:
    return {
        "captions": "caption_intelligence",
        "transcripts": "transcript_intelligence",
        "comments": "audience_intelligence",
    }[dim]


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
