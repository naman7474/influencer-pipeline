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


from pipeline.contact_extract import extract_email_from_text


def _score_professionalism_instagram(cip: dict, tracker) -> float:
    """IG-tuned professionalism (0-100). Existing logic, factored out so the
    YT path can diverge cleanly."""
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
    return min(prof_score, 100)


def _score_professionalism_youtube(cip: dict, tracker) -> float:
    """YT-tuned professionalism (0-100).

    YouTube doesn't expose `is_business` (no creator-type flag like IG) and
    `verified` is inconsistent across our scrape sources. Instead we read:
      - is_business: channel-age proxy (>= 3 years tells us this is an
        established, intentional creator). 25 pts.
      - is_verified: BD's `verified` flag OR mega-tier (>= 1M subs is
        effectively a verified channel). 25 pts.
      - has_email: regex-extract email from the channel `bio`/`Description`
        which is where YT creators put `business@…` contacts. 25 pts.
      - audio_quality: tier-driven — mega/macro creators run pro production.
        Falls back to Whisper-confidence when our transcripts came from
        Whisper rather than youtube-transcript-api. 25 pts max.

    Tracks the same coverage keys (is_business, is_verified, has_email,
    audio_quality) so the confidence envelope stays comparable across
    platforms.
    """
    profile = cip.get("profile") or {}

    # ── Channel-age proxy → is_business ──
    created = (
        profile.get("channel_created_at")
        or profile.get("created_date")
        or profile.get("published_at")
    )
    age_years = _channel_age_years(created)
    if age_years is not None:
        tracker.mark_present("is_business")
    else:
        tracker.mark_default("is_business", reason="no_created_date")
    is_business_proxy = age_years is not None and age_years >= 3

    # ── Verified ──
    raw_verified = profile.get("is_verified")
    if raw_verified is not None:
        tracker.mark_present("is_verified")
    else:
        tracker.mark_default("is_verified", reason="missing")
    # Mega-tier YT channels are effectively verified — being publicly visible
    # at 1M+ subs without verification is rare. Bridges the gap when BD
    # doesn't surface the flag.
    is_verified_proxy = bool(raw_verified) or profile.get("tier") == "mega"

    # ── Email extraction from bio ──
    # The scraper now writes profile.email when bio contained a match —
    # honor that, but also re-scan in case profile.email is missing.
    explicit_email = profile.get("email")
    bio = (profile.get("bio") or "").strip()
    has_email = bool(explicit_email or extract_email_from_text(bio))
    if explicit_email or bio:
        tracker.mark_present("has_email")
    else:
        tracker.mark_default("has_email", reason="no_bio")

    # ── Audio quality ──
    # If we have Whisper-tier transcripts, use confidence as before.
    # Otherwise (most YT creators, where youtube-transcript-api worked),
    # use tier as a proxy: macro/mega → professional, mid → semi, smaller → casual.
    transcripts_list = cip.get("transcripts") or []
    has_whisper = any(t.get("avg_confidence") is not None for t in transcripts_list)
    if has_whisper:
        audio_quality = _classify_audio_quality(transcripts_list)
        tracker.mark_present("audio_quality")
    else:
        tier = profile.get("tier", "nano")
        audio_quality = {
            "mega": "professional",
            "macro": "semi_professional",
            "mid": "casual",
            "micro": "casual",
            "nano": "raw",
        }.get(tier, "casual")
        if transcripts_list:
            # We have transcripts but no confidence — partial credit.
            tracker.mark_present("audio_quality")
        else:
            tracker.mark_default("audio_quality", reason="no_transcripts")

    quality_map = {
        "professional": 25,
        "semi_professional": 18,
        "casual": 10,
        "raw": 5,
    }
    prof_score = (
        (25 if is_business_proxy else 0)
        + (25 if is_verified_proxy else 0)
        + (25 if has_email else 0)
        + quality_map.get(audio_quality, 10)
    )
    return min(prof_score, 100)


def _channel_age_years(created_iso: str | None) -> float | None:
    """Parse an ISO 8601 timestamp; return age in years from now."""
    if not created_iso:
        return None
    from datetime import datetime, timezone

    try:
        dt = datetime.fromisoformat(str(created_iso).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    delta = datetime.now(timezone.utc) - dt
    return delta.days / 365.25


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
    if cip.get("platform") == "youtube":
        scores["professionalism"] = _score_professionalism_youtube(
            cip, tracker
        )
    else:
        scores["professionalism"] = _score_professionalism_instagram(
            cip, tracker
        )

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


# ──────────────────────────────────────────────────────────────
# YouTube pipeline orchestration
# ──────────────────────────────────────────────────────────────


def build_youtube_creator_intelligence_profile(
    channel_url: str,
    brightdata_token: str,
    gemini_api_key: str,
    openai_api_key: str,
    youtube_api_key: str | None = None,
    num_videos: int = 20,
    num_transcripts: int = 5,
) -> dict:
    """YouTube CIP — parallel to build_creator_intelligence_profile.

    Stages:
      1. Resolve the URL -> (channel_id, canonical url) via handle_resolver.
      2. Bright Data channel scrape: profile / subs / about / external links.
      3. Bright Data video discovery: top N recent videos.
      4. Transcription: prefer inline captions from Bright Data; fall back
         to Whisper only when captions are absent.
      5. Gemini analysis: captions (video description + title + tags),
         transcripts, top-video comments.
      6. Scoring: compute_creator_scores() with YT ER benchmarks and YT
         inputs (views_per_sub, watch_through_proxy, upload cadence).

    Returns the CIP dict ready for store_youtube_cip(). On any stage
    failure the function returns a partial CIP with an `error` key —
    mirrors the IG flow so the caller can ignore/retry.
    """
    from pipeline.youtube.handle_resolver import resolve as resolve_yt
    from pipeline.youtube.youtube_api import YouTubeAPIClient
    from pipeline.youtube.scraper_channels import (
        scrape_channels,
        extract_channel_metrics,
    )
    from pipeline.youtube.scraper_videos import (
        scrape_videos_discovery,
        select_top_videos,
        extract_video_metrics,
        aggregate_channel_metrics,
    )
    from pipeline.youtube.scraper_comments import (
        scrape_comments as scrape_yt_comments,
        select_top_videos_for_comments,
        extract_comment_metrics,
    )
    from pipeline.confidence import YT_ER_BENCHMARKS

    logger.info(f"Starting YT CIP build for {channel_url}")
    bd_client = BrightdataClient(api_token=brightdata_token)
    yt_api = YouTubeAPIClient(api_key=youtube_api_key)

    cip: dict = {
        "platform": "youtube",
        "profile_url": channel_url,
        "pipeline_version": PIPELINE_VERSION,
        "scraped_at": datetime.now().isoformat(),
        "data_provenance": "brightdata_scraper_api+youtube_data_api_v3",
    }

    try:
        # ── STEP 1: resolve URL / handle ──
        resolved = resolve_yt(channel_url, api=yt_api)
        canonical_url = resolved.url
        cip["resolved"] = {
            "channel_id": resolved.channel_id,
            "handle": resolved.handle,
            "url": canonical_url,
        }

        # ── STEP 2: channel scrape ──
        logger.info("  [Step 1/6] Scraping YouTube channel...")
        raw_channels = scrape_channels(bd_client, [canonical_url])
        if not raw_channels:
            return {**cip, "error": "no channel data from brightdata"}
        profile = extract_channel_metrics(raw_channels[0])
        # Prefer YT Data API stats for headline numbers when available —
        # Bright Data's scraped subscriber counts can drift by hours/days.
        if resolved.channel_id and yt_api.available:
            stats = yt_api.fetch_channel_stats([resolved.channel_id]).get(
                resolved.channel_id
            )
            if stats:
                profile["followers_or_subs"] = (
                    stats.get("subscriber_count") or profile["followers_or_subs"]
                )
                profile["posts_or_videos_count"] = (
                    stats.get("video_count") or profile["posts_or_videos_count"]
                )
                profile["topic_categories"] = stats.get("topic_categories") or []
        cip["profile"] = profile

        # ── STEP 3: video discovery (API-primary, BD fallback) ──
        logger.info("  [Step 2/6] Discovering recent videos...")
        raw_videos = scrape_videos_discovery(
            bd_client,
            canonical_url,
            num_videos,
            yt_api=yt_api,
            channel_id=resolved.channel_id,
        )
        top_for_transcripts = select_top_videos(raw_videos, num_transcripts)
        videos = [extract_video_metrics(v) for v in raw_videos]
        cip["videos"] = videos

        # Channel-level aggregate metrics for scoring
        agg = aggregate_channel_metrics(videos)
        subs = profile.get("followers_or_subs") or 0
        avg_views = agg.get("avg_view_count") or 0
        avg_views_per_sub = round(avg_views / max(subs, 1), 3)
        cip["channel_metrics"] = {
            **agg,
            "avg_views_per_sub": avg_views_per_sub,
        }

        # ── STEP 4: transcripts (tiered: youtube-transcript-api → BD → Whisper) ──
        from pipeline.youtube.transcripts import fetch_transcript

        logger.info("  [Step 3/6] Gathering transcripts...")
        transcripts: list[dict] = []
        for v in [extract_video_metrics(x) for x in top_for_transcripts]:
            vid = v.get("video_id")
            vurl = v.get("url")
            if not vid or not vurl:
                continue
            tr = fetch_transcript(
                video_id=vid,
                video_url=vurl,
                bd_client=bd_client,
                openai_key=openai_api_key,
            )
            if tr is None:
                continue
            transcripts.append(
                {
                    "video_id": tr["video_id"],
                    # `post_id` alias so analyze_transcripts (IG-shaped) works.
                    "post_id": tr["video_id"],
                    "transcript_text": tr["text"],
                    "caption_source": tr["source"],
                }
            )
        cip["transcripts"] = transcripts

        # ── STEP 5: comments (API-primary, BD fallback) + Gemini analysis ──
        logger.info("  [Step 4/6] Scraping comments...")
        comment_video_urls = select_top_videos_for_comments(videos, top_n=5)
        raw_comments = (
            scrape_yt_comments(bd_client, comment_video_urls, yt_api=yt_api)
            if comment_video_urls
            else []
        )
        comment_metrics = extract_comment_metrics(
            raw_comments,
            creator_channel_id=resolved.channel_id,
            creator_handle=profile.get("handle") or "",
        )
        cip["comments"] = comment_metrics

        logger.info("  [Step 5/6] Running Gemini analysis...")
        gemini_client = init_gemini(gemini_api_key)
        creator_handle = profile.get("handle") or ""
        try:
            cip["caption_intelligence"] = analyze_captions(
                gemini_client,
                handle=creator_handle,
                bio=profile.get("bio") or "",
                category=profile.get("category") or "",
                # YT "captions" = video descriptions (titles fall back when description empty).
                captions=[
                    v.get("description") or v.get("title") or ""
                    for v in videos
                ],
            )
        except Exception as e:  # noqa: BLE001
            cip["caption_intelligence"] = {"_llm_failure": True, "error": str(e)}
        try:
            cip["transcript_intelligence"] = analyze_transcripts(
                gemini_client, handle=creator_handle, transcripts=transcripts
            )
        except Exception as e:  # noqa: BLE001
            cip["transcript_intelligence"] = {"_llm_failure": True, "error": str(e)}
        try:
            cip["audience_intelligence"] = analyze_comments(
                gemini_client,
                handle=creator_handle,
                comment_texts=comment_metrics.get("_comment_texts", []),
                comment_timestamps=comment_metrics.get("_comment_timestamps", []),
                commenter_handles=comment_metrics.get("_commenter_handles", []),
                comment_hour_distribution=comment_metrics.get(
                    "comment_hour_distribution_utc", {}
                ),
                num_posts_with_comments=len(videos),
            )
        except Exception as e:  # noqa: BLE001
            cip["audience_intelligence"] = {"_llm_failure": True, "error": str(e)}

        # ── STEP 6: scoring ──
        logger.info("  [Step 6/6] Computing YT CPI...")
        llm_status = _summarize_llm_status(cip)
        # Adapt IG-shaped `posts` view so compute_creator_scores keeps working.
        # YT ER uses views as denominator — we precompute it here rather than
        # forking the full scorer.
        total_likes = sum(v.get("like_count") or 0 for v in videos)
        total_comments = sum(v.get("comment_count") or 0 for v in videos)
        total_views = sum(v.get("view_count") or 0 for v in videos)
        yt_er = (
            (total_likes + total_comments) / total_views
            if total_views > 0
            else 0.0
        )
        cip["posts"] = {
            "avg_engagement_rate": yt_er,
            "posts_per_week": _posts_per_week_from_cadence(
                agg.get("upload_cadence_days")
            ),
            "content_mix": agg.get("content_mix", {}),
            "num_posts": len(videos),
        }
        cip["scores"] = compute_creator_scores(
            cip, llm_status=llm_status, er_benchmarks=YT_ER_BENCHMARKS
        )
        # Attach YT-specific metrics so db.upsert_creator_score_platform
        # can route them onto the YT creator_scores columns.
        cip["scores"]["avg_views_per_sub"] = avg_views_per_sub
        cip["scores"]["watch_through_proxy"] = agg.get("watch_through_proxy")
        cip["scores"]["upload_cadence_days"] = agg.get("upload_cadence_days")

        return cip

    except Exception as e:  # noqa: BLE001
        logger.exception("YouTube CIP build failed")
        return {**cip, "error": str(e)}


def _posts_per_week_from_cadence(cadence_days: float | None) -> float | None:
    if not cadence_days or cadence_days <= 0:
        return None
    return round(7.0 / cadence_days, 2)


def build_youtube_creator_intelligence_profile_batch(
    channel_urls: list[str],
    brightdata_token: str,
    gemini_api_key: str,
    openai_api_key: str,
    youtube_api_key: str | None = None,
    num_videos: int = 20,
    num_transcripts: int = 5,
    max_workers: int = 8,
) -> list[dict]:
    """Batch YT CIP builder for N creators.

    Wins vs. looping the single-creator builder:
    1. One Bright Data channel trigger for all URLs (instead of N).
       Pricing is per-record returned, so the savings are wall-clock, not
       dollars — one HTTP round-trip and one snapshot poll instead of N.
    2. Per-creator stages 3–7 (video discovery, comments, transcripts,
       Gemini, scoring) run in a ThreadPoolExecutor; these are I/O-bound
       so threads are fine.

    Returns a list of CIPs in the SAME ORDER as input. Failed creators
    have `{'error': ...}` on their dict — same shape as the single-creator
    error path.
    """
    from concurrent.futures import ThreadPoolExecutor
    from pipeline.youtube.handle_resolver import resolve as resolve_yt
    from pipeline.youtube.youtube_api import YouTubeAPIClient
    from pipeline.youtube.scraper_channels import (
        scrape_channels,
        extract_channel_metrics,
    )

    if not channel_urls:
        return []

    logger.info(f"Batch YT CIP build: {len(channel_urls)} creators")
    bd_client = BrightdataClient(api_token=brightdata_token)
    yt_api = YouTubeAPIClient(api_key=youtube_api_key)

    # ── Stage 1: resolve URLs in parallel ──
    # Handle resolution hits the YT API (1 unit) per handle-form URL.
    # Channel-ID-form URLs short-circuit without an API call.
    resolved_by_url: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(resolve_yt, u, yt_api): u for u in channel_urls}
        for fut, u in futures.items():
            try:
                resolved_by_url[u] = fut.result()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"handle resolve failed for {u}: {e}")
                resolved_by_url[u] = None

    canonical_urls = [
        resolved_by_url[u].url if resolved_by_url.get(u) else u
        for u in channel_urls
    ]

    # ── Stage 2: ONE Bright Data channel trigger for all URLs ──
    # Up to ~500 URLs per trigger. Returns one record per URL (or an error
    # record for bad URLs — we tolerate partial failures).
    try:
        raw_channels = scrape_channels(bd_client, canonical_urls)
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Batch channel scrape failed: {e}")
        raw_channels = []

    # Index BD results by url so we can match them back to input order.
    raw_by_url: dict[str, dict] = {}
    for rec in raw_channels or []:
        u = rec.get("url") or rec.get("profile_url") or rec.get("channel_url")
        if u:
            raw_by_url[u] = rec
        elif rec.get("channel_id"):
            # Sometimes BD doesn't echo the input URL; match on channel_id
            for csp_url, res in resolved_by_url.items():
                if res and res.channel_id == rec.get("channel_id"):
                    raw_by_url[canonical_urls[channel_urls.index(csp_url)]] = rec
                    break

    # ── Stages 3–7: per-creator, parallel ──
    def _one(idx: int) -> dict:
        url = channel_urls[idx]
        canonical_url = canonical_urls[idx]
        resolved = resolved_by_url.get(url)
        raw_channel = raw_by_url.get(canonical_url)
        if raw_channel is None:
            # Couldn't scrape this creator's channel — still try the API
            # path so we at least get canonical stats.
            raw_channel = {}
        return _build_yt_cip_with_preloaded_channel(
            original_url=url,
            canonical_url=canonical_url,
            resolved=resolved,
            raw_channel=raw_channel,
            bd_client=bd_client,
            yt_api=yt_api,
            gemini_api_key=gemini_api_key,
            openai_api_key=openai_api_key,
            num_videos=num_videos,
            num_transcripts=num_transcripts,
        )

    results: list[dict | None] = [None] * len(channel_urls)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_one, i): i for i in range(len(channel_urls))}
        for fut, i in futures.items():
            try:
                results[i] = fut.result()
            except Exception as e:  # noqa: BLE001
                logger.exception(f"Batch CIP build failed for index {i}")
                results[i] = {
                    "platform": "youtube",
                    "profile_url": channel_urls[i],
                    "error": str(e),
                }

    return [r for r in results if r is not None]


def _build_yt_cip_with_preloaded_channel(
    *,
    original_url: str,
    canonical_url: str,
    resolved,
    raw_channel: dict,
    bd_client: BrightdataClient,
    yt_api,
    gemini_api_key: str,
    openai_api_key: str,
    num_videos: int,
    num_transcripts: int,
) -> dict:
    """Stages 3–7 of the YT CIP, given a pre-fetched channel record.

    Factored out of `build_youtube_creator_intelligence_profile` so the
    batch orchestrator can run it concurrently per creator after a single
    batched Bright Data channel scrape. Behavior must match the
    single-creator path exactly for a batch-of-1.
    """
    from pipeline.youtube.scraper_channels import extract_channel_metrics
    from pipeline.youtube.scraper_videos import (
        scrape_videos_discovery,
        select_top_videos,
        extract_video_metrics,
        aggregate_channel_metrics,
    )
    from pipeline.youtube.scraper_comments import (
        scrape_comments as scrape_yt_comments,
        select_top_videos_for_comments,
        extract_comment_metrics,
    )
    from pipeline.youtube.transcripts import fetch_transcript
    from pipeline.confidence import YT_ER_BENCHMARKS

    cip: dict = {
        "platform": "youtube",
        "profile_url": original_url,
        "pipeline_version": PIPELINE_VERSION,
        "scraped_at": datetime.now().isoformat(),
        "data_provenance": "brightdata_scraper_api+youtube_data_api_v3",
        "resolved": {
            "channel_id": getattr(resolved, "channel_id", None),
            "handle": getattr(resolved, "handle", None),
            "url": canonical_url,
        },
    }

    try:
        # Profile — use the pre-scraped channel record; enrich with API.
        profile = extract_channel_metrics(raw_channel or {})
        channel_id = getattr(resolved, "channel_id", None)
        if channel_id and yt_api.available:
            stats = yt_api.fetch_channel_stats([channel_id]).get(channel_id)
            if stats:
                profile["followers_or_subs"] = (
                    stats.get("subscriber_count")
                    or profile.get("followers_or_subs")
                    or 0
                )
                profile["posts_or_videos_count"] = (
                    stats.get("video_count")
                    or profile.get("posts_or_videos_count")
                    or 0
                )
                profile["topic_categories"] = (
                    stats.get("topic_categories") or []
                )
        cip["profile"] = profile

        # Videos
        raw_videos = scrape_videos_discovery(
            bd_client,
            canonical_url,
            num_videos,
            yt_api=yt_api,
            channel_id=channel_id,
        )
        top_for_transcripts = select_top_videos(raw_videos, num_transcripts)
        videos = [extract_video_metrics(v) for v in raw_videos]
        cip["videos"] = videos

        agg = aggregate_channel_metrics(videos)
        subs = profile.get("followers_or_subs") or 0
        avg_views = agg.get("avg_view_count") or 0
        avg_views_per_sub = round(avg_views / max(subs, 1), 3)
        cip["channel_metrics"] = {
            **agg,
            "avg_views_per_sub": avg_views_per_sub,
        }

        # Transcripts (tiered)
        transcripts: list[dict] = []
        for v in [extract_video_metrics(x) for x in top_for_transcripts]:
            vid, vurl = v.get("video_id"), v.get("url")
            if not vid or not vurl:
                continue
            tr = fetch_transcript(
                video_id=vid, video_url=vurl,
                bd_client=bd_client, openai_key=openai_api_key,
            )
            if tr is None:
                continue
            transcripts.append({
                "video_id": tr["video_id"],
                "transcript_text": tr["text"],
                "caption_source": tr["source"],
            })
        cip["transcripts"] = transcripts

        # Comments + Gemini
        comment_video_urls = select_top_videos_for_comments(videos, top_n=5)
        raw_comments = (
            scrape_yt_comments(bd_client, comment_video_urls, yt_api=yt_api)
            if comment_video_urls else []
        )
        comment_metrics = extract_comment_metrics(
            raw_comments,
            creator_channel_id=channel_id,
            creator_handle=profile.get("handle") or "",
        )
        cip["comments"] = comment_metrics

        gemini_client = init_gemini(gemini_api_key)
        creator_handle = profile.get("handle") or ""
        try:
            cip["caption_intelligence"] = analyze_captions(
                gemini_client,
                handle=creator_handle,
                bio=profile.get("bio") or "",
                category=profile.get("category") or "",
                captions=[
                    v.get("description") or v.get("title") or ""
                    for v in videos
                ],
            )
        except Exception as e:  # noqa: BLE001
            cip["caption_intelligence"] = {"_llm_failure": True, "error": str(e)}
        try:
            cip["transcript_intelligence"] = analyze_transcripts(
                gemini_client, handle=creator_handle, transcripts=transcripts
            )
        except Exception as e:  # noqa: BLE001
            cip["transcript_intelligence"] = {"_llm_failure": True, "error": str(e)}
        try:
            cip["audience_intelligence"] = analyze_comments(
                gemini_client,
                handle=creator_handle,
                comment_texts=comment_metrics.get("_comment_texts", []),
                comment_timestamps=comment_metrics.get("_comment_timestamps", []),
                commenter_handles=comment_metrics.get("_commenter_handles", []),
                comment_hour_distribution=comment_metrics.get(
                    "comment_hour_distribution_utc", {}
                ),
                num_posts_with_comments=len(videos),
            )
        except Exception as e:  # noqa: BLE001
            cip["audience_intelligence"] = {"_llm_failure": True, "error": str(e)}

        # Scoring
        llm_status = _summarize_llm_status(cip)
        total_likes = sum(v.get("like_count") or 0 for v in videos)
        total_comments = sum(v.get("comment_count") or 0 for v in videos)
        total_views = sum(v.get("view_count") or 0 for v in videos)
        yt_er = (
            (total_likes + total_comments) / total_views
            if total_views > 0 else 0.0
        )
        cip["posts"] = {
            "avg_engagement_rate": yt_er,
            "posts_per_week": _posts_per_week_from_cadence(
                agg.get("upload_cadence_days")
            ),
            "content_mix": agg.get("content_mix", {}),
            "num_posts": len(videos),
        }
        cip["scores"] = compute_creator_scores(
            cip, llm_status=llm_status, er_benchmarks=YT_ER_BENCHMARKS
        )
        cip["scores"]["avg_views_per_sub"] = avg_views_per_sub
        cip["scores"]["watch_through_proxy"] = agg.get("watch_through_proxy")
        cip["scores"]["upload_cadence_days"] = agg.get("upload_cadence_days")

        return cip

    except Exception as e:  # noqa: BLE001
        logger.exception("YT CIP build failed in worker")
        return {**cip, "error": str(e)}
