import copy
import logging
from datetime import datetime

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
    gemini_api_key: str,
    openai_api_key: str,
    num_posts: int = 5,
    num_reels: int = 10,
    num_comment_posts: int = 5,
    days_back: int = 90,
    er_benchmarks: dict | None = None,
    commerce_signal: dict | None = None,
) -> dict:
    """
    Full pipeline: Scrape -> Transcribe -> Analyse -> Compute -> Return CIP.

    Default scrape mix is reels-heavy (10 reels + 5 posts) since influencer
    evaluation hinges on reels. Comments are sourced from the top reels,
    not top posts.

    Apify cost per creator: ~$0.066
      - Profile: $0.002
      - Posts (5): $0.0135
      - Reels (10): $0.025
      - Comments (50): $0.025

    Whisper cost per creator (legacy pod path): ~$0.030. Replaced by Modal
    serverless GPU sidecar in Phase 3.

    Gemini cost per creator: ~$0.002 (3 sequential calls — merged to 1 in Phase 4).
    """
    logger.info(f"Starting CIP build for {profile_url}")

    gemini_client = init_gemini(gemini_api_key)

    # Hybrid IG-comment policy: if the creator has a commerce signal
    # (affiliate sales data), that's a stronger audience-purchasing-power
    # proxy than comment-derived authenticity/sentiment, so skip comment
    # scraping entirely. Otherwise fall back to a thin scrape (~10 comments
    # total via 1 per reel) — enough for language detection without paying
    # for the full 25-comment pull.
    if commerce_signal:
        comments_per_reel = 0
        num_comment_posts = 0
    else:
        comments_per_reel = 1
    logger.info(
        "  IG comment policy: commerce_signal=%s → comments_per_reel=%d, num_comment_posts=%d",
        "yes" if commerce_signal else "no",
        comments_per_reel,
        num_comment_posts,
    )

    # Apply the per-reel comment policy before the first bundle.fetch() call
    # so the cached run honors it.
    try:
        from pipeline import apify_instagram_bundle
        apify_instagram_bundle.set_default_comments_per_reel(comments_per_reel)
    except Exception as e:  # noqa: BLE001
        logger.debug("apify bundle setter unavailable: %s", e)

    cip = {
        "profile_url": profile_url,
        "pipeline_version": PIPELINE_VERSION,
        "scraped_at": datetime.now().isoformat(),
        "data_provenance": "apify_instagram_scraper",
        "commerce_signal": commerce_signal,  # passes through for downstream consumers
    }

    # ── STEP 1: Profile Scrape ──
    logger.info("  [Step 1/6] Scraping profile...")
    raw_profiles = scrape_profiles([profile_url])

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
        profile_url,
        num_posts=num_posts,
        days_back=days_back,
    )

    logger.info(
        f"  Posts Discovery returned {len(raw_posts)} posts."
    )

    cip["_raw_posts"] = raw_posts  # needed by store_full_cip for DB insert
    # NOTE: engagement metrics are computed below over posts + reels (reels
    # carry the dominant signal; excluding them gives a misleadingly low ER).

    # ── STEP 3a: Reels Discovery ──
    logger.info(f"  [Step 3a/6] Discovering {num_reels} reels from profile...")
    raw_reels = scrape_reels_discovery(profile_url, num_reels=num_reels)

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
        top_reels = []
        logger.info("  [Step 3a/6] No reels found, skipping...")
        cip["reels"] = {}
    # Raw reel records (same shape as posts: post_id/url/description/likes/
    # views) for per-video analysis + raw-content persistence. Reels carry the
    # dominant signal and are where comments live, so they MUST be analysed.
    cip["_raw_reels"] = top_reels

    # Engagement metrics over posts + reels combined. Reels are the dominant
    # content for most creators; computing ER from the handful of static posts
    # alone produced an unrealistically low headline rate (e.g. 0.07% when the
    # reels engage 3-5%).
    post_metrics = extract_post_metrics(
        raw_posts + top_reels, followers, handle=handle
    )
    cip["posts"] = post_metrics

    # ── STEP 3b: Comments ──
    # Rebased onto top reels (was: top posts) — reels carry the dominant
    # evaluation signal so comments on reels give us the audience signal
    # closest to the assets we score on. Falls back to top posts if the
    # reel scrape returned nothing.
    if top_reels:
        comment_urls = select_top_posts_for_comments(
            top_reels, top_n=num_comment_posts
        )
    else:
        comment_urls = select_top_posts_for_comments(
            raw_posts, top_n=num_comment_posts
        )

    if comment_urls:
        logger.info(
            f"  [Step 3b/6] Scraping comments from {len(comment_urls)} sources..."
        )
        raw_comments = scrape_comments(comment_urls)
        comment_metrics = extract_comment_metrics(raw_comments, handle)
        cip["comments"] = comment_metrics
    else:
        logger.info("  [Step 3b/6] No commented posts found, skipping...")
        cip["comments"] = {}

    # ── STEP 4: Whisper Transcription ──
    # In async mode we defer transcription to background jobs — the
    # critical-path scoring runs without transcripts and an
    # ``audience_refresh`` job re-scores once they land. The CIP carries
    # `_async_transcribe_pending=True` so the finalize step knows to
    # enqueue the fanout.
    video_urls_for_whisper = cip.get("reels", {}).get(
        "video_urls_for_whisper", []
    )

    from pipeline import whisper_client

    if not video_urls_for_whisper:
        logger.info("  [Step 4/6] No video URLs available, skipping Whisper...")
        cip["transcripts"] = []
    elif whisper_client.is_async_mode():
        logger.info(
            "  [Step 4/6] WHISPER_ASYNC=1: deferring %d transcripts to background jobs",
            len(video_urls_for_whisper),
        )
        cip["transcripts"] = []
        cip["_async_transcribe_pending"] = video_urls_for_whisper
    else:
        logger.info(
            f"  [Step 4/6] Transcribing {len(video_urls_for_whisper)} reels "
            "via Modal Whisper..."
        )
        transcripts = transcribe_reels_whisper(
            video_urls_for_whisper, openai_api_key
        )
        cip["transcripts"] = transcripts

    # ── STEP 5: LLM Analysis ──
    # Two modes:
    #   LLM_MERGED=1 → one Sonnet 4.6 call, split into the three keys
    #   default     → legacy three sequential Gemini calls
    from pipeline import llm as merged_llm

    captions = post_metrics.get("_captions", [])
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

    comments_dict = cip.get("comments") or {}
    comment_texts = comments_dict.get("_comment_texts") or []

    if merged_llm.is_merged_mode():
        logger.info("  [Step 5/6] Running merged LLM (Claude Sonnet 4.6)...")
        merged_input_comments: list[dict] = []
        commenter_handles = comments_dict.get("_commenter_handles") or []
        comment_timestamps = comments_dict.get("_comment_timestamps") or []
        for i, txt in enumerate(comment_texts[:50]):
            merged_input_comments.append(
                {
                    "user": commenter_handles[i] if i < len(commenter_handles) else None,
                    "text": txt,
                    "timestamp": comment_timestamps[i] if i < len(comment_timestamps) else None,
                }
            )
        merged = merged_llm.evaluate_creator(
            handle=handle,
            bio=profile_metrics.get("bio") or "",
            category=profile_metrics.get("category") or "",
            captions=captions,
            transcripts=valid_transcripts,
            comments=merged_input_comments or None,
            comment_hour_distribution=comments_dict.get(
                "comment_hour_distribution_utc"
            ),
        )
        dims = merged_llm.split_into_dimensions(merged)
        for key, payload in dims.items():
            cip[key] = payload
    else:
        logger.info("  [Step 5/6] Running Gemini analysis (legacy three-call)...")
        if captions:
            cip["caption_intelligence"] = analyze_captions(
                gemini_client,
                handle=handle,
                bio=profile_metrics.get("bio", ""),
                category=profile_metrics.get("category", ""),
                captions=captions,
            )
        if valid_transcripts:
            cip["transcript_intelligence"] = analyze_transcripts(
                gemini_client,
                handle=handle,
                transcripts=valid_transcripts,
            )
        if comment_texts:
            cip["audience_intelligence"] = analyze_comments(
                gemini_client,
                handle=handle,
                comment_texts=comment_texts,
                comment_timestamps=comments_dict.get("_comment_timestamps"),
                commenter_handles=comments_dict.get("_commenter_handles"),
                comment_hour_distribution=comments_dict.get(
                    "comment_hour_distribution_utc"
                ),
                num_posts_with_comments=len(comment_texts),
            )

    # ── STEP 5b: Per-video analysis (LLM_PER_POST, runs ALONGSIDE the
    # creator-level path above; stashes payloads on the CIP for db.store) ──
    from pipeline import llm_post

    if llm_post.is_per_post_mode():
        logger.info("  [Step 5b] Running per-video analysis (OpenRouter)...")
        payloads, item_meta = llm_post.run_per_post_analysis(
            handle,
            "instagram",
            posts=(cip.get("_raw_posts") or []) + (cip.get("_raw_reels") or []),
            transcripts=cip.get("transcripts", []),
            comments_by_post=(cip.get("comments") or {}).get("_comments_by_post"),
        )
        cip["_post_payloads"] = payloads
        cip["_post_item_meta"] = item_meta
        logger.info(f"  [Step 5b] Analysed {len(payloads)} posts per-video")

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
        t["avg_confidence"]
        for t in transcripts
        if t.get("transcript_text")
        and not t.get("is_likely_music", False)
        and isinstance(t.get("avg_confidence"), (int, float))
    ]
    # Some transcript sources (e.g. OpenAI whisper-1) don't return a
    # confidence score — fall back to "casual" rather than crashing.
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
    gemini_api_key: str,
    openai_api_key: str,
    youtube_api_key: str | None = None,
    num_videos: int = 20,
    num_transcripts: int = 5,
) -> dict:
    """YouTube CIP — parallel to build_creator_intelligence_profile.

    Stages:
      1. Resolve the URL -> (channel_id, canonical url) via handle_resolver.
      2. YouTube Data API channel fetch: stats, description, links.
      3. YouTube Data API video discovery: top N recent videos.
      4. Transcription: tier-1 youtube-transcript-api, tier-2 Whisper.
      5. Gemini analysis: captions (description + title + tags),
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
    yt_api = YouTubeAPIClient(api_key=youtube_api_key)

    cip: dict = {
        "platform": "youtube",
        "profile_url": channel_url,
        "pipeline_version": PIPELINE_VERSION,
        "scraped_at": datetime.now().isoformat(),
        "data_provenance": "youtube_data_api_v3",
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
        if not resolved.channel_id:
            return {**cip, "error": "could not resolve channel_id"}

        # ── STEP 2: channel scrape via YT API ──
        logger.info("  [Step 1/6] Fetching YouTube channel via API...")
        raw_channels = scrape_channels(
            [canonical_url],
            yt_api=yt_api,
            channel_ids=[resolved.channel_id],
        )
        if not raw_channels:
            return {**cip, "error": "no channel data from youtube api"}
        profile = extract_channel_metrics(raw_channels[0])
        cip["profile"] = profile

        # ── STEP 3: video discovery via YT API ──
        logger.info("  [Step 2/6] Discovering recent videos...")
        raw_videos = scrape_videos_discovery(
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

        # ── STEP 4: transcripts (tier-1 inline; Modal tier-2 sync or async) ──
        from pipeline.youtube.transcripts import _try_transcript_api, fetch_transcript
        from pipeline import whisper_client

        logger.info("  [Step 3/6] Gathering transcripts...")
        async_mode = whisper_client.is_async_mode()
        transcripts: list[dict] = []
        async_pending: list[dict] = []
        for v in [extract_video_metrics(x) for x in top_for_transcripts]:
            vid = v.get("video_id")
            vurl = v.get("url")
            if not vid or not vurl:
                continue
            t1 = _try_transcript_api(vid)
            if t1 is not None:
                transcripts.append(
                    {
                        "video_id": t1["video_id"],
                        "post_id": t1["video_id"],
                        "transcript_text": t1["text"],
                        "caption_source": t1["source"],
                    }
                )
                continue
            if async_mode:
                async_pending.append(
                    {
                        "video_url": vurl,
                        "video_id": vid,
                        "caption": v.get("title") or v.get("description") or "",
                        "duration_seconds": v.get("duration_seconds"),
                    }
                )
                continue
            # Sync fallback: Modal Whisper via the tiered fetcher.
            tr = fetch_transcript(
                video_id=vid,
                video_url=vurl,
                openai_key=openai_api_key,
                duration_seconds=v.get("duration_seconds"),
            )
            if tr is None:
                continue
            transcripts.append(
                {
                    "video_id": tr["video_id"],
                    "post_id": tr["video_id"],
                    "transcript_text": tr["text"],
                    "caption_source": tr["source"],
                }
            )
        if async_pending:
            logger.info(
                "  [Step 3/6] WHISPER_ASYNC=1: deferring %d transcripts to background jobs",
                len(async_pending),
            )
            cip["_async_transcribe_pending"] = async_pending
        cip["transcripts"] = transcripts

        # ── STEP 5: comments via YT API + Gemini analysis ──
        logger.info("  [Step 4/6] Scraping comments...")
        comment_video_urls = select_top_videos_for_comments(videos, top_n=5)
        raw_comments = (
            scrape_yt_comments(comment_video_urls, yt_api=yt_api)
            if comment_video_urls
            else []
        )
        comment_metrics = extract_comment_metrics(
            raw_comments,
            creator_channel_id=resolved.channel_id,
            creator_handle=profile.get("handle") or "",
        )
        cip["comments"] = comment_metrics

        from pipeline import llm as merged_llm

        creator_handle = profile.get("handle") or ""
        yt_captions = [
            v.get("description") or v.get("title") or "" for v in videos
        ]
        yt_comment_texts = comment_metrics.get("_comment_texts", []) or []
        yt_commenter_handles = comment_metrics.get("_commenter_handles", []) or []
        yt_comment_timestamps = comment_metrics.get("_comment_timestamps", []) or []

        if merged_llm.is_merged_mode():
            logger.info("  [Step 5/6] Running merged LLM (Claude Sonnet 4.6)...")
            merged_input_comments: list[dict] = []
            for i, txt in enumerate(yt_comment_texts[:50]):
                merged_input_comments.append(
                    {
                        "user": yt_commenter_handles[i]
                        if i < len(yt_commenter_handles)
                        else None,
                        "text": txt,
                        "timestamp": yt_comment_timestamps[i]
                        if i < len(yt_comment_timestamps)
                        else None,
                    }
                )
            merged = merged_llm.evaluate_creator(
                handle=creator_handle,
                bio=profile.get("bio") or "",
                category=profile.get("category") or "",
                captions=yt_captions,
                transcripts=transcripts,
                comments=merged_input_comments or None,
                comment_hour_distribution=comment_metrics.get(
                    "comment_hour_distribution_utc", {}
                ),
            )
            for key, payload in merged_llm.split_into_dimensions(merged).items():
                cip[key] = payload
        else:
            logger.info("  [Step 5/6] Running Gemini analysis (legacy three-call)...")
            gemini_client = init_gemini(gemini_api_key)
            try:
                cip["caption_intelligence"] = analyze_captions(
                    gemini_client,
                    handle=creator_handle,
                    bio=profile.get("bio") or "",
                    category=profile.get("category") or "",
                    captions=yt_captions,
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
                    comment_texts=yt_comment_texts,
                    comment_timestamps=yt_comment_timestamps,
                    commenter_handles=yt_commenter_handles,
                    comment_hour_distribution=comment_metrics.get(
                        "comment_hour_distribution_utc", {}
                    ),
                    num_posts_with_comments=len(videos),
                )
            except Exception as e:  # noqa: BLE001
                cip["audience_intelligence"] = {"_llm_failure": True, "error": str(e)}

        # ── STEP 5b: Per-video analysis (LLM_PER_POST) ──
        from pipeline import llm_post

        if llm_post.is_per_post_mode():
            logger.info("  [Step 5b] Running per-video analysis (OpenRouter)...")
            payloads, item_meta = llm_post.run_per_post_analysis(
                creator_handle,
                "youtube",
                posts=videos,
                transcripts=transcripts,
                comments_by_post=comment_metrics.get("_comments_by_post"),
            )
            cip["_post_payloads"] = payloads
            cip["_post_item_meta"] = item_meta
            logger.info(f"  [Step 5b] Analysed {len(payloads)} videos per-video")

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
    gemini_api_key: str,
    openai_api_key: str,
    youtube_api_key: str | None = None,
    num_videos: int = 20,
    num_transcripts: int = 5,
    max_workers: int = 8,
    *,
    on_complete=None,
) -> list[dict]:
    """Batch YT CIP builder for N creators.

    Per-creator stages run in a ThreadPoolExecutor — these are I/O-bound
    against the YouTube Data API so threads are fine.

    Returns a list of CIPs in the SAME ORDER as input. Failed creators
    have `{'error': ...}` on their dict — same shape as the single-creator
    error path.
    """
    from concurrent.futures import ThreadPoolExecutor
    from pipeline.youtube.handle_resolver import resolve as resolve_yt
    from pipeline.youtube.api_pool import YouTubeAPIPool
    from pipeline.youtube.scraper_channels import scrape_channels

    if not channel_urls:
        return []

    logger.info(f"Batch YT CIP build: {len(channel_urls)} creators")

    # Multi-key API pool for stats, video discovery, comments. Required —
    # the BrightData fallback was removed in the pipeline rewrite.
    yt_pool = YouTubeAPIPool()
    if not yt_pool.available:
        raise RuntimeError(
            "Batch YT pipeline requires at least one YT API key — set "
            "YOUTUBE_API_KEYS (comma-separated) or YOUTUBE_API_KEY."
        )
    yt_api = yt_pool  # downstream methods are signature-compatible

    # ── Stage 1: resolve URLs in parallel ──
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

    # ── Stage 2: channel records via YT API (batched 50/call inside the pool) ──
    try:
        channel_ids = [
            (resolved_by_url.get(u).channel_id if resolved_by_url.get(u) else None)
            for u in channel_urls
        ]
        # Drop unresolved URLs from the API fetch; we still pass the
        # original URL list to scrape_channels so output order matches input.
        resolvable = [
            (url, cid)
            for url, cid in zip(channel_urls, channel_ids)
            if cid
        ]
        if resolvable:
            urls_for_api, ids_for_api = zip(*resolvable)
            raw_channels = scrape_channels(
                list(urls_for_api), yt_api=yt_pool, channel_ids=list(ids_for_api)
            )
        else:
            raw_channels = []
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
            yt_api=yt_api,
            gemini_api_key=gemini_api_key,
            openai_api_key=openai_api_key,
            num_videos=num_videos,
            num_transcripts=num_transcripts,
        )

    # Stream results as workers finish (instead of blocking on futures[0]).
    # `on_complete(cip)` fires immediately after each channel completes so
    # callers can persist per-channel — important so a 6-hour batch isn't
    # invisible until the slowest channel finishes.
    from concurrent.futures import as_completed

    completed = 0
    total = len(channel_urls)
    results: list[dict | None] = [None] * total
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_one, i): i for i in range(total)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                cip = fut.result()
            except Exception as e:  # noqa: BLE001
                logger.exception(f"Batch CIP build failed for index {i}")
                cip = {
                    "platform": "youtube",
                    "profile_url": channel_urls[i],
                    "error": str(e),
                }
            results[i] = cip
            completed += 1
            if on_complete is not None:
                try:
                    on_complete(cip)
                except Exception as cb_err:  # noqa: BLE001
                    logger.error(f"on_complete callback failed: {cb_err}")
            if completed % 10 == 0 or completed == total:
                logger.info(
                    f"  YT batch progress: {completed}/{total} channels done"
                )

    return [r for r in results if r is not None]


def _build_yt_cip_with_preloaded_channel(
    *,
    original_url: str,
    canonical_url: str,
    resolved,
    raw_channel: dict,
    yt_api,
    gemini_api_key: str,
    openai_api_key: str,
    num_videos: int,
    num_transcripts: int,
) -> dict:
    """Stages 3–7 of the YT CIP, given a pre-fetched channel record.

    Factored out of `build_youtube_creator_intelligence_profile` so the
    batch orchestrator can run it concurrently per creator after a single
    batched channel fetch.
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

    record_provenance = (raw_channel or {}).get(
        "_data_provenance", "youtube_data_api_v3"
    )
    cip: dict = {
        "platform": "youtube",
        "profile_url": original_url,
        "pipeline_version": PIPELINE_VERSION,
        "scraped_at": datetime.now().isoformat(),
        "data_provenance": record_provenance,
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
                # Backfill handle from the API-returned `customUrl` (e.g.
                # "@mkbhd"). Channel-id-form input URLs short-circuit the
                # handle resolver, so without this the profile.handle stays
                # null and the JSON output collides on `None.json`.
                if not profile.get("handle"):
                    custom_url = (stats.get("custom_url") or "").lstrip("@")
                    if custom_url:
                        profile["handle"] = custom_url
                if not profile.get("display_name") and stats.get("title"):
                    profile["display_name"] = stats["title"]
                if not profile.get("bio") and stats.get("description"):
                    profile["bio"] = stats["description"]
                if not profile.get("country") and stats.get("country"):
                    profile["country"] = stats["country"]
                if not profile.get("channel_created_at") and stats.get("published_at"):
                    profile["channel_created_at"] = stats["published_at"]
        cip["profile"] = profile

        # Videos
        raw_videos = scrape_videos_discovery(
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
                video_id=vid,
                video_url=vurl,
                openai_key=openai_api_key,
                duration_seconds=v.get("duration_seconds"),
            )
            if tr is None:
                continue
            transcripts.append({
                "video_id": tr["video_id"],
                # `post_id` alias so analyze_transcripts (IG-shaped) works.
                "post_id": tr["video_id"],
                "transcript_text": tr["text"],
                "caption_source": tr["source"],
            })
        cip["transcripts"] = transcripts

        # Comments + Gemini
        comment_video_urls = select_top_videos_for_comments(videos, top_n=5)
        raw_comments = (
            scrape_yt_comments(comment_video_urls, yt_api=yt_api)
            if comment_video_urls else []
        )
        comment_metrics = extract_comment_metrics(
            raw_comments,
            creator_channel_id=channel_id,
            creator_handle=profile.get("handle") or "",
        )
        cip["comments"] = comment_metrics

        from pipeline import llm as merged_llm

        creator_handle = profile.get("handle") or ""
        yt_captions = [
            v.get("description") or v.get("title") or "" for v in videos
        ]
        yt_comment_texts = comment_metrics.get("_comment_texts", []) or []
        yt_commenter_handles = comment_metrics.get("_commenter_handles", []) or []
        yt_comment_timestamps = comment_metrics.get("_comment_timestamps", []) or []

        if merged_llm.is_merged_mode():
            merged_input_comments: list[dict] = []
            for i, txt in enumerate(yt_comment_texts[:50]):
                merged_input_comments.append(
                    {
                        "user": yt_commenter_handles[i]
                        if i < len(yt_commenter_handles)
                        else None,
                        "text": txt,
                        "timestamp": yt_comment_timestamps[i]
                        if i < len(yt_comment_timestamps)
                        else None,
                    }
                )
            merged = merged_llm.evaluate_creator(
                handle=creator_handle,
                bio=profile.get("bio") or "",
                category=profile.get("category") or "",
                captions=yt_captions,
                transcripts=transcripts,
                comments=merged_input_comments or None,
                comment_hour_distribution=comment_metrics.get(
                    "comment_hour_distribution_utc", {}
                ),
            )
            for key, payload in merged_llm.split_into_dimensions(merged).items():
                cip[key] = payload
        else:
            gemini_client = init_gemini(gemini_api_key)
            try:
                cip["caption_intelligence"] = analyze_captions(
                    gemini_client,
                    handle=creator_handle,
                    bio=profile.get("bio") or "",
                    category=profile.get("category") or "",
                    captions=yt_captions,
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
                    comment_texts=yt_comment_texts,
                    comment_timestamps=yt_comment_timestamps,
                    commenter_handles=yt_commenter_handles,
                    comment_hour_distribution=comment_metrics.get(
                        "comment_hour_distribution_utc", {}
                    ),
                    num_posts_with_comments=len(videos),
                )
            except Exception as e:  # noqa: BLE001
                cip["audience_intelligence"] = {"_llm_failure": True, "error": str(e)}

        # ── Per-video analysis (LLM_PER_POST) ──
        from pipeline import llm_post

        if llm_post.is_per_post_mode():
            payloads, item_meta = llm_post.run_per_post_analysis(
                creator_handle,
                "youtube",
                posts=videos,
                transcripts=transcripts,
                comments_by_post=comment_metrics.get("_comments_by_post"),
            )
            cip["_post_payloads"] = payloads
            cip["_post_item_meta"] = item_meta

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
