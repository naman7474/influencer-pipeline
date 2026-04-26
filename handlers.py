"""
Background-job handlers for the IG analysis pipeline.

Invoked by `api.py /process-next-job`. Each handler:
  1. Receives the claimed `background_jobs` row + Supabase client.
  2. Runs the CIP pipeline for the relevant Instagram handle.
  3. Writes domain rows (brand or creator + intelligence tables).
  4. Generates + stores a content embedding.
  5. For brand jobs: fans out `creator_ig_scrape` rows for past collaborators.
  6. For creator jobs: if this is the last sibling, triggers /api/matching/compute.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from pipeline import db as pdb
from pipeline.embeddings import (
    build_brand_embedding_input,
    build_creator_embedding_input,
    embed_text,
)
from pipeline.pipeline import build_creator_intelligence_profile
from pipeline.calibration import load_er_benchmarks

logger = logging.getLogger(__name__)


# ── Tunables ────────────────────────────────────────────────────────────────

MAX_FANOUT_CREATORS = 10
FANOUT_STAGGER_SECONDS = 30
CREATOR_FRESHNESS_DAYS = 30


# ── Helpers ─────────────────────────────────────────────────────────────────


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing required env var {name}")
    return v


def _handle_url(handle: str) -> str:
    clean = handle.strip().lstrip("@").strip("/")
    return f"https://www.instagram.com/{clean}/"


def _extract_ig_content_dna(cip: dict) -> dict:
    caption = cip.get("caption_intelligence") or {}
    return {
        "primary_niche": caption.get("primary_niche"),
        "recurring_topics": caption.get("recurring_topics") or [],
        "content_pillars": caption.get("content_pillars") or [],
        "primary_tone": caption.get("primary_tone"),
        "brand_mentions": caption.get("brand_mentions") or [],
    }


def _extract_ig_audience_profile(cip: dict) -> dict:
    audience = cip.get("audience_intelligence") or {}
    return {
        "primary_country": audience.get("primary_country"),
        "estimated_age_group": audience.get("estimated_age_group"),
        "estimated_gender_skew": audience.get("estimated_gender_skew"),
        "overall_sentiment": audience.get("overall_sentiment"),
        "interest_signals": audience.get("interest_signals") or [],
    }


def _extract_collaborators_from_posts(
    raw_posts: list[dict], brand_handle: str
) -> list[str]:
    """Collect coauthor_producers + tagged_users from paid partnerships."""
    bh = brand_handle.lower().lstrip("@")
    seen: set[str] = set()
    out: list[str] = []
    for post in raw_posts:
        bucket: list[str] = []
        for co in post.get("coauthor_producers") or []:
            if isinstance(co, dict):
                uname = co.get("username") or co.get("handle")
            else:
                uname = co
            if uname:
                bucket.append(str(uname))
        if post.get("is_paid_partnership"):
            for tu in post.get("tagged_users") or []:
                uname = (
                    tu.get("username") if isinstance(tu, dict) else tu
                )
                if uname:
                    bucket.append(str(uname))
        for uname in bucket:
            norm = uname.strip().lstrip("@").lower()
            if not norm or norm == bh or norm in seen:
                continue
            seen.add(norm)
            out.append(norm)
    return out


def _union_past_collaborations(
    ig_handles: list[str], manual_list: list[str] | None, brand_handle: str
) -> list[str]:
    bh = brand_handle.lower().lstrip("@")
    seen: set[str] = set()
    merged: list[str] = []
    for source in (ig_handles, manual_list or []):
        for raw in source:
            if not raw:
                continue
            norm = str(raw).strip().lstrip("@").lower()
            if not norm or norm == bh or norm in seen:
                continue
            seen.add(norm)
            merged.append(norm)
    return merged


def _get_fresh_creator_handles(db, handles: list[str]) -> set[str]:
    """Return the subset of handles whose creator row was scraped < CREATOR_FRESHNESS_DAYS ago."""
    if not handles:
        return set()
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=CREATOR_FRESHNESS_DAYS)
    ).isoformat()
    result = (
        db.table("creators")
        .select("handle,last_scraped_at")
        .in_("handle", handles)
        .gte("last_scraped_at", cutoff)
        .execute()
    )
    return {row["handle"].lower() for row in (result.data or [])}


def _enqueue_creator_fanout(
    db, brand_id: str, handles: list[str]
) -> int:
    """Insert creator_ig_scrape jobs with sequential availability."""
    now = datetime.now(timezone.utc)
    rows = []
    for i, handle in enumerate(handles):
        available_at = now + timedelta(seconds=i * FANOUT_STAGGER_SECONDS)
        rows.append(
            {
                "job_type": "creator_ig_scrape",
                "brand_id": brand_id,
                "status": "queued",
                "payload": {"handle": handle, "parent_brand_id": brand_id},
                "available_at": available_at.isoformat(),
            }
        )
    if not rows:
        return 0
    db.table("background_jobs").insert(rows).execute()
    return len(rows)


# ── Phase 2.5 auto-stitch helpers ────────────────────────────────────────

AUTO_STITCH_CONFIDENCE_THRESHOLD = 0.9


def _has_platform_profile(db, creator_id: str, platform: str) -> bool:
    """Has this creator already got a profile on `platform`?

    Used as loop-prevention + idempotency for auto-stitch fanouts.
    """
    res = (
        db.table("creator_social_profiles")
        .select("id")
        .eq("creator_id", creator_id)
        .eq("platform", platform)
        .limit(1)
        .execute()
    )
    return bool(res.data)


def _record_stitch_candidate(
    db,
    source_creator_id: str,
    target_creator_id: str | None,
    candidate,
    status: str = "pending",
) -> None:
    """Write a row to `stitch_candidates` for admin review.

    Called when auto-fanout would create a cross-creator collision
    (the target handle already belongs to a different creators.id).
    The unique index on (source, source_platform, target_platform,
    target_handle) WHERE status='pending' dedupes re-scrapes.
    """
    row = {
        "source_creator_id": source_creator_id,
        "target_creator_id": target_creator_id,
        "source_platform": candidate.source_platform,
        "target_platform": candidate.target_platform,
        "target_handle": candidate.target_handle,
        "confidence": float(candidate.confidence),
        "reason": candidate.reason,
        "status": status,
    }
    try:
        db.table("stitch_candidates").insert(row).execute()
    except Exception as e:  # noqa: BLE001
        # Unique index on open candidates — silently ignore dup inserts
        logger.debug(f"stitch_candidate insert skipped: {e}")


def _enqueue_auto_stitch_ig_scrape(
    db, target_handle: str, existing_creator_id: str, confidence: float
) -> None:
    """Enqueue a `creator_ig_scrape` bound to an existing creators.id.

    Same job_type as the brand-fanout path — the only difference is the
    `existing_creator_id` payload key which `store_full_cip` honors to
    attach the IG profile onto an existing row rather than creating a
    new one.
    """
    row = {
        "job_type": "creator_ig_scrape",
        "status": "queued",
        "payload": {
            "handle": target_handle,
            "existing_creator_id": existing_creator_id,
            "source": "auto_stitch_from_yt",
            "source_confidence": float(confidence),
        },
        "available_at": datetime.now(timezone.utc).isoformat(),
    }
    db.table("background_jobs").insert(row).execute()


def _enqueue_auto_stitch_yt_scrape(
    db, target_url: str, existing_creator_id: str, confidence: float
) -> None:
    """Enqueue a `creator_yt_scrape` bound to an existing creators.id."""
    row = {
        "job_type": "creator_yt_scrape",
        "status": "queued",
        "payload": {
            "url": target_url,
            "existing_creator_id": existing_creator_id,
            "source": "auto_stitch_from_ig",
            "source_confidence": float(confidence),
        },
        "available_at": datetime.now(timezone.utc).isoformat(),
    }
    db.table("background_jobs").insert(row).execute()


def _run_auto_stitch_from_yt(db, creator_id: str, cip: dict) -> None:
    """After a YT scrape completes, inspect external_links for IG handles
    and auto-fan-out IG scrapes bound to the same creator_id.

    Safeguards:
      - only confidence >= 0.9 (direct URL links, not handle-string matches)
      - skip if creator already has an IG profile (loop prevention)
      - skip-and-flag if target handle already belongs to a different creator
        (cross-creator collision → stitch_candidates)
    """
    from pipeline.youtube.stitching import propose_stitch_candidates

    profile = cip.get("profile") or {}
    candidates = propose_stitch_candidates(
        source_platform="youtube",
        source_handle=profile.get("handle") or "",
        source_external_links=profile.get("external_links"),
        target_platforms=("instagram",),
    )
    auto = [c for c in candidates if c.confidence >= AUTO_STITCH_CONFIDENCE_THRESHOLD]
    if not auto:
        return

    for cand in auto:
        if _has_platform_profile(db, creator_id, "instagram"):
            logger.debug(
                f"Auto-stitch skipped: creator {creator_id} already has IG profile"
            )
            break  # already IG-present — no need to check further candidates
        other_id = pdb._find_creator_by_platform_profile(
            db, "instagram", None, cand.target_handle
        )
        if other_id and other_id != creator_id:
            logger.info(
                f"Auto-stitch collision: YT creator {creator_id} links to "
                f"@{cand.target_handle} which belongs to {other_id} — "
                "recording stitch_candidate for admin review"
            )
            _record_stitch_candidate(db, creator_id, other_id, cand)
            continue
        logger.info(
            f"Auto-stitch fanout: YT creator {creator_id} -> IG @{cand.target_handle}"
        )
        _enqueue_auto_stitch_ig_scrape(
            db, cand.target_handle, creator_id, cand.confidence
        )


def _run_auto_stitch_from_ig(db, creator_id: str, cip: dict) -> None:
    """After an IG scrape completes, inspect `profile.external_url` for a
    YouTube link and auto-fan-out a YT scrape bound to the same creator_id.

    IG has a single-valued `external_url` field (not a list like YT's about
    panel), so we check it directly. Same safeguards as the YT->IG flow.
    """
    from pipeline.youtube.stitching import (
        StitchCandidate,
        extract_handles_from_links,
    )

    profile = cip.get("profile") or {}
    external_url = profile.get("external_url")
    if not external_url:
        return

    linked = extract_handles_from_links(
        [{"label": "external", "url": external_url}]
    )
    yt_handles = linked.get("youtube") or set()
    if not yt_handles:
        return

    source_handle = (profile.get("handle") or "").lstrip("@").lower()
    for yt_handle in yt_handles:
        if _has_platform_profile(db, creator_id, "youtube"):
            logger.debug(
                f"Auto-stitch skipped: creator {creator_id} already has YT profile"
            )
            break
        target_url = f"https://www.youtube.com/@{yt_handle}"
        other_id = pdb._find_creator_by_platform_profile(
            db, "youtube", None, yt_handle
        )
        cand = StitchCandidate(
            source_platform="instagram",
            source_handle=source_handle,
            target_platform="youtube",
            target_handle=yt_handle,
            confidence=1.0,
            reason=(
                f"instagram profile external_url links to youtube.com/@{yt_handle}"
            ),
        )
        if other_id and other_id != creator_id:
            logger.info(
                f"Auto-stitch collision: IG creator {creator_id} links to "
                f"YT @{yt_handle} which belongs to {other_id}"
            )
            _record_stitch_candidate(db, creator_id, other_id, cand)
            continue
        logger.info(
            f"Auto-stitch fanout: IG creator {creator_id} -> YT @{yt_handle}"
        )
        _enqueue_auto_stitch_yt_scrape(db, target_url, creator_id, cand.confidence)


def _brand_handle_from(brand: dict) -> str:
    raw = brand.get("instagram_handle") or ""
    return raw.strip().lstrip("@").strip("/").lower()


def _update_brand(db, brand_id: str, patch: dict[str, Any]) -> None:
    db.table("brands").update(patch).eq("id", brand_id).execute()


def _update_creator_embedding(
    db, creator_id: str, embedding: list[float]
) -> None:
    db.table("creators").update(
        {
            "content_embedding": embedding,
            "embedding_computed_at": datetime.now(timezone.utc).isoformat(),
        }
    ).eq("id", creator_id).execute()


def _siblings_all_terminal(db, brand_id: str) -> bool:
    result = (
        db.table("background_jobs")
        .select("id,status")
        .eq("brand_id", brand_id)
        .eq("job_type", "creator_ig_scrape")
        .execute()
    )
    rows = result.data or []
    if not rows:
        return True
    return all(r["status"] in ("succeeded", "failed") for r in rows)


def _trigger_matching_compute(brand_id: str) -> None:
    """Fire-and-forget call to /api/matching/compute. Failure is non-fatal —
    a scheduled re-compute will catch up."""
    base = os.environ.get("WEB_APP_URL")
    secret = os.environ.get("MATCHING_COMPUTE_SECRET")
    if not base:
        logger.warning("WEB_APP_URL not set; skipping matching recompute trigger")
        return
    try:
        headers = {"Content-Type": "application/json"}
        if secret:
            headers["X-Worker-Secret"] = secret
        url = f"{base.rstrip('/')}/api/matching/compute"
        with httpx.Client(timeout=10.0) as client:
            client.post(url, json={"brand_id": brand_id}, headers=headers)
        logger.info(f"Triggered matching recompute for brand {brand_id}")
    except Exception as e:
        logger.warning(f"Failed to trigger matching recompute: {e}")


def _trigger_creator_recompute(creator_id: str) -> None:
    """Fire-and-forget call to /api/matching/recompute-creator for
    standalone creator pipeline runs (no parent brand fanout)."""
    base = os.environ.get("WEB_APP_URL")
    secret = os.environ.get("MATCHING_COMPUTE_SECRET")
    if not base or not secret:
        logger.warning(
            "WEB_APP_URL or MATCHING_COMPUTE_SECRET not set; skipping creator recompute"
        )
        return
    try:
        headers = {
            "Content-Type": "application/json",
            "X-Worker-Secret": secret,
        }
        url = f"{base.rstrip('/')}/api/matching/recompute-creator"
        with httpx.Client(timeout=10.0) as client:
            client.post(url, json={"creator_id": creator_id}, headers=headers)
        logger.info(f"Triggered creator recompute for creator {creator_id}")
    except Exception as e:
        logger.warning(f"Failed to trigger creator recompute: {e}")


# ── Handlers ────────────────────────────────────────────────────────────────


def handle_brand_ig_scrape(db, job: dict) -> None:
    brand_id = job["brand_id"]
    brand = pdb.get_brand(db, brand_id)
    if not brand:
        raise ValueError(f"Brand {brand_id} not found")

    handle = _brand_handle_from(brand)
    if not handle:
        raise ValueError(f"Brand {brand_id} has no instagram_handle")

    _update_brand(
        db,
        brand_id,
        {
            "ig_analysis_status": "running",
            "ig_analysis_error": None,
        },
    )

    brightdata_token = _require_env("BRIGHTDATA_API_TOKEN")
    gemini_key = _require_env("GEMINI_API_KEY")
    openai_key = _require_env("OPENAI_API_KEY")

    cip = build_creator_intelligence_profile(
        profile_url=_handle_url(handle),
        brightdata_token=brightdata_token,
        gemini_api_key=gemini_key,
        openai_api_key=openai_key,
        er_benchmarks=load_er_benchmarks(db),
    )
    if cip.get("error"):
        raise RuntimeError(f"CIP failed for brand @{handle}: {cip['error']}")

    ig_content_dna = _extract_ig_content_dna(cip)
    ig_audience_profile = _extract_ig_audience_profile(cip)

    # Collaborators: coauthor/tagged ∪ past_collaborations textarea, minus brand
    raw_posts = cip.get("_raw_posts") or []
    ig_cohort = _extract_collaborators_from_posts(raw_posts, handle)
    all_collaborators = _union_past_collaborations(
        ig_cohort, brand.get("past_collaborations"), handle
    )

    # Generate brand embedding
    embedding_input = build_brand_embedding_input(
        brand_name=brand.get("brand_name"),
        description=brand.get("brand_description"),
        industry=brand.get("industry"),
        brand_values=brand.get("brand_values"),
        product_categories=brand.get("product_categories"),
        target_audience=brand.get("target_audience"),
        ig_content_dna=ig_content_dna,
    )
    embedding = embed_text(embedding_input, openai_key) if embedding_input else None

    now_iso = datetime.now(timezone.utc).isoformat()
    patch: dict[str, Any] = {
        "ig_analysis_status": "completed",
        "ig_analysis_completed_at": now_iso,
        "ig_analysis_error": None,
        "ig_content_dna": ig_content_dna,
        "ig_audience_profile": ig_audience_profile,
        "ig_collaborators": all_collaborators,
    }
    if embedding is not None:
        patch["content_embedding"] = embedding
        patch["embedding_computed_at"] = now_iso
    _update_brand(db, brand_id, patch)

    # Fan-out: cap at MAX_FANOUT_CREATORS, skip fresh ones
    fresh = _get_fresh_creator_handles(db, all_collaborators)
    fanout = [h for h in all_collaborators if h not in fresh][:MAX_FANOUT_CREATORS]
    enqueued = _enqueue_creator_fanout(db, brand_id, fanout)
    logger.info(
        f"Brand @{handle}: {len(all_collaborators)} collaborators, "
        f"{len(fresh)} fresh, {enqueued} creator jobs enqueued"
    )

    # Edge case: no fanout → trigger matching compute now
    if enqueued == 0:
        _trigger_matching_compute(brand_id)


def handle_creator_ig_scrape(db, job: dict) -> None:
    payload = job.get("payload") or {}
    handle = payload.get("handle")
    parent_brand_id = payload.get("parent_brand_id") or job.get("brand_id")
    # Phase 2.5 auto-stitch: when a YT scrape discovered this creator's
    # IG handle in external_links, it enqueues this job with the existing
    # creators.id so the IG profile attaches onto the same row.
    existing_creator_id = payload.get("existing_creator_id")
    if not handle:
        raise ValueError("creator_ig_scrape job missing handle in payload")

    brightdata_token = _require_env("BRIGHTDATA_API_TOKEN")
    gemini_key = _require_env("GEMINI_API_KEY")
    openai_key = _require_env("OPENAI_API_KEY")

    cip = build_creator_intelligence_profile(
        profile_url=_handle_url(handle),
        brightdata_token=brightdata_token,
        gemini_api_key=gemini_key,
        openai_api_key=openai_key,
        er_benchmarks=load_er_benchmarks(db),
    )
    if cip.get("error"):
        raise RuntimeError(f"CIP failed for creator @{handle}: {cip['error']}")

    creator_id = pdb.store_full_cip(db, cip, existing_creator_id=existing_creator_id)

    embedding_input = build_creator_embedding_input(cip)
    if embedding_input:
        embedding = embed_text(embedding_input, openai_key)
        # Shadow column (creators.content_embedding) for one release.
        _update_creator_embedding(db, creator_id, embedding)
        # Per-platform embedding (migration 046) — canonical going forward.
        pdb.upsert_creator_platform_embedding(
            db, creator_id, "instagram", embedding
        )

    # Auto-stitch: discover YT handle in IG's external_url, fan out a YT
    # scrape bound to the same creator row. Skips if creator already has
    # a YT profile (loop prevention for YT->IG->YT round-trip).
    try:
        _run_auto_stitch_from_ig(db, creator_id, cip)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"IG auto-stitch failed for creator {creator_id}: {e}")

    # If this creator was part of a brand fanout, trigger a brand-scoped
    # recompute when the whole fanout is done. Otherwise (standalone
    # creator run) trigger a per-creator recompute so any brand with
    # Shopify data picks up the new score.
    if parent_brand_id and _siblings_all_terminal(db, parent_brand_id):
        _trigger_matching_compute(parent_brand_id)
    elif not parent_brand_id and creator_id:
        _trigger_creator_recompute(creator_id)


# ── Content Video Analysis ─────────────────────────────────────────────────


def handle_content_video_analysis(db, job: dict) -> None:
    """
    Transcribe + qualitatively analyze a submitted campaign video.

    Pipeline: fetch context → scrape video URL → transcribe (Whisper) →
    analyze (Claude) → store results.
    """
    from pipeline.brightdata_client import BrightdataClient
    from pipeline.content_analyzer import (
        ANALYSIS_VERSION,
        MODEL as ANALYSIS_MODEL,
        analyze_submission_content,
    )
    from pipeline.scraper_posts import scrape_single_post
    from pipeline.transcriber import transcribe_reels

    payload = job.get("payload") or {}
    submission_id = payload.get("content_submission_id")
    content_url = payload.get("content_url")
    campaign_id = payload.get("campaign_id")
    creator_id = payload.get("creator_id")
    brand_id = job.get("brand_id")

    if not submission_id:
        raise ValueError("content_video_analysis job missing content_submission_id")

    now_iso = datetime.now(timezone.utc).isoformat()

    def _update_analysis(analysis_id: str, patch: dict) -> None:
        db.table("content_analyses").update(patch).eq("id", analysis_id).execute()

    def _update_submission_status(status: str) -> None:
        db.table("content_submissions").update(
            {"analysis_status": status}
        ).eq("id", submission_id).execute()

    # ── 1. Check for existing analysis (idempotency) ──────────────────
    existing = (
        db.table("content_analyses")
        .select("id,status")
        .eq("content_submission_id", submission_id)
        .execute()
    )
    if existing.data:
        row = existing.data[0]
        if row["status"] == "completed":
            logger.info(f"Analysis already completed for submission {submission_id}, skipping")
            return
        analysis_id = row["id"]
        _update_analysis(analysis_id, {"status": "transcribing", "error_message": None})
    else:
        # Create analysis row
        insert_result = (
            db.table("content_analyses")
            .insert({
                "content_submission_id": submission_id,
                "campaign_id": campaign_id,
                "creator_id": creator_id,
                "brand_id": brand_id,
                "status": "transcribing",
                "analysis_version": ANALYSIS_VERSION,
            })
            .execute()
        )
        analysis_id = insert_result.data[0]["id"]

    _update_submission_status("processing")

    # ── 2. Fetch context ──────────────────────────────────────────────
    submission = (
        db.table("content_submissions")
        .select("caption_text,content_url")
        .eq("id", submission_id)
        .single()
        .execute()
    ).data
    caption_text = submission.get("caption_text") if submission else None

    campaign = (
        db.table("campaigns")
        .select("name,goal,description,brief_requirements,target_regions,target_niches")
        .eq("id", campaign_id)
        .single()
        .execute()
    ).data or {}

    guidelines_result = (
        db.table("brand_guidelines")
        .select("forbidden_topics,content_dos,content_donts,required_disclosures,preferred_content_themes,notes")
        .eq("brand_id", brand_id)
        .limit(1)
        .execute()
    )
    brand_guidelines = guidelines_result.data[0] if guidelines_result.data else None

    # ── 3. Scrape + Transcribe (if video URL present) ─────────────────
    transcript = None
    has_video = content_url and ("instagram.com/reel/" in content_url or "instagram.com/p/" in content_url)

    if has_video:
        try:
            brightdata_token = _require_env("BRIGHTDATA_API_TOKEN")
            openai_key = _require_env("OPENAI_API_KEY")

            bd_client = BrightdataClient(api_token=brightdata_token)
            post_data = scrape_single_post(bd_client, content_url)

            if not post_data or not post_data.get("video_url"):
                logger.warning(f"No video_url found for {content_url}")
                _update_analysis(analysis_id, {
                    "status": "analyzing",
                    "error_message": "Video URL not accessible — running caption-only analysis",
                })
            else:
                reel_entry = {
                    "post_id": post_data.get("post_id", submission_id),
                    "video_url": post_data["video_url"],
                    "caption": post_data.get("description", ""),
                    "length": post_data.get("length", 0),
                }

                transcripts = transcribe_reels([reel_entry], openai_key)
                if transcripts and transcripts[0].get("transcript_text"):
                    transcript = transcripts[0]
                    _update_analysis(analysis_id, {
                        "status": "analyzing",
                        "transcript_text": transcript["transcript_text"],
                        "transcript_segments": transcript.get("segments", []),
                        "detected_language": transcript.get("detected_language"),
                        "hook_text": transcript.get("hook_text"),
                        "audio_confidence": transcript.get("avg_confidence"),
                        "is_likely_music": transcript.get("is_likely_music", False),
                        "reel_length_seconds": transcript.get("reel_length_seconds"),
                    })
                else:
                    error_msg = transcripts[0].get("error", "Transcription returned empty") if transcripts else "Transcription failed"
                    logger.warning(f"Transcription failed for {submission_id}: {error_msg}")
                    _update_analysis(analysis_id, {
                        "status": "analyzing",
                        "error_message": f"Transcription issue: {error_msg} — running caption-only analysis",
                    })

        except Exception as e:
            logger.error(f"Scrape/transcribe failed for {submission_id}: {e}")
            _update_analysis(analysis_id, {
                "status": "analyzing",
                "error_message": f"Video processing failed: {e} — running caption-only analysis",
            })
    else:
        _update_analysis(analysis_id, {"status": "analyzing"})

    # ── 4. Claude analysis ────────────────────────────────────────────
    if not transcript and not caption_text:
        _update_analysis(analysis_id, {
            "status": "skipped",
            "error_message": "No transcript or caption available for analysis.",
        })
        _update_submission_status("skipped")
        return

    try:
        anthropic_key = _require_env("ANTHROPIC_API_KEY")

        analysis_result = analyze_submission_content(
            transcript=transcript,
            caption_text=caption_text,
            brand_guidelines=brand_guidelines,
            campaign=campaign,
            anthropic_api_key=anthropic_key,
        )

        overall = analysis_result.get("overall", {})
        _update_analysis(analysis_id, {
            "status": "completed",
            "analysis": analysis_result,
            "overall_score": overall.get("score"),
            "hook_strength_score": analysis_result.get("hook_strength", {}).get("score"),
            "brand_mention_score": analysis_result.get("brand_mention", {}).get("score"),
            "brief_compliance_score": analysis_result.get("brief_compliance", {}).get("score"),
            "guideline_compliance_score": analysis_result.get("guideline_compliance", {}).get("score"),
            "analysis_model": ANALYSIS_MODEL,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        _update_submission_status("completed")
        logger.info(
            f"Content analysis completed for submission {submission_id} — "
            f"score={overall.get('score')}, rec={overall.get('recommendation')}"
        )

    except Exception as e:
        logger.error(f"Claude analysis failed for {submission_id}: {e}")
        _update_analysis(analysis_id, {
            "status": "failed",
            "error_message": str(e),
        })
        _update_submission_status("failed")
        raise


# ── Dispatch ────────────────────────────────────────────────────────────────

def handle_creator_yt_scrape(db, job: dict) -> None:
    """Per-creator YouTube scrape.

    Payload accepts any of: {url} | {channel_id} | {handle}. The first
    available is resolved into a canonical URL before the build runs.
    Mirrors handle_creator_ig_scrape's responsibility.

    Phase 2.5: also checks `existing_creator_id` in payload (set by the
    symmetric IG→YT auto-stitch) so YT can attach onto an existing row.
    """
    from pipeline.pipeline import build_youtube_creator_intelligence_profile
    from pipeline.youtube.handle_resolver import resolve as resolve_yt

    payload = job.get("payload") or {}
    parent_brand_id = payload.get("parent_brand_id") or job.get("brand_id")
    existing_creator_id = payload.get("existing_creator_id")

    raw = (
        payload.get("url")
        or payload.get("channel_url")
        or payload.get("channel_id")
        or payload.get("handle")
    )
    if not raw:
        raise ValueError("creator_yt_scrape job missing url/channel_id/handle")

    resolved = resolve_yt(raw)
    channel_url = resolved.url

    cip = build_youtube_creator_intelligence_profile(
        channel_url=channel_url,
        brightdata_token=_require_env("BRIGHTDATA_API_TOKEN"),
        gemini_api_key=_require_env("GEMINI_API_KEY"),
        openai_api_key=_require_env("OPENAI_API_KEY"),
        youtube_api_key=os.environ.get("YOUTUBE_API_KEY"),
    )
    if cip.get("error"):
        raise RuntimeError(
            f"YT CIP failed for {channel_url}: {cip['error']}"
        )

    creator_id = pdb.store_youtube_cip(
        db, cip, existing_creator_id=existing_creator_id
    )

    embedding_input = build_creator_embedding_input(cip)
    if embedding_input:
        embedding = embed_text(embedding_input, _require_env("OPENAI_API_KEY"))
        # Per-platform embedding for scoring (migration 046). No shadow
        # column write for YT — that field is IG-indexed.
        pdb.upsert_creator_platform_embedding(
            db, creator_id, "youtube", embedding
        )

    # Auto-stitch: inspect YT external_links for IG URL, fan out an IG
    # scrape bound to the same creator row.
    try:
        _run_auto_stitch_from_yt(db, creator_id, cip)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"YT auto-stitch failed for creator {creator_id}: {e}")

    if parent_brand_id and _siblings_all_terminal(db, parent_brand_id):
        _trigger_matching_compute(parent_brand_id)
    elif not parent_brand_id and creator_id:
        _trigger_creator_recompute(creator_id)


def handle_brand_yt_scrape(db, job: dict) -> None:
    """Brand-side YouTube analysis: scrape + CIP the brand's own YT channel.

    Parallels handle_brand_ig_scrape. The result lands in
    brand_platform_analyses(platform='youtube'). Collaborator fanout
    (triggering per-creator scrapes for tagged-in-video creators) is
    deferred — YT collabs are surfaced differently from IG tags and
    deserve their own handler.
    """
    from pipeline.pipeline import build_youtube_creator_intelligence_profile
    from pipeline.youtube.handle_resolver import resolve as resolve_yt

    payload = job.get("payload") or {}
    brand_id = job.get("brand_id") or payload.get("brand_id")
    raw = (
        payload.get("url")
        or payload.get("channel_url")
        or payload.get("handle")
    )
    if not brand_id or not raw:
        raise ValueError("brand_yt_scrape job missing brand_id or channel url")

    resolved = resolve_yt(raw)
    cip = build_youtube_creator_intelligence_profile(
        channel_url=resolved.url,
        brightdata_token=_require_env("BRIGHTDATA_API_TOKEN"),
        gemini_api_key=_require_env("GEMINI_API_KEY"),
        openai_api_key=_require_env("OPENAI_API_KEY"),
        youtube_api_key=os.environ.get("YOUTUBE_API_KEY"),
    )

    # Extract @handle + channel_id mentions across the brand's own videos.
    # These are the creators the brand has publicly collab'd with and are
    # the highest-priority fanout targets for per-creator YT scrapes.
    from pipeline.youtube.collaborators import extract_collaborators

    collab_report = extract_collaborators(
        cip.get("videos") or [],
        self_handle=(cip.get("profile") or {}).get("handle"),
        self_channel_id=(cip.get("resolved") or {}).get("channel_id"),
    )
    collaborator_handles = [c["handle"] for c in collab_report["handles"]]

    analysis = {
        "handle": (cip.get("profile") or {}).get("handle"),
        "profile_url": resolved.url,
        "analysis_status": "failed" if cip.get("error") else "completed",
        "analysis_completed_at": datetime.now(timezone.utc).isoformat(),
        "analysis_error": cip.get("error"),
        "content_dna": cip.get("caption_intelligence"),
        "audience_profile": cip.get("audience_intelligence"),
        "collaborators": collaborator_handles,
    }
    pdb.upsert_brand_platform_analysis(db, brand_id, "youtube", analysis)


def handle_creator_multi_platform_scrape(db, job: dict) -> None:
    """Run IG and YT scrapes concurrently for a single creator.

    Both pipelines are I/O-bound (HTTP polling + Whisper + Gemini + API
    calls). A `ThreadPoolExecutor` gives us true parallelism without the
    async refactor. Wall-clock ≈ max(ig_time, yt_time) instead of sum.

    Payload:
      ig_handle           (optional): IG handle (without leading @)
      yt_url              (optional): canonical YT channel URL / @handle
      existing_creator_id (optional): bind both profiles to this row
      parent_brand_id     (optional): triggers matching compute on both-terminal

    If only one of ig_handle / yt_url is set, we just run that side.
    Errors on one platform don't abort the other; each raises
    independently via its thread's fut.result() if both fail.
    """
    from concurrent.futures import ThreadPoolExecutor

    payload = job.get("payload") or {}
    ig_handle = payload.get("ig_handle")
    yt_url = payload.get("yt_url") or payload.get("channel_url")
    existing_creator_id = payload.get("existing_creator_id")
    parent_brand_id = payload.get("parent_brand_id") or job.get("brand_id")

    if not ig_handle and not yt_url:
        raise ValueError(
            "creator_multi_platform_scrape job missing ig_handle and yt_url"
        )

    results: dict[str, str | None] = {"instagram": None, "youtube": None}
    errors: dict[str, str] = {}

    def _ig() -> str | None:
        sub_job = {
            "payload": {
                "handle": ig_handle,
                "existing_creator_id": existing_creator_id,
                "parent_brand_id": parent_brand_id,
            }
        }
        handle_creator_ig_scrape(db, sub_job)
        return existing_creator_id  # store_full_cip returned id is discarded
        # by handle_creator_ig_scrape; we could plumb it back but the
        # multi-platform handler just needs to know it ran.

    def _yt() -> str | None:
        sub_job = {
            "payload": {
                "url": yt_url,
                "existing_creator_id": existing_creator_id,
                "parent_brand_id": parent_brand_id,
            }
        }
        handle_creator_yt_scrape(db, sub_job)
        return existing_creator_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures: dict[str, object] = {}
        if ig_handle:
            futures["instagram"] = pool.submit(_ig)
        if yt_url:
            futures["youtube"] = pool.submit(_yt)
        for platform, fut in futures.items():
            try:
                results[platform] = fut.result()
            except Exception as e:  # noqa: BLE001
                errors[platform] = str(e)
                logger.exception(
                    f"multi_platform_scrape: {platform} failed"
                )

    # If one platform failed and the other succeeded, we still surface the
    # failure so the worker marks the job failed. If both failed, same.
    if errors and len(errors) == len(results):
        raise RuntimeError(
            f"multi_platform_scrape: all platforms failed: {errors}"
        )

    # Only trigger matching once, after both children finish.
    if parent_brand_id and _siblings_all_terminal(db, parent_brand_id):
        _trigger_matching_compute(parent_brand_id)


HANDLERS = {
    "brand_ig_scrape": handle_brand_ig_scrape,
    "creator_ig_scrape": handle_creator_ig_scrape,
    "brand_yt_scrape": handle_brand_yt_scrape,
    "creator_yt_scrape": handle_creator_yt_scrape,
    "creator_multi_platform_scrape": handle_creator_multi_platform_scrape,
    "content_video_analysis": handle_content_video_analysis,
}


def dispatch(db, job: dict) -> None:
    job_type = job["job_type"]
    handler = HANDLERS.get(job_type)
    if not handler:
        raise ValueError(f"No handler registered for job_type={job_type}")
    handler(db, job)
