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
    )
    if cip.get("error"):
        raise RuntimeError(f"CIP failed for creator @{handle}: {cip['error']}")

    creator_id = pdb.store_full_cip(db, cip)

    embedding_input = build_creator_embedding_input(cip)
    if embedding_input:
        embedding = embed_text(embedding_input, openai_key)
        _update_creator_embedding(db, creator_id, embedding)

    # If this was the last sibling in the fanout, recompute matches.
    if parent_brand_id and _siblings_all_terminal(db, parent_brand_id):
        _trigger_matching_compute(parent_brand_id)


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

HANDLERS = {
    "brand_ig_scrape": handle_brand_ig_scrape,
    "creator_ig_scrape": handle_creator_ig_scrape,
    "content_video_analysis": handle_content_video_analysis,
}


def dispatch(db, job: dict) -> None:
    job_type = job["job_type"]
    handler = HANDLERS.get(job_type)
    if not handler:
        raise ValueError(f"No handler registered for job_type={job_type}")
    handler(db, job)
