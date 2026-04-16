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


# ── Dispatch ────────────────────────────────────────────────────────────────

HANDLERS = {
    "brand_ig_scrape": handle_brand_ig_scrape,
    "creator_ig_scrape": handle_creator_ig_scrape,
}


def dispatch(db, job: dict) -> None:
    job_type = job["job_type"]
    handler = HANDLERS.get(job_type)
    if not handler:
        raise ValueError(f"No handler registered for job_type={job_type}")
    handler(db, job)
