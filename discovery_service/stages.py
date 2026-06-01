"""Pure-ish stage functions for the discovery pipeline.

Each stage:
  - Reads `discovery_requests` for state when needed
  - Mutates the in-progress row's `status` + counters
  - Returns the data the next stage consumes

Keeping the stages here (rather than inline in `app.py`) makes them
unit-testable against a fake Supabase client; `app.py` is just the Modal
function definition that wires them together.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Iterable

from supabase import Client

from pipeline.discovery_service.brand_match_client import BrandMatchClient
from pipeline.discovery_service.search import (
    CandidateChannel,
    filter_candidates,
    search_keyword,
)
from pipeline.youtube.api_pool import load_api_keys_from_env

logger = logging.getLogger(__name__)


# ── Status helpers ──────────────────────────────────────────────────

def _set_status(
    db: Client,
    request_id: str,
    *,
    status: str | None = None,
    error_text: str | None = None,
    **counters: int,
) -> None:
    """Update discovery_requests with new status + optional counters.

    All optional updates are merged into a single PATCH so each stage
    boundary is one DB call. Service-role key bypasses RLS.
    """
    payload: dict = {}
    if status is not None:
        payload["status"] = status
        if status == "searching":
            payload["started_at"] = datetime.utcnow().isoformat()
        elif status in ("completed", "failed"):
            payload["completed_at"] = datetime.utcnow().isoformat()
    if error_text is not None:
        payload["error_text"] = error_text[:1000]
    payload.update(counters)
    if not payload:
        return
    try:
        db.table("discovery_requests").update(payload).eq(
            "id", request_id
        ).execute()
    except Exception as e:  # noqa: BLE001
        # Status updates are not allowed to crash the worker. The next
        # write will reconcile state.
        logger.warning(f"_set_status({request_id}) failed: {e}")


def _bump_counter(
    db: Client, request_id: str, column: str, delta: int = 1
) -> None:
    """Atomic-ish counter increment via SQL function.

    Falls back to a read-modify-write on any failure; the discovery is
    single-spawn so contention is minimal.
    """
    try:
        db.rpc(
            "increment_discovery_counter",
            {
                "p_request_id": request_id,
                "p_column": column,
                "p_delta": delta,
            },
        ).execute()
    except Exception:
        # No RPC defined — do read-modify-write. Tolerable for a counter
        # only one container is bumping.
        try:
            res = (
                db.table("discovery_requests")
                .select(column)
                .eq("id", request_id)
                .limit(1)
                .execute()
            )
            current = (res.data or [{}])[0].get(column) or 0
            db.table("discovery_requests").update(
                {column: current + delta}
            ).eq("id", request_id).execute()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"bump_counter({column}) failed: {e}")


# ── Brand / candidate helpers ───────────────────────────────────────


def _load_competitor_substrings(db: Client, brand_id: str) -> list[str]:
    """Return the brand's competitor_brands as lowercased substrings
    safe for substring filtering against a YT channel title.

    Mirrors web/src/app/api/discover/search/route.ts::sanitizeCompetitorBrands —
    including the @-stripping. The Settings UI stores entries like
    `@Unacademy` but YT channel titles read "Unacademy NEET..." with no
    leading `@`. Keep the `@` and the substring never matches, and we
    end up scraping every competitor channel into the brand's DB.
    """
    try:
        res = (
            db.table("brands")
            .select("competitor_brands")
            .eq("id", brand_id)
            .limit(1)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"load competitor_brands failed: {e}")
        return []
    raw = ((res.data or [{}])[0] or {}).get("competitor_brands") or []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        t = item.strip().lstrip("@").lower()
        if len(t) <= 2 or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _existing_channel_ids(
    db: Client, channel_ids: Iterable[str]
) -> set[str]:
    """Look up which of these YT channel IDs we already have a creator for.

    Reads `creator_social_profiles.platform_user_id` since that's where
    YouTube channel IDs land in this schema.
    """
    ids = [c for c in channel_ids if c]
    if not ids:
        return set()
    out: set[str] = set()
    # PostgREST `.in_` has a URL-length cap; batch in 100s.
    for i in range(0, len(ids), 100):
        batch = ids[i : i + 100]
        try:
            res = (
                db.table("creator_social_profiles")
                .select("platform_user_id")
                .eq("platform", "youtube")
                .in_("platform_user_id", batch)
                .execute()
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"existing-channel lookup failed: {e}")
            continue
        for row in res.data or []:
            v = row.get("platform_user_id")
            if v:
                out.add(v)
    return out


# ── Stage implementations ───────────────────────────────────────────


def stage_search(
    db: Client,
    request_id: str,
    query: str,
    *,
    pool,
    channel_max: int = 200,
    video_max: int = 200,
    region_code: str | None = None,
) -> list[CandidateChannel]:
    """Run channel+video keyword search, merge + dedup.

    Updates `discovery_requests.status='searching'` on entry and
    `candidates_total` on exit (pre-filter count).
    """
    _set_status(db, request_id, status="searching")
    candidates = search_keyword(
        query,
        pool,
        channel_max=channel_max,
        video_max=video_max,
        region_code=region_code,
    )
    _set_status(db, request_id, candidates_total=len(candidates))
    return candidates


def stage_filter(
    db: Client,
    request_id: str,
    brand_id: str,
    candidates: list[CandidateChannel],
    *,
    pool=None,
    user_filters: dict | None = None,
) -> tuple[list[CandidateChannel], set[str]]:
    """Apply competitor-name, already-in-DB, and (optional) tier filters.

    Returns (survivors, existing_channel_ids). Survivors are channels we
    will deep-scrape. `existing_channel_ids` is reported back so the
    caller can attach them to the discovery_request after deep-scrape
    completes (so the user sees the full ~250 creators for the query).

    `user_filters` carries the filter chips active on the Discover page
    when the user clicked Search YouTube. We apply the cheaply-checkable
    ones BEFORE deep-scrape so we don't burn ~30s of compute per creator
    on candidates the user would have filtered out anyway. Currently
    handled here: tiers, min/max followers. Niche/audience filters can't
    be applied at this stage (we don't know niche until LLM analysis
    runs) — the matching engine sorts those out later.
    """
    _set_status(db, request_id, status="profiling")
    competitors = _load_competitor_substrings(db, brand_id)

    # Order matters: drop competitor channels BEFORE checking which are
    # already in DB. Earlier versions filtered competitors only from the
    # "new candidates" stream but `_existing_channel_ids` still picked up
    # competitors that we'd already scraped in a previous run — then
    # `_tag_existing_creators` happily stamped them with the new
    # discovery_request_id, surfacing them in this discovery's results
    # alongside the legitimate creators. Drop them up-front and the leak
    # closes on both paths.
    non_competitor_candidates = [
        c
        for c in candidates
        if not any(
            s and s in (c.title or "").lower() for s in competitors
        )
    ]
    existing = _existing_channel_ids(
        db, (c.channel_id for c in non_competitor_candidates)
    )
    new_candidates = filter_candidates(
        non_competitor_candidates,
        competitor_substrings=[],  # already filtered above
        existing_channel_ids=existing,
    )

    # ── Optional: drop candidates outside the user's tier / follower
    #              range BEFORE we burn ~30s of compute per candidate.
    survivors = new_candidates
    user_filters = user_filters or {}
    wants_tier = (
        isinstance(user_filters.get("tiers"), list)
        and len(user_filters["tiers"]) > 0
    )
    min_followers = _safe_int(user_filters.get("min_followers"))
    max_followers = _safe_int(user_filters.get("max_followers"))

    if pool is not None and (wants_tier or min_followers or max_followers):
        survivors = _apply_subscriber_filters(
            new_candidates,
            pool=pool,
            tiers=user_filters.get("tiers") if wants_tier else None,
            min_followers=min_followers,
            max_followers=max_followers,
        )

    logger.info(
        f"stage_filter: {len(candidates)} candidates → "
        f"{len(new_candidates)} new (vs DB) → "
        f"{len(survivors)} after tier/follower filter, "
        f"{len(existing)} already in DB, "
        f"competitor_substrings={len(competitors)}"
    )
    # candidates_total stays as the full pre-filter count so the UI loader
    # shows the realistic universe size, not just new-to-DB rows.
    _set_status(db, request_id, candidates_profiled=len(survivors))
    return survivors, existing


def _safe_int(v) -> int | None:
    """Coerce JSONB-ish numerics to int; None on garbage / missing."""
    if v is None:
        return None
    try:
        out = int(v)
        return out if out > 0 else None
    except (TypeError, ValueError):
        return None


def _apply_subscriber_filters(
    candidates: list[CandidateChannel],
    *,
    pool,
    tiers: list[str] | None,
    min_followers: int | None,
    max_followers: int | None,
) -> list[CandidateChannel]:
    """Batch-fetch subscriber counts + filter.

    One channels.list call per ~50 channel IDs (~1 quota unit each) →
    negligible cost vs the deep-scrape we'd skip.
    """
    from pipeline.youtube.scraper_channels import classify_creator_tier

    ids = [c.channel_id for c in candidates]
    if not ids:
        return []

    # Fetch stats for all candidates. fetch_channel_stats handles batching
    # internally and returns {channel_id: {subscriber_count, ...}}.
    stats = pool.fetch_channel_stats(ids)

    tiers_set = set(t.lower() for t in (tiers or []) if isinstance(t, str))
    out: list[CandidateChannel] = []
    for c in candidates:
        s = stats.get(c.channel_id) or {}
        subs = s.get("subscriber_count")
        try:
            subs = int(subs) if subs is not None else None
        except (TypeError, ValueError):
            subs = None
        # Channels with hidden subscriber counts come back as None or 0.
        # Be permissive: when we can't classify, let them through and let
        # the user's filter on the result page drop them — better than
        # silently culling potentially-good creators.
        if subs is None:
            c.subscriber_count = None
            out.append(c)
            continue
        c.subscriber_count = subs
        if min_followers is not None and subs < min_followers:
            continue
        if max_followers is not None and subs > max_followers:
            continue
        if tiers_set:
            tier = classify_creator_tier(subs)
            if tier not in tiers_set:
                continue
        out.append(c)
    return out


def _scrape_one_creator(
    candidate: CandidateChannel,
    *,
    db: Client,
    request_id: str,
    brand_id: str,
    gemini_api_key: str,
    openai_api_key: str,
    youtube_api_key: str | None,
    num_videos: int,
    num_transcripts: int,
) -> str | None:
    """Build + store one creator's CIP. Returns creator_id on success.

    Heavy reuse of the existing pipeline: `build_youtube_creator_intelligence_profile`
    handles channel + videos + transcripts + LLM analysis + scoring in
    one synchronous call. `store_youtube_cip` writes everything to the
    right tables — INCLUDING calling `_mark_for_embedding` which only
    enqueues an embedding (creates a creator_embeddings row with
    embedding=NULL). A separate `scripts/embed_creators.py` worker is
    supposed to pick those up and compute the actual vector.
    Discovery callers typically don't have that worker running, so we
    compute the embedding inline right after the CIP write — otherwise
    the new creator never appears in normal hybrid search results.
    """
    from pipeline.pipeline import build_youtube_creator_intelligence_profile
    from pipeline.db import store_youtube_cip, upsert_creator_platform_embedding
    from pipeline.embeddings import build_creator_embedding_input, embed_text

    url = candidate.channel_url or (
        f"https://www.youtube.com/channel/{candidate.channel_id}"
    )

    try:
        cip = build_youtube_creator_intelligence_profile(
            channel_url=url,
            gemini_api_key=gemini_api_key,
            openai_api_key=openai_api_key,
            youtube_api_key=youtube_api_key,
            num_videos=num_videos,
            num_transcripts=num_transcripts,
        )
    except Exception as e:  # noqa: BLE001
        logger.error(
            f"build_youtube_cip failed for {candidate.channel_id}: {e}"
        )
        return None

    if cip.get("error"):
        logger.warning(
            f"YT CIP partial for {candidate.channel_id}: {cip['error']}"
        )
        # Still try to store the partial — handle / profile may be enough.
        if not cip.get("profile"):
            return None

    # Retry once on `Server disconnected` — supabase-py's shared httpx
    # connection pool occasionally drops a connection mid-flight under
    # concurrent load. A single retry with a fresh attempt clears it
    # ~95% of the time and is cheaper than re-running the whole scrape.
    creator_id: str | None = None
    for attempt in (1, 2):
        try:
            creator_id = store_youtube_cip(db, cip)
            break
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            transient = "Server disconnected" in msg or "Connection" in msg
            if attempt == 1 and transient:
                time.sleep(0.5)
                continue
            logger.error(
                f"store_youtube_cip failed for {candidate.channel_id} "
                f"(attempt {attempt}): {e}"
            )
            return None
    if not creator_id:
        return None

    # ── Compute embedding inline ──
    # Without this, store_youtube_cip's `_mark_for_embedding` leaves the
    # vector NULL and the creator is invisible to hybrid search. The
    # nightly batch worker would eventually catch up, but the user
    # expects the creator visible immediately after discovery completes.
    try:
        content_text = build_creator_embedding_input(cip)
        if content_text.strip():
            embedding = embed_text(content_text, openai_api_key)
            upsert_creator_platform_embedding(
                db,
                creator_id=creator_id,
                platform="youtube",
                embedding=embedding,
                content_text=content_text,
            )
    except Exception as e:  # noqa: BLE001
        # Embedding failure doesn't unwind the creator — they still
        # exist in the DB and will show up in filter-based searches.
        # The next nightly embed worker will pick up the NULL row.
        logger.warning(
            f"inline embedding failed for {creator_id}: {e}"
        )

    # Tag the creator with this discovery's provenance.
    try:
        db.table("creators").update(
            {
                "discovery_request_id": request_id,
                "discovered_at": datetime.utcnow().isoformat(),
            }
        ).eq("id", creator_id).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"discovery_request_id tag failed for {creator_id}: {e}"
        )

    _bump_counter(db, request_id, "candidates_scraped")
    return creator_id


def stage_parallel_scrape(
    db: Client,
    request_id: str,
    brand_id: str,
    survivors: list[CandidateChannel],
    *,
    parallelism: int = 50,
    num_videos: int = 5,
    num_transcripts: int = 5,
) -> list[str]:
    """Run per-creator deep scrape in parallel via ThreadPoolExecutor.

    Each `build_youtube_creator_intelligence_profile` is sync I/O-bound
    (YouTube API + Modal Whisper + Gemini + OpenAI) so threads release
    the GIL on every network call. 50 threads × ~30s wall-clock per
    creator → ~2-3 min for 200 creators.

    Returns the list of successfully-scraped creator_ids (in completion
    order; UI sorts by match_score after stage_brand_match).
    """
    _set_status(db, request_id, status="scraping")

    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    # Resolve YouTube keys from the same env-var trio the pool uses
    # (`YOUTUBE_API_KEYS` comma-separated, `YOUTUBE_API_KEY_1..N`, or
    # legacy `YOUTUBE_API_KEY`). With 12 keys × ~10K quota each we have
    # huge headroom; distributing one-key-per-thread keeps any single
    # key well under its daily quota even on a 200-creator discovery.
    yt_keys = load_api_keys_from_env()
    if not yt_keys:
        msg = "no YouTube API keys configured — set YOUTUBE_API_KEYS"
        _set_status(db, request_id, status="failed", error_text=msg)
        raise RuntimeError(msg)

    if not gemini_api_key or not openai_api_key:
        # Without these, LLM + embedding stages inside the CIP build will
        # silently no-op and the resulting `creators` row will be very
        # thin. Surface this loud and early so deploys don't ship blind.
        msg = "GEMINI_API_KEY and OPENAI_API_KEY required for discovery"
        _set_status(db, request_id, status="failed", error_text=msg)
        raise RuntimeError(msg)

    creator_ids: list[str] = []
    errs = 0
    parallelism = max(1, min(parallelism, len(survivors) or 1))

    with ThreadPoolExecutor(max_workers=parallelism) as exe:
        futures = {
            # Round-robin a YouTube key per submission so per-key quota
            # spreads evenly across the 12 keys. Each creator costs ~11
            # quota units; 200 creators × 11 / 12 keys ≈ 184 units/key.
            exe.submit(
                _scrape_one_creator,
                c,
                db=db,
                request_id=request_id,
                brand_id=brand_id,
                gemini_api_key=gemini_api_key,
                openai_api_key=openai_api_key,
                youtube_api_key=yt_keys[i % len(yt_keys)],
                num_videos=num_videos,
                num_transcripts=num_transcripts,
            ): c
            for i, c in enumerate(survivors)
        }
        for fut in as_completed(futures):
            try:
                cid = fut.result()
            except Exception as e:  # noqa: BLE001
                logger.error(
                    f"per-creator scrape worker error: {e}"
                )
                cid = None
            if cid:
                creator_ids.append(cid)
            else:
                errs += 1

    logger.info(
        f"stage_parallel_scrape: {len(creator_ids)} OK, {errs} failed "
        f"out of {len(survivors)} attempts"
    )
    return creator_ids


def stage_brand_match(
    db: Client,
    request_id: str,
    brand_id: str,
    creator_ids: list[str],
) -> dict:
    """Trigger brand-match scoring for the new creators via Next.js.

    Updates status to 'matching' on entry, bumps candidates_matched on
    exit. Returns the JSON the Next.js endpoint sent back.
    """
    _set_status(db, request_id, status="matching")
    client = BrandMatchClient()
    result = client.compute_batch(creator_ids, brand_id)
    _set_status(
        db,
        request_id,
        candidates_matched=int(result.get("computed", 0)),
    )
    return result


def stage_complete(db: Client, request_id: str) -> None:
    _set_status(db, request_id, status="completed")


def stage_failed(
    db: Client, request_id: str, error_text: str
) -> None:
    _set_status(db, request_id, status="failed", error_text=error_text)
