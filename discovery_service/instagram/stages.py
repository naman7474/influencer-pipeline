"""Stage functions for the Instagram discovery pipeline.

Mirrors `pipeline.discovery_service.stages` but driven by Apify (search
+ profile + posts/reels) instead of the YouTube Data API. Reuses every
helper that's platform-agnostic — competitor substring loader, status
FSM updater, brand-match callback, embedding helpers — so the
behaviour is identical end-to-end and tier-filter / dedup logic stays
in one place.

Per Phase 5 plan: comments are explicitly skipped (set via
`set_default_comments_per_reel(0)` before each creator's CIP build,
backed by a `commerce_signal={"discovery_no_comments": True}` payload
to the existing builder so its internal policy aligns).
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Iterable

from supabase import Client

from pipeline.discovery_service.brand_match_client import BrandMatchClient
from pipeline.discovery_service.instagram.search import (
    IgCandidate,
    filter_ig_candidates,
    search_ig_users,
)
# Reuse YT helpers — they're platform-agnostic. Single source of truth
# for status transitions, counter increments, competitor sanitization.
from pipeline.discovery_service.stages import (
    _bump_counter,
    _load_competitor_substrings,
    _set_status,
    stage_brand_match,
    stage_complete,
    stage_failed,
)

logger = logging.getLogger(__name__)


# ── Existing-in-DB lookup ───────────────────────────────────────────


def _existing_ig_handles(db: Client, usernames: Iterable[str]) -> set[str]:
    """Look up which of these IG usernames we already have a creator for.

    Reads `creator_social_profiles.handle` filtered by `platform='instagram'`.
    Case-insensitive comparison (IG usernames are case-insensitive in
    practice) — we normalize to lower in both directions.
    """
    handles = [u.strip().lower() for u in usernames if u and u.strip()]
    if not handles:
        return set()
    out: set[str] = set()
    # PostgREST `.in_` has a URL-length cap; batch in 100s.
    for i in range(0, len(handles), 100):
        batch = handles[i : i + 100]
        try:
            res = (
                db.table("creator_social_profiles")
                .select("handle")
                .eq("platform", "instagram")
                .in_("handle", batch)
                .execute()
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"existing-handle lookup failed: {e}")
            continue
        for row in res.data or []:
            v = row.get("handle")
            if v:
                out.add(str(v).lower())
    return out


# ── Stage implementations ───────────────────────────────────────────


def stage_search_ig(
    db: Client,
    request_id: str,
    query: str,
    *,
    max_results: int = 100,
) -> list[IgCandidate]:
    """Run the Instagram keyword search. Updates status to `searching`."""
    _set_status(db, request_id, status="searching")
    candidates = search_ig_users(query, max_results=max_results)
    _set_status(db, request_id, candidates_total=len(candidates))
    return candidates


def stage_filter_ig(
    db: Client,
    request_id: str,
    brand_id: str,
    candidates: list[IgCandidate],
    *,
    user_filters: dict | None = None,
) -> tuple[list[IgCandidate], set[str]]:
    """Apply competitor + already-in-DB + (optional) tier/follower filters.

    Mirrors the YT-side `stage_filter` semantics:
      1. Drop competitor channels up-front (by username/full_name/bio)
         so they're not re-tagged with this discovery_request_id.
      2. Identify already-in-DB usernames (returned separately so the
         caller can tag them, just like YT existing_ids).
      3. Apply tier / min-max follower filters using the
         `follower_count` already returned by the Apify search actor —
         no extra round-trip needed (this is one of the perks of the
         `instagram-search-users` actor over `instagram-hashtag-scraper`).
    """
    _set_status(db, request_id, status="profiling")
    competitors = _load_competitor_substrings(db, brand_id)

    # Drop competitor channels FIRST so they're not re-tagged below.
    non_competitor: list[IgCandidate] = []
    for c in candidates:
        haystack = " ".join(
            x.lower()
            for x in (c.username, c.full_name or "", c.biography or "")
            if x
        )
        if any(s and s in haystack for s in competitors):
            continue
        non_competitor.append(c)

    # Existing-in-DB lookup over the non-competitor set.
    existing = _existing_ig_handles(
        db, (c.username for c in non_competitor)
    )
    new_candidates = filter_ig_candidates(
        non_competitor,
        competitor_substrings=[],  # already filtered above
        existing_handles=existing,
    )

    # Apply user filters (tier, min/max followers) on follower_count
    # already returned by the search actor. No extra Apify call needed.
    user_filters = user_filters or {}
    new_candidates = _apply_subscriber_filters_ig(
        new_candidates, user_filters
    )

    logger.info(
        "stage_filter_ig: %d candidates → %d new (vs DB), %d in DB, "
        "competitor_substrings=%d",
        len(candidates),
        len(new_candidates),
        len(existing),
        len(competitors),
    )
    _set_status(db, request_id, candidates_profiled=len(new_candidates))
    return new_candidates, existing


def _apply_subscriber_filters_ig(
    candidates: list[IgCandidate],
    user_filters: dict,
) -> list[IgCandidate]:
    """Drop candidates outside the user's tier / follower range.

    Uses follower_count from the search actor (no extra Apify call).
    Permissive when follower_count is None — let the candidate through
    so a private/restricted profile isn't silently dropped.
    """
    from pipeline.scraper_profiles import classify_creator_tier

    tiers_raw = user_filters.get("tiers") or []
    tiers_set = (
        set(str(t).lower() for t in tiers_raw if isinstance(t, str))
        if isinstance(tiers_raw, list)
        else set()
    )
    min_f = _safe_int(user_filters.get("min_followers"))
    max_f = _safe_int(user_filters.get("max_followers"))

    if not tiers_set and not min_f and not max_f:
        return candidates

    out: list[IgCandidate] = []
    for c in candidates:
        subs = c.follower_count
        if subs is None:
            out.append(c)
            continue
        if min_f is not None and subs < min_f:
            continue
        if max_f is not None and subs > max_f:
            continue
        if tiers_set:
            tier = classify_creator_tier(subs)
            if tier not in tiers_set:
                continue
        out.append(c)
    return out


def _safe_int(v) -> int | None:
    if v is None:
        return None
    try:
        n = int(v)
        return n if n > 0 else None
    except (TypeError, ValueError):
        return None


def _scrape_one_ig_creator(
    candidate: IgCandidate,
    *,
    db: Client,
    request_id: str,
    brand_id: str,
    gemini_api_key: str,
    openai_api_key: str,
    num_posts: int,
    num_reels: int,
) -> str | None:
    """Build + store one IG creator's CIP. Returns creator_id on success.

    Heavy reuse of the existing IG pipeline:
      - `build_creator_intelligence_profile` (pipeline/pipeline.py) does
        scrape → analyse → score in one synchronous call. It supports
        skipping comments via `commerce_signal={...}` (truthy → 0 comments).
      - `store_full_cip` (pipeline/db.py) handles all the table writes.
      - Inline embedding (matching the YT discovery fix) so the new
        creator is immediately searchable via hybrid search.
    """
    from pipeline.pipeline import build_creator_intelligence_profile
    from pipeline.db import (
        store_full_cip,
        upsert_creator_platform_embedding,
    )
    from pipeline.embeddings import build_creator_embedding_input, embed_text
    from pipeline import apify_instagram_bundle

    profile_url = (
        candidate.profile_url
        or f"https://www.instagram.com/{candidate.username}/"
    )

    # Belt-and-suspenders: enforce comments_per_reel=0 BEFORE the builder
    # runs. The builder will re-set it from the commerce_signal policy,
    # but our explicit set protects against any code path that bypasses
    # commerce_signal handling.
    try:
        apify_instagram_bundle.set_default_comments_per_reel(0)
    except Exception:
        pass

    try:
        # Truthy `commerce_signal` forces the builder's internal comment
        # policy to comments_per_reel=0. The dict is round-tripped into
        # `cip["commerce_signal"]` purely as metadata — never read for
        # scoring, so the placeholder is safe.
        cip = build_creator_intelligence_profile(
            profile_url=profile_url,
            gemini_api_key=gemini_api_key,
            openai_api_key=openai_api_key,
            num_posts=num_posts,
            num_reels=num_reels,
            num_comment_posts=0,
            commerce_signal={"discovery_no_comments": True},
        )
    except Exception as e:  # noqa: BLE001
        logger.error(
            f"build_ig_cip failed for @{candidate.username}: {e}"
        )
        return None

    if cip.get("error"):
        logger.warning(
            f"IG CIP partial for @{candidate.username}: {cip['error']}"
        )
        if not cip.get("profile"):
            return None

    # Retry-once on transient Supabase disconnects (same lesson as YT
    # discovery — shared httpx pool occasionally drops a connection
    # under concurrent load; a single retry clears it ~95% of the time).
    import time as _time

    creator_id: str | None = None
    for attempt in (1, 2):
        try:
            creator_id = store_full_cip(db, cip)
            break
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            transient = "Server disconnected" in msg or "Connection" in msg
            if attempt == 1 and transient:
                _time.sleep(0.5)
                continue
            logger.error(
                f"store_full_cip failed for @{candidate.username} "
                f"(attempt {attempt}): {e}"
            )
            return None
    if not creator_id:
        return None

    # Inline embedding — without this the creator is invisible to hybrid
    # search until the nightly batch worker catches up. Same lesson as YT.
    try:
        content_text = build_creator_embedding_input(cip)
        if content_text.strip():
            embedding = embed_text(content_text, openai_api_key)
            upsert_creator_platform_embedding(
                db,
                creator_id=creator_id,
                platform="instagram",
                embedding=embedding,
                content_text=content_text,
            )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"inline embedding failed for @{candidate.username}: {e}"
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
            f"discovery_request_id tag failed for @{candidate.username}: {e}"
        )

    _bump_counter(db, request_id, "candidates_scraped")
    return creator_id


def stage_parallel_scrape_ig(
    db: Client,
    request_id: str,
    brand_id: str,
    survivors: list[IgCandidate],
    *,
    parallelism: int = 10,
    num_posts: int = 5,
    num_reels: int = 10,
) -> list[str]:
    """Per-creator deep scrape in parallel via ThreadPoolExecutor.

    Conservative default parallelism (10) — Apify Instagram actors are
    rate-limited per token and a few of the calls (`scrape_profiles`,
    `scrape_posts_discovery`, `scrape_reels_discovery`) run sequentially
    inside `build_creator_intelligence_profile`, so the effective
    concurrency upstream of Apify is ~3x this number. 10 keeps us
    comfortably under Apify's typical 30 req/s cap.
    """
    _set_status(db, request_id, status="scraping")

    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")

    if not gemini_api_key or not openai_api_key:
        msg = "GEMINI_API_KEY and OPENAI_API_KEY required for IG discovery"
        _set_status(db, request_id, status="failed", error_text=msg)
        raise RuntimeError(msg)

    creator_ids: list[str] = []
    errs = 0
    parallelism = max(1, min(parallelism, len(survivors) or 1))

    with ThreadPoolExecutor(max_workers=parallelism) as exe:
        futures = {
            exe.submit(
                _scrape_one_ig_creator,
                c,
                db=db,
                request_id=request_id,
                brand_id=brand_id,
                gemini_api_key=gemini_api_key,
                openai_api_key=openai_api_key,
                num_posts=num_posts,
                num_reels=num_reels,
            ): c
            for c in survivors
        }
        for fut in as_completed(futures):
            try:
                cid = fut.result()
            except Exception as e:  # noqa: BLE001
                logger.error(f"per-creator IG worker error: {e}")
                cid = None
            if cid:
                creator_ids.append(cid)
            else:
                errs += 1

    logger.info(
        "stage_parallel_scrape_ig: %d OK, %d failed out of %d",
        len(creator_ids),
        errs,
        len(survivors),
    )
    return creator_ids


def stage_brand_match_ig(
    db: Client,
    request_id: str,
    brand_id: str,
    creator_ids: list[str],
) -> dict:
    """Trigger brand-match scoring via the shared compute-batch endpoint.

    Identical to the YT-side stage — the matching engine doesn't care
    about platform; it scores any creator_id present in the leaderboard
    MV. Kept as a thin alias here for symmetry with YT's stage_brand_match.
    """
    return stage_brand_match(db, request_id, brand_id, creator_ids)


def resolve_existing_ig_creator_ids(
    db: Client, usernames: Iterable[str]
) -> list[str]:
    """Map a set of IG usernames to creators.id via creator_social_profiles.

    Mirrors the YT-side resolver but keyed on `handle` instead of
    `platform_user_id`. Used by app.py to tag pre-existing creators
    with the discovery_request_id so they surface in the result set.
    """
    handles = [u.strip().lower() for u in usernames if u and u.strip()]
    out: list[str] = []
    if not handles:
        return out
    for i in range(0, len(handles), 100):
        batch = handles[i : i + 100]
        try:
            res = (
                db.table("creator_social_profiles")
                .select("creator_id, handle")
                .eq("platform", "instagram")
                .in_("handle", batch)
                .execute()
            )
        except Exception:
            continue
        for row in res.data or []:
            cid = row.get("creator_id")
            if cid:
                out.append(cid)
    return out


# Re-export the platform-agnostic terminal-state helpers so app.py can
# import everything from one place.
__all__ = [
    "stage_search_ig",
    "stage_filter_ig",
    "stage_parallel_scrape_ig",
    "stage_brand_match_ig",
    "stage_complete",
    "stage_failed",
    "resolve_existing_ig_creator_ids",
]
