"""Instagram keyword search → de-duplicated candidate username list.

Calls the `patient_discovery/instagram-search-users` Apify actor, which
takes a free-text query and returns rich user profile metadata
(username, full name, biography, follower count, verified flag, avatar
URL). The discovery service downstream feeds the survivors through the
existing IG pipeline (`pipeline.apify_instagram_bundle.fetch`).

Actor reference: <https://apify.com/patient_discovery/instagram-search-users>

The actor id can be overridden via env var
`APIFY_ACTOR_IG_SEARCH_USERS` (default:
`patient_discovery/instagram-search-users`) so we can swap to a
different search actor without a code deploy.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, Optional

from pipeline.apify_client import ApifyClient, make_default_client

logger = logging.getLogger(__name__)


DEFAULT_ACTOR_ID = "patient_discovery/instagram-search-users"


@dataclass
class IgCandidate:
    """One result from the IG keyword search.

    Populated from the actor's dataset rows; downstream stages augment
    with deep-scrape data (`pipeline.apify_instagram_bundle.fetch`) and
    inferred fields (`pipeline.llm.evaluate_creator`).
    """

    username: str
    full_name: Optional[str] = None
    biography: Optional[str] = None
    follower_count: Optional[int] = None
    is_verified: bool = False
    profile_pic_url: Optional[str] = None
    profile_url: Optional[str] = None


def search_ig_users(
    query: str,
    *,
    max_results: int = 100,
    client: Optional[ApifyClient] = None,
    actor_id: Optional[str] = None,
) -> list[IgCandidate]:
    """Run the Instagram user-search Apify actor and return candidates.

    Synchronous: blocks until the Apify run finishes. Typical wall-clock
    is 30-90s. Dedups results by username case-insensitively (the actor
    sometimes returns a creator twice when their handle matches both
    primary and alias indices).

    `query` is passed verbatim. Quoting / escaping is the actor's
    responsibility.
    """
    if not query.strip():
        return []

    actor = actor_id or os.environ.get(
        "APIFY_ACTOR_IG_SEARCH_USERS", DEFAULT_ACTOR_ID
    )
    apify = client or make_default_client()

    # Input schema mirrors the actor's documented fields:
    #   - search: free-text query
    #   - searchType: "user" → user-search lane (vs "place" / "hashtag")
    #   - resultsLimit: server-side cap
    # The actor accepts a dict at the top level (not a list).
    payload: dict = {
        "search": query.strip(),
        "searchType": "user",
        "resultsLimit": max(1, min(500, int(max_results))),
    }

    try:
        items = apify.trigger_and_wait(actor, payload)
    except Exception as e:  # noqa: BLE001
        logger.error(
            "patient_discovery/instagram-search-users failed for "
            "'%s': %s",
            query,
            e,
        )
        return []

    return _dedup_candidates(_translate_items(items))


def _translate_items(items: Iterable[dict]) -> list[IgCandidate]:
    """Map raw Apify dataset items → IgCandidate list.

    The actor's field names aren't stable across versions; accept the
    common variants so a minor actor update doesn't silently produce
    empty candidates.
    """
    out: list[IgCandidate] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        username = (
            item.get("username")
            or item.get("user_name")
            or item.get("handle")
            or ""
        )
        username = str(username).strip().lstrip("@")
        if not username:
            continue
        out.append(
            IgCandidate(
                username=username,
                full_name=_first_str(item, "full_name", "fullName", "displayName"),
                biography=_first_str(item, "biography", "bio", "about"),
                follower_count=_first_int(
                    item,
                    "follower_count",
                    "followers",
                    "followersCount",
                    "edge_followed_by",
                ),
                is_verified=bool(
                    item.get("is_verified")
                    or item.get("isVerified")
                    or item.get("verified")
                ),
                profile_pic_url=_first_str(
                    item, "profile_pic_url", "profilePicUrl", "avatar"
                ),
                profile_url=_first_str(
                    item, "profile_url", "profileUrl", "url"
                )
                or f"https://www.instagram.com/{username}/",
            )
        )
    return out


def _first_str(d: dict, *keys: str) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _first_int(d: dict, *keys: str) -> Optional[int]:
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        # Some actors return `edge_followed_by` as `{"count": N}`.
        if isinstance(v, dict):
            v = v.get("count")
        try:
            n = int(v)
            return n if n >= 0 else None
        except (TypeError, ValueError):
            continue
    return None


def _dedup_candidates(candidates: list[IgCandidate]) -> list[IgCandidate]:
    seen: set[str] = set()
    out: list[IgCandidate] = []
    for c in candidates:
        key = c.username.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    logger.info(
        "search_ig_users dedup: %d raw → %d unique",
        len(candidates),
        len(out),
    )
    return out


def filter_ig_candidates(
    candidates: Iterable[IgCandidate],
    *,
    competitor_substrings: list[str],
    existing_handles: set[str],
) -> list[IgCandidate]:
    """Drop candidates that are already in DB or look like a competitor.

    Mirrors `pipeline.discovery_service.search.filter_candidates`.
    `competitor_substrings` is the brand's `competitor_brands` array
    lower-cased + `@`-stripped (see Phase 1C / 5G).
    `existing_handles` is the set of `creators.handle` values for IG
    creators we've already scraped — we don't re-scrape them, but the
    caller tags them with the discovery_request_id so they still
    surface in this discovery's result set.
    """
    out: list[IgCandidate] = []
    for c in candidates:
        if c.username.lower() in existing_handles:
            continue
        # Match competitor substring against username AND display name —
        # IG channels often hide the brand in display_name only.
        haystack = " ".join(
            x.lower()
            for x in (c.username, c.full_name or "", c.biography or "")
            if x
        )
        if any(s and s in haystack for s in competitor_substrings):
            continue
        out.append(c)
    return out
