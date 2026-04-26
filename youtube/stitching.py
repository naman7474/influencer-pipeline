"""Cross-platform creator stitching.

After a YT channel is ingested, the `creator_social_profiles.external_links`
column often contains the creator's Instagram handle / URL ("find me on IG:
@foo"). This module extracts those signals and proposes candidate
merges to an existing IG-only `creators` row so the platform sees one
canonical creator with both platform profiles rather than two disconnected
records.

The merge itself is intentionally *not* performed here — stitching produces
candidates with a confidence score. An admin UI (or automatic merge when
confidence is high) does the actual DB update via a follow-up path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

_IG_URL_RE = re.compile(r"(?:https?://)?(?:www\.)?instagram\.com/([A-Za-z0-9_.]+)", re.I)
_TT_URL_RE = re.compile(r"(?:https?://)?(?:www\.)?tiktok\.com/@([A-Za-z0-9_.]+)", re.I)
_YT_URL_RE = re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/@([A-Za-z0-9_.\-]+)", re.I)


@dataclass(frozen=True)
class StitchCandidate:
    """A proposed merge between two (platform, handle) identities."""

    source_platform: str
    source_handle: str
    target_platform: str
    target_handle: str
    confidence: float          # 0..1 — 1.0 = direct link, 0.3 = handle match only
    reason: str                # human-readable, for the admin UI


def extract_handles_from_links(
    external_links: Iterable[dict | str] | None,
) -> dict[str, set[str]]:
    """Scan `external_links` for IG / TikTok / YT mentions.

    Accepts either list-of-{label,url} (Bright Data's typical shape) or a
    flat list of URL strings. Returns `{platform: {handle, ...}}`.
    """
    out: dict[str, set[str]] = {"instagram": set(), "tiktok": set(), "youtube": set()}
    if not external_links:
        return out

    for item in external_links:
        url = item.get("url", "") if isinstance(item, dict) else str(item or "")
        if not url:
            continue
        m = _IG_URL_RE.search(url)
        if m:
            out["instagram"].add(m.group(1).lower().rstrip("/"))
            continue
        m = _TT_URL_RE.search(url)
        if m:
            out["tiktok"].add(m.group(1).lower())
            continue
        m = _YT_URL_RE.search(url)
        if m:
            out["youtube"].add(m.group(1).lower())
    return out


def propose_stitch_candidates(
    source_platform: str,
    source_handle: str,
    source_external_links: Iterable[dict | str] | None,
    *,
    target_platforms: Iterable[str] = ("instagram",),
) -> list[StitchCandidate]:
    """Produce stitch candidates from a YT profile's external links.

    Returns candidates pointing at IG/TikTok targets. Confidence is 1.0 for
    a direct link found in external_links (the creator themselves posted
    their IG URL on their YT about page — high signal).
    """
    handles_by_platform = extract_handles_from_links(source_external_links)
    candidates: list[StitchCandidate] = []
    for target_platform in target_platforms:
        for target_handle in handles_by_platform.get(target_platform, set()):
            candidates.append(
                StitchCandidate(
                    source_platform=source_platform,
                    source_handle=source_handle.lstrip("@").lower(),
                    target_platform=target_platform,
                    target_handle=target_handle,
                    confidence=1.0,
                    reason=(
                        f"{source_platform} profile links to "
                        f"{target_platform}.com/{target_handle}"
                    ),
                )
            )
    return candidates


def propose_handle_match_candidate(
    source_platform: str,
    source_handle: str,
    target_platform: str,
    target_handle: str,
) -> StitchCandidate | None:
    """Propose a low-confidence candidate based on handle string similarity.

    Use only when no direct link was found. Same string (case-insensitive,
    `@`-stripped, `_yt`/`_ig` suffixes tolerated) is a weak-but-useful
    signal. Returns None if the handles don't match.
    """
    if not source_handle or not target_handle:
        return None

    def _normalize(h: str) -> str:
        h = h.lstrip("@").lower()
        for suffix in ("_yt", "_ig", "_tt", "_official", "official"):
            if h.endswith(suffix):
                h = h[: -len(suffix)]
        return h

    if _normalize(source_handle) != _normalize(target_handle):
        return None

    return StitchCandidate(
        source_platform=source_platform,
        source_handle=source_handle.lstrip("@").lower(),
        target_platform=target_platform,
        target_handle=target_handle.lstrip("@").lower(),
        confidence=0.3,
        reason=(
            f"handle string match: @{source_handle} == @{target_handle}"
            " (low-confidence; needs admin review)"
        ),
    )


def find_stitch_candidates_in_db(
    db,
    source_platform: str,
    source_creator_id: str,
) -> list[StitchCandidate]:
    """Query helper: look up source's external_links in the DB and propose
    candidates against existing handles on other platforms.

    Requires a supabase client — minimal wiring so callers can integrate
    this into the YT ingestion path or a scheduled job.
    """
    source_row = (
        db.table("creator_social_profiles")
        .select("platform, handle, external_links")
        .eq("creator_id", source_creator_id)
        .eq("platform", source_platform)
        .limit(1)
        .execute()
    )
    rows = source_row.data or []
    if not rows:
        return []
    row = rows[0]

    linked = extract_handles_from_links(row.get("external_links"))
    candidates: list[StitchCandidate] = []
    for platform, handles in linked.items():
        if platform == source_platform:
            continue
        for handle in handles:
            # Is there already a creator row on the target platform with
            # this handle? If so, propose a merge.
            res = (
                db.table("creator_social_profiles")
                .select("creator_id")
                .eq("platform", platform)
                .eq("handle", handle)
                .limit(1)
                .execute()
            )
            if res.data:
                candidates.append(
                    StitchCandidate(
                        source_platform=source_platform,
                        source_handle=row["handle"],
                        target_platform=platform,
                        target_handle=handle,
                        confidence=1.0,
                        reason=(
                            f"{source_platform} creator's external link resolves "
                            f"to existing {platform} creator @{handle}"
                        ),
                    )
                )
    return candidates
