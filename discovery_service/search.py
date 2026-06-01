"""YouTube keyword search → de-duplicated channel-ID list.

Combines channel-search and video-search results so the discovery has good
coverage of both already-popular channels (channel-search) and long-tail
creators whose individual uploads match (video-search). Each `search.list`
call costs 100 quota units; the multi-key pool absorbs the cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from pipeline.youtube.api_pool import YouTubeAPIPool

logger = logging.getLogger(__name__)


@dataclass
class CandidateChannel:
    """One row out of the keyword-search merge stage.

    `source` is which lane contributed the channel first; useful for
    logging the channel-vs-video balance after dedup.

    `subscriber_count` is populated lazily in stage_filter via a cheap
    batch `channels.list` call before deep-scrape. Used for tier /
    min-max-follower filtering so we don't burn 30s of compute on a
    creator whose subscriber count puts them outside the user's filter.
    """

    channel_id: str
    title: str | None = None
    description: str | None = None
    thumbnail: str | None = None
    channel_url: str | None = None
    source: str = "channel"  # 'channel' | 'video'
    subscriber_count: int | None = None


def search_keyword(
    query: str,
    pool: YouTubeAPIPool,
    *,
    channel_max: int = 200,
    video_max: int = 200,
    region_code: str | None = None,
) -> list[CandidateChannel]:
    """Search YouTube for `query`, merge channel + video lanes, dedup.

    Returns up to `channel_max + video_max` candidates worst-case, but in
    practice ~30–60% dedup means ~250–300 unique channels for a 200/200
    split. Order roughly preserves search-relevance rank, channels first
    then video-discovered channels appended.

    Pure function — no DB writes. The caller is responsible for inserting
    `creators` rows once survivors are filtered.

    Quota cost: ~(channel_max / 50) * 100 + (video_max / 50) * 100 units
    across all keys in the pool. For 200/200 → ~1600 units (≈1.3% of
    daily 120K headroom with 12 keys).
    """
    if not query.strip():
        return []

    # ── Lane 1: channel-search ──
    try:
        chan_rows = pool.search_keyword_channels(
            query,
            max_results=channel_max,
            region_code=region_code,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"channel-search failed for '{query}': {e}")
        chan_rows = []

    # ── Lane 2: video-search → unique channel IDs ──
    try:
        vid_channel_ids = pool.search_keyword_video_channels(
            query,
            max_videos=video_max,
            region_code=region_code,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"video-search failed for '{query}': {e}")
        vid_channel_ids = []

    # ── Merge, channel-search first to preserve its ranking ──
    seen: set[str] = set()
    out: list[CandidateChannel] = []

    for row in chan_rows:
        cid = row.get("channel_id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(
            CandidateChannel(
                channel_id=cid,
                title=row.get("title"),
                description=row.get("description"),
                thumbnail=row.get("thumbnail"),
                channel_url=row.get("channel_url"),
                source="channel",
            )
        )

    for cid in vid_channel_ids:
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(
            CandidateChannel(
                channel_id=cid,
                channel_url=f"https://www.youtube.com/channel/{cid}",
                source="video",
            )
        )

    logger.info(
        f"search_keyword '{query}': channel_lane={len(chan_rows)} "
        f"video_lane={len(vid_channel_ids)} after_dedup={len(out)}"
    )
    return out


def filter_candidates(
    candidates: Iterable[CandidateChannel],
    *,
    competitor_substrings: list[str],
    existing_channel_ids: set[str],
) -> list[CandidateChannel]:
    """Drop candidates that are already in DB or match a competitor name.

    `competitor_substrings` is the current brand's `competitor_brands`
    array, lowercased + trimmed. A match against the candidate's `title`
    (case-insensitive substring) excludes the channel.

    `existing_channel_ids` skips creators we've already scraped — the
    Modal call still records them as part of the discovery_request's
    creator pool (via the per-creator dedup pass in stages.py), but they
    don't get re-scraped.
    """
    out: list[CandidateChannel] = []
    for c in candidates:
        if c.channel_id in existing_channel_ids:
            continue
        title_lower = (c.title or "").lower()
        if any(s and s in title_lower for s in competitor_substrings):
            continue
        out.append(c)
    return out
