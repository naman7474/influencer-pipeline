"""Extract collaborator channels mentioned across a YouTube creator's videos.

On Instagram, collaborators are structured data (`coauthor_producers`, tagged
users). YouTube has no equivalent — collabs show up as:
  - @handle mentions in video titles / descriptions
  - /channel/UC… URLs in descriptions ("subscribe to my friend")
  - Featured channels on the About page (Bright Data channel dataset
    sometimes returns these)

This module pulls those signals out of the video + channel records so the
brand-side YT scrape can fan out to the creator's actual collaborators, same
way the IG path already fans out to `coauthor_producers`.
"""

from __future__ import annotations

import re
from collections import Counter

# Valid YT handle: letters, numbers, dots, hyphens, underscores.
# Length 3..30 per YouTube's published rules (reject bare "@1" false
# positives like "@1st").
_HANDLE_MENTION_RE = re.compile(r"@([A-Za-z0-9_.\-]{3,30})")
_CHANNEL_URL_RE = re.compile(
    r"(?:youtube\.com/channel/(UC[0-9A-Za-z_\-]{22}))"
    r"|(?:youtube\.com/@([A-Za-z0-9_.\-]{3,30}))"
)


def extract_mentions_from_text(text: str | None) -> tuple[set[str], set[str]]:
    """Return (handles, channel_ids) mentioned in `text`.

    Handles are lowercased, `@`-stripped. Channel ids are the raw `UC…`.
    """
    handles: set[str] = set()
    channel_ids: set[str] = set()
    if not text:
        return handles, channel_ids

    for m in _HANDLE_MENTION_RE.finditer(text):
        handles.add(m.group(1).lower())

    for m in _CHANNEL_URL_RE.finditer(text):
        cid, handle = m.group(1), m.group(2)
        if cid:
            channel_ids.add(cid)
        if handle:
            handles.add(handle.lower())

    return handles, channel_ids


def extract_collaborators(
    videos: list[dict],
    self_handle: str | None = None,
    self_channel_id: str | None = None,
    min_mentions: int = 2,
) -> dict:
    """Aggregate collaborator signals across all of a creator's videos.

    Args:
        videos: records emitted by `extract_video_metrics` — expects
            `title`, `description` keys.
        self_handle / self_channel_id: exclude self-mentions so we don't
            fanout back to the creator we just scraped.
        min_mentions: require a collaborator to appear in at least this
            many videos before we fan out. Defaults to 2 to filter random
            one-off mentions ("shout out to @fan123"). Callers that want
            every signal can pass 1.

    Returns:
        {
          "handles": [{"handle": "mkbhd", "count": 3}, ...],
          "channel_ids": [{"channel_id": "UC...", "count": 2}, ...],
          "total_videos_scanned": 20,
        }
    """
    handle_counter: Counter[str] = Counter()
    channel_counter: Counter[str] = Counter()
    self_handle_norm = (self_handle or "").lstrip("@").lower()

    for v in videos:
        text = " ".join(
            filter(
                None,
                [v.get("title"), v.get("description")],
            )
        )
        handles, channel_ids = extract_mentions_from_text(text)
        handle_counter.update(handles)
        channel_counter.update(channel_ids)

    # Drop self-mentions
    if self_handle_norm:
        handle_counter.pop(self_handle_norm, None)
    if self_channel_id:
        channel_counter.pop(self_channel_id, None)

    handles_out = [
        {"handle": h, "count": c}
        for h, c in handle_counter.most_common()
        if c >= min_mentions
    ]
    channels_out = [
        {"channel_id": cid, "count": c}
        for cid, c in channel_counter.most_common()
        if c >= min_mentions
    ]

    return {
        "handles": handles_out,
        "channel_ids": channels_out,
        "total_videos_scanned": len(videos),
    }
