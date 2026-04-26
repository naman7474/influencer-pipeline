"""Normalize any YouTube URL / handle into a canonical (channel_id, handle, url) tuple.

The rest of the pipeline keys creators by channel_id, so ingestion needs to
resolve whatever the user pastes — `@handle`, `youtube.com/c/custom`,
`youtu.be/<vid>`, etc. — into a `UC…` channel id.

Flow:
  1. Parse the URL / handle locally; if it already contains a `UC…` id, done.
  2. Otherwise hit the Data API (channels.list?forHandle=) — 1 quota unit.
  3. If the API isn't configured, return the handle form and let Bright Data
     fill in the channel_id later (its channel dataset returns the id).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from pipeline.youtube.youtube_api import YouTubeAPIClient

_CHANNEL_ID_RE = re.compile(r"(UC[0-9A-Za-z_-]{22})")
_HANDLE_URL_RE = re.compile(r"youtube\.com/@([0-9A-Za-z_.\-]+)", re.I)
_CUSTOM_URL_RE = re.compile(r"youtube\.com/c/([0-9A-Za-z_.\-]+)", re.I)
_USER_URL_RE = re.compile(r"youtube\.com/user/([0-9A-Za-z_.\-]+)", re.I)


@dataclass
class ResolvedChannel:
    channel_id: Optional[str]   # UCxxxxxxxxxxxxxxxxxxxxxx — canonical
    handle: Optional[str]       # @-stripped, lowercase
    url: str                    # canonical URL for Bright Data scrape


def resolve(raw: str, api: Optional[YouTubeAPIClient] = None) -> ResolvedChannel:
    """Resolve any user-pasted YT input to a ResolvedChannel.

    Never raises — callers should check `channel_id is None` and decide
    whether that's fatal for their flow.
    """
    if not raw:
        return ResolvedChannel(None, None, "")

    raw = raw.strip()

    # Bare `@handle` or `handle`
    if raw.startswith("@") or not raw.startswith("http"):
        handle = raw.lstrip("@")
        channel_id = _resolve_handle(handle, api)
        return ResolvedChannel(
            channel_id=channel_id,
            handle=handle.lower(),
            url=(
                f"https://www.youtube.com/channel/{channel_id}"
                if channel_id
                else f"https://www.youtube.com/@{handle}"
            ),
        )

    # URL forms
    match = _CHANNEL_ID_RE.search(raw)
    if match:
        channel_id = match.group(1)
        return ResolvedChannel(
            channel_id=channel_id,
            handle=None,
            url=f"https://www.youtube.com/channel/{channel_id}",
        )

    match = _HANDLE_URL_RE.search(raw)
    if match:
        handle = match.group(1)
        channel_id = _resolve_handle(handle, api)
        return ResolvedChannel(
            channel_id=channel_id,
            handle=handle.lower(),
            url=(
                f"https://www.youtube.com/channel/{channel_id}"
                if channel_id
                else f"https://www.youtube.com/@{handle}"
            ),
        )

    # Custom URL (/c/name) and legacy /user/name — both need the search
    # endpoint to resolve, which is 100 units. Skip the API and let
    # Bright Data's channel scraper resolve it on its side.
    match = _CUSTOM_URL_RE.search(raw) or _USER_URL_RE.search(raw)
    if match:
        return ResolvedChannel(
            channel_id=None,
            handle=match.group(1).lower(),
            url=raw,
        )

    return ResolvedChannel(channel_id=None, handle=None, url=raw)


def _resolve_handle(
    handle: str, api: Optional[YouTubeAPIClient]
) -> Optional[str]:
    if not handle:
        return None
    if api is None:
        api = YouTubeAPIClient()
    if not api.available:
        return None
    return api.resolve_handle_to_channel_id(handle)
