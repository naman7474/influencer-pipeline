"""
OpenAI embedding helpers for brand + creator content fingerprints.

Uses text-embedding-3-small (1536-dim) to match the existing agent memory
embeddings in `web/src/lib/agent/memory/embeddings.ts`. That alignment is
load-bearing: the HNSW indexes added in 20260415_brand_ig_analysis.sql
were created with vector(1536), so any model change must migrate the
indexes too.
"""

from __future__ import annotations

import logging
from typing import Iterable

import httpx

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
_EMBEDDING_URL = "https://api.openai.com/v1/embeddings"


def _compact_text(text: str, max_chars: int = 7500) -> str:
    """Trim overly long input. Embedding models truncate internally, but we
    keep a cap here so the HTTP payload stays predictable."""
    clean = " ".join(text.split())
    return clean[:max_chars]


def embed_text(text: str, openai_api_key: str, timeout: float = 30.0) -> list[float]:
    """Call OpenAI embeddings API. Returns a 1536-dim vector."""
    payload = {"model": EMBEDDING_MODEL, "input": _compact_text(text)}
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(_EMBEDDING_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    vector = data["data"][0]["embedding"]
    if len(vector) != EMBEDDING_DIM:
        raise ValueError(
            f"Unexpected embedding dim {len(vector)} (wanted {EMBEDDING_DIM})"
        )
    return vector


def build_brand_embedding_input(
    *,
    brand_name: str | None,
    description: str | None,
    industry: str | None,
    brand_values: Iterable[str] | None,
    product_categories: Iterable[str] | None,
    target_audience: str | None,
    ig_content_dna: dict | None,
) -> str:
    """Compose the text blob that represents a brand's content DNA."""
    parts: list[str] = []
    if brand_name:
        parts.append(f"Brand: {brand_name}")
    if industry:
        parts.append(f"Industry: {industry}")
    if description:
        parts.append(f"Description: {description}")
    if product_categories:
        parts.append("Categories: " + ", ".join(product_categories))
    if brand_values:
        parts.append("Values: " + ", ".join(brand_values))
    if target_audience:
        parts.append(f"Target audience: {target_audience}")
    if ig_content_dna:
        niche = ig_content_dna.get("primary_niche")
        if niche:
            parts.append(f"Niche: {niche}")
        topics = ig_content_dna.get("recurring_topics") or []
        if topics:
            parts.append("Topics: " + ", ".join(topics))
        pillars = ig_content_dna.get("content_pillars") or []
        if pillars:
            parts.append("Pillars: " + ", ".join(pillars))
        tone = ig_content_dna.get("primary_tone")
        if tone:
            parts.append(f"Tone: {tone}")
    return "\n".join(parts)


def build_creator_embedding_input(cip: dict) -> str:
    """Compose the text blob that represents a creator's content DNA."""
    parts: list[str] = []
    profile = cip.get("profile") or {}
    handle = profile.get("handle")
    if handle:
        parts.append(f"Creator: @{handle}")
    if profile.get("category"):
        parts.append(f"Category: {profile['category']}")
    if profile.get("bio"):
        parts.append(f"Bio: {profile['bio']}")

    caption = cip.get("caption_intelligence") or {}
    if caption.get("primary_niche"):
        parts.append(f"Niche: {caption['primary_niche']}")
    topics = caption.get("recurring_topics") or []
    if topics:
        parts.append("Topics: " + ", ".join(topics))
    pillars = caption.get("content_pillars") or []
    if pillars:
        parts.append("Pillars: " + ", ".join(pillars))
    if caption.get("primary_tone"):
        parts.append(f"Tone: {caption['primary_tone']}")

    audience = cip.get("audience_intelligence") or {}
    if audience.get("estimated_age_group"):
        parts.append(f"Audience age: {audience['estimated_age_group']}")
    if audience.get("primary_country"):
        parts.append(f"Primary country: {audience['primary_country']}")

    return "\n".join(parts)
