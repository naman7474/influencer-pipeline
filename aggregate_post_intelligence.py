"""Roll per-post intelligence rows up into creator-level distributions + medians.

Pure functions, no I/O. Consumes the list of per-post payloads produced by
``pipeline.llm_post.analyze_posts_batch`` (plus optional per-item engagement)
and produces the row written to ``creator_intelligence_distributions`` — the
distribution buckets that drive the profile pie/bar charts and the
median-emphasised metrics the SOP cares about ("stable medians > one-off
virality").

Design note — AUGMENT, not replace: the per-post path runs ALONGSIDE the
existing creator-level LLM call. The legacy caption/transcript/audience tables
stay populated by that call (and keep feeding the leaderboard MV + matching
engine + the language pies), so this aggregator only needs to emit the NEW
per-post dimensions. Defaulted/failed payloads contribute nothing (their fields
are None) and are excluded from both distributions and medians.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Optional


# Music-only / no-verbal-hook posts use this hook_style; excluded from the
# hook-style pie so it doesn't drown out real stylistic signal.
_HOOK_STYLE_EXCLUDE = {"music"}


def _is_real(payload: dict) -> bool:
    """A payload counts toward aggregation only if it wasn't a defaulted stub."""
    return not payload.get("_defaulted")


def _distribution(
    payloads: list[dict],
    field: str,
    *,
    exclude: Optional[set[str]] = None,
) -> list[dict]:
    """Count a categorical field across payloads → sorted [{label,count,pct}]."""
    exclude = exclude or set()
    counts: Counter[str] = Counter()
    for p in payloads:
        v = p.get(field)
        if v is None:
            continue
        label = str(v).strip()
        if not label or label.lower() in {"none", "null"} or label in exclude:
            continue
        counts[label] += 1
    total = sum(counts.values())
    if total == 0:
        return []
    return [
        {"label": label, "count": count, "pct": round(count * 100 / total, 1)}
        for label, count in counts.most_common()
    ]


def _comment_distribution(payloads: list[dict], field: str) -> list[dict]:
    """Count a categorical field nested under comment_classification."""
    counts: Counter[str] = Counter()
    for p in payloads:
        cc = p.get("comment_classification") or {}
        v = cc.get(field)
        if v is None:
            continue
        label = str(v).strip()
        if not label or label.lower() in {"none", "null"}:
            continue
        counts[label] += 1
    total = sum(counts.values())
    if total == 0:
        return []
    return [
        {"label": label, "count": count, "pct": round(count * 100 / total, 1)}
        for label, count in counts.most_common()
    ]


def _comment_field_values(payloads: list[dict], field: str) -> list[float]:
    out: list[float] = []
    for p in payloads:
        cc = p.get("comment_classification") or {}
        v = cc.get(field)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _median(values: list[float]) -> Optional[float]:
    clean = [v for v in values if isinstance(v, (int, float))]
    if not clean:
        return None
    return round(statistics.median(clean), 4)


def aggregate_posts(
    post_payloads: list[dict],
    engagement_by_item: Optional[dict[str, dict]] = None,
) -> dict:
    """Aggregate per-post payloads into a creator_intelligence_distributions row.

    Args:
        post_payloads: list of per-post dicts from ``analyze_posts_batch``
            (each a PostIntelligencePayload dump + item_id, possibly defaulted).
        engagement_by_item: optional {item_id: {engagement_rate: float, ...}}
            from the scraped rows — used for the engagement-rate median.

    Returns a dict shaped for ``upsert_creator_distributions`` (the jsonb
    bucket arrays + median columns + posts_analyzed). Never raises on empty.
    """
    engagement_by_item = engagement_by_item or {}
    real = [p for p in post_payloads if _is_real(p)]

    # Distributions (only real payloads contribute).
    intent_dist = _distribution(real, "post_intent")
    pillar_dist = _distribution(real, "content_pillar")
    hook_dist = _distribution(real, "hook_style", exclude=_HOOK_STYLE_EXCLUDE)
    cta_dist = _distribution(real, "cta_type")
    orientation_dist = _distribution(real, "content_orientation")
    trigger_dist = _distribution(real, "emotional_trigger")
    audience_intent_dist = _comment_distribution(real, "audience_intent")

    # Medians (exclude nulls).
    median_hook = _median([
        p["hook_quality"] for p in real
        if isinstance(p.get("hook_quality"), (int, float))
    ])
    median_discussion = _median(_comment_field_values(real, "discussion_pct"))
    median_sentiment = _median(_comment_field_values(real, "sentiment_score"))

    eng_rates: list[float] = []
    for p in real:
        eng = engagement_by_item.get(str(p.get("item_id") or ""))
        if eng and isinstance(eng.get("engagement_rate"), (int, float)):
            eng_rates.append(float(eng["engagement_rate"]))
    median_engagement = _median(eng_rates)

    return {
        "intent_distribution": intent_dist,
        "pillar_distribution": pillar_dist,
        "hook_style_distribution": hook_dist,
        "cta_distribution": cta_dist,
        "orientation_distribution": orientation_dist,
        "dominant_orientation": orientation_dist[0]["label"] if orientation_dist else None,
        "emotional_trigger_distribution": trigger_dist,
        "audience_intent_distribution": audience_intent_dist,
        "median_hook_quality": median_hook,
        "median_engagement_rate": median_engagement,
        "median_discussion_pct": median_discussion,
        "median_sentiment_score": median_sentiment,
        "posts_analyzed": len(real),
    }


def build_post_rows(
    owner_id: str,
    platform: str,
    post_payloads: list[dict],
    item_meta: Optional[dict[str, dict]] = None,
    owner_col: str = "creator_id",
) -> list[dict]:
    """Build post_intelligence rows from per-post payloads + scraped metadata.

    ``owner_col`` is "creator_id" (default) or "brand_id" — lets the SAME
    per-video pipeline store a brand's own content under brand_id (migration
    083). ``item_meta`` maps item_id -> {content_type, views, likes,
    comments_count, engagement_rate, has_transcript, comment_sample_size}.
    Missing metadata is fine. One row per payload.
    """
    item_meta = item_meta or {}
    rows: list[dict] = []
    for p in post_payloads:
        iid = str(p.get("item_id") or "").strip()
        if not iid:
            continue
        cc = p.get("comment_classification") or {}
        demo = p.get("demographics_signal") or {}
        meta = item_meta.get(iid, {})
        rows.append({
            owner_col: owner_id,
            "platform": platform,
            "item_id": iid,
            "content_type": meta.get("content_type"),
            "post_intent": p.get("post_intent"),
            "content_pillar": p.get("content_pillar"),
            "hook_style": p.get("hook_style"),
            "hook_quality": p.get("hook_quality"),
            "emotional_trigger": p.get("emotional_trigger"),
            "cta_type": p.get("cta_type"),
            "content_orientation": p.get("content_orientation"),
            "comment_class_emoji_pct": cc.get("emoji_only_pct"),
            "comment_class_link_pct": cc.get("link_trigger_pct"),
            "comment_class_discussion_pct": cc.get("discussion_pct"),
            "discussion_quality": cc.get("discussion_quality"),
            "audience_intent": cc.get("audience_intent"),
            "audience_sentiment_score": cc.get("sentiment_score"),
            "views": meta.get("views"),
            "likes": meta.get("likes"),
            "comments_count": meta.get("comments_count"),
            "engagement_rate": meta.get("engagement_rate"),
            "demo_age_signal": demo.get("estimated_age_group"),
            "demo_gender_signal": demo.get("estimated_gender_skew"),
            "interest_signals": demo.get("interest_signals") or [],
            "raw_llm_response": p,
            "data_quality": {
                "was_defaulted": bool(p.get("_defaulted")),
                "default_reason": p.get("_default_reason"),
            },
            "has_transcript": bool(meta.get("has_transcript")),
            "comment_sample_size": meta.get("comment_sample_size", 0),
            "prompt_version": p.get("_prompt_version"),
        })
    return rows
