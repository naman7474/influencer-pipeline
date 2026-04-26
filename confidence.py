"""
Coverage tracking for CPI scoring.

CoverageTracker records whether each input feeding a sub-score was
actually present in the CIP or silently defaulted. The resulting
confidence envelope lets downstream consumers distinguish a creator
scored on full data from one whose score is mostly fallback neutrals.

Usage:
    tracker = CoverageTracker()
    value = _pull(cip, ["posts", "avg_engagement_rate"], default=0,
                  key="avg_er", tracker=tracker)
    ...
    envelope = tracker.to_dict(subscore_weights=CPI_WEIGHTS)
    scores["confidence"] = envelope
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# Sub-score keys — must match the keys emitted in compute_creator_scores.
ENGAGEMENT_QUALITY = "engagement_quality"
CONTENT_QUALITY = "content_quality"
AUDIENCE_AUTHENTICITY = "audience_authenticity"
GROWTH_TRAJECTORY = "growth_trajectory"
PROFESSIONALISM = "professionalism"

# CPI weights — kept here so the confidence envelope can mirror the
# composite without importing from pipeline.py and creating a cycle.
CPI_WEIGHTS: dict[str, float] = {
    ENGAGEMENT_QUALITY: 0.30,
    CONTENT_QUALITY: 0.25,
    AUDIENCE_AUTHENTICITY: 0.20,
    GROWTH_TRAJECTORY: 0.15,
    PROFESSIONALISM: 0.10,
}

# Every input key used by compute_creator_scores, grouped by which
# sub-score it feeds. New inputs must be registered here or they will
# not be reflected in coverage math.
SCORE_INPUT_MAP: dict[str, str] = {
    # Engagement Quality
    "avg_er": ENGAGEMENT_QUALITY,
    "reply_rate": ENGAGEMENT_QUALITY,
    "rewatch": ENGAGEMENT_QUALITY,
    "llm_engagement_quality": ENGAGEMENT_QUALITY,
    "conv_depth": ENGAGEMENT_QUALITY,
    "community_feel": ENGAGEMENT_QUALITY,
    # Content Quality
    "hook_quality": CONTENT_QUALITY,
    "consistency_stddev": CONTENT_QUALITY,
    "posts_per_week": CONTENT_QUALITY,
    "storytelling": CONTENT_QUALITY,
    "educational_density": CONTENT_QUALITY,
    # Audience Authenticity
    "authenticity_score": AUDIENCE_AUTHENTICITY,
    "substantive_pct": AUDIENCE_AUTHENTICITY,
    "follower_ratio": AUDIENCE_AUTHENTICITY,
    # Growth Trajectory
    "engagement_trend": GROWTH_TRAJECTORY,
    # Professionalism
    "is_business": PROFESSIONALISM,
    "is_verified": PROFESSIONALISM,
    "has_email": PROFESSIONALISM,
    "audio_quality": PROFESSIONALISM,
}

# Confidence tier thresholds. Single source of truth — migrations
# 033 reference these through a CHECK constraint on the tier column.
TIER_HIGH_MIN = 0.80
TIER_MEDIUM_MIN = 0.50


# ── Platform-specific ER benchmarks ─────────────────────────
# IG benchmark = (likes + comments) / followers; YT benchmark =
# (likes + comments) / views. The denominators differ, so the
# tier cutoffs differ too — YT ratios run ~2–3x higher because
# views < followers for any given piece of content.

IG_ER_BENCHMARKS: dict[str, float] = {
    "nano": 0.06,
    "micro": 0.04,
    "mid": 0.025,
    "macro": 0.015,
    "mega": 0.01,
}

YT_ER_BENCHMARKS: dict[str, float] = {
    "nano": 0.10,
    "micro": 0.07,
    "mid": 0.05,
    "macro": 0.03,
    "mega": 0.02,
}

# Views-per-sub is YT's watch-time proxy — how actively is the sub
# base actually watching? <0.1 = weak channel, >=0.5 = strong.
YT_VIEWS_PER_SUB_STRONG = 0.5
YT_VIEWS_PER_SUB_WEAK = 0.1


class CoverageTracker:
    """Records which scoring inputs were real vs. defaulted."""

    def __init__(self) -> None:
        # key -> "present" | "default"
        self._status: dict[str, str] = {}
        # key -> reason string (only for defaulted keys)
        self._reasons: dict[str, str] = {}
        # dimension -> error string (captions / transcripts / comments)
        self._llm_failures: dict[str, str] = {}
        # dimension -> bool (True when dimension is available and usable)
        self._llm_calls_succeeded: dict[str, bool] = {}
        # arbitrary data-quality flags from upstream scrapers
        self._data_quality_flags: set[str] = set()

    def mark_present(self, key: str) -> None:
        if key not in SCORE_INPUT_MAP:
            logger.warning(
                f"CoverageTracker.mark_present: unknown key {key!r}; "
                "register it in SCORE_INPUT_MAP"
            )
        self._status[key] = "present"
        self._reasons.pop(key, None)

    def mark_default(self, key: str, reason: str = "missing") -> None:
        if key not in SCORE_INPUT_MAP:
            logger.warning(
                f"CoverageTracker.mark_default: unknown key {key!r}; "
                "register it in SCORE_INPUT_MAP"
            )
        # First write wins for presence, but a later explicit default
        # (e.g., llm_failure) should override an earlier "missing".
        prior = self._status.get(key)
        if prior == "present":
            return
        self._status[key] = "default"
        self._reasons[key] = reason

    def mark_llm_failure(self, dimension: str, error: str) -> None:
        """Record that an LLM dimension failed so callers can mark
        every downstream input as defaulted with reason='llm_failure'."""
        self._llm_failures[dimension] = error
        self._llm_calls_succeeded[dimension] = False

    def mark_llm_success(self, dimension: str) -> None:
        self._llm_calls_succeeded.setdefault(dimension, True)

    def add_data_quality_flag(self, flag: str) -> None:
        if flag:
            self._data_quality_flags.add(flag)

    # ── Read helpers ──

    def is_defaulted(self, key: str) -> bool:
        return self._status.get(key) == "default"

    def coverage_for(self, keys: Iterable[str]) -> float:
        """Fraction of keys whose value was actually present."""
        keys = list(keys)
        if not keys:
            return 1.0
        present = sum(
            1 for k in keys if self._status.get(k) == "present"
        )
        return round(present / len(keys), 3)

    def _keys_for_subscore(self, subscore: str) -> list[str]:
        return [k for k, v in SCORE_INPUT_MAP.items() if v == subscore]

    def to_dict(
        self,
        subscore_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Build the confidence envelope attached to scores["confidence"].

        Shape:
          {
            "per_subscore": {<subscore>: {"coverage": 0.83, "present": 5,
                                         "total": 6, "missing": [...]}, ...},
            "overall_coverage": 0.72,
            "tier": "high" | "medium" | "low",
            "missing_inputs": [...],
            "llm_calls_succeeded": {"captions": true, ...},
            "llm_failures": {"captions": "..."},
            "data_quality_flags": [...],
          }
        """
        weights = subscore_weights or CPI_WEIGHTS

        per_subscore: dict[str, dict[str, Any]] = {}
        weighted_sum = 0.0
        weight_total = 0.0
        all_missing: list[str] = []

        for subscore, weight in weights.items():
            keys = self._keys_for_subscore(subscore)
            if not keys:
                continue
            coverage = self.coverage_for(keys)
            missing = [
                k for k in keys
                if self._status.get(k) != "present"
            ]
            per_subscore[subscore] = {
                "coverage": coverage,
                "present": len(keys) - len(missing),
                "total": len(keys),
                "missing": missing,
            }
            weighted_sum += coverage * weight
            weight_total += weight
            all_missing.extend(missing)

        overall = (
            round(weighted_sum / weight_total, 3) if weight_total else 1.0
        )
        tier = _coverage_tier(overall)

        return {
            "per_subscore": per_subscore,
            "overall_coverage": overall,
            "tier": tier,
            "missing_inputs": all_missing,
            "llm_calls_succeeded": dict(self._llm_calls_succeeded),
            "llm_failures": dict(self._llm_failures),
            "data_quality_flags": sorted(self._data_quality_flags),
            "default_reasons": dict(self._reasons),
        }


def _coverage_tier(coverage: float) -> str:
    if coverage >= TIER_HIGH_MIN:
        return "high"
    if coverage >= TIER_MEDIUM_MIN:
        return "medium"
    return "low"


# ── Convenience reader ──

_MISSING = object()


def pull(
    cip: dict,
    path: list[str],
    *,
    default: Any,
    key: str,
    tracker: CoverageTracker,
    present_predicate=None,
) -> Any:
    """
    Walk `path` through `cip`. If the leaf is present (and passes
    `present_predicate` when supplied), record it present and return
    it; otherwise record defaulted and return `default`.

    A predicate lets callers treat sentinel values (e.g.,
    `engagement_trend == "insufficient_data"`) as "defaulted" while
    still letting downstream math see the sentinel.
    """
    node: Any = cip
    for part in path:
        if not isinstance(node, dict):
            node = _MISSING
            break
        node = node.get(part, _MISSING)
        if node is _MISSING:
            break

    if node is _MISSING or node is None:
        tracker.mark_default(key, reason="missing")
        return default

    if present_predicate is not None and not present_predicate(node):
        tracker.mark_default(key, reason="sentinel")
        # Still return the real value; callers map it to a default.
        return node

    tracker.mark_present(key)
    return node
