"""
Fraud / audience-quality flag computation (W4).

Each flag returns:
  {
    "code": str,
    "severity": "info" | "warn" | "critical",
    "reason": str,
    "evidence": dict,     # raw values that tripped the rule
  }

Rules are data-driven so thresholds can be calibrated later (W7). For
now they start at industry-reported defaults (HypeAuditor, Modash).

Downstream:
  - pipeline.pipeline.compute_creator_scores attaches the list to
    `cip["fraud_flags"]` and the codes-only list to
    `scores["fraud_flag_codes"]`.
  - The matching engine (W5) excludes creators with `critical` codes
    from default ranking.
"""

from __future__ import annotations

from typing import Any, Callable


# ── Thresholds. Single place to calibrate. ────────────────────────
THRESH_SUSPICIOUS_HIGH_ER = 0.08
THRESH_HIGH_ER_MIN_FOLLOWERS = 10_000

THRESH_DEAD_AUDIENCE_ER = 0.005
THRESH_DEAD_AUDIENCE_MIN_FOLLOWERS = 1_000

THRESH_VIEWS_LIKES_HIGH = 200
THRESH_VIEWS_LIKES_LOW = 5

THRESH_HOUR_CLUSTER_SHARE = 0.35

THRESH_LOW_SUBSTANTIVE_PCT = 0.10
THRESH_LOW_SUBSTANTIVE_AUTH = 0.40

THRESH_ENGAGEMENT_BAIT = 0.70


def compute_fraud_flags(cip: dict) -> list[dict]:
    """Run every rule. Returns the full flag list (possibly empty)."""
    flags: list[dict] = []
    for rule in _RULES:
        try:
            result = rule(cip)
        except Exception as e:
            # Defensive: never let a fraud rule crash the pipeline.
            result = None
            _ = e
        if result:
            flags.append(result)
    return flags


def flag_codes(flags: list[dict]) -> list[str]:
    return [f["code"] for f in flags]


def has_critical(flags: list[dict]) -> bool:
    return any(f.get("severity") == "critical" for f in flags)


# ── Rule implementations ──────────────────────────────────────────

def _rule_suspicious_high_er(cip: dict) -> dict | None:
    followers = (cip.get("profile") or {}).get("followers") or 0
    avg_er = (cip.get("posts") or {}).get("avg_engagement_rate")
    if avg_er is None:
        return None
    if (
        avg_er > THRESH_SUSPICIOUS_HIGH_ER
        and followers > THRESH_HIGH_ER_MIN_FOLLOWERS
    ):
        return {
            "code": "suspicious_high_er",
            "severity": "warn",
            "reason": (
                f"Avg ER {avg_er:.3%} exceeds the {THRESH_SUSPICIOUS_HIGH_ER:.0%} "
                f"threshold for creators with > {THRESH_HIGH_ER_MIN_FOLLOWERS:,} followers"
            ),
            "evidence": {"avg_engagement_rate": avg_er, "followers": followers},
        }
    return None


def _rule_dead_audience(cip: dict) -> dict | None:
    followers = (cip.get("profile") or {}).get("followers") or 0
    avg_er = (cip.get("posts") or {}).get("avg_engagement_rate")
    if avg_er is None:
        return None
    if (
        avg_er < THRESH_DEAD_AUDIENCE_ER
        and followers > THRESH_DEAD_AUDIENCE_MIN_FOLLOWERS
    ):
        return {
            "code": "dead_audience",
            "severity": "critical",
            "reason": (
                f"Avg ER {avg_er:.3%} below {THRESH_DEAD_AUDIENCE_ER:.1%} "
                f"on a {followers:,}-follower account — audience is likely bought/inactive"
            ),
            "evidence": {"avg_engagement_rate": avg_er, "followers": followers},
        }
    return None


def _rule_views_likes_anomaly(cip: dict) -> dict | None:
    ratio = (cip.get("reels") or {}).get("avg_views_to_likes_ratio")
    if ratio is None:
        return None
    if ratio > THRESH_VIEWS_LIKES_HIGH or ratio < THRESH_VIEWS_LIKES_LOW:
        return {
            "code": "views_likes_anomaly",
            "severity": "warn",
            "reason": (
                f"Views-to-likes ratio {ratio:.1f} outside "
                f"[{THRESH_VIEWS_LIKES_LOW}, {THRESH_VIEWS_LIKES_HIGH}] "
                "— may indicate inauthentic distribution or vote manipulation"
            ),
            "evidence": {"avg_views_to_likes_ratio": ratio},
        }
    return None


def _rule_comment_hour_clustering(cip: dict) -> dict | None:
    dist = (cip.get("comments") or {}).get("comment_hour_distribution_utc")
    if not dist or not isinstance(dist, dict):
        return None
    total = 0.0
    peak_hour = None
    peak_share = 0.0
    # Normalize values to a share if they look like counts.
    try:
        numeric = {k: float(v) for k, v in dist.items()}
    except (TypeError, ValueError):
        return None
    total = sum(numeric.values())
    if total <= 0:
        return None
    for hour, raw in numeric.items():
        share = raw / total
        if share > peak_share:
            peak_share = share
            peak_hour = hour
    if peak_share > THRESH_HOUR_CLUSTER_SHARE:
        return {
            "code": "comment_hour_clustering",
            "severity": "warn",
            "reason": (
                f"{peak_share:.0%} of comments posted in a single UTC hour "
                f"({peak_hour}) — inconsistent with organic global audience"
            ),
            "evidence": {"peak_hour_utc": peak_hour, "peak_share": round(peak_share, 3)},
        }
    return None


def _rule_low_substantive_comments(cip: dict) -> dict | None:
    auth = (cip.get("audience_intelligence") or {}).get("audience_authenticity") or {}
    sub_pct = auth.get("substantive_comment_percentage")
    auth_score = auth.get("authenticity_score")
    if sub_pct is None or auth_score is None:
        return None
    if (
        sub_pct < THRESH_LOW_SUBSTANTIVE_PCT
        and auth_score < THRESH_LOW_SUBSTANTIVE_AUTH
    ):
        return {
            "code": "low_substantive_comments",
            "severity": "info",
            "reason": (
                f"Only {sub_pct:.0%} of comments are substantive and "
                f"authenticity score is {auth_score:.2f}"
            ),
            "evidence": {
                "substantive_comment_percentage": sub_pct,
                "authenticity_score": auth_score,
            },
        }
    return None


def _rule_engagement_bait_high(cip: dict) -> dict | None:
    signals = (cip.get("caption_intelligence") or {}).get("authenticity_signals") or {}
    bait = signals.get("engagement_bait_score")
    if bait is None:
        return None
    if bait > THRESH_ENGAGEMENT_BAIT:
        return {
            "code": "engagement_bait_high",
            "severity": "warn",
            "reason": (
                f"Engagement-bait score {bait:.2f} exceeds "
                f"{THRESH_ENGAGEMENT_BAIT:.2f} — captions lean on manipulative CTAs"
            ),
            "evidence": {"engagement_bait_score": bait},
        }
    return None


_RULES: list[Callable[[dict], Any]] = [
    _rule_suspicious_high_er,
    _rule_dead_audience,
    _rule_views_likes_anomaly,
    _rule_comment_hour_clustering,
    _rule_low_substantive_comments,
    _rule_engagement_bait_high,
]
