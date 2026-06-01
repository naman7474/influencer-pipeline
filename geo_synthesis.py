"""Synthetic brand_shopify_geo rows for brands without Shopify.

When a brand hasn't connected Shopify, brand_shopify_geo is empty for that
brand → v_brand_geo_gaps returns nothing → matching engine's
computeAudienceGeo floors at 0.3 (web/src/lib/matching/engine.ts:1167),
which caps the composite match score around 35-45% even for a perfectly
niche-aligned creator.

We synthesize state-level rows from the onboarding signals we DO have
(shipping_zones, target_regions) and the brand's IG audience profile,
tagging them source='synthetic'. Real Shopify data takes precedence in
v_brand_geo_gaps if it ever arrives (migration 20260502_brand_synthetic_geo).

The state taxonomy mirrors STATE_ZONE in web/src/lib/geo/india.ts. Both
must stay in sync — the matcher uses the TS table to resolve creator
audience to state, then joins against these brand_shopify_geo state slugs.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# Mirror of STATE_ZONE keys in web/src/lib/geo/india.ts.
INDIAN_STATES: tuple[str, ...] = (
    # North
    "delhi",
    "uttar pradesh",
    "haryana",
    "punjab",
    "rajasthan",
    "himachal pradesh",
    "uttarakhand",
    "jammu and kashmir",
    "ladakh",
    "chandigarh",
    # South
    "tamil nadu",
    "karnataka",
    "kerala",
    "andhra pradesh",
    "telangana",
    "puducherry",
    # East
    "west bengal",
    "odisha",
    "bihar",
    "jharkhand",
    "assam",
    "meghalaya",
    "manipur",
    "mizoram",
    "nagaland",
    "tripura",
    "arunachal pradesh",
    "sikkim",
    # West
    "maharashtra",
    "gujarat",
    "goa",
    "madhya pradesh",
    "chhattisgarh",
)

# Common city → state backtrack (subset of CITY_ZONE in india.ts).
_CITY_STATE: dict[str, str] = {
    "delhi": "delhi",
    "new delhi": "delhi",
    "noida": "uttar pradesh",
    "gurgaon": "haryana",
    "gurugram": "haryana",
    "ghaziabad": "uttar pradesh",
    "lucknow": "uttar pradesh",
    "jaipur": "rajasthan",
    "chandigarh": "chandigarh",
    "amritsar": "punjab",
    "ludhiana": "punjab",
    "bangalore": "karnataka",
    "bengaluru": "karnataka",
    "chennai": "tamil nadu",
    "hyderabad": "telangana",
    "kochi": "kerala",
    "thiruvananthapuram": "kerala",
    "coimbatore": "tamil nadu",
    "mumbai": "maharashtra",
    "pune": "maharashtra",
    "nashik": "maharashtra",
    "nagpur": "maharashtra",
    "ahmedabad": "gujarat",
    "surat": "gujarat",
    "vadodara": "gujarat",
    "indore": "madhya pradesh",
    "bhopal": "madhya pradesh",
    "kolkata": "west bengal",
    "patna": "bihar",
    "ranchi": "jharkhand",
    "bhubaneswar": "odisha",
    "guwahati": "assam",
}

# Population-weight floor for synthetic rows. Real shopify rows often
# exceed 0.2 in big states; we under-weight synthetic so it produces
# meaningful but not overconfident scores.
_DEFAULT_POP_WEIGHT = 0.10
_DEFAULT_GAP_SCORE = 0.5  # ∈ [0,1]; engine.ts:1088 weights as 0.6 + g*0.4 = 0.8.


def resolve_state(location: Optional[str]) -> Optional[str]:
    """Return canonical lowercase state slug or None. Mirrors india.ts::resolveState."""
    if not location:
        return None
    key = location.strip().lower()
    if not key:
        return None
    if key in INDIAN_STATES:
        return key
    if key in _CITY_STATE:
        return _CITY_STATE[key]
    for state in INDIAN_STATES:
        if state in key:
            return state
    for city, state in _CITY_STATE.items():
        if city in key:
            return state
    return None


def _pan_india_states() -> list[str]:
    """States to seed when shipping_zones contains 'All India' or similar."""
    return [
        "maharashtra", "karnataka", "tamil nadu", "delhi", "telangana",
        "gujarat", "uttar pradesh", "west bengal", "rajasthan", "kerala",
        "haryana", "punjab", "madhya pradesh", "andhra pradesh", "odisha",
    ]


def derive_synthetic_geo_rows(
    *,
    brand_id: str,
    shipping_zones: Optional[Iterable[str]],
    target_regions: Optional[Iterable[str]],
    ig_audience_profile: Optional[dict],
) -> list[dict]:
    """
    Build a list of brand_shopify_geo upsert dicts (source='synthetic').

    Strategy:
      1. shipping_zones strings ('Delhi', 'Mumbai', 'All India') → states
      2. target_regions same path
      3. ig_audience_profile.primary_country → if 'IN'/'india', boost
         pan-India seed; otherwise (creator's audience is foreign) don't
         emit India rows since the brand probably ships abroad and we
         don't have a non-India geo system yet.

    Each emitted row sets problem_type='awareness_gap' (the brand has no
    proven sales in this state — every state is technically an awareness
    gap until Shopify says otherwise).
    """
    state_weights: dict[str, float] = {}

    def add(state: Optional[str], weight: float) -> None:
        if not state:
            return
        state_weights[state] = max(state_weights.get(state, 0.0), weight)

    # --- shipping_zones ---
    for raw in shipping_zones or []:
        s = (raw or "").strip().lower()
        if not s:
            continue
        if s in ("all india", "pan india", "india"):
            for st in _pan_india_states():
                add(st, _DEFAULT_POP_WEIGHT)
            break
        add(resolve_state(raw), _DEFAULT_POP_WEIGHT * 1.5)

    # --- target_regions ---
    for raw in target_regions or []:
        add(resolve_state(raw), _DEFAULT_POP_WEIGHT)

    # --- IG audience signal ---
    audience = ig_audience_profile or {}
    primary_country = (audience.get("primary_country") or "").strip().lower()
    if primary_country in ("", "in", "india") and not state_weights:
        # No structured signal at all — at least seed pan-India so the
        # brand isn't stuck with an empty geo table.
        for st in _pan_india_states():
            add(st, _DEFAULT_POP_WEIGHT)

    # --- emit rows ---
    rows: list[dict] = []
    for state, weight in state_weights.items():
        rows.append({
            "brand_id": brand_id,
            "state": state,
            "city": None,
            "country": "IN",
            "sessions": 0,
            "orders": 0,
            "revenue": 0,
            "population_weight": round(weight, 3),
            "gap_score": _DEFAULT_GAP_SCORE,
            "problem_type": "awareness_gap",
            "source": "synthetic",
        })

    logger.info(
        "synthetic geo for brand %s: %d states from shipping_zones=%s target_regions=%s",
        brand_id, len(rows), list(shipping_zones or []), list(target_regions or []),
    )
    return rows
