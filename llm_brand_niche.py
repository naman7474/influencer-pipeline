"""Brand niche + brand_type classification.

After IG content is scraped + caption-analyzed, infer the brand's primary
and secondary niche from the SAME 12-niche enum used for creators
(pipeline.llm_captions.CAPTION_NICHES) AND the brand's business model
(d2c | b2b | services | agency | marketplace | other) used by the
matching engine to pick a weight profile. Both are inferred in a single
LLM call to keep onboarding latency low.

Inputs blend onboarding-time signals (description, product_categories,
target audience, shopify_connected) with IG-derived signals (content_dna
recurring topics + content pillars) so the classifier sees both stated
identity and observed behavior.
"""

from __future__ import annotations

import logging
from typing import Optional

from pipeline.llm_captions import CAPTION_NICHES
from pipeline.llm_client import call_gemini_json

logger = logging.getLogger(__name__)

BRAND_TYPES = ("d2c", "b2b", "services", "agency", "marketplace", "other")

_PROMPT = """You are an influencer-marketing analyst classifying a BRAND.

Output two classifications: primary/secondary niche AND business model.

NICHE — pick ONE primary_niche and OPTIONALLY ONE secondary_niche from:
{niches}

Niche rules:
- primary_niche MUST be in the enum. If no good fit, pick the closest.
- secondary_niche MUST be in the enum or null.
- Lowercase, exact match.
- Decide based on what the brand SELLS and what content they POST, not on
  marketing copy adjectives.
- An EdTech / online learning / coaching brand → "education".
- A finance / fintech / wealth-management brand → "finance".
- A parenting / babycare / kids brand → "parenting".
- A travel / hospitality brand → "travel".
- A gaming / OTT / media brand → "entertainment".
- A wellness / mental health / supplements brand → "health".

BRAND TYPE — pick ONE from: {brand_types}

Brand-type rules:
- "d2c": consumer-facing, sells physical goods directly to end users
  (own website, Shopify-style storefront, retail/CPG/fashion/beauty).
  Cues: shopify_connected=true, product_categories like apparel/jewelry/
  beauty/food/home, retail price points (₹100–₹50,000 typical).
- "b2b": sells to other businesses — enterprise SaaS, APIs, business
  software, B2B services, devtools.
- "services": sells professional services to consumers (clinics, salons,
  fitness studios, consultancies serving individuals).
- "agency": marketing/creative/PR/influencer agency that represents
  OTHER brands.
- "marketplace": a platform aggregating other brands' inventory
  (Amazon-style, food delivery, two-sided platforms).
- "other": only if none of the above clearly applies.

Brand-type cues to weight heavily:
- shopify_connected=true → very strong "d2c" prior, override only with
  contrary evidence.
- product_categories listing physical goods → "d2c".
- description mentioning "platform", "marketplace", "two-sided" → "marketplace".
- description mentioning "for businesses", "SaaS", "API" → "b2b".
- description mentioning "we manage", "we create content for brands" → "agency".

INPUTS:
BRAND NAME: {brand_name}
INDUSTRY (self-reported): {industry}
PRODUCT CATEGORIES (self-reported): {product_categories}
SHOPIFY CONNECTED: {shopify_connected}
TARGET AUDIENCE: {target_audience}
DESCRIPTION: {description}

OBSERVED IG CONTENT (from caption analysis):
- recurring_topics: {recurring_topics}
- content_pillars: {content_pillars}
- primary_tone: {primary_tone}

Respond with ONLY this JSON, no prose:
{{
  "primary_niche": "<one of the niche enum>",
  "secondary_niche": "<one of the niche enum or null>",
  "brand_type": "<one of the brand_type enum>",
  "confidence": 0.0-1.0,
  "reasoning": "1 short sentence covering both decisions"
}}"""


def classify_brand_niche(
    gemini_client,
    *,
    brand_name: Optional[str],
    industry: Optional[str],
    product_categories: Optional[list[str]],
    target_audience: Optional[str],
    description: Optional[str],
    ig_content_dna: Optional[dict],
    shopify_connected: Optional[bool] = None,
) -> dict:
    """
    Returns:
      {"primary_niche": str|None, "secondary_niche": str|None,
       "brand_type": str|None, "confidence": float, "reasoning": str}
    On LLM failure (or if the model returned an out-of-enum value), returns
    None for the offending field(s) — caller decides whether to skip
    persistence or retry. Never raises.
    """
    dna = ig_content_dna or {}
    prompt = _PROMPT.format(
        niches=", ".join(CAPTION_NICHES),
        brand_types=", ".join(BRAND_TYPES),
        brand_name=brand_name or "(unknown)",
        industry=industry or "(unknown)",
        product_categories=", ".join(product_categories or []) or "(none)",
        shopify_connected=(
            "true" if shopify_connected else
            "false" if shopify_connected is False else
            "(unknown)"
        ),
        target_audience=target_audience or "(unknown)",
        description=(description or "(unknown)")[:1500],
        recurring_topics=", ".join(dna.get("recurring_topics") or []) or "(none)",
        content_pillars=", ".join(dna.get("content_pillars") or []) or "(none)",
        primary_tone=dna.get("primary_tone") or "(unknown)",
    )

    result = call_gemini_json(
        gemini_client,
        prompt,
        dimension="brand_niche_classification",
    )

    # Tolerate LLMFailure shape (has "error" key) and out-of-enum values.
    if not isinstance(result, dict) or result.get("error"):
        return {
            "primary_niche": None,
            "secondary_niche": None,
            "brand_type": None,
            "confidence": 0.0,
            "reasoning": (
                result.get("error") if isinstance(result, dict) else "llm_failure"
            ),
        }

    primary = (result.get("primary_niche") or "").strip().lower() or None
    secondary = result.get("secondary_niche")
    secondary = (secondary or "").strip().lower() if secondary else None
    brand_type = (result.get("brand_type") or "").strip().lower() or None

    if primary not in CAPTION_NICHES:
        logger.warning(
            "brand_niche_classification: out-of-enum primary=%r — discarding",
            primary,
        )
        primary = None
    if secondary and secondary not in CAPTION_NICHES:
        secondary = None
    if brand_type not in BRAND_TYPES:
        # If the LLM returned something off-enum but we have a strong shopify
        # signal, fall back to "d2c"; otherwise leave null and let the caller
        # decide. A null brand_type in DB defaults to 'other' via the column
        # default in migration 057.
        logger.warning(
            "brand_niche_classification: out-of-enum brand_type=%r", brand_type,
        )
        brand_type = "d2c" if shopify_connected else None

    return {
        "primary_niche": primary,
        "secondary_niche": secondary,
        "brand_type": brand_type,
        "confidence": float(result.get("confidence") or 0.0),
        "reasoning": (result.get("reasoning") or "")[:300],
    }
