"""
Pydantic v2 schemas for Gemini intelligence payloads.

These models sit at the LLM boundary: every LLM call parses its JSON
through one of these schemas before the CIP touches it. The goal is
to give a single source of truth that both Python scorers and the
TypeScript matching engine (via export_json_schema.py → Zod) can
agree on.

Design notes:
- `extra="ignore"` — Gemini is free to emit extra fields; we ignore
  them. Breaking on unexpected keys would cause constant churn.
- Every numeric field that should be a decimal fraction goes through
  `coerce_decimal_percentage` which turns `60` into `0.6`. Gemini
  drifts between percentages and decimals constantly.
- Dimension-level schemas do NOT force every sub-object to be present;
  they fill missing sub-objects with empty instances so downstream
  `.get(...)` chains continue to work exactly as they did in v1.0.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Helpers ──────────────────────────────────────────────────────

def coerce_decimal_percentage(v: Any) -> Optional[float]:
    """Coerce Gemini percentage drift (60 → 0.6) into [0, 1] decimals."""
    if v is None or v == "":
        return None
    try:
        n = float(v)
    except (TypeError, ValueError):
        return None
    if n != n:  # NaN
        return None
    if n > 1.0:
        n = n / 100.0
    if n < 0.0:
        return 0.0
    if n > 1.0:
        return 1.0
    return round(n, 3)


def _coerce_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1", "y")
    return bool(v)


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# ── Shared envelope ───────────────────────────────────────────────

class DataQualityEnvelope(_Base):
    """Matches the data_quality jsonb column on intelligence rows."""

    confidence: float = 0.0
    coverage_percentage: float = 0.0
    was_defaulted: bool = True
    missing_fields: list[str] = Field(default_factory=list)
    sample_size: int = 0
    schema_version: str = "1.0"


# ── Caption payload ──────────────────────────────────────────────

class NicheClassification(_Base):
    primary_niche: Optional[str] = None
    secondary_niche: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None

    @field_validator("confidence", mode="before")
    @classmethod
    def _dec(cls, v): return coerce_decimal_percentage(v)


class ToneProfile(_Base):
    primary_tone: Optional[str] = None
    secondary_tone: Optional[str] = None
    formality_score: Optional[float] = None
    humor_score: Optional[float] = None
    authenticity_feel: Optional[float] = None

    @field_validator(
        "formality_score", "humor_score", "authenticity_feel",
        mode="before",
    )
    @classmethod
    def _dec(cls, v): return coerce_decimal_percentage(v)


class LanguageAnalysis(_Base):
    languages_detected: list[str] = Field(default_factory=list)
    primary_language: Optional[str] = None
    language_mix_percentages: dict[str, float] = Field(default_factory=dict)
    uses_transliteration: bool = False
    script_types: list[str] = Field(default_factory=list)

    @field_validator("language_mix_percentages", mode="before")
    @classmethod
    def _normalize_mix(cls, v):
        if not isinstance(v, dict) or not v:
            return {}
        out = {}
        for k, val in v.items():
            try:
                fv = float(val)
            except (TypeError, ValueError):
                continue
            out[k] = fv
        # If any value > 1.0 we assume percentage input and rescale.
        if any(x > 1.0 for x in out.values()):
            out = {k: round(v / 100.0, 3) for k, v in out.items()}
        return out

    @field_validator("uses_transliteration", mode="before")
    @classmethod
    def _to_bool(cls, v): return _coerce_bool(v) or False


class CtaPatterns(_Base):
    dominant_cta_style: str = "none"
    cta_frequency: Optional[float] = None
    conversion_oriented: bool = False
    cta_examples: list[str] = Field(default_factory=list)

    @field_validator("cta_frequency", mode="before")
    @classmethod
    def _dec(cls, v): return coerce_decimal_percentage(v)

    @field_validator("conversion_oriented", mode="before")
    @classmethod
    def _to_bool(cls, v): return _coerce_bool(v) or False


class BrandMentions(_Base):
    organic_brand_mentions: list[str] = Field(default_factory=list)
    paid_brand_mentions: list[str] = Field(default_factory=list)
    brand_categories: list[str] = Field(default_factory=list)


class ContentThemes(_Base):
    recurring_topics: list[str] = Field(default_factory=list)
    content_pillars: list[str] = Field(default_factory=list)
    seasonal_patterns: Optional[str] = None


class AuthenticitySignals(_Base):
    personal_storytelling_frequency: Optional[float] = None
    vulnerability_openness: Optional[float] = None
    product_mentions_feel_natural: bool = False
    uses_personal_anecdotes: bool = False
    engagement_bait_score: Optional[float] = None

    @field_validator(
        "personal_storytelling_frequency",
        "vulnerability_openness",
        "engagement_bait_score",
        mode="before",
    )
    @classmethod
    def _dec(cls, v): return coerce_decimal_percentage(v)

    @field_validator(
        "product_mentions_feel_natural", "uses_personal_anecdotes",
        mode="before",
    )
    @classmethod
    def _to_bool(cls, v): return _coerce_bool(v) or False


class CaptionIntelligencePayload(_Base):
    niche_classification: NicheClassification = Field(
        default_factory=NicheClassification
    )
    tone_profile: ToneProfile = Field(default_factory=ToneProfile)
    language_analysis: LanguageAnalysis = Field(default_factory=LanguageAnalysis)
    cta_patterns: CtaPatterns = Field(default_factory=CtaPatterns)
    brand_mentions: BrandMentions = Field(default_factory=BrandMentions)
    content_themes: ContentThemes = Field(default_factory=ContentThemes)
    authenticity_signals: AuthenticitySignals = Field(
        default_factory=AuthenticitySignals
    )


# ── Transcript payload ───────────────────────────────────────────

class SpeakingLanguage(_Base):
    primary_spoken_language: Optional[str] = None
    languages_spoken: list[str] = Field(default_factory=list)
    accent_notes: Optional[str] = None
    caption_vs_spoken_mismatch: bool = False

    @field_validator("caption_vs_spoken_mismatch", mode="before")
    @classmethod
    def _to_bool(cls, v): return _coerce_bool(v) or False


class HookItem(_Base):
    post_id: Optional[str] = None
    hook_text: Optional[str] = None
    hook_type: Optional[str] = None
    hook_quality: Optional[float] = None

    @field_validator("hook_quality", mode="before")
    @classmethod
    def _dec(cls, v): return coerce_decimal_percentage(v)


class HookAnalysis(_Base):
    hooks: list[HookItem] = Field(default_factory=list)
    avg_hook_quality: Optional[float] = None
    dominant_hook_style: Optional[str] = None

    @field_validator("avg_hook_quality", mode="before")
    @classmethod
    def _dec(cls, v): return coerce_decimal_percentage(v)


class ContentDepth(_Base):
    avg_word_count_per_reel: Optional[int] = None
    vocabulary_complexity: Optional[str] = None
    educational_density: Optional[float] = None
    storytelling_score: Optional[float] = None
    filler_word_frequency: Optional[float] = None

    @field_validator("avg_word_count_per_reel", mode="before")
    @classmethod
    def _coerce_int(cls, v):
        if v is None or v == "":
            return None
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    @field_validator(
        "educational_density", "storytelling_score", "filler_word_frequency",
        mode="before",
    )
    @classmethod
    def _dec(cls, v): return coerce_decimal_percentage(v)


class AudioProduction(_Base):
    overall_quality_assessment: Optional[str] = "casual"
    uses_background_music: bool = False
    voiceover_vs_oncamera: Optional[str] = None
    pacing: Optional[str] = None

    @field_validator("uses_background_music", mode="before")
    @classmethod
    def _to_bool(cls, v): return _coerce_bool(v) or False


class RegionalSignals(_Base):
    cultural_references: list[str] = Field(default_factory=list)
    local_places_mentioned: list[str] = Field(default_factory=list)
    regional_language_phrases: list[str] = Field(default_factory=list)
    estimated_region: Optional[str] = None


class TranscriptIntelligencePayload(_Base):
    data_quality: Optional[str] = None
    speaking_language: SpeakingLanguage = Field(default_factory=SpeakingLanguage)
    hook_analysis: HookAnalysis = Field(default_factory=HookAnalysis)
    brand_mention_analysis: list[dict] = Field(default_factory=list)
    content_depth: ContentDepth = Field(default_factory=ContentDepth)
    audio_production: AudioProduction = Field(default_factory=AudioProduction)
    regional_signals: RegionalSignals = Field(default_factory=RegionalSignals)


# ── Audience payload ─────────────────────────────────────────────

class AudienceLanguageDistribution(_Base):
    languages: dict[str, float] = Field(default_factory=dict)
    primary_audience_language: Optional[str] = None
    multilingual_audience: bool = False

    @field_validator("languages", mode="before")
    @classmethod
    def _normalize(cls, v):
        if not isinstance(v, dict) or not v:
            return {}
        out: dict[str, float] = {}
        for k, val in v.items():
            try:
                fv = float(val)
            except (TypeError, ValueError):
                continue
            out[k] = fv
        if any(x > 1.0 for x in out.values()):
            out = {k: round(v / 100.0, 3) for k, v in out.items()}
        return out

    @field_validator("multilingual_audience", mode="before")
    @classmethod
    def _to_bool(cls, v): return _coerce_bool(v) or False


class AudienceGeoSplit(_Base):
    domestic_percentage: Optional[float] = None
    primary_country: Optional[str] = None

    @field_validator("domestic_percentage", mode="before")
    @classmethod
    def _dec(cls, v): return coerce_decimal_percentage(v)


class AudienceGeographyInference(_Base):
    regions: list[str] = Field(default_factory=list)
    domestic_vs_international_split: AudienceGeoSplit = Field(
        default_factory=AudienceGeoSplit
    )

    @field_validator("regions", mode="before")
    @classmethod
    def _coerce_regions(cls, v):
        """Gemini sometimes returns regions as a list of strings, other times
        as a list of `{region, confidence, signals}` objects (matches the
        prompt template). Accept both — extract `.region` from objects."""
        if not v:
            return []
        out: list[str] = []
        for item in v:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                name = item.get("region") or item.get("name")
                if name:
                    out.append(str(name))
        return out


class AudienceAuthenticity(_Base):
    authenticity_score: Optional[float] = None
    emoji_only_percentage: Optional[float] = None
    generic_comment_percentage: Optional[float] = None
    substantive_comment_percentage: Optional[float] = None
    suspicious_patterns: list[str] = Field(default_factory=list)

    @field_validator(
        "authenticity_score",
        "emoji_only_percentage",
        "generic_comment_percentage",
        "substantive_comment_percentage",
        mode="before",
    )
    @classmethod
    def _dec(cls, v): return coerce_decimal_percentage(v)


class AudienceSentiment(_Base):
    overall_sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    common_positive_themes: list[str] = Field(default_factory=list)
    common_negative_themes: list[str] = Field(default_factory=list)

    @field_validator("sentiment_score", mode="before")
    @classmethod
    def _dec(cls, v):
        # Sentiment can legitimately be [-1, 1]; do not clamp to [0,1].
        if v is None or v == "":
            return None
        try:
            n = float(v)
        except (TypeError, ValueError):
            return None
        # Gemini sometimes emits [0, 100] on a signed axis — rescale.
        if abs(n) > 1.0:
            n = max(-1.0, min(1.0, n / 100.0))
        return round(n, 3)


class AudienceDemographics(_Base):
    estimated_age_group: Optional[str] = None
    estimated_gender_skew: Optional[str] = None
    interest_signals: list[str] = Field(default_factory=list)


class EngagementQuality(_Base):
    quality_score: Optional[float] = None
    conversation_depth: Optional[str] = None
    community_feel: Optional[str] = None

    @field_validator("quality_score", mode="before")
    @classmethod
    def _dec(cls, v): return coerce_decimal_percentage(v)


class AudienceIntelligencePayload(_Base):
    audience_language_distribution: AudienceLanguageDistribution = Field(
        default_factory=AudienceLanguageDistribution
    )
    audience_geography_inference: AudienceGeographyInference = Field(
        default_factory=AudienceGeographyInference
    )
    audience_authenticity: AudienceAuthenticity = Field(
        default_factory=AudienceAuthenticity
    )
    audience_sentiment: AudienceSentiment = Field(
        default_factory=AudienceSentiment
    )
    audience_demographics_inference: AudienceDemographics = Field(
        default_factory=AudienceDemographics
    )
    engagement_quality: EngagementQuality = Field(
        default_factory=EngagementQuality
    )


# ── LLM failure sentinel ─────────────────────────────────────────

class LLMFailure(dict):
    """Dict-compatible sentinel returned when an LLM dimension fails.

    Downstream consumers test `obj.get("_llm_failure")` or use the
    `is_llm_failure` helper. Inherits from `dict` so existing code
    that calls `.get()` does not break.
    """

    def __init__(
        self,
        *,
        dimension: str,
        error: str,
        prompt_snippet: str = "",
    ) -> None:
        super().__init__(
            _llm_failure=True,
            dimension=dimension,
            error=error,
            prompt_snippet=prompt_snippet,
        )


def is_llm_failure(obj: Any) -> bool:
    return isinstance(obj, dict) and bool(obj.get("_llm_failure"))
