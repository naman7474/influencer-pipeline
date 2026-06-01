"""Pydantic schema for the single merged LLM call (Phase 4).

Composes the existing three dimension payloads (Caption / Transcript /
Audience Intelligence) into one structured output. The merged call is
made with a single Anthropic Claude Sonnet 4.6 request; the response
is split back into the three CIP keys downstream so storage + scoring
code stay untouched.

``transcript`` and ``audience`` are Optional so creators with no
transcripts (Whisper deferred) or no comments (commerce-signal short
circuit) still parse cleanly.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from pipeline.schemas.intelligence import (
    AudienceIntelligencePayload,
    CaptionIntelligencePayload,
    TranscriptIntelligencePayload,
)


class MergedIntelligence(BaseModel):
    """Single-call LLM output — composes the three dimension schemas."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    caption: CaptionIntelligencePayload = Field(
        default_factory=CaptionIntelligencePayload
    )
    transcript: Optional[TranscriptIntelligencePayload] = None
    audience: Optional[AudienceIntelligencePayload] = None


def merged_tool_schema() -> dict:
    """JSON Schema (Draft 2020-12) for Anthropic's ``tools[].input_schema``.

    Anthropic requires top-level type=object plus a properties map; we
    flatten the Pydantic model_json_schema() so $defs/refs are resolved
    inline (the SDK accepts refs, but inlined is more portable).
    """
    raw = MergedIntelligence.model_json_schema(mode="serialization")
    return _inline_defs(raw)


def _inline_defs(schema: dict) -> dict:
    """Walk a JSON Schema and replace ``$ref`` pointers with the inline
    definition from ``$defs``. Anthropic's tool input_schema accepts refs
    but inlined is simpler to debug and copy-paste.
    """
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node and node["$ref"].startswith("#/$defs/"):
                key = node["$ref"].split("/")[-1]
                target = defs.get(key, {})
                return resolve(target)
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)
