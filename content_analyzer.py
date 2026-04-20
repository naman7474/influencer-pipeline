"""
Content video analysis using Claude for deep qualitative assessment.

Takes a transcription result + brand/campaign context and produces
a structured analysis covering hook strength, brand compliance,
brief alignment, cultural signals, and overall quality scoring.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

ANALYSIS_VERSION = "1.0"
MODEL = "claude-sonnet-4-20250514"


def _build_system_prompt() -> str:
    return """You are an expert influencer marketing content analyst. You analyze creator-submitted campaign content (Instagram Reels) for brand-creator fit, compliance, and quality.

You will receive:
1. A transcript of the reel (from speech-to-text)
2. The caption text the creator wrote
3. Brand guidelines (forbidden topics, dos/donts, required disclosures, preferred themes)
4. Campaign brief requirements and objectives

Your job is to produce a structured JSON analysis covering these dimensions:

1. **hook_strength** — How compelling is the opening 3 seconds? Does it stop the scroll?
2. **brand_mention** — Is the brand mention natural or scripted? Organic or forced?
3. **brief_compliance** — Does the content hit each specific campaign requirement?
4. **guideline_compliance** — Any forbidden topics? Follows dos/donts? Required disclosures present?
5. **language_tone** — What language(s) spoken? Matches target audience? Tone aligns with brand?
6. **content_depth** — Surface-level or genuine value? Pacing engaging or rushed/dragging?
7. **cultural_signals** — Regional references, cultural context, audience-relevant markers
8. **cta_effectiveness** — Clear call-to-action? Natural integration?
9. **production_quality** — Audio clarity, delivery quality signals from the transcript
10. **overall** — Weighted composite with strengths, improvement areas, and recommendation

## Scoring guidelines
- Scores are 0-100 integers
- 90-100: Exceptional — ready to go live as-is
- 75-89: Strong — minor improvements possible but approvable
- 60-74: Adequate — meets minimum bar, some clear improvement areas
- 40-59: Needs work — revision recommended before approval
- 0-39: Significant issues — likely reject or major revision needed

## Output format
Return ONLY valid JSON matching this schema (no markdown, no explanation outside JSON):
{
  "hook_strength": {
    "score": <int>,
    "label": "strong" | "good" | "adequate" | "weak",
    "hook_type": "question" | "statement" | "visual" | "story" | "shock" | "none",
    "assessment": "<1-2 sentences>"
  },
  "brand_mention": {
    "score": <int>,
    "naturalness": "organic" | "natural" | "slightly_scripted" | "scripted" | "forced" | "absent",
    "feels_scripted": <bool>,
    "assessment": "<1-2 sentences>"
  },
  "brief_compliance": {
    "score": <int>,
    "requirements": [
      {"requirement": "<from brief>", "met": <bool>, "evidence": "<quote or explanation>"}
    ],
    "assessment": "<1-2 sentences>"
  },
  "guideline_compliance": {
    "score": <int>,
    "forbidden_topics_flagged": ["<topic>" ...],
    "disclosures_present": ["#ad" ...],
    "disclosures_missing": ["<required disclosure>" ...],
    "dos_followed": ["<guideline>" ...],
    "donts_violated": ["<guideline>" ...],
    "assessment": "<1-2 sentences>"
  },
  "language_tone": {
    "score": <int>,
    "spoken_languages": ["Hindi", "English" ...],
    "primary_language": "<lang>",
    "tone": "casual" | "professional" | "energetic" | "storytelling" | "educational" | "humorous",
    "tone_brand_alignment": "aligned" | "mostly_aligned" | "neutral" | "misaligned",
    "assessment": "<1-2 sentences>"
  },
  "content_depth": {
    "score": <int>,
    "depth_level": "deep" | "moderate" | "surface",
    "pacing": "well-paced" | "slightly_fast" | "slightly_slow" | "rushed" | "dragging",
    "value_provided": <bool>,
    "assessment": "<1-2 sentences>"
  },
  "cultural_signals": {
    "score": <int>,
    "regional_references": ["<reference>" ...],
    "cultural_markers": ["<marker>" ...],
    "audience_relevance": "high" | "moderate" | "low" | "neutral",
    "assessment": "<1-2 sentences>"
  },
  "cta_effectiveness": {
    "score": <int>,
    "cta_present": <bool>,
    "cta_text": "<extracted CTA or null>",
    "cta_style": "link_in_bio" | "swipe_up" | "comment" | "dm" | "discount_code" | "none",
    "natural_integration": <bool>,
    "assessment": "<1-2 sentences>"
  },
  "production_quality": {
    "score": <int>,
    "audio_clarity": "excellent" | "good" | "fair" | "poor",
    "delivery_quality": "polished" | "natural" | "rough" | "unedited",
    "assessment": "<1-2 sentences>"
  },
  "overall": {
    "score": <int>,
    "tier": "exceptional" | "strong" | "adequate" | "needs_work" | "poor",
    "summary": "<2-3 sentence overall assessment>",
    "strengths": ["<strength 1>", "<strength 2>" ...],
    "improvement_areas": ["<area 1>", "<area 2>" ...],
    "recommendation": "approve" | "approve_with_notes" | "revision_requested" | "reject",
    "confidence": <float 0.0-1.0>
  }
}"""


def _build_user_prompt(
    transcript: dict | None,
    caption_text: str | None,
    brand_guidelines: dict | None,
    campaign: dict,
) -> str:
    sections: list[str] = []

    # Campaign context
    sections.append("## Campaign Context")
    sections.append(f"- **Name**: {campaign.get('name', 'N/A')}")
    sections.append(f"- **Goal**: {campaign.get('goal', 'N/A')}")
    if campaign.get("description"):
        sections.append(f"- **Description**: {campaign['description']}")
    if campaign.get("target_regions"):
        sections.append(f"- **Target Regions**: {', '.join(campaign['target_regions'])}")
    if campaign.get("target_niches"):
        sections.append(f"- **Target Niches**: {', '.join(campaign['target_niches'])}")

    # Brief requirements
    brief = campaign.get("brief_requirements")
    if brief:
        sections.append("\n## Brief Requirements")
        if isinstance(brief, list):
            for i, req in enumerate(brief, 1):
                sections.append(f"{i}. {req}")
        elif isinstance(brief, dict):
            for key, val in brief.items():
                sections.append(f"- **{key}**: {val}")
        else:
            sections.append(str(brief))

    # Brand guidelines
    if brand_guidelines:
        sections.append("\n## Brand Guidelines")
        if brand_guidelines.get("forbidden_topics"):
            sections.append(f"- **Forbidden Topics**: {', '.join(brand_guidelines['forbidden_topics'])}")
        if brand_guidelines.get("content_dos"):
            sections.append("- **Content Dos**:")
            for d in brand_guidelines["content_dos"]:
                sections.append(f"  - {d}")
        if brand_guidelines.get("content_donts"):
            sections.append("- **Content Don'ts**:")
            for d in brand_guidelines["content_donts"]:
                sections.append(f"  - {d}")
        if brand_guidelines.get("required_disclosures"):
            sections.append(f"- **Required Disclosures**: {', '.join(brand_guidelines['required_disclosures'])}")
        if brand_guidelines.get("preferred_content_themes"):
            sections.append(f"- **Preferred Themes**: {', '.join(brand_guidelines['preferred_content_themes'])}")
        if brand_guidelines.get("notes"):
            sections.append(f"- **Notes**: {brand_guidelines['notes']}")

    # Transcript
    sections.append("\n## Creator Content")
    if transcript and transcript.get("transcript_text"):
        sections.append(f"### Transcript (spoken words)")
        sections.append(transcript["transcript_text"])
        if transcript.get("hook_text"):
            sections.append(f"\n**Hook (first 3 seconds):** {transcript['hook_text']}")
        if transcript.get("detected_language"):
            sections.append(f"**Detected Language:** {transcript['detected_language']}")
        if transcript.get("is_likely_music"):
            sections.append("**Note:** This transcript may be background music/song lyrics, not creator speech.")
    else:
        sections.append("### Transcript")
        sections.append("*No transcript available (video could not be transcribed or is caption-only)*")

    if caption_text:
        sections.append(f"\n### Caption")
        sections.append(caption_text)
    else:
        sections.append("\n### Caption")
        sections.append("*No caption provided*")

    sections.append("\n---\nAnalyze this content and return the structured JSON assessment.")
    return "\n".join(sections)


def analyze_submission_content(
    transcript: dict | None,
    caption_text: str | None,
    brand_guidelines: dict | None,
    campaign: dict,
    anthropic_api_key: str,
    model: str = MODEL,
) -> dict[str, Any]:
    """
    Run Claude analysis on a content submission.

    Args:
        transcript: Dict from transcriber with transcript_text, hook_text, etc. None for caption-only.
        caption_text: The caption text from the submission.
        brand_guidelines: Brand guidelines dict (forbidden_topics, content_dos, etc.).
        campaign: Campaign dict (name, goal, description, brief_requirements, target_regions, etc.).
        anthropic_api_key: Anthropic API key.
        model: Claude model to use.

    Returns:
        Structured analysis dict matching the schema.
    """
    client = anthropic.Anthropic(api_key=anthropic_api_key)

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(transcript, caption_text, brand_guidelines, campaign)

    logger.info(f"Calling Claude ({model}) for content analysis")

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    response_text = message.content[0].text

    # Parse JSON response
    analysis = _parse_analysis_response(response_text)

    # Validate and fill defaults
    analysis = _validate_analysis(analysis)

    logger.info(
        f"Content analysis complete — overall_score={analysis.get('overall', {}).get('score')}, "
        f"recommendation={analysis.get('overall', {}).get('recommendation')}"
    )
    return analysis


def _parse_analysis_response(text: str) -> dict:
    """Parse Claude's JSON response, handling potential formatting issues."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting the outermost JSON object (same approach as llm_client.py)
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in Claude response: {text[:200]}")

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break

    raise ValueError(f"Could not parse JSON from Claude response: {text[:300]}")


_REQUIRED_KEYS = [
    "hook_strength",
    "brand_mention",
    "brief_compliance",
    "guideline_compliance",
    "language_tone",
    "content_depth",
    "cultural_signals",
    "cta_effectiveness",
    "production_quality",
    "overall",
]


def _validate_analysis(analysis: dict) -> dict:
    """Ensure all required top-level keys exist with at least a score."""
    for key in _REQUIRED_KEYS:
        if key not in analysis or analysis[key] is None:
            analysis[key] = {"score": 0, "assessment": "Analysis unavailable for this dimension."}
            logger.warning(f"Analysis missing key '{key}', defaulting to score=0")
        elif "score" not in analysis[key]:
            analysis[key]["score"] = 0

    # Ensure overall has required sub-fields
    overall = analysis.get("overall", {})
    overall.setdefault("score", 0)
    overall.setdefault("tier", "needs_work")
    overall.setdefault("summary", "Analysis incomplete.")
    overall.setdefault("strengths", [])
    overall.setdefault("improvement_areas", [])
    overall.setdefault("recommendation", "revision_requested")
    overall.setdefault("confidence", 0.5)
    analysis["overall"] = overall

    return analysis
