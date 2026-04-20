"""Tests for pipeline/content_analyzer.py — prompt building, JSON parsing, validation."""

import json
import pytest

from pipeline.content_analyzer import (
    _build_user_prompt,
    _parse_analysis_response,
    _validate_analysis,
    _REQUIRED_KEYS,
)


# ── Prompt building ──────────────────────────────────────────────────────


class TestBuildUserPrompt:
    def test_includes_campaign_name_and_goal(self):
        campaign = {"name": "Summer Glow", "goal": "awareness"}
        prompt = _build_user_prompt(None, None, None, campaign)
        assert "Summer Glow" in prompt
        assert "awareness" in prompt

    def test_includes_brief_requirements_list(self):
        campaign = {
            "name": "Test",
            "goal": "ugc",
            "brief_requirements": [
                "Show product unboxing",
                "Mention discount code",
            ],
        }
        prompt = _build_user_prompt(None, None, None, campaign)
        assert "Show product unboxing" in prompt
        assert "Mention discount code" in prompt

    def test_includes_brief_requirements_dict(self):
        campaign = {
            "name": "Test",
            "goal": "ugc",
            "brief_requirements": {"tone": "casual", "length": "30s"},
        }
        prompt = _build_user_prompt(None, None, None, campaign)
        assert "tone" in prompt
        assert "casual" in prompt

    def test_includes_brand_guidelines(self):
        guidelines = {
            "forbidden_topics": ["politics", "alcohol"],
            "content_dos": ["Show product in use"],
            "content_donts": ["No competitor mentions"],
            "required_disclosures": ["#ad"],
            "preferred_content_themes": ["wellness"],
        }
        prompt = _build_user_prompt(None, None, guidelines, {"name": "T", "goal": "g"})
        assert "politics" in prompt
        assert "alcohol" in prompt
        assert "Show product in use" in prompt
        assert "No competitor mentions" in prompt
        assert "#ad" in prompt
        assert "wellness" in prompt

    def test_includes_transcript(self):
        transcript = {
            "transcript_text": "Hey everyone, check out this amazing product",
            "hook_text": "Hey everyone",
            "detected_language": "en",
            "is_likely_music": False,
        }
        prompt = _build_user_prompt(
            transcript, None, None, {"name": "T", "goal": "g"}
        )
        assert "Hey everyone, check out this amazing product" in prompt
        assert "Hey everyone" in prompt
        assert "en" in prompt

    def test_includes_caption(self):
        prompt = _build_user_prompt(
            None,
            "Love this serum! #ad @brand",
            None,
            {"name": "T", "goal": "g"},
        )
        assert "Love this serum! #ad @brand" in prompt

    def test_handles_no_transcript_no_caption(self):
        prompt = _build_user_prompt(None, None, None, {"name": "T", "goal": "g"})
        assert "No transcript available" in prompt
        assert "No caption provided" in prompt

    def test_music_flag_noted(self):
        transcript = {
            "transcript_text": "la la la",
            "is_likely_music": True,
        }
        prompt = _build_user_prompt(
            transcript, None, None, {"name": "T", "goal": "g"}
        )
        assert "music" in prompt.lower()

    def test_includes_target_regions_and_niches(self):
        campaign = {
            "name": "T",
            "goal": "g",
            "target_regions": ["Mumbai", "Delhi"],
            "target_niches": ["Beauty", "Fashion"],
        }
        prompt = _build_user_prompt(None, None, None, campaign)
        assert "Mumbai" in prompt
        assert "Delhi" in prompt
        assert "Beauty" in prompt
        assert "Fashion" in prompt

    def test_includes_description(self):
        campaign = {
            "name": "T",
            "goal": "g",
            "description": "Launch campaign for new moisturizer",
        }
        prompt = _build_user_prompt(None, None, None, campaign)
        assert "Launch campaign for new moisturizer" in prompt


# ── JSON parsing ─────────────────────────────────────────────────────────


class TestParseAnalysisResponse:
    def test_parses_clean_json(self):
        obj = {"overall": {"score": 80}}
        result = _parse_analysis_response(json.dumps(obj))
        assert result == obj

    def test_parses_json_with_surrounding_text(self):
        text = 'Here is the analysis:\n{"overall": {"score": 80}}\nDone!'
        result = _parse_analysis_response(text)
        assert result["overall"]["score"] == 80

    def test_parses_json_with_markdown_code_block(self):
        text = '```json\n{"overall": {"score": 80}}\n```'
        result = _parse_analysis_response(text)
        assert result["overall"]["score"] == 80

    def test_raises_on_no_json(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            _parse_analysis_response("No JSON here at all")

    def test_handles_nested_braces(self):
        obj = {
            "hook_strength": {"score": 78, "details": {"sub": "value"}},
            "overall": {"score": 80},
        }
        result = _parse_analysis_response(json.dumps(obj))
        assert result["hook_strength"]["details"]["sub"] == "value"

    def test_handles_strings_with_braces(self):
        obj = {
            "overall": {
                "score": 80,
                "summary": "Content has {good} structure",
            }
        }
        result = _parse_analysis_response(json.dumps(obj))
        assert result["overall"]["summary"] == "Content has {good} structure"


# ── Validation ───────────────────────────────────────────────────────────


class TestValidateAnalysis:
    def test_complete_analysis_passes_through(self):
        analysis = {key: {"score": 80, "assessment": "Good"} for key in _REQUIRED_KEYS}
        analysis["overall"] = {
            "score": 80,
            "tier": "strong",
            "summary": "Good content",
            "strengths": ["A"],
            "improvement_areas": ["B"],
            "recommendation": "approve",
            "confidence": 0.85,
        }
        result = _validate_analysis(analysis)
        assert result["overall"]["score"] == 80
        assert result["overall"]["recommendation"] == "approve"

    def test_fills_missing_keys_with_defaults(self):
        analysis = {}
        result = _validate_analysis(analysis)
        for key in _REQUIRED_KEYS:
            assert key in result
            assert result[key]["score"] == 0

    def test_fills_missing_overall_subfields(self):
        analysis = {"overall": {"score": 75}}
        # Add other required keys
        for key in _REQUIRED_KEYS:
            if key != "overall":
                analysis[key] = {"score": 50}
        result = _validate_analysis(analysis)
        assert result["overall"]["tier"] == "needs_work"  # default
        assert result["overall"]["summary"] == "Analysis incomplete."
        assert result["overall"]["strengths"] == []
        assert result["overall"]["improvement_areas"] == []
        assert result["overall"]["recommendation"] == "revision_requested"
        assert result["overall"]["confidence"] == 0.5

    def test_preserves_existing_overall_subfields(self):
        analysis = {
            "overall": {
                "score": 90,
                "tier": "exceptional",
                "summary": "Amazing",
                "strengths": ["Hook"],
                "improvement_areas": [],
                "recommendation": "approve",
                "confidence": 0.95,
            }
        }
        for key in _REQUIRED_KEYS:
            if key != "overall":
                analysis[key] = {"score": 90}
        result = _validate_analysis(analysis)
        assert result["overall"]["tier"] == "exceptional"
        assert result["overall"]["confidence"] == 0.95

    def test_fills_missing_score_in_category(self):
        analysis = {"hook_strength": {"assessment": "No score provided"}}
        for key in _REQUIRED_KEYS:
            if key != "hook_strength":
                analysis[key] = {"score": 50}
        result = _validate_analysis(analysis)
        assert result["hook_strength"]["score"] == 0
        assert result["hook_strength"]["assessment"] == "No score provided"

    def test_handles_none_category_values(self):
        analysis = {"hook_strength": None}
        for key in _REQUIRED_KEYS:
            if key != "hook_strength":
                analysis[key] = {"score": 50}
        result = _validate_analysis(analysis)
        assert result["hook_strength"]["score"] == 0
