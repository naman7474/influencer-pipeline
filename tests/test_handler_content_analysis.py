"""Tests for the content_video_analysis handler in pipeline/handlers.py.

These tests mock all external dependencies (Supabase, BrightData, Whisper, Claude)
to verify the handler orchestration logic and error handling paths.
"""

from unittest.mock import MagicMock, patch
import os
import pytest

# Patch environment variables before importing handlers
os.environ.setdefault("BRIGHTDATA_API_TOKEN", "test-bd-token")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")

from pipeline.handlers import handle_content_video_analysis


def make_mock_db():
    """Create a mock Supabase client with chainable query builder."""
    db = MagicMock()
    tables = {}

    def mock_table(name):
        if name not in tables:
            tables[name] = MagicMock()
        return tables[name]

    db.table = mock_table
    return db, tables


SAMPLE_JOB = {
    "brand_id": "brand-1",
    "payload": {
        "content_submission_id": "sub-1",
        "content_url": "https://www.instagram.com/reel/ABC123/",
        "campaign_id": "camp-1",
        "creator_id": "creator-1",
    },
}

SAMPLE_ANALYSIS = {
    "hook_strength": {"score": 78},
    "brand_mention": {"score": 85},
    "brief_compliance": {"score": 90},
    "guideline_compliance": {"score": 95},
    "language_tone": {"score": 82},
    "content_depth": {"score": 70},
    "cultural_signals": {"score": 65},
    "cta_effectiveness": {"score": 75},
    "production_quality": {"score": 80},
    "overall": {
        "score": 80,
        "tier": "strong",
        "summary": "Good content",
        "strengths": [],
        "improvement_areas": [],
        "recommendation": "approve",
        "confidence": 0.85,
    },
}


class TestHandleContentVideoAnalysis:
    def test_raises_on_missing_submission_id(self):
        db, _ = make_mock_db()
        job = {"brand_id": "b1", "payload": {}}
        with pytest.raises(ValueError, match="missing content_submission_id"):
            handle_content_video_analysis(db, job)

    def test_raises_on_empty_payload(self):
        db, _ = make_mock_db()
        job = {"brand_id": "b1", "payload": None}
        with pytest.raises(ValueError, match="missing content_submission_id"):
            handle_content_video_analysis(db, job)

    @patch("pipeline.content_analyzer.analyze_submission_content", return_value=SAMPLE_ANALYSIS)
    @patch(
        "pipeline.transcriber.transcribe_reels",
        return_value=[
            {
                "post_id": "ABC123",
                "transcript_text": "Hey everyone",
                "hook_text": "Hey",
                "detected_language": "en",
                "segments": [],
                "avg_confidence": 0.85,
                "is_likely_music": False,
                "reel_length_seconds": 30,
            }
        ],
    )
    @patch(
        "pipeline.scraper_posts.scrape_single_post",
        return_value={
            "post_id": "ABC123",
            "video_url": "https://cdn.ig.com/v.mp4",
            "description": "caption",
            "length": 30,
        },
    )
    @patch("pipeline.brightdata_client.BrightdataClient")
    def test_full_happy_path(self, mock_bd_class, mock_scrape, mock_transcribe, mock_analyze):
        db, tables = make_mock_db()

        # No existing analysis
        empty_result = MagicMock()
        empty_result.data = []
        tables["content_analyses"].select.return_value.eq.return_value.execute.return_value = empty_result

        # Insert returns id
        insert_result = MagicMock()
        insert_result.data = [{"id": "analysis-1"}]
        tables["content_analyses"].insert.return_value.execute.return_value = insert_result

        # Update returns success
        tables["content_analyses"].update.return_value.eq.return_value.execute.return_value = MagicMock(data=[{}])

        # Submission data
        sub_result = MagicMock()
        sub_result.data = {
            "caption_text": "Check out this product! #ad",
            "content_url": "https://www.instagram.com/reel/ABC123/",
        }
        tables["content_submissions"].select.return_value.eq.return_value.single.return_value.execute.return_value = sub_result
        tables["content_submissions"].update.return_value.eq.return_value.execute.return_value = MagicMock(data=[{}])

        # Campaign data
        camp_result = MagicMock()
        camp_result.data = {
            "name": "Summer Campaign",
            "goal": "awareness",
            "description": "Promote new product",
            "brief_requirements": ["Show product"],
            "target_regions": ["Mumbai"],
            "target_niches": ["Beauty"],
        }
        tables["campaigns"].select.return_value.eq.return_value.single.return_value.execute.return_value = camp_result

        # Brand guidelines
        guidelines_result = MagicMock()
        guidelines_result.data = [
            {
                "forbidden_topics": [],
                "content_dos": ["Be natural"],
                "content_donts": [],
                "required_disclosures": ["#ad"],
                "preferred_content_themes": ["wellness"],
                "notes": None,
            }
        ]
        tables["brand_guidelines"].select.return_value.eq.return_value.limit.return_value.execute.return_value = guidelines_result

        handle_content_video_analysis(db, SAMPLE_JOB)

        # Verify scrape was called
        mock_scrape.assert_called_once()

        # Verify transcribe was called
        mock_transcribe.assert_called_once()

        # Verify Claude analysis was called
        mock_analyze.assert_called_once()

    def test_skips_already_completed_analysis(self):
        db, tables = make_mock_db()

        # Pre-populate the content_analyses table mock
        existing_result = MagicMock()
        existing_result.data = [{"id": "analysis-1", "status": "completed"}]
        tables["content_analyses"].select.return_value.eq.return_value.execute.return_value = existing_result

        # Should not raise, should just return early
        handle_content_video_analysis(db, SAMPLE_JOB)

        # content_submissions table should never have been accessed
        assert "content_submissions" not in tables

    @patch("pipeline.content_analyzer.analyze_submission_content", return_value=SAMPLE_ANALYSIS)
    def test_caption_only_skips_video_processing(self, mock_analyze):
        """When content_url is None, skip video scraping and transcription."""
        db, tables = make_mock_db()

        job = {
            "brand_id": "brand-1",
            "payload": {
                "content_submission_id": "sub-1",
                "content_url": None,
                "campaign_id": "camp-1",
                "creator_id": "creator-1",
            },
        }

        # No existing analysis
        empty_result = MagicMock()
        empty_result.data = []
        tables["content_analyses"].select.return_value.eq.return_value.execute.return_value = empty_result

        # Insert
        insert_result = MagicMock()
        insert_result.data = [{"id": "analysis-1"}]
        tables["content_analyses"].insert.return_value.execute.return_value = insert_result

        # Update
        tables["content_analyses"].update.return_value.eq.return_value.execute.return_value = MagicMock(data=[{}])

        # Submission with caption only
        sub_result = MagicMock()
        sub_result.data = {
            "caption_text": "Amazing product! #ad @brand",
            "content_url": None,
        }
        tables["content_submissions"].select.return_value.eq.return_value.single.return_value.execute.return_value = sub_result
        tables["content_submissions"].update.return_value.eq.return_value.execute.return_value = MagicMock(data=[{}])

        # Campaign
        camp_result = MagicMock()
        camp_result.data = {"name": "Test", "goal": "ugc", "brief_requirements": []}
        tables["campaigns"].select.return_value.eq.return_value.single.return_value.execute.return_value = camp_result

        # No guidelines
        guidelines_result = MagicMock()
        guidelines_result.data = []
        tables["brand_guidelines"].select.return_value.eq.return_value.limit.return_value.execute.return_value = guidelines_result

        handle_content_video_analysis(db, job)

        # Claude should still be called with caption-only analysis
        mock_analyze.assert_called_once()

    def test_skips_when_no_content(self):
        """When neither transcript nor caption is available, status should be 'skipped'."""
        db, tables = make_mock_db()

        job = {
            "brand_id": "brand-1",
            "payload": {
                "content_submission_id": "sub-1",
                "content_url": None,
                "campaign_id": "camp-1",
                "creator_id": "creator-1",
            },
        }

        # No existing analysis
        empty_result = MagicMock()
        empty_result.data = []
        tables["content_analyses"].select.return_value.eq.return_value.execute.return_value = empty_result

        # Insert
        insert_result = MagicMock()
        insert_result.data = [{"id": "analysis-1"}]
        tables["content_analyses"].insert.return_value.execute.return_value = insert_result

        # Update
        tables["content_analyses"].update.return_value.eq.return_value.execute.return_value = MagicMock(data=[{}])

        # Submission with NO caption and NO video
        sub_result = MagicMock()
        sub_result.data = {
            "caption_text": None,
            "content_url": None,
        }
        tables["content_submissions"].select.return_value.eq.return_value.single.return_value.execute.return_value = sub_result
        tables["content_submissions"].update.return_value.eq.return_value.execute.return_value = MagicMock(data=[{}])

        # Campaign
        camp_result = MagicMock()
        camp_result.data = {"name": "Test", "goal": "ugc", "brief_requirements": []}
        tables["campaigns"].select.return_value.eq.return_value.single.return_value.execute.return_value = camp_result

        # No guidelines
        guidelines_result = MagicMock()
        guidelines_result.data = []
        tables["brand_guidelines"].select.return_value.eq.return_value.limit.return_value.execute.return_value = guidelines_result

        # Should not raise — should just skip
        handle_content_video_analysis(db, job)

        # Verify the analysis was marked as skipped
        update_calls = tables["content_analyses"].update.call_args_list
        # At least one update should set status to 'skipped'
        found_skipped = any(
            call.args[0].get("status") == "skipped"
            for call in update_calls
            if call.args
        )
        assert found_skipped, "Expected analysis to be marked as 'skipped'"
