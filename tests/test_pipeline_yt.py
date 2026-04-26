"""Tests for the YouTube orchestrator in pipeline.pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from pipeline.pipeline import (
    build_youtube_creator_intelligence_profile,
    _posts_per_week_from_cadence,
)


class TestPostsPerWeekFromCadence:
    def test_none(self):
        assert _posts_per_week_from_cadence(None) is None

    def test_zero(self):
        assert _posts_per_week_from_cadence(0) is None

    def test_negative(self):
        assert _posts_per_week_from_cadence(-1) is None

    def test_daily_cadence(self):
        assert _posts_per_week_from_cadence(1.0) == 7.0

    def test_weekly_cadence(self):
        assert _posts_per_week_from_cadence(7.0) == 1.0

    def test_monthly_cadence(self):
        out = _posts_per_week_from_cadence(30.0)
        assert abs(out - 0.23) < 0.01


class TestBuildYoutubeCreatorIntelligenceProfile:
    """Integration-ish test: all external deps mocked.

    These tests patch at the import site inside the function, which requires
    patching `pipeline.youtube.*` module functions before the call.
    """

    def _patch_all(self, monkeypatch):
        """Swap out every external call made by build_youtube_creator_intelligence_profile."""
        # BrightdataClient + Gemini client are built in-function; patch the
        # classes / factories that the module uses.
        monkeypatch.setattr(
            "pipeline.pipeline.BrightdataClient", MagicMock()
        )
        monkeypatch.setattr("pipeline.pipeline.init_gemini", MagicMock())

        # Handle resolver
        resolved = MagicMock()
        resolved.channel_id = "UCtest"
        resolved.handle = "test"
        resolved.url = "https://www.youtube.com/channel/UCtest"
        monkeypatch.setattr(
            "pipeline.youtube.handle_resolver.resolve",
            MagicMock(return_value=resolved),
        )

        # YouTubeAPIClient — available=False so the API-enrichment branch
        # is skipped.
        yt_api = MagicMock()
        yt_api.available = False
        monkeypatch.setattr(
            "pipeline.youtube.youtube_api.YouTubeAPIClient",
            MagicMock(return_value=yt_api),
        )

        # Scrapers
        monkeypatch.setattr(
            "pipeline.youtube.scraper_channels.scrape_channels",
            MagicMock(return_value=[{"handle": "test", "subscribers": 10000}]),
        )
        monkeypatch.setattr(
            "pipeline.youtube.scraper_channels.extract_channel_metrics",
            MagicMock(
                return_value={
                    "handle": "test",
                    "platform_user_id": "UCtest",
                    "followers_or_subs": 10000,
                    "posts_or_videos_count": 100,
                    "tier": "micro",
                    "data_quality_flags": [],
                }
            ),
        )

        video_record = {
            "video_id": "v1",
            "url": "https://yt.com/watch?v=v1",
            "view_count": 1000,
            "like_count": 100,
            "comment_count": 20,
            "duration_seconds": 300,
            "is_short": False,
            "top_comments": [],
        }
        monkeypatch.setattr(
            "pipeline.youtube.scraper_videos.scrape_videos_discovery",
            MagicMock(return_value=[video_record]),
        )
        monkeypatch.setattr(
            "pipeline.youtube.scraper_videos.select_top_videos",
            MagicMock(return_value=[video_record]),
        )
        monkeypatch.setattr(
            "pipeline.youtube.scraper_videos.extract_video_metrics",
            MagicMock(return_value=video_record),
        )
        monkeypatch.setattr(
            "pipeline.youtube.scraper_videos.aggregate_channel_metrics",
            MagicMock(
                return_value={
                    "avg_view_count": 1000,
                    "watch_through_proxy": 0.1,
                    "upload_cadence_days": 3.5,
                    "content_mix": {"youtube_long": 1},
                    "top_comments_from_videos": [],
                }
            ),
        )

        monkeypatch.setattr(
            "pipeline.youtube.scraper_comments.scrape_comments",
            MagicMock(return_value=[]),
        )
        monkeypatch.setattr(
            "pipeline.youtube.scraper_comments.select_top_videos_for_comments",
            MagicMock(return_value=[]),
        )
        monkeypatch.setattr(
            "pipeline.youtube.scraper_comments.extract_comment_metrics",
            MagicMock(return_value={"_comment_texts": []}),
        )

        # LLM analyzers
        monkeypatch.setattr(
            "pipeline.pipeline.analyze_captions",
            MagicMock(return_value={"primary_niche": "tech"}),
        )
        monkeypatch.setattr(
            "pipeline.pipeline.analyze_transcripts",
            MagicMock(return_value={"avg_hook_quality": 0.8}),
        )
        monkeypatch.setattr(
            "pipeline.pipeline.analyze_comments",
            MagicMock(return_value={"authenticity_score": 0.9}),
        )

        # Transcriber (not called when captions are absent AND no URL;
        # but we still patch for the path where it is called)
        monkeypatch.setattr(
            "pipeline.pipeline.transcribe_reels_whisper",
            MagicMock(return_value=[]),
        )

        # Scorer
        monkeypatch.setattr(
            "pipeline.pipeline.compute_creator_scores",
            MagicMock(
                return_value={
                    "cpi": 70,
                    "engagement_quality": 60,
                    "content_quality": 70,
                    "audience_authenticity": 80,
                    "growth_trajectory": 60,
                    "professionalism": 80,
                    "confidence": {"tier": "high", "overall_coverage": 0.9},
                    "scoring_inputs": {},
                }
            ),
        )

    def test_happy_path(self, monkeypatch):
        self._patch_all(monkeypatch)
        cip = build_youtube_creator_intelligence_profile(
            channel_url="https://www.youtube.com/@test",
            brightdata_token="x",
            gemini_api_key="y",
            openai_api_key="z",
        )
        assert cip["platform"] == "youtube"
        assert cip["profile"]["handle"] == "test"
        assert cip["resolved"]["channel_id"] == "UCtest"
        assert cip["scores"]["cpi"] == 70
        assert cip["scores"]["avg_views_per_sub"] == 0.1  # 1000/10000
        assert cip["scores"]["upload_cadence_days"] == 3.5
        # posts_per_week derived from cadence (7/3.5 = 2.0)
        assert cip["posts"]["posts_per_week"] == 2.0
        assert cip["posts"]["avg_engagement_rate"] == (100 + 20) / 1000
        assert "error" not in cip

    def test_error_returns_error_key(self, monkeypatch):
        self._patch_all(monkeypatch)
        # Force the channel scrape to raise
        monkeypatch.setattr(
            "pipeline.youtube.scraper_channels.scrape_channels",
            MagicMock(side_effect=RuntimeError("brightdata down")),
        )
        cip = build_youtube_creator_intelligence_profile(
            channel_url="https://www.youtube.com/@test",
            brightdata_token="x",
            gemini_api_key="y",
            openai_api_key="z",
        )
        assert cip["error"] == "brightdata down"
        assert cip["platform"] == "youtube"

    def test_empty_channel_scrape(self, monkeypatch):
        self._patch_all(monkeypatch)
        monkeypatch.setattr(
            "pipeline.youtube.scraper_channels.scrape_channels",
            MagicMock(return_value=[]),
        )
        cip = build_youtube_creator_intelligence_profile(
            channel_url="https://www.youtube.com/@test",
            brightdata_token="x",
            gemini_api_key="y",
            openai_api_key="z",
        )
        assert "no channel data" in cip["error"]

    def test_tier1_captions_skip_whisper(self, monkeypatch):
        """fetch_transcript tier 1 (youtube-transcript-api) returns text →
        the orchestrator stores it and never reaches Whisper."""
        self._patch_all(monkeypatch)
        # Patch fetch_transcript so it always returns tier-1-shaped output
        # — that's what we'd get if youtube-transcript-api worked.
        monkeypatch.setattr(
            "pipeline.youtube.transcripts.fetch_transcript",
            MagicMock(
                return_value={
                    "video_id": "v1",
                    "source": "youtube_transcript_api",
                    "text": "Hello viewers",
                }
            ),
        )
        # Whisper would only be reached if fetch_transcript returned None;
        # ensure it's never called.
        whisper_mock = MagicMock(return_value=None)
        monkeypatch.setattr(
            "pipeline.youtube.transcripts._try_whisper", whisper_mock
        )
        cip = build_youtube_creator_intelligence_profile(
            channel_url="https://www.youtube.com/@test",
            brightdata_token="x",
            gemini_api_key="y",
            openai_api_key="z",
        )
        whisper_mock.assert_not_called()
        assert len(cip["transcripts"]) >= 1
        assert cip["transcripts"][0]["caption_source"] == "youtube_transcript_api"

    def test_llm_failure_marked(self, monkeypatch):
        self._patch_all(monkeypatch)
        monkeypatch.setattr(
            "pipeline.pipeline.analyze_captions",
            MagicMock(side_effect=RuntimeError("gemini down")),
        )
        cip = build_youtube_creator_intelligence_profile(
            channel_url="https://www.youtube.com/@test",
            brightdata_token="x",
            gemini_api_key="y",
            openai_api_key="z",
        )
        assert cip["caption_intelligence"]["_llm_failure"] is True
        assert "gemini down" in cip["caption_intelligence"]["error"]

    def test_yt_api_stats_merged_when_available(self, monkeypatch):
        self._patch_all(monkeypatch)
        yt_api = MagicMock()
        yt_api.available = True
        yt_api.fetch_channel_stats.return_value = {
            "UCtest": {
                "subscriber_count": 99_999,
                "video_count": 500,
                "topic_categories": ["Tech"],
            }
        }
        monkeypatch.setattr(
            "pipeline.youtube.youtube_api.YouTubeAPIClient",
            MagicMock(return_value=yt_api),
        )
        cip = build_youtube_creator_intelligence_profile(
            channel_url="https://www.youtube.com/@test",
            brightdata_token="x",
            gemini_api_key="y",
            openai_api_key="z",
            youtube_api_key="real-key",
        )
        assert cip["profile"]["followers_or_subs"] == 99_999
        assert cip["profile"]["posts_or_videos_count"] == 500
        assert cip["profile"]["topic_categories"] == ["Tech"]
