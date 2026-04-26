"""Tests for YouTube handlers added to pipeline.handlers."""

from unittest.mock import MagicMock

import pytest

from pipeline import handlers


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("BRIGHTDATA_API_TOKEN", "bd")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    monkeypatch.setenv("YOUTUBE_API_KEY", "yt")


class TestDispatchRegistry:
    def test_yt_handlers_registered(self):
        assert "brand_yt_scrape" in handlers.HANDLERS
        assert "creator_yt_scrape" in handlers.HANDLERS
        assert "brand_ig_scrape" in handlers.HANDLERS  # unchanged
        assert "creator_ig_scrape" in handlers.HANDLERS


class TestHandleCreatorYtScrape:
    def test_missing_url_raises(self):
        with pytest.raises(ValueError, match="missing"):
            handlers.handle_creator_yt_scrape(MagicMock(), {"payload": {}})

    def test_happy_path(self, monkeypatch):
        resolved = MagicMock()
        resolved.channel_id = "UCx"
        resolved.handle = "x"
        resolved.url = "https://www.youtube.com/channel/UCx"
        monkeypatch.setattr(
            "pipeline.youtube.handle_resolver.resolve",
            MagicMock(return_value=resolved),
        )
        cip = {
            "platform": "youtube",
            "profile": {"handle": "x"},
            "scores": {"cpi": 50},
        }
        monkeypatch.setattr(
            "pipeline.pipeline.build_youtube_creator_intelligence_profile",
            MagicMock(return_value=cip),
        )
        monkeypatch.setattr(
            handlers.pdb, "store_youtube_cip", MagicMock(return_value="c-1")
        )
        monkeypatch.setattr(
            handlers, "build_creator_embedding_input", MagicMock(return_value="text")
        )
        monkeypatch.setattr(
            handlers, "embed_text", MagicMock(return_value=[0.1, 0.2])
        )
        monkeypatch.setattr(
            handlers, "_update_creator_embedding", MagicMock()
        )
        monkeypatch.setattr(
            handlers, "_trigger_creator_recompute", MagicMock()
        )
        monkeypatch.setattr(
            handlers, "_siblings_all_terminal", MagicMock(return_value=False)
        )

        handlers.handle_creator_yt_scrape(
            MagicMock(), {"payload": {"url": "https://youtube.com/@x"}}
        )

    def test_cip_error_raises(self, monkeypatch):
        resolved = MagicMock()
        resolved.url = "https://yt.com/@x"
        monkeypatch.setattr(
            "pipeline.youtube.handle_resolver.resolve",
            MagicMock(return_value=resolved),
        )
        monkeypatch.setattr(
            "pipeline.pipeline.build_youtube_creator_intelligence_profile",
            MagicMock(return_value={"error": "boom"}),
        )
        with pytest.raises(RuntimeError, match="boom"):
            handlers.handle_creator_yt_scrape(
                MagicMock(), {"payload": {"url": "https://youtube.com/@x"}}
            )

    def test_parent_brand_fanout_triggers_matching(self, monkeypatch):
        resolved = MagicMock()
        resolved.url = "u"
        monkeypatch.setattr(
            "pipeline.youtube.handle_resolver.resolve",
            MagicMock(return_value=resolved),
        )
        monkeypatch.setattr(
            "pipeline.pipeline.build_youtube_creator_intelligence_profile",
            MagicMock(return_value={"profile": {}, "scores": {}}),
        )
        monkeypatch.setattr(
            handlers.pdb, "store_youtube_cip", MagicMock(return_value="c-1")
        )
        monkeypatch.setattr(
            handlers, "build_creator_embedding_input", MagicMock(return_value=None)
        )
        monkeypatch.setattr(
            handlers, "_siblings_all_terminal", MagicMock(return_value=True)
        )
        matching_mock = MagicMock()
        monkeypatch.setattr(handlers, "_trigger_matching_compute", matching_mock)
        monkeypatch.setattr(
            handlers, "_trigger_creator_recompute", MagicMock()
        )

        handlers.handle_creator_yt_scrape(
            MagicMock(),
            {
                "payload": {
                    "url": "https://yt.com/@x",
                    "parent_brand_id": "brand-1",
                }
            },
        )
        matching_mock.assert_called_once_with("brand-1")


class TestHandleBrandYtScrape:
    def test_missing_brand_or_url_raises(self):
        with pytest.raises(ValueError):
            handlers.handle_brand_yt_scrape(MagicMock(), {"payload": {}})
        with pytest.raises(ValueError):
            handlers.handle_brand_yt_scrape(
                MagicMock(), {"brand_id": "b", "payload": {}}
            )

    def test_writes_brand_platform_analysis(self, monkeypatch):
        resolved = MagicMock()
        resolved.url = "https://www.youtube.com/@brand"
        monkeypatch.setattr(
            "pipeline.youtube.handle_resolver.resolve",
            MagicMock(return_value=resolved),
        )
        cip = {
            "profile": {"handle": "brand"},
            "caption_intelligence": {"primary_niche": "tech"},
            "audience_intelligence": {"primary_country": "US"},
        }
        monkeypatch.setattr(
            "pipeline.pipeline.build_youtube_creator_intelligence_profile",
            MagicMock(return_value=cip),
        )
        upsert_mock = MagicMock()
        monkeypatch.setattr(
            handlers.pdb, "upsert_brand_platform_analysis", upsert_mock
        )

        handlers.handle_brand_yt_scrape(
            MagicMock(),
            {"brand_id": "brand-1", "payload": {"url": "https://yt.com/@x"}},
        )
        upsert_mock.assert_called_once()
        args = upsert_mock.call_args.args
        assert args[1] == "brand-1"
        assert args[2] == "youtube"
        analysis = args[3]
        assert analysis["analysis_status"] == "completed"
        assert analysis["handle"] == "brand"
        assert analysis["content_dna"]["primary_niche"] == "tech"

    def test_error_marks_analysis_failed(self, monkeypatch):
        resolved = MagicMock()
        resolved.url = "u"
        monkeypatch.setattr(
            "pipeline.youtube.handle_resolver.resolve",
            MagicMock(return_value=resolved),
        )
        monkeypatch.setattr(
            "pipeline.pipeline.build_youtube_creator_intelligence_profile",
            MagicMock(return_value={"error": "brightdata down"}),
        )
        upsert_mock = MagicMock()
        monkeypatch.setattr(
            handlers.pdb, "upsert_brand_platform_analysis", upsert_mock
        )

        handlers.handle_brand_yt_scrape(
            MagicMock(),
            {"brand_id": "brand-1", "payload": {"url": "u"}},
        )
        analysis = upsert_mock.call_args.args[3]
        assert analysis["analysis_status"] == "failed"
        assert analysis["analysis_error"] == "brightdata down"
