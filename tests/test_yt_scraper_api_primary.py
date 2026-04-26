"""Tests for Phase 2.5 API-primary flip on scraper_videos + scraper_comments."""

from unittest.mock import MagicMock

from pipeline.youtube.scraper_videos import (
    scrape_videos_discovery,
    _iso8601_duration_to_seconds,
    _api_record_to_bd_shape,
)
from pipeline.youtube.scraper_comments import (
    scrape_comments,
    _video_id_from_url,
)


class TestIsoDuration:
    def test_none(self):
        assert _iso8601_duration_to_seconds(None) == 0

    def test_empty(self):
        assert _iso8601_duration_to_seconds("") == 0

    def test_seconds_only(self):
        assert _iso8601_duration_to_seconds("PT30S") == 30

    def test_minutes_seconds(self):
        assert _iso8601_duration_to_seconds("PT2M30S") == 150

    def test_hours(self):
        assert _iso8601_duration_to_seconds("PT1H2M3S") == 3723

    def test_malformed(self):
        assert _iso8601_duration_to_seconds("not-iso") == 0


class TestApiRecordNormalization:
    def test_minimum_record(self):
        out = _api_record_to_bd_shape({"video_id": "abc"})
        assert out["video_id"] == "abc"
        assert out["url"] == "https://www.youtube.com/watch?v=abc"
        assert out["top_comments"] == []

    def test_full_record(self):
        rec = {
            "video_id": "abc",
            "title": "T",
            "description": "D",
            "tags": ["a", "b"],
            "category_id": 22,
            "duration_iso8601": "PT5M",
            "view_count": 1000,
            "like_count": 50,
            "comment_count": 10,
            "published_at": "2026-01-01T00:00:00Z",
            "has_captions": True,
            "live_broadcast_content": "none",
        }
        out = _api_record_to_bd_shape(rec)
        assert out["views"] == 1000
        assert out["view_count"] == 1000
        assert out["likes"] == 50
        assert out["num_comments"] == 10
        assert out["length"] == 300
        assert out["duration_seconds"] == 300
        assert out["has_captions"] is True
        assert out["is_live"] is False

    def test_live_broadcast(self):
        out = _api_record_to_bd_shape(
            {"video_id": "v", "live_broadcast_content": "live"}
        )
        assert out["is_live"] is True
        assert out["is_livestream"] is True


class TestScrapeVideosDiscovery:
    def test_api_primary_when_available(self):
        yt_api = MagicMock()
        yt_api.available = True
        yt_api.list_channel_uploads.return_value = [
            {
                "video_id": "v1",
                "title": "T",
                "view_count": 100,
                "duration_iso8601": "PT1M",
            }
        ]
        bd = MagicMock()
        out = scrape_videos_discovery(
            bd, "https://yt.com/@x", num_videos=5,
            yt_api=yt_api, channel_id="UCx",
        )
        assert len(out) == 1
        assert out[0]["video_id"] == "v1"
        # Bright Data should NOT have been called
        bd.scrape_and_wait.assert_not_called()

    def test_falls_back_to_bd_when_no_channel_id(self):
        yt_api = MagicMock()
        yt_api.available = True
        bd = MagicMock()
        bd.scrape_and_wait.return_value = [{"video_id": "bd_v"}]
        out = scrape_videos_discovery(
            bd, "https://yt.com/@x", num_videos=5,
            yt_api=yt_api, channel_id=None,
        )
        assert out[0]["video_id"] == "bd_v"
        bd.scrape_and_wait.assert_called_once()

    def test_falls_back_to_bd_when_no_yt_api(self):
        bd = MagicMock()
        bd.scrape_and_wait.return_value = []
        scrape_videos_discovery(bd, "https://yt.com/@x", num_videos=5)
        bd.scrape_and_wait.assert_called_once()

    def test_prefer_bd_env_var(self, monkeypatch):
        monkeypatch.setenv("YT_SCRAPER_PREFER_BRIGHTDATA", "1")
        yt_api = MagicMock()
        yt_api.available = True
        bd = MagicMock()
        bd.scrape_and_wait.return_value = []
        scrape_videos_discovery(
            bd, "https://yt.com/@x", num_videos=5,
            yt_api=yt_api, channel_id="UCx",
        )
        bd.scrape_and_wait.assert_called_once()
        yt_api.list_channel_uploads.assert_not_called()


class TestVideoIdFromUrl:
    def test_watch_url(self):
        assert _video_id_from_url("https://yt.com/watch?v=abc") == "abc"

    def test_watch_with_params(self):
        assert _video_id_from_url("https://yt.com/watch?v=abc&t=30") == "abc"

    def test_short_url(self):
        assert _video_id_from_url("https://youtu.be/abc") == "abc"

    def test_shorts_url(self):
        assert _video_id_from_url("https://yt.com/shorts/abc") == "abc"

    def test_empty(self):
        assert _video_id_from_url("") is None

    def test_unknown(self):
        assert _video_id_from_url("https://example.com") is None


class TestScrapeComments:
    def test_api_primary(self):
        yt_api = MagicMock()
        yt_api.available = True
        yt_api.list_comment_threads.return_value = [
            {"author": "u", "text": "t"}
        ]
        bd = MagicMock()
        out = scrape_comments(
            bd, ["https://yt.com/watch?v=v1"], yt_api=yt_api
        )
        assert len(out) == 1
        assert out[0]["author"] == "u"
        bd.trigger_and_wait.assert_not_called()

    def test_api_aggregates_across_videos(self):
        yt_api = MagicMock()
        yt_api.available = True
        yt_api.list_comment_threads.side_effect = [
            [{"author": "u1", "text": "a"}],
            [{"author": "u2", "text": "b"}],
        ]
        bd = MagicMock()
        out = scrape_comments(
            bd,
            [
                "https://yt.com/watch?v=v1",
                "https://youtu.be/v2",
            ],
            yt_api=yt_api,
        )
        assert len(out) == 2

    def test_api_skips_invalid_urls(self):
        yt_api = MagicMock()
        yt_api.available = True
        yt_api.list_comment_threads.return_value = []
        bd = MagicMock()
        out = scrape_comments(bd, ["https://example.com"], yt_api=yt_api)
        assert out == []
        yt_api.list_comment_threads.assert_not_called()

    def test_bd_fallback(self):
        bd = MagicMock()
        bd.trigger_and_wait.return_value = [{"author": "bd"}]
        out = scrape_comments(bd, ["https://yt.com/watch?v=v1"])
        assert out[0]["author"] == "bd"

    def test_prefer_bd_env(self, monkeypatch):
        monkeypatch.setenv("YT_SCRAPER_PREFER_BRIGHTDATA", "1")
        yt_api = MagicMock()
        yt_api.available = True
        bd = MagicMock()
        bd.trigger_and_wait.return_value = []
        scrape_comments(bd, ["https://yt.com/watch?v=v1"], yt_api=yt_api)
        bd.trigger_and_wait.assert_called_once()
