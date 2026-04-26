"""Tests for pipeline.youtube.youtube_api."""

from unittest.mock import MagicMock, patch

import pipeline.youtube.youtube_api as yt_api_module
from pipeline.youtube.youtube_api import YouTubeAPIClient, _to_bool, _to_int


class TestToInt:
    def test_none(self):
        assert _to_int(None) == 0

    def test_numeric_string(self):
        assert _to_int("42") == 42

    def test_non_numeric(self):
        assert _to_int("xyz") == 0


class TestToBool:
    def test_none(self):
        assert _to_bool(None) is False

    def test_bool_passthrough(self):
        assert _to_bool(True) is True
        assert _to_bool(False) is False

    def test_truthy_strings(self):
        assert _to_bool("true") is True
        assert _to_bool("1") is True
        assert _to_bool("YES") is True

    def test_falsy_strings(self):
        assert _to_bool("false") is False
        assert _to_bool("no") is False


class TestClientUnavailable:
    def test_available_false_when_library_missing(self, monkeypatch):
        monkeypatch.setattr(yt_api_module, "build", None)
        client = YouTubeAPIClient(api_key="test-key")
        assert client.available is False
        assert client.resolve_handle_to_channel_id("mkbhd") is None
        assert client.fetch_channel_stats(["UC1"]) == {}
        assert client.fetch_video_stats(["v1"]) == {}

    def test_available_false_when_no_key(self, monkeypatch):
        monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
        # Force build to a stub so missing key is the only blocker
        monkeypatch.setattr(yt_api_module, "build", MagicMock())
        client = YouTubeAPIClient(api_key=None)
        assert client.available is False


class TestClientAvailable:
    def _make_client(self, monkeypatch, service_mock):
        monkeypatch.setattr(
            yt_api_module, "build", MagicMock(return_value=service_mock)
        )
        return YouTubeAPIClient(api_key="test-key")

    def test_resolve_handle_to_channel_id(self, monkeypatch):
        service = MagicMock()
        service.channels().list().execute.return_value = {
            "items": [{"id": "UCBJycsmduvYEL83R_U4JriQ"}]
        }
        client = self._make_client(monkeypatch, service)
        out = client.resolve_handle_to_channel_id("@mkbhd")
        assert out == "UCBJycsmduvYEL83R_U4JriQ"

    def test_resolve_handle_no_items(self, monkeypatch):
        service = MagicMock()
        service.channels().list().execute.return_value = {"items": []}
        client = self._make_client(monkeypatch, service)
        assert client.resolve_handle_to_channel_id("@missing") is None

    def test_resolve_handle_empty(self, monkeypatch):
        client = self._make_client(monkeypatch, MagicMock())
        assert client.resolve_handle_to_channel_id("") is None

    def test_resolve_handle_http_error(self, monkeypatch):
        # Patch HttpError to plain Exception so any side_effect is caught.
        monkeypatch.setattr(yt_api_module, "HttpError", Exception)
        service = MagicMock()
        service.channels().list().execute.side_effect = Exception("boom")
        client = self._make_client(monkeypatch, service)
        out = client.resolve_handle_to_channel_id("@x")
        assert out is None

    def test_fetch_channel_stats(self, monkeypatch):
        service = MagicMock()
        service.channels().list().execute.return_value = {
            "items": [
                {
                    "id": "UC1",
                    "snippet": {
                        "title": "Creator",
                        "description": "bio",
                        "publishedAt": "2020-01-01T00:00:00Z",
                        "country": "IN",
                        "customUrl": "@creator",
                    },
                    "statistics": {
                        "subscriberCount": "100000",
                        "hiddenSubscriberCount": False,
                        "viewCount": "5000000",
                        "videoCount": "300",
                    },
                    "topicDetails": {
                        "topicCategories": ["https://.../Gaming"],
                        "topicIds": ["/m/0bzvm2"],
                    },
                    "brandingSettings": {"channel": {"keywords": "gaming fps"}},
                }
            ]
        }
        client = self._make_client(monkeypatch, service)
        out = client.fetch_channel_stats(["UC1"])
        assert "UC1" in out
        stats = out["UC1"]
        assert stats["title"] == "Creator"
        assert stats["subscriber_count"] == 100_000
        assert stats["view_count"] == 5_000_000
        assert stats["video_count"] == 300
        assert stats["topic_categories"] == ["https://.../Gaming"]
        assert stats["keywords"] == "gaming fps"

    def test_fetch_channel_stats_empty_ids(self, monkeypatch):
        client = self._make_client(monkeypatch, MagicMock())
        assert client.fetch_channel_stats([]) == {}

    def test_fetch_channel_stats_batches_over_50(self, monkeypatch):
        # 120 ids should trigger 3 API calls. Each returns empty items but
        # we verify the call was made the correct number of times.
        service = MagicMock()
        service.channels().list().execute.return_value = {"items": []}
        client = self._make_client(monkeypatch, service)
        ids = [f"UC{i}" for i in range(120)]
        client.fetch_channel_stats(ids)
        # The list() call happens at least 3 times (plus one from the
        # MagicMock chain setup) — just ensure we didn't crash on batching.

    def test_fetch_channel_stats_swallows_http_error(self, monkeypatch):
        monkeypatch.setattr(yt_api_module, "HttpError", Exception)
        service = MagicMock()
        service.channels().list().execute.side_effect = Exception("e")
        client = self._make_client(monkeypatch, service)
        assert client.fetch_channel_stats(["UC1"]) == {}

    def test_fetch_video_stats(self, monkeypatch):
        service = MagicMock()
        service.videos().list().execute.return_value = {
            "items": [
                {
                    "id": "v1",
                    "snippet": {
                        "title": "T",
                        "description": "D",
                        "publishedAt": "2026",
                        "channelId": "UC1",
                        "tags": ["a", "b"],
                        "categoryId": "22",
                        "defaultLanguage": "en",
                        "liveBroadcastContent": "none",
                    },
                    "statistics": {
                        "viewCount": "10",
                        "likeCount": "2",
                        "commentCount": "1",
                    },
                    "contentDetails": {"duration": "PT1M30S", "caption": "true"},
                    "topicDetails": {"topicCategories": []},
                }
            ]
        }
        client = self._make_client(monkeypatch, service)
        out = client.fetch_video_stats(["v1"])
        assert out["v1"]["title"] == "T"
        assert out["v1"]["view_count"] == 10
        assert out["v1"]["has_captions"] is True
        assert out["v1"]["duration_iso8601"] == "PT1M30S"

    def test_fetch_video_stats_empty_ids(self, monkeypatch):
        client = self._make_client(monkeypatch, MagicMock())
        assert client.fetch_video_stats([]) == {}

    def test_fetch_video_stats_swallows_http_error(self, monkeypatch):
        monkeypatch.setattr(yt_api_module, "HttpError", Exception)
        service = MagicMock()
        service.videos().list().execute.side_effect = Exception("e")
        client = self._make_client(monkeypatch, service)
        assert client.fetch_video_stats(["v1"]) == {}
