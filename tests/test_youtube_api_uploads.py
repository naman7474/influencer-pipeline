"""Tests for YouTubeAPIClient.list_channel_uploads + list_comment_threads."""

from unittest.mock import MagicMock

import pipeline.youtube.youtube_api as yt_api_module
from pipeline.youtube.youtube_api import YouTubeAPIClient


def _make_client(monkeypatch, service_mock):
    monkeypatch.setattr(
        yt_api_module, "build", MagicMock(return_value=service_mock)
    )
    return YouTubeAPIClient(api_key="test-key")


class TestFetchUploadsPlaylistId:
    def test_unavailable_client(self, monkeypatch):
        monkeypatch.setattr(yt_api_module, "build", None)
        c = YouTubeAPIClient(api_key="k")
        assert c._fetch_uploads_playlist_id("UC1") is None

    def test_empty_channel_id(self, monkeypatch):
        c = _make_client(monkeypatch, MagicMock())
        assert c._fetch_uploads_playlist_id("") is None

    def test_happy_path(self, monkeypatch):
        svc = MagicMock()
        svc.channels().list().execute.return_value = {
            "items": [
                {
                    "contentDetails": {
                        "relatedPlaylists": {"uploads": "UU1_uploads"}
                    }
                }
            ]
        }
        c = _make_client(monkeypatch, svc)
        assert c._fetch_uploads_playlist_id("UC1") == "UU1_uploads"

    def test_no_items(self, monkeypatch):
        svc = MagicMock()
        svc.channels().list().execute.return_value = {"items": []}
        c = _make_client(monkeypatch, svc)
        assert c._fetch_uploads_playlist_id("UC1") is None

    def test_http_error(self, monkeypatch):
        monkeypatch.setattr(yt_api_module, "HttpError", Exception)
        svc = MagicMock()
        svc.channels().list().execute.side_effect = Exception("boom")
        c = _make_client(monkeypatch, svc)
        assert c._fetch_uploads_playlist_id("UC1") is None


class TestListChannelUploads:
    def test_unavailable(self, monkeypatch):
        monkeypatch.setattr(yt_api_module, "build", None)
        c = YouTubeAPIClient(api_key="k")
        assert c.list_channel_uploads("UC1") == []

    def test_empty_channel_id(self, monkeypatch):
        c = _make_client(monkeypatch, MagicMock())
        assert c.list_channel_uploads("") == []

    def test_no_uploads_playlist(self, monkeypatch):
        svc = MagicMock()
        svc.channels().list().execute.return_value = {"items": []}
        c = _make_client(monkeypatch, svc)
        assert c.list_channel_uploads("UC1") == []

    def test_happy_path(self, monkeypatch):
        svc = MagicMock()
        # channels.list → uploads playlist id
        svc.channels().list().execute.return_value = {
            "items": [
                {"contentDetails": {"relatedPlaylists": {"uploads": "UU1"}}}
            ]
        }
        # playlistItems.list → two video ids
        svc.playlistItems().list().execute.return_value = {
            "items": [
                {"contentDetails": {"videoId": "v1"}},
                {"contentDetails": {"videoId": "v2"}},
            ]
        }
        # videos.list (called by fetch_video_stats) → full records
        svc.videos().list().execute.return_value = {
            "items": [
                {
                    "id": "v1",
                    "snippet": {
                        "title": "T1",
                        "publishedAt": "2026-01-01T00:00:00Z",
                        "channelId": "UC1",
                    },
                    "statistics": {
                        "viewCount": "100",
                        "likeCount": "10",
                        "commentCount": "2",
                    },
                    "contentDetails": {
                        "duration": "PT2M",
                        "caption": "true",
                    },
                    "topicDetails": {"topicCategories": []},
                },
                {
                    "id": "v2",
                    "snippet": {
                        "title": "T2",
                        "publishedAt": "2026-01-02T00:00:00Z",
                        "channelId": "UC1",
                    },
                    "statistics": {
                        "viewCount": "200",
                        "likeCount": "20",
                        "commentCount": "4",
                    },
                    "contentDetails": {
                        "duration": "PT5M",
                        "caption": "false",
                    },
                    "topicDetails": {"topicCategories": []},
                },
            ]
        }
        c = _make_client(monkeypatch, svc)
        out = c.list_channel_uploads("UC1", limit=10)
        assert len(out) == 2
        # Preserves uploads-order (v1 before v2)
        assert out[0]["video_id"] == "v1"
        assert out[0]["view_count"] == 100
        assert out[1]["video_id"] == "v2"

    def test_pagination_stops_at_limit(self, monkeypatch):
        """Ensure we stop pulling playlist pages once we've hit `limit`."""
        svc = MagicMock()
        svc.channels().list().execute.return_value = {
            "items": [{"contentDetails": {"relatedPlaylists": {"uploads": "UU"}}}]
        }

        call_count = {"n": 0}

        def playlist_execute():
            call_count["n"] += 1
            # First page returns 50 items + pageToken; caller should STOP
            # pulling because limit=50.
            return {
                "items": [
                    {"contentDetails": {"videoId": f"v{i}"}} for i in range(50)
                ],
                "nextPageToken": "next",
            }

        svc.playlistItems().list().execute.side_effect = playlist_execute
        svc.videos().list().execute.return_value = {
            "items": [
                {
                    "id": f"v{i}",
                    "snippet": {"title": "t", "channelId": "UC"},
                    "statistics": {"viewCount": "1"},
                    "contentDetails": {"duration": "PT1M"},
                }
                for i in range(50)
            ]
        }
        c = _make_client(monkeypatch, svc)
        out = c.list_channel_uploads("UC", limit=50)
        assert len(out) == 50


class TestListCommentThreads:
    def test_unavailable(self, monkeypatch):
        monkeypatch.setattr(yt_api_module, "build", None)
        c = YouTubeAPIClient(api_key="k")
        assert c.list_comment_threads("v1") == []

    def test_empty_video_id(self, monkeypatch):
        c = _make_client(monkeypatch, MagicMock())
        assert c.list_comment_threads("") == []

    def test_http_error_swallowed(self, monkeypatch):
        monkeypatch.setattr(yt_api_module, "HttpError", Exception)
        svc = MagicMock()
        svc.commentThreads().list().execute.side_effect = Exception("429")
        c = _make_client(monkeypatch, svc)
        assert c.list_comment_threads("v1") == []

    def test_normalization(self, monkeypatch):
        svc = MagicMock()
        svc.commentThreads().list().execute.return_value = {
            "items": [
                {
                    "snippet": {
                        "topLevelComment": {
                            "snippet": {
                                "authorDisplayName": "@u1",
                                "authorChannelId": {"value": "UC_u1"},
                                "textDisplay": "nice video",
                                "publishedAt": "2026-01-01T00:00:00Z",
                                "likeCount": 5,
                            }
                        },
                        "totalReplyCount": 1,
                    },
                    "replies": {
                        "comments": [
                            {
                                "snippet": {
                                    "authorDisplayName": "creator",
                                    "authorChannelId": {"value": "UC_creator"},
                                    "textDisplay": "thanks!",
                                    "publishedAt": "2026-01-01T01:00:00Z",
                                    "likeCount": 10,
                                }
                            }
                        ]
                    },
                }
            ]
        }
        c = _make_client(monkeypatch, svc)
        out = c.list_comment_threads("v1", order="relevance", max_results=5)
        assert len(out) == 1
        t = out[0]
        assert t["author"] == "@u1"
        assert t["author_channel_id"] == "UC_u1"
        assert t["text"] == "nice video"
        assert t["like_count"] == 5
        assert t["reply_count"] == 1
        assert len(t["replies"]) == 1
        assert t["replies"][0]["author_channel_id"] == "UC_creator"

    def test_no_replies(self, monkeypatch):
        svc = MagicMock()
        svc.commentThreads().list().execute.return_value = {
            "items": [
                {
                    "snippet": {
                        "topLevelComment": {
                            "snippet": {
                                "authorDisplayName": "u",
                                "textDisplay": "t",
                                "publishedAt": "2026",
                                "likeCount": 0,
                            }
                        }
                    }
                }
            ]
        }
        c = _make_client(monkeypatch, svc)
        out = c.list_comment_threads("v1")
        assert out[0]["replies"] == []
        assert out[0]["reply_count"] == 0
