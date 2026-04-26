"""Tests for build_youtube_creator_intelligence_profile_batch + existing_creator_id."""

from unittest.mock import MagicMock, patch

import pytest

from pipeline import db as pdb
from pipeline.pipeline import (
    build_youtube_creator_intelligence_profile_batch,
)


class TestStoreFullCipExistingCreatorId:
    def test_passes_through_when_set(self, monkeypatch):
        db = MagicMock()
        # Mock all the downstream writers
        monkeypatch.setattr(
            pdb, "upsert_social_profile", MagicMock(return_value="csp")
        )
        monkeypatch.setattr(pdb, "upsert_posts", MagicMock())
        monkeypatch.setattr(pdb, "insert_creator_scores", MagicMock())
        monkeypatch.setattr(pdb, "insert_caption_intelligence", MagicMock())
        monkeypatch.setattr(pdb, "insert_transcript_intelligence", MagicMock())
        monkeypatch.setattr(pdb, "insert_audience_intelligence", MagicMock())

        upsert_creator_mock = MagicMock()
        monkeypatch.setattr(pdb, "upsert_creator", upsert_creator_mock)

        cip = {"profile": {"handle": "x"}, "_raw_posts": [], "scores": {"cpi": 50}}
        out = pdb.store_full_cip(db, cip, existing_creator_id="existing-id")
        assert out == "existing-id"
        # upsert_creator should NOT have been called (we used existing id)
        upsert_creator_mock.assert_not_called()
        pdb.upsert_social_profile.assert_called_once()
        args = pdb.upsert_social_profile.call_args.args
        assert args[1] == "existing-id"
        assert args[2] == "instagram"

    def test_without_existing_creator_id_uses_upsert_creator(self, monkeypatch):
        db = MagicMock()
        monkeypatch.setattr(
            pdb, "upsert_creator", MagicMock(return_value="new-id")
        )
        monkeypatch.setattr(pdb, "upsert_posts", MagicMock())
        monkeypatch.setattr(pdb, "insert_creator_scores", MagicMock())
        monkeypatch.setattr(pdb, "upsert_social_profile", MagicMock())
        monkeypatch.setattr(pdb, "insert_caption_intelligence", MagicMock())
        monkeypatch.setattr(pdb, "insert_transcript_intelligence", MagicMock())
        monkeypatch.setattr(pdb, "insert_audience_intelligence", MagicMock())

        cip = {"profile": {"handle": "x"}, "_raw_posts": [], "scores": {"cpi": 60}}
        out = pdb.store_full_cip(db, cip)
        assert out == "new-id"
        pdb.upsert_creator.assert_called_once()
        pdb.upsert_social_profile.assert_not_called()


class TestStoreYoutubeCipExistingCreatorId:
    def test_bypasses_lookup_when_set(self, monkeypatch):
        db = MagicMock()
        monkeypatch.setattr(pdb, "upsert_social_profile", MagicMock())
        monkeypatch.setattr(pdb, "upsert_youtube_videos", MagicMock())
        monkeypatch.setattr(pdb, "upsert_creator_score_platform", MagicMock())
        find_mock = MagicMock()
        create_mock = MagicMock()
        monkeypatch.setattr(pdb, "_find_creator_by_platform_profile", find_mock)
        monkeypatch.setattr(pdb, "_create_youtube_creator_shell", create_mock)

        cip = {
            "profile": {"handle": "x"},
            "resolved": {"channel_id": "UCx"},
            "videos": [],
            "scores": {"cpi": 70},
        }
        out = pdb.store_youtube_cip(db, cip, existing_creator_id="e-id")
        assert out == "e-id"
        find_mock.assert_not_called()
        create_mock.assert_not_called()
        pdb.upsert_social_profile.assert_called_once()
        assert pdb.upsert_social_profile.call_args.args[1] == "e-id"


class TestBatchOrchestrator:
    def test_empty_list_returns_empty(self):
        out = build_youtube_creator_intelligence_profile_batch(
            [], "bd", "g", "o"
        )
        assert out == []

    def test_per_creator_fanout(self, monkeypatch):
        """Batch channel scrape returns 2 records → 2 CIPs back."""
        # Mock BrightdataClient constructor and scrape_channels
        monkeypatch.setattr(
            "pipeline.pipeline.BrightdataClient",
            MagicMock(return_value=MagicMock()),
        )
        # YouTube API unavailable — batch falls back to BD-only path
        yt_api_mock = MagicMock()
        yt_api_mock.available = False
        monkeypatch.setattr(
            "pipeline.youtube.youtube_api.YouTubeAPIClient",
            MagicMock(return_value=yt_api_mock),
        )
        # Resolve: return dummy ResolvedChannel per URL
        def fake_resolve(url, api=None):
            r = MagicMock()
            r.url = url
            r.handle = url.split("@")[-1]
            r.channel_id = None
            return r
        monkeypatch.setattr(
            "pipeline.youtube.handle_resolver.resolve", fake_resolve
        )
        # scrape_channels returns one record per URL
        monkeypatch.setattr(
            "pipeline.youtube.scraper_channels.scrape_channels",
            MagicMock(
                return_value=[
                    {"url": "https://www.youtube.com/@a", "handle": "a"},
                    {"url": "https://www.youtube.com/@b", "handle": "b"},
                ]
            ),
        )
        # Stub the per-creator worker so we don't actually run the full
        # pipeline — we're testing the batch shape, not the worker logic.
        worker_mock = MagicMock(
            side_effect=[
                {"platform": "youtube", "profile_url": "u1", "scores": {"cpi": 1}},
                {"platform": "youtube", "profile_url": "u2", "scores": {"cpi": 2}},
            ]
        )
        monkeypatch.setattr(
            "pipeline.pipeline._build_yt_cip_with_preloaded_channel",
            worker_mock,
        )

        out = build_youtube_creator_intelligence_profile_batch(
            ["https://www.youtube.com/@a", "https://www.youtube.com/@b"],
            brightdata_token="bd",
            gemini_api_key="g",
            openai_api_key="o",
            max_workers=2,
        )
        assert len(out) == 2
        # Worker called once per URL
        assert worker_mock.call_count == 2

    def test_per_creator_error_isolation(self, monkeypatch):
        """One creator's worker raises → other succeeds, failure reported."""
        monkeypatch.setattr(
            "pipeline.pipeline.BrightdataClient",
            MagicMock(return_value=MagicMock()),
        )
        yt_api_mock = MagicMock()
        yt_api_mock.available = False
        monkeypatch.setattr(
            "pipeline.youtube.youtube_api.YouTubeAPIClient",
            MagicMock(return_value=yt_api_mock),
        )
        monkeypatch.setattr(
            "pipeline.youtube.handle_resolver.resolve",
            MagicMock(return_value=MagicMock(url="u", handle="h", channel_id=None)),
        )
        monkeypatch.setattr(
            "pipeline.youtube.scraper_channels.scrape_channels",
            MagicMock(return_value=[{"url": "u"}]),
        )

        def worker(**kwargs):
            if kwargs["original_url"] == "bad":
                raise RuntimeError("worker boom")
            return {"platform": "youtube", "profile_url": kwargs["original_url"]}

        monkeypatch.setattr(
            "pipeline.pipeline._build_yt_cip_with_preloaded_channel", worker
        )

        out = build_youtube_creator_intelligence_profile_batch(
            ["good", "bad"],
            brightdata_token="bd",
            gemini_api_key="g",
            openai_api_key="o",
        )
        # Both return entries; the bad one has an error key
        assert len(out) == 2
        errors = [c for c in out if c.get("error")]
        assert len(errors) == 1
        assert "worker boom" in errors[0]["error"]

    def test_batched_channel_scrape_failure_degrades_gracefully(self, monkeypatch):
        """BD channel scrape raises → we still try per-creator workers
        with empty channel records (API path can still work)."""
        monkeypatch.setattr(
            "pipeline.pipeline.BrightdataClient",
            MagicMock(return_value=MagicMock()),
        )
        yt_api_mock = MagicMock()
        yt_api_mock.available = False
        monkeypatch.setattr(
            "pipeline.youtube.youtube_api.YouTubeAPIClient",
            MagicMock(return_value=yt_api_mock),
        )
        monkeypatch.setattr(
            "pipeline.youtube.handle_resolver.resolve",
            MagicMock(return_value=MagicMock(url="u", handle="h", channel_id=None)),
        )
        monkeypatch.setattr(
            "pipeline.youtube.scraper_channels.scrape_channels",
            MagicMock(side_effect=RuntimeError("BD down")),
        )
        worker_mock = MagicMock(
            return_value={"platform": "youtube", "profile_url": "u"}
        )
        monkeypatch.setattr(
            "pipeline.pipeline._build_yt_cip_with_preloaded_channel",
            worker_mock,
        )

        out = build_youtube_creator_intelligence_profile_batch(
            ["u"], brightdata_token="bd",
            gemini_api_key="g", openai_api_key="o",
        )
        assert len(out) == 1
        # Worker was still called (with empty raw_channel)
        worker_mock.assert_called_once()
