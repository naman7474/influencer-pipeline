"""Tests for Phase 2.5 auto-stitch: YT → IG and IG → YT fanouts."""

from unittest.mock import MagicMock, patch

import pytest

from pipeline import handlers
from pipeline.youtube.stitching import StitchCandidate


class TestHasPlatformProfile:
    def test_returns_true_when_row_exists(self):
        db = MagicMock()
        res = MagicMock()
        res.data = [{"id": "csp-1"}]
        db.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = res
        assert handlers._has_platform_profile(db, "c-1", "instagram") is True

    def test_returns_false_when_empty(self):
        db = MagicMock()
        res = MagicMock()
        res.data = []
        db.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = res
        assert handlers._has_platform_profile(db, "c-1", "youtube") is False


class TestRecordStitchCandidate:
    def test_writes_row(self):
        db = MagicMock()
        cand = StitchCandidate(
            source_platform="youtube",
            source_handle="mkbhd",
            target_platform="instagram",
            target_handle="mkbhd",
            confidence=1.0,
            reason="direct link",
        )
        handlers._record_stitch_candidate(
            db, "source-id", "target-id", cand, status="pending"
        )
        db.table.assert_called_with("stitch_candidates")
        args = db.table.return_value.insert.call_args.args
        row = args[0]
        assert row["source_creator_id"] == "source-id"
        assert row["target_creator_id"] == "target-id"
        assert row["source_platform"] == "youtube"
        assert row["target_platform"] == "instagram"
        assert row["target_handle"] == "mkbhd"
        assert row["confidence"] == 1.0
        assert row["status"] == "pending"

    def test_swallows_duplicate_error(self):
        db = MagicMock()
        db.table.return_value.insert.return_value.execute.side_effect = (
            RuntimeError("unique violation")
        )
        cand = StitchCandidate(
            source_platform="youtube", source_handle="x",
            target_platform="instagram", target_handle="y",
            confidence=1.0, reason="r",
        )
        # Should not raise
        handlers._record_stitch_candidate(db, "s", None, cand)


class TestRunAutoStitchFromYt:
    def test_no_external_links_no_fanout(self, monkeypatch):
        db = MagicMock()
        cip = {"profile": {"handle": "x", "external_links": []}}
        enqueue_mock = MagicMock()
        monkeypatch.setattr(handlers, "_enqueue_auto_stitch_ig_scrape", enqueue_mock)
        handlers._run_auto_stitch_from_yt(db, "c-1", cip)
        enqueue_mock.assert_not_called()

    def test_ig_link_fanout_happy_path(self, monkeypatch):
        db = MagicMock()
        # Creator has no IG profile yet
        monkeypatch.setattr(
            handlers, "_has_platform_profile",
            MagicMock(return_value=False),
        )
        # IG handle doesn't collide with another existing creator
        monkeypatch.setattr(
            handlers.pdb, "_find_creator_by_platform_profile",
            MagicMock(return_value=None),
        )
        enqueue_mock = MagicMock()
        monkeypatch.setattr(
            handlers, "_enqueue_auto_stitch_ig_scrape", enqueue_mock
        )
        record_mock = MagicMock()
        monkeypatch.setattr(
            handlers, "_record_stitch_candidate", record_mock
        )

        cip = {
            "profile": {
                "handle": "mkbhd",
                "external_links": [
                    {"label": "IG", "url": "https://instagram.com/mkbhd"}
                ],
            }
        }
        handlers._run_auto_stitch_from_yt(db, "c-1", cip)
        enqueue_mock.assert_called_once()
        args = enqueue_mock.call_args.args
        assert args[1] == "mkbhd"           # target_handle
        assert args[2] == "c-1"             # existing_creator_id
        assert args[3] == 1.0               # confidence
        record_mock.assert_not_called()

    def test_ig_profile_already_exists_no_fanout(self, monkeypatch):
        db = MagicMock()
        monkeypatch.setattr(
            handlers, "_has_platform_profile",
            MagicMock(return_value=True),
        )
        enqueue_mock = MagicMock()
        monkeypatch.setattr(
            handlers, "_enqueue_auto_stitch_ig_scrape", enqueue_mock
        )

        cip = {
            "profile": {
                "handle": "mkbhd",
                "external_links": [
                    {"label": "IG", "url": "https://instagram.com/mkbhd"}
                ],
            }
        }
        handlers._run_auto_stitch_from_yt(db, "c-1", cip)
        enqueue_mock.assert_not_called()

    def test_cross_creator_collision_flags_stitch_candidate(self, monkeypatch):
        db = MagicMock()
        monkeypatch.setattr(
            handlers, "_has_platform_profile",
            MagicMock(return_value=False),
        )
        # IG handle already belongs to creator 'other-id'
        monkeypatch.setattr(
            handlers.pdb, "_find_creator_by_platform_profile",
            MagicMock(return_value="other-id"),
        )
        enqueue_mock = MagicMock()
        record_mock = MagicMock()
        monkeypatch.setattr(
            handlers, "_enqueue_auto_stitch_ig_scrape", enqueue_mock
        )
        monkeypatch.setattr(
            handlers, "_record_stitch_candidate", record_mock
        )

        cip = {
            "profile": {
                "handle": "mkbhd",
                "external_links": [
                    {"label": "IG", "url": "https://instagram.com/mkbhd"}
                ],
            }
        }
        handlers._run_auto_stitch_from_yt(db, "c-1", cip)
        enqueue_mock.assert_not_called()
        record_mock.assert_called_once()
        args = record_mock.call_args.args
        # Args: (db, source_creator_id, target_creator_id, candidate)
        assert args[1] == "c-1"
        assert args[2] == "other-id"

    def test_same_creator_no_collision(self, monkeypatch):
        """_find_creator_by_platform_profile returns the same id — not a collision."""
        db = MagicMock()
        monkeypatch.setattr(
            handlers, "_has_platform_profile",
            MagicMock(return_value=False),
        )
        monkeypatch.setattr(
            handlers.pdb, "_find_creator_by_platform_profile",
            MagicMock(return_value="c-1"),  # same as source
        )
        enqueue_mock = MagicMock()
        monkeypatch.setattr(
            handlers, "_enqueue_auto_stitch_ig_scrape", enqueue_mock
        )

        cip = {
            "profile": {
                "handle": "mkbhd",
                "external_links": [
                    {"url": "https://instagram.com/mkbhd"}
                ],
            }
        }
        handlers._run_auto_stitch_from_yt(db, "c-1", cip)
        # Not a collision — fanout proceeds
        enqueue_mock.assert_called_once()


class TestRunAutoStitchFromIg:
    def test_no_external_url_no_fanout(self, monkeypatch):
        db = MagicMock()
        enqueue_mock = MagicMock()
        monkeypatch.setattr(
            handlers, "_enqueue_auto_stitch_yt_scrape", enqueue_mock
        )
        handlers._run_auto_stitch_from_ig(db, "c-1", {"profile": {}})
        enqueue_mock.assert_not_called()

    def test_non_yt_url_no_fanout(self, monkeypatch):
        db = MagicMock()
        enqueue_mock = MagicMock()
        monkeypatch.setattr(
            handlers, "_enqueue_auto_stitch_yt_scrape", enqueue_mock
        )
        cip = {"profile": {"external_url": "https://example.com/shop"}}
        handlers._run_auto_stitch_from_ig(db, "c-1", cip)
        enqueue_mock.assert_not_called()

    def test_yt_url_fanout(self, monkeypatch):
        db = MagicMock()
        monkeypatch.setattr(
            handlers, "_has_platform_profile",
            MagicMock(return_value=False),
        )
        monkeypatch.setattr(
            handlers.pdb, "_find_creator_by_platform_profile",
            MagicMock(return_value=None),
        )
        enqueue_mock = MagicMock()
        monkeypatch.setattr(
            handlers, "_enqueue_auto_stitch_yt_scrape", enqueue_mock
        )

        cip = {
            "profile": {
                "handle": "mkbhd",
                "external_url": "https://www.youtube.com/@mkbhd",
            }
        }
        handlers._run_auto_stitch_from_ig(db, "c-1", cip)
        enqueue_mock.assert_called_once()
        args = enqueue_mock.call_args.args
        assert "youtube.com/@mkbhd" in args[1]
        assert args[2] == "c-1"

    def test_yt_already_present_no_fanout(self, monkeypatch):
        db = MagicMock()
        monkeypatch.setattr(
            handlers, "_has_platform_profile",
            MagicMock(return_value=True),
        )
        enqueue_mock = MagicMock()
        monkeypatch.setattr(
            handlers, "_enqueue_auto_stitch_yt_scrape", enqueue_mock
        )
        cip = {
            "profile": {
                "handle": "x",
                "external_url": "https://youtube.com/@x",
            }
        }
        handlers._run_auto_stitch_from_ig(db, "c-1", cip)
        enqueue_mock.assert_not_called()

    def test_ig_yt_collision_flags(self, monkeypatch):
        db = MagicMock()
        monkeypatch.setattr(
            handlers, "_has_platform_profile",
            MagicMock(return_value=False),
        )
        monkeypatch.setattr(
            handlers.pdb, "_find_creator_by_platform_profile",
            MagicMock(return_value="other-creator"),
        )
        enqueue_mock = MagicMock()
        record_mock = MagicMock()
        monkeypatch.setattr(handlers, "_enqueue_auto_stitch_yt_scrape", enqueue_mock)
        monkeypatch.setattr(handlers, "_record_stitch_candidate", record_mock)

        cip = {
            "profile": {
                "handle": "x",
                "external_url": "https://youtube.com/@claimed",
            }
        }
        handlers._run_auto_stitch_from_ig(db, "c-1", cip)
        enqueue_mock.assert_not_called()
        record_mock.assert_called_once()


class TestEnqueueAutoStitchJobs:
    def test_ig_job_shape(self):
        db = MagicMock()
        handlers._enqueue_auto_stitch_ig_scrape(db, "target_handle", "creator-1", 1.0)
        db.table.assert_called_with("background_jobs")
        row = db.table.return_value.insert.call_args.args[0]
        assert row["job_type"] == "creator_ig_scrape"
        assert row["payload"]["handle"] == "target_handle"
        assert row["payload"]["existing_creator_id"] == "creator-1"
        assert row["payload"]["source"] == "auto_stitch_from_yt"
        assert row["payload"]["source_confidence"] == 1.0
        assert row["status"] == "queued"

    def test_yt_job_shape(self):
        db = MagicMock()
        handlers._enqueue_auto_stitch_yt_scrape(
            db, "https://youtube.com/@x", "creator-1", 1.0
        )
        row = db.table.return_value.insert.call_args.args[0]
        assert row["job_type"] == "creator_yt_scrape"
        assert row["payload"]["url"] == "https://youtube.com/@x"
        assert row["payload"]["existing_creator_id"] == "creator-1"
        assert row["payload"]["source"] == "auto_stitch_from_ig"
