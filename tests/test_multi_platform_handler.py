"""Tests for handle_creator_multi_platform_scrape (Phase 2.5 parallel runner)."""

from unittest.mock import MagicMock

import pytest

from pipeline import handlers


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("BRIGHTDATA_API_TOKEN", "bd")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENAI_API_KEY", "o")


class TestMultiPlatformHandler:
    def test_registered_in_dispatch(self):
        assert "creator_multi_platform_scrape" in handlers.HANDLERS

    def test_requires_at_least_one_platform(self):
        with pytest.raises(ValueError, match="missing ig_handle and yt_url"):
            handlers.handle_creator_multi_platform_scrape(
                MagicMock(), {"payload": {}}
            )

    def test_runs_both_in_parallel(self, monkeypatch):
        ig_mock = MagicMock()
        yt_mock = MagicMock()
        monkeypatch.setattr(handlers, "handle_creator_ig_scrape", ig_mock)
        monkeypatch.setattr(handlers, "handle_creator_yt_scrape", yt_mock)
        monkeypatch.setattr(
            handlers, "_siblings_all_terminal",
            MagicMock(return_value=False),
        )

        handlers.handle_creator_multi_platform_scrape(
            MagicMock(),
            {
                "payload": {
                    "ig_handle": "mkbhd",
                    "yt_url": "https://youtube.com/@mkbhd",
                    "existing_creator_id": "c-1",
                }
            },
        )
        ig_mock.assert_called_once()
        yt_mock.assert_called_once()
        # Both sub-jobs should carry the existing_creator_id
        assert (
            ig_mock.call_args.args[1]["payload"]["existing_creator_id"]
            == "c-1"
        )
        assert (
            yt_mock.call_args.args[1]["payload"]["existing_creator_id"]
            == "c-1"
        )

    def test_ig_only(self, monkeypatch):
        ig_mock = MagicMock()
        yt_mock = MagicMock()
        monkeypatch.setattr(handlers, "handle_creator_ig_scrape", ig_mock)
        monkeypatch.setattr(handlers, "handle_creator_yt_scrape", yt_mock)
        handlers.handle_creator_multi_platform_scrape(
            MagicMock(), {"payload": {"ig_handle": "x"}}
        )
        ig_mock.assert_called_once()
        yt_mock.assert_not_called()

    def test_yt_only(self, monkeypatch):
        ig_mock = MagicMock()
        yt_mock = MagicMock()
        monkeypatch.setattr(handlers, "handle_creator_ig_scrape", ig_mock)
        monkeypatch.setattr(handlers, "handle_creator_yt_scrape", yt_mock)
        handlers.handle_creator_multi_platform_scrape(
            MagicMock(), {"payload": {"yt_url": "https://youtube.com/@x"}}
        )
        ig_mock.assert_not_called()
        yt_mock.assert_called_once()

    def test_one_platform_failure_does_not_abort_other(self, monkeypatch):
        """IG fails, YT succeeds → handler completes, doesn't raise."""
        monkeypatch.setattr(
            handlers, "handle_creator_ig_scrape",
            MagicMock(side_effect=RuntimeError("ig boom")),
        )
        yt_mock = MagicMock()
        monkeypatch.setattr(handlers, "handle_creator_yt_scrape", yt_mock)
        monkeypatch.setattr(
            handlers, "_siblings_all_terminal", MagicMock(return_value=False)
        )
        # Should not raise — one success is enough
        handlers.handle_creator_multi_platform_scrape(
            MagicMock(),
            {
                "payload": {
                    "ig_handle": "x",
                    "yt_url": "https://youtube.com/@x",
                }
            },
        )
        yt_mock.assert_called_once()

    def test_all_platform_failures_raise(self, monkeypatch):
        monkeypatch.setattr(
            handlers, "handle_creator_ig_scrape",
            MagicMock(side_effect=RuntimeError("ig boom")),
        )
        monkeypatch.setattr(
            handlers, "handle_creator_yt_scrape",
            MagicMock(side_effect=RuntimeError("yt boom")),
        )
        with pytest.raises(RuntimeError, match="all platforms failed"):
            handlers.handle_creator_multi_platform_scrape(
                MagicMock(),
                {
                    "payload": {
                        "ig_handle": "x",
                        "yt_url": "https://youtube.com/@x",
                    }
                },
            )

    def test_parent_brand_triggers_matching_once(self, monkeypatch):
        monkeypatch.setattr(
            handlers, "handle_creator_ig_scrape", MagicMock()
        )
        monkeypatch.setattr(
            handlers, "handle_creator_yt_scrape", MagicMock()
        )
        monkeypatch.setattr(
            handlers, "_siblings_all_terminal",
            MagicMock(return_value=True),
        )
        matching_mock = MagicMock()
        monkeypatch.setattr(
            handlers, "_trigger_matching_compute", matching_mock
        )

        handlers.handle_creator_multi_platform_scrape(
            MagicMock(),
            {
                "payload": {
                    "ig_handle": "x",
                    "yt_url": "https://youtube.com/@x",
                    "parent_brand_id": "brand-1",
                }
            },
        )
        matching_mock.assert_called_once_with("brand-1")
