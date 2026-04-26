"""Tests for YT-aware professionalism scoring (Phase 2.5 follow-up)."""

from datetime import datetime, timezone, timedelta

from pipeline.confidence import CoverageTracker
from pipeline.pipeline import (
    _score_professionalism_youtube,
    _channel_age_years,
)


class TestChannelAgeYears:
    def test_none(self):
        assert _channel_age_years(None) is None

    def test_invalid_string(self):
        assert _channel_age_years("not-a-date") is None

    def test_recent(self):
        dt = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        age = _channel_age_years(dt)
        assert age is not None
        assert 0.9 < age < 1.1

    def test_ten_years(self):
        dt = (datetime.now(timezone.utc) - timedelta(days=365 * 10)).isoformat()
        age = _channel_age_years(dt)
        assert age is not None
        assert 9.5 < age < 10.5

    def test_z_suffix_normalization(self):
        dt = (datetime.now(timezone.utc) - timedelta(days=730)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        age = _channel_age_years(dt)
        assert age is not None
        assert 1.9 < age < 2.1


class TestProfessionalismYouTube:
    def _cip(self, profile: dict, transcripts: list | None = None) -> dict:
        return {
            "platform": "youtube",
            "profile": profile,
            "transcripts": transcripts or [],
        }

    def test_mkbhd_like_creator_scores_high(self):
        """Mega tier + verified + email in bio + 17yo channel → near-max."""
        old_date = (
            datetime.now(timezone.utc) - timedelta(days=365 * 17)
        ).isoformat()
        cip = self._cip({
            "tier": "mega",
            "is_verified": True,
            "bio": "MKBHD: Quality Tech Videos\nbusiness@mkbhd.com\nNYC",
            "channel_created_at": old_date,
        })
        score = _score_professionalism_youtube(cip, CoverageTracker())
        # is_business (age >= 3) 25 + is_verified 25 + has_email 25
        # + audio_quality (tier=mega → professional) 25 = 100
        assert score == 100

    def test_unverified_mega_still_proxies_via_tier(self):
        """Even if BD didn't surface a verified flag, mega-tier carries it."""
        cip = self._cip({
            "tier": "mega",
            "is_verified": None,  # Missing flag
            "bio": "tech reviews",
            "channel_created_at": (
                datetime.now(timezone.utc) - timedelta(days=365 * 5)
            ).isoformat(),
        })
        score = _score_professionalism_youtube(cip, CoverageTracker())
        # 25 (age >= 3) + 25 (mega-tier proxy) + 0 (no email) + 25 (audio_quality) = 75
        assert score == 75

    def test_young_nano_creator_scores_low(self):
        cip = self._cip({
            "tier": "nano",
            "is_verified": False,
            "bio": "",
            "channel_created_at": (
                datetime.now(timezone.utc) - timedelta(days=180)
            ).isoformat(),
        })
        score = _score_professionalism_youtube(cip, CoverageTracker())
        # 0 (age < 3) + 0 (not verified, not mega) + 0 (no bio/email) + 5 (raw) = 5
        assert score == 5

    def test_email_extracted_from_bio(self):
        cip = self._cip({
            "tier": "micro",
            "is_verified": False,
            "bio": "DM business+yt@example.org for sponsorships",
            "channel_created_at": (
                datetime.now(timezone.utc) - timedelta(days=365 * 4)
            ).isoformat(),
        })
        score = _score_professionalism_youtube(cip, CoverageTracker())
        # 25 (age) + 0 (not verified, not mega) + 25 (email) + 10 (casual) = 60
        assert score == 60

    def test_no_email_in_bio(self):
        cip = self._cip({
            "tier": "micro",
            "is_verified": False,
            "bio": "Just a regular tech enthusiast",
            "channel_created_at": (
                datetime.now(timezone.utc) - timedelta(days=365 * 4)
            ).isoformat(),
        })
        score = _score_professionalism_youtube(cip, CoverageTracker())
        # 25 + 0 + 0 + 10 = 35
        assert score == 35

    def test_tracker_marks_coverage(self):
        cip = self._cip(
            {
                "tier": "mid",
                "is_verified": True,
                "bio": "creator@example.com",
                "channel_created_at": (
                    datetime.now(timezone.utc) - timedelta(days=365 * 3)
                ).isoformat(),
            },
            transcripts=[{"transcript_text": "hello"}],
        )
        tracker = CoverageTracker()
        _score_professionalism_youtube(cip, tracker)
        env = tracker.to_dict()
        prof = env["per_subscore"].get("professionalism")
        assert prof is not None
        # All four inputs (is_business, is_verified, has_email, audio_quality)
        # marked present
        assert prof["missing"] == []

    def test_whisper_confidence_used_when_available(self):
        """If transcripts came from Whisper (have avg_confidence), use that
        instead of the tier-based proxy."""
        cip = self._cip(
            {"tier": "macro", "channel_created_at": "2010-01-01T00:00:00Z"},
            transcripts=[
                {"transcript_text": "x", "avg_confidence": 0.95},
                {"transcript_text": "y", "avg_confidence": 0.92},
            ],
        )
        # avg_confidence 0.93 → "professional" → 25 pts
        # 25 (age) + 0 (not verified, not mega) + 0 (no bio) + 25 (whisper-based prof) = 50
        score = _score_professionalism_youtube(cip, CoverageTracker())
        assert score == 50
