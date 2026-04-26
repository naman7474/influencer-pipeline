"""Tests for pipeline.youtube.stitching."""

from unittest.mock import MagicMock

from pipeline.youtube.stitching import (
    StitchCandidate,
    extract_handles_from_links,
    find_stitch_candidates_in_db,
    propose_handle_match_candidate,
    propose_stitch_candidates,
)


class TestExtractHandlesFromLinks:
    def test_none(self):
        out = extract_handles_from_links(None)
        assert out == {"instagram": set(), "tiktok": set(), "youtube": set()}

    def test_empty(self):
        out = extract_handles_from_links([])
        assert out == {"instagram": set(), "tiktok": set(), "youtube": set()}

    def test_instagram_url(self):
        out = extract_handles_from_links(
            [{"label": "IG", "url": "https://www.instagram.com/mkbhd/"}]
        )
        assert "mkbhd" in out["instagram"]

    def test_tiktok_url(self):
        out = extract_handles_from_links(
            [{"url": "https://www.tiktok.com/@mkbhd"}]
        )
        assert "mkbhd" in out["tiktok"]

    def test_youtube_url(self):
        out = extract_handles_from_links(
            [{"url": "https://www.youtube.com/@somechannel"}]
        )
        assert "somechannel" in out["youtube"]

    def test_plain_string_list(self):
        out = extract_handles_from_links(
            ["https://instagram.com/x", "https://tiktok.com/@y"]
        )
        assert "x" in out["instagram"]
        assert "y" in out["tiktok"]

    def test_case_insensitive(self):
        out = extract_handles_from_links(
            [{"url": "HTTPS://INSTAGRAM.COM/MKBHD"}]
        )
        assert "mkbhd" in out["instagram"]

    def test_missing_url_key(self):
        out = extract_handles_from_links([{"label": "IG"}])  # no url
        assert out["instagram"] == set()

    def test_non_matching_url_ignored(self):
        out = extract_handles_from_links([{"url": "https://example.com/x"}])
        assert not any(out[k] for k in out)

    def test_non_dict_non_string_ignored(self):
        out = extract_handles_from_links([None, 0, {"url": ""}])
        assert not any(out[k] for k in out)


class TestProposeStitchCandidates:
    def test_ig_target_from_yt(self):
        cands = propose_stitch_candidates(
            source_platform="youtube",
            source_handle="mkbhd",
            source_external_links=[{"url": "https://instagram.com/mkbhd"}],
        )
        assert len(cands) == 1
        assert cands[0].target_platform == "instagram"
        assert cands[0].target_handle == "mkbhd"
        assert cands[0].confidence == 1.0
        assert "instagram.com/mkbhd" in cands[0].reason

    def test_no_matching_links(self):
        cands = propose_stitch_candidates(
            source_platform="youtube",
            source_handle="x",
            source_external_links=[],
        )
        assert cands == []

    def test_custom_target_platforms(self):
        cands = propose_stitch_candidates(
            source_platform="youtube",
            source_handle="x",
            source_external_links=[
                {"url": "https://tiktok.com/@x"},
                {"url": "https://instagram.com/y"},
            ],
            target_platforms=("tiktok",),
        )
        assert len(cands) == 1
        assert cands[0].target_platform == "tiktok"


class TestProposeHandleMatchCandidate:
    def test_exact_match(self):
        c = propose_handle_match_candidate(
            "instagram", "mkbhd", "youtube", "mkbhd"
        )
        assert c is not None
        assert c.confidence == 0.3

    def test_suffix_normalization(self):
        c = propose_handle_match_candidate(
            "instagram", "mkbhd", "youtube", "mkbhd_yt"
        )
        assert c is not None

        c2 = propose_handle_match_candidate(
            "instagram", "mkbhd_ig", "youtube", "mkbhd"
        )
        assert c2 is not None

    def test_mismatch_returns_none(self):
        assert (
            propose_handle_match_candidate(
                "instagram", "a", "youtube", "b"
            )
            is None
        )

    def test_empty_returns_none(self):
        assert propose_handle_match_candidate("a", "", "b", "x") is None
        assert propose_handle_match_candidate("a", "x", "b", "") is None


class TestFindStitchCandidatesInDb:
    def test_returns_empty_when_source_missing(self):
        db = MagicMock()
        res = MagicMock(); res.data = []
        db.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = res
        out = find_stitch_candidates_in_db(db, "youtube", "nonexistent")
        assert out == []

    def test_finds_candidate_via_external_link(self):
        db = MagicMock()
        # First call: source row with external_links
        source_res = MagicMock()
        source_res.data = [
            {
                "platform": "youtube",
                "handle": "mkbhd",
                "external_links": [
                    {"label": "IG", "url": "https://instagram.com/mkbhd"}
                ],
            }
        ]
        # Second call: existing IG creator with that handle
        target_res = MagicMock()
        target_res.data = [{"creator_id": "ig-creator"}]

        call_count = {"n": 0}

        def select_side_effect(*_args, **_kwargs):
            chain = MagicMock()
            chain.eq.return_value = chain
            chain.limit.return_value = chain
            chain.execute.return_value = (
                source_res if call_count["n"] == 0 else target_res
            )
            call_count["n"] += 1
            return chain

        db.table.return_value.select.side_effect = select_side_effect

        cands = find_stitch_candidates_in_db(db, "youtube", "c-yt-1")
        assert len(cands) == 1
        assert cands[0].target_platform == "instagram"
        assert cands[0].target_handle == "mkbhd"


class TestStitchCandidateDataclass:
    def test_is_frozen(self):
        c = StitchCandidate(
            source_platform="youtube",
            source_handle="x",
            target_platform="instagram",
            target_handle="x",
            confidence=1.0,
            reason="test",
        )
        import dataclasses

        with __import__("pytest").raises((dataclasses.FrozenInstanceError, AttributeError)):
            c.confidence = 0.5  # type: ignore[misc]
