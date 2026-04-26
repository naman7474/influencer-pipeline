"""Tests for pipeline.db multi-platform helpers (migration 043)."""

from unittest.mock import MagicMock

import pytest

from pipeline import db as pdb


def _make_db_mock(select_result=None, insert_result=None, upsert_result=None):
    """Build a chainable Supabase client mock.

    The real supabase-py client uses `.table(x).upsert(row, ...).execute()`
    etc. This fixture returns a mock whose chains resolve to the provided
    results.
    """
    db = MagicMock()
    table = db.table.return_value

    # For upsert → execute returns upsert_result
    if upsert_result is not None:
        upsert_chain = table.upsert.return_value
        upsert_chain.execute.return_value = upsert_result

    # For insert → execute returns insert_result
    if insert_result is not None:
        insert_chain = table.insert.return_value
        insert_chain.execute.return_value = insert_result

    # For select → execute returns select_result. The select chain can
    # also be filtered with .eq() / .limit() / .maybeSingle() — we return
    # a chainable mock where every accessor yields the same chain and
    # .execute() resolves to the configured value.
    if select_result is not None:
        select_chain = MagicMock()
        select_chain.eq.return_value = select_chain
        select_chain.limit.return_value = select_chain
        select_chain.maybe_single.return_value = select_chain
        select_chain.execute.return_value = select_result
        table.select.return_value = select_chain

    return db


class TestUpsertSocialProfile:
    def test_writes_expected_row(self):
        result = MagicMock()
        result.data = [{"id": "csp-1"}]
        db = _make_db_mock(upsert_result=result)
        profile = {
            "handle": "mkbhd",
            "platform_user_id": "UCabc",
            "profile_url": "https://youtube.com/@mkbhd",
            "display_name": "MKBHD",
            "bio": "Quality tech videos",
            "avatar_url": "https://img",
            "category": "Tech",
            "country": "US",
            "is_verified": True,
            "is_business": False,
            "followers_or_subs": 20_000_000,
            "posts_or_videos_count": 1600,
            "avg_engagement": 0.05,
            "external_links": [{"label": "IG", "url": "x"}],
        }
        csp_id = pdb.upsert_social_profile(db, "creator-1", "youtube", profile)
        assert csp_id == "csp-1"
        db.table.assert_called_with("creator_social_profiles")
        args, kwargs = db.table.return_value.upsert.call_args
        row = args[0]
        assert row["creator_id"] == "creator-1"
        assert row["platform"] == "youtube"
        assert row["handle"] == "mkbhd"
        assert row["followers_or_subs"] == 20_000_000
        assert kwargs["on_conflict"] == "creator_id,platform"

    def test_ig_profile_alias_fields(self):
        result = MagicMock()
        result.data = [{"id": "csp-2"}]
        db = _make_db_mock(upsert_result=result)
        profile = {
            "handle": "igcreator",
            "instagram_id": "123456",
            "followers": 50_000,
            "posts_count": 200,
            "biography": "IG bio",
            "brightdata_avg_engagement": 0.03,
        }
        pdb.upsert_social_profile(db, "creator-2", "instagram", profile)
        row = db.table.return_value.upsert.call_args.args[0]
        assert row["platform_user_id"] == "123456"
        assert row["followers_or_subs"] == 50_000
        assert row["posts_or_videos_count"] == 200
        assert row["bio"] == "IG bio"
        assert row["avg_engagement"] == 0.03


class TestUpsertYoutubeVideos:
    def test_empty_list_returns_empty(self):
        db = _make_db_mock()
        assert pdb.upsert_youtube_videos(db, "c1", []) == []

    def test_skips_rows_without_video_id(self):
        db = _make_db_mock()
        result = MagicMock()
        result.data = []
        db.table.return_value.upsert.return_value.execute.return_value = result
        ids = pdb.upsert_youtube_videos(db, "c1", [{"title": "no id"}])
        # No valid rows -> shortcut returns []
        assert ids == []

    def test_upserts_valid_rows(self):
        result = MagicMock()
        result.data = [{"id": "ytv-1"}]
        db = _make_db_mock(upsert_result=result)
        videos = [
            {
                "video_id": "abc",
                "url": "https://yt.com/watch?v=abc",
                "title": "Title",
                "description": "desc",
                "tags": ["a"],
                "category_id": 22,
                "is_short": False,
                "is_livestream": False,
                "duration_seconds": 100,
                "view_count": 1000,
                "like_count": 10,
                "comment_count": 2,
                "thumbnail_url": "https://img",
                "has_captions": True,
                "caption_source": "youtube_auto",
                "published_at": "2026-01-01T00:00:00Z",
            }
        ]
        out = pdb.upsert_youtube_videos(db, "c1", videos)
        assert out == ["ytv-1"]
        args, kwargs = db.table.return_value.upsert.call_args
        assert kwargs["on_conflict"] == "creator_id,video_id"


class TestUpsertCreatorScorePlatform:
    def test_writes_score_row(self):
        db = _make_db_mock()
        db.table.return_value.insert.return_value.execute.return_value = MagicMock()
        pdb.upsert_creator_score_platform(
            db, "c1", "youtube", {"cpi": 75.0}
        )
        db.table.assert_called_with("creator_scores")
        args, _ = db.table.return_value.insert.call_args
        row = args[0]
        assert row["creator_id"] == "c1"
        assert row["platform"] == "youtube"
        assert row["cpi"] == 75.0
        assert "computed_at" in row


class TestUpsertBrandPlatformAnalysis:
    def test_writes_analysis_row(self):
        db = _make_db_mock()
        db.table.return_value.upsert.return_value.execute.return_value = MagicMock()
        pdb.upsert_brand_platform_analysis(
            db,
            "brand-1",
            "youtube",
            {
                "handle": "brandyt",
                "analysis_status": "completed",
                "content_dna": {"topics": ["tech"]},
                "collaborators": ["creator1"],
            },
        )
        db.table.assert_called_with("brand_platform_analyses")
        args, kwargs = db.table.return_value.upsert.call_args
        row = args[0]
        assert row["brand_id"] == "brand-1"
        assert row["platform"] == "youtube"
        assert row["analysis_status"] == "completed"
        assert row["collaborators"] == ["creator1"]
        assert kwargs["on_conflict"] == "brand_id,platform"


class TestFindCreatorByPlatformProfile:
    def test_finds_by_platform_user_id(self):
        result = MagicMock()
        result.data = [{"creator_id": "c1"}]
        db = _make_db_mock(select_result=result)
        cid = pdb._find_creator_by_platform_profile(
            db, "youtube", "UCabc", None
        )
        assert cid == "c1"

    def test_falls_back_to_handle(self):
        # First (by platform_user_id) returns empty; second (by handle) returns row.
        db = MagicMock()
        empty_res = MagicMock()
        empty_res.data = []
        hit_res = MagicMock()
        hit_res.data = [{"creator_id": "c2"}]

        # We'll return different select chains on successive calls by
        # re-configuring the mock based on call count.
        call_count = {"n": 0}

        def select_side_effect(*_args, **_kwargs):
            chain = MagicMock()
            chain.eq.return_value = chain
            chain.limit.return_value = chain
            # First call chain resolves to empty, second to hit
            chain.execute.return_value = (
                empty_res if call_count["n"] == 0 else hit_res
            )
            call_count["n"] += 1
            return chain

        db.table.return_value.select.side_effect = select_side_effect

        cid = pdb._find_creator_by_platform_profile(
            db, "youtube", "UCabc", "mkbhd"
        )
        assert cid == "c2"
        assert call_count["n"] == 2

    def test_returns_none_when_nothing_found(self):
        empty = MagicMock()
        empty.data = []
        db = _make_db_mock(select_result=empty)
        cid = pdb._find_creator_by_platform_profile(
            db, "youtube", "UCx", "handle"
        )
        assert cid is None

    def test_no_ids_returns_none(self):
        db = _make_db_mock()
        assert (
            pdb._find_creator_by_platform_profile(db, "youtube", None, None)
            is None
        )


class TestCreateYoutubeCreatorShell:
    def test_creates_creator_row(self):
        db = MagicMock()
        # handle collision check returns empty
        existing = MagicMock()
        existing.data = []
        db.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = existing
        # insert returns id
        insert_res = MagicMock()
        insert_res.data = [{"id": "new-creator"}]
        db.table.return_value.insert.return_value.execute.return_value = insert_res

        cid = pdb._create_youtube_creator_shell(
            db,
            {
                "handle": "mkbhd",
                "display_name": "MKBHD",
                "bio": "Tech",
                "followers_or_subs": 100,
                "posts_or_videos_count": 10,
                "tier": "micro",
            },
        )
        assert cid == "new-creator"
        row = db.table.return_value.insert.call_args.args[0]
        assert row["handle"] == "mkbhd"

    def test_suffixes_on_handle_collision(self):
        db = MagicMock()
        existing = MagicMock()
        existing.data = [{"id": "existing"}]
        db.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = existing
        insert_res = MagicMock()
        insert_res.data = [{"id": "new"}]
        db.table.return_value.insert.return_value.execute.return_value = insert_res
        pdb._create_youtube_creator_shell(db, {"handle": "collide"})
        row = db.table.return_value.insert.call_args.args[0]
        assert row["handle"] == "collide_yt"


class TestStoreYoutubeCip:
    def test_happy_path_new_creator(self, monkeypatch):
        db = MagicMock()
        # Configure all chain resolutions. For _find_creator_by_platform_profile
        # we return None → shell creator is created.
        lookup_res = MagicMock(); lookup_res.data = []
        db.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = lookup_res
        db.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = lookup_res
        creator_insert = MagicMock(); creator_insert.data = [{"id": "c-new"}]
        score_insert = MagicMock()
        db.table.return_value.insert.return_value.execute.side_effect = [
            creator_insert,  # _create_youtube_creator_shell
            score_insert,    # upsert_creator_score_platform
        ]
        upsert_res = MagicMock(); upsert_res.data = [{"id": "csp-new"}]
        db.table.return_value.upsert.return_value.execute.return_value = upsert_res

        cip = {
            "profile": {
                "handle": "ytcreator",
                "platform_user_id": "UCx",
                "followers_or_subs": 1000,
                "tier": "nano",
            },
            "resolved": {"channel_id": "UCx"},
            "videos": [],  # empty path
            "scores": {
                "cpi": 60,
                "engagement_quality": 50,
                "avg_views_per_sub": 0.3,
            },
            "pipeline_version": "1.1",
        }
        cid = pdb.store_youtube_cip(db, cip)
        assert cid == "c-new"

    def test_skips_llm_failure_blocks(self):
        db = MagicMock()
        lookup_res = MagicMock(); lookup_res.data = [{"creator_id": "c-exist"}]
        db.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = lookup_res
        upsert_res = MagicMock(); upsert_res.data = [{"id": "csp"}]
        db.table.return_value.upsert.return_value.execute.return_value = upsert_res

        cip = {
            "profile": {"handle": "x", "platform_user_id": "UCx"},
            "resolved": {"channel_id": "UCx"},
            "videos": [],
            "scores": {},  # no scores
            "caption_intelligence": {"_llm_failure": True},
            "transcript_intelligence": {"_llm_failure": True},
            "audience_intelligence": {"_llm_failure": True},
        }
        cid = pdb.store_youtube_cip(db, cip)
        assert cid == "c-exist"
        # Score insert should NOT have been called
        db.table.return_value.insert.assert_not_called()
