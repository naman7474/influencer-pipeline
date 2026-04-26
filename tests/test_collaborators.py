"""Tests for pipeline.youtube.collaborators."""

from pipeline.youtube.collaborators import (
    extract_collaborators,
    extract_mentions_from_text,
)


class TestExtractMentionsFromText:
    def test_empty_text(self):
        h, c = extract_mentions_from_text("")
        assert h == set() and c == set()

    def test_none(self):
        h, c = extract_mentions_from_text(None)
        assert h == set() and c == set()

    def test_handle_mention(self):
        h, c = extract_mentions_from_text("shout out to @mkbhd thanks!")
        assert "mkbhd" in h
        assert c == set()

    def test_multiple_handles(self):
        h, _ = extract_mentions_from_text("@alice and @bob")
        assert h == {"alice", "bob"}

    def test_channel_url(self):
        h, c = extract_mentions_from_text(
            "see https://www.youtube.com/channel/UCBJycsmduvYEL83R_U4JriQ"
        )
        assert "UCBJycsmduvYEL83R_U4JriQ" in c

    def test_handle_url(self):
        h, c = extract_mentions_from_text(
            "see https://www.youtube.com/@mkbhd"
        )
        assert "mkbhd" in h

    def test_short_mentions_filtered(self):
        # @1st has only 3 chars — it's at the minimum length boundary but
        # regex accepts 3+. @a is below and rejected.
        h, _ = extract_mentions_from_text("@a @ab")  # both < 3 chars
        assert h == set()

    def test_case_normalization(self):
        h, _ = extract_mentions_from_text("@MKBHD")
        assert h == {"mkbhd"}

    def test_dot_and_hyphen_in_handle(self):
        h, _ = extract_mentions_from_text("@foo.bar @foo-baz")
        assert "foo.bar" in h
        assert "foo-baz" in h


class TestExtractCollaborators:
    def test_empty_videos(self):
        out = extract_collaborators([])
        assert out == {
            "handles": [],
            "channel_ids": [],
            "total_videos_scanned": 0,
        }

    def test_collab_threshold(self):
        # @friend mentioned twice = above default threshold (2)
        # @oneshot mentioned once = below threshold, filtered out
        videos = [
            {"title": "shout to @friend", "description": ""},
            {"title": "", "description": "thanks @friend"},
            {"title": "@oneshot thanks", "description": ""},
        ]
        out = extract_collaborators(videos)
        handles = {h["handle"] for h in out["handles"]}
        assert "friend" in handles
        assert "oneshot" not in handles
        assert out["total_videos_scanned"] == 3

    def test_min_mentions_override(self):
        videos = [{"title": "@someone", "description": ""}]
        out = extract_collaborators(videos, min_mentions=1)
        assert any(h["handle"] == "someone" for h in out["handles"])

    def test_self_handle_excluded(self):
        videos = [
            {"title": "I @me @friend", "description": "also @friend"},
            {"title": "@me", "description": "@friend"},
        ]
        out = extract_collaborators(
            videos, self_handle="me", min_mentions=1
        )
        handles = {h["handle"] for h in out["handles"]}
        assert "me" not in handles
        assert "friend" in handles

    def test_self_channel_id_excluded(self):
        videos = [
            {"title": "", "description": "https://youtube.com/channel/UCself"},
            {"title": "", "description": "https://youtube.com/channel/UCother"},
        ]
        out = extract_collaborators(
            videos, self_channel_id="UCself", min_mentions=1
        )
        cids = {c["channel_id"] for c in out["channel_ids"]}
        assert "UCself" not in cids

    def test_counts_sorted_desc(self):
        videos = [
            {"title": "@alice @bob", "description": ""},
            {"title": "@alice", "description": ""},
            {"title": "@alice", "description": ""},
            {"title": "@bob", "description": ""},
        ]
        out = extract_collaborators(videos, min_mentions=1)
        # @alice has count 3, @bob has count 2 — alice comes first
        assert out["handles"][0]["handle"] == "alice"
        assert out["handles"][0]["count"] == 3
