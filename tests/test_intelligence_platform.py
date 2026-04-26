"""Tests for per-platform intelligence + embedding writes (migrations 046/047)."""

from unittest.mock import MagicMock

from pipeline import db as pdb


class TestInsertCaptionIntelligencePlatform:
    def test_default_platform_instagram(self):
        db = MagicMock()
        pdb.insert_caption_intelligence(db, "c-1", {"niche_classification": {}})
        row = db.table.return_value.upsert.call_args.args[0]
        assert row["platform"] == "instagram"

    def test_explicit_platform_youtube(self):
        db = MagicMock()
        pdb.insert_caption_intelligence(
            db, "c-1", {"niche_classification": {}}, platform="youtube"
        )
        row = db.table.return_value.upsert.call_args.args[0]
        assert row["platform"] == "youtube"


class TestInsertTranscriptIntelligencePlatform:
    def test_default_platform_instagram(self):
        db = MagicMock()
        pdb.insert_transcript_intelligence(db, "c-1", {})
        row = db.table.return_value.upsert.call_args.args[0]
        assert row["platform"] == "instagram"

    def test_explicit_platform_youtube(self):
        db = MagicMock()
        pdb.insert_transcript_intelligence(db, "c-1", {}, platform="youtube")
        row = db.table.return_value.upsert.call_args.args[0]
        assert row["platform"] == "youtube"


class TestInsertAudienceIntelligencePlatform:
    def test_default_platform_instagram(self):
        db = MagicMock()
        pdb.insert_audience_intelligence(db, "c-1", {})
        row = db.table.return_value.upsert.call_args.args[0]
        assert row["platform"] == "instagram"

    def test_explicit_platform_youtube(self):
        db = MagicMock()
        pdb.insert_audience_intelligence(db, "c-1", {}, platform="youtube")
        row = db.table.return_value.upsert.call_args.args[0]
        assert row["platform"] == "youtube"


class TestUpsertCreatorPlatformEmbedding:
    def test_empty_embedding_noop(self):
        db = MagicMock()
        pdb.upsert_creator_platform_embedding(db, "c-1", "instagram", [])
        db.table.assert_not_called()

    def test_none_embedding_noop(self):
        db = MagicMock()
        pdb.upsert_creator_platform_embedding(db, "c-1", "instagram", None)
        db.table.assert_not_called()

    def test_writes_row(self):
        db = MagicMock()
        emb = [0.1] * 1536
        pdb.upsert_creator_platform_embedding(db, "c-1", "youtube", emb)
        db.table.assert_called_with("creator_content_embeddings")
        args, kwargs = db.table.return_value.upsert.call_args
        row = args[0]
        assert row["creator_id"] == "c-1"
        assert row["platform"] == "youtube"
        assert row["embedding"] == emb
        assert "computed_at" in row
        assert kwargs["on_conflict"] == "creator_id,platform"


class TestStoreYoutubeCipPlatformScoped:
    """store_youtube_cip should pass platform='youtube' to intelligence inserters."""

    def test_intelligence_writes_tagged_youtube(self, monkeypatch):
        db = MagicMock()
        lookup_res = MagicMock(); lookup_res.data = []
        db.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = lookup_res
        db.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = lookup_res
        creator_insert = MagicMock(); creator_insert.data = [{"id": "c-new"}]
        db.table.return_value.insert.return_value.execute.return_value = creator_insert
        upsert_res = MagicMock(); upsert_res.data = [{"id": "csp"}]
        db.table.return_value.upsert.return_value.execute.return_value = upsert_res

        caption_mock = MagicMock()
        transcript_mock = MagicMock()
        audience_mock = MagicMock()
        monkeypatch.setattr(pdb, "insert_caption_intelligence", caption_mock)
        monkeypatch.setattr(pdb, "insert_transcript_intelligence", transcript_mock)
        monkeypatch.setattr(pdb, "insert_audience_intelligence", audience_mock)

        cip = {
            "profile": {"handle": "x"},
            "resolved": {"channel_id": "UCx"},
            "videos": [],
            "scores": {},
            "caption_intelligence": {"primary_niche": "tech"},
            "transcript_intelligence": {"avg_hook_quality": 0.8},
            "audience_intelligence": {"primary_country": "US"},
        }
        pdb.store_youtube_cip(db, cip)

        assert caption_mock.call_args.kwargs["platform"] == "youtube"
        assert transcript_mock.call_args.kwargs["platform"] == "youtube"
        assert audience_mock.call_args.kwargs["platform"] == "youtube"
