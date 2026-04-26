"""Tests for pipeline.youtube.transcripts tiered fetcher."""

from unittest.mock import MagicMock, patch

import pytest

import pipeline.youtube.transcripts as tr_module
from pipeline.youtube.transcripts import fetch_transcript


class TestFetchTranscript:
    def test_empty_video_id_returns_none(self):
        assert fetch_transcript("", "https://yt.com/watch?v=x") is None

    def test_tier1_success(self, monkeypatch):
        """youtube-transcript-api 1.x returns text → return early."""
        # 1.x: YouTubeTranscriptApi() is an instance with .fetch(video_id)
        # returning a FetchedTranscript-like object with .to_raw_data().
        fake_fetched = MagicMock()
        fake_fetched.to_raw_data.return_value = [
            {"text": "Hello ", "start": 0.0, "duration": 1.0},
            {"text": "world", "start": 1.0, "duration": 1.0},
        ]
        fake_instance = MagicMock()
        fake_instance.fetch.return_value = fake_fetched
        fake_class = MagicMock(return_value=fake_instance)
        fake_class.fetch = MagicMock()  # so hasattr(cls,"fetch") is True
        fake_module = MagicMock()
        fake_module.YouTubeTranscriptApi = fake_class

        import sys
        monkeypatch.setitem(sys.modules, "youtube_transcript_api", fake_module)

        out = fetch_transcript("abc123", "https://yt.com/watch?v=abc123")
        assert out is not None
        assert out["video_id"] == "abc123"
        assert out["source"] == "youtube_transcript_api"
        assert "Hello" in out["text"]
        # Segments stored in legacy raw shape regardless of which API path ran.
        assert out["segments"][0]["text"] == "Hello "

    def test_tier1_library_missing_falls_through(self, monkeypatch):
        import sys
        # Force ImportError on the dynamic import
        original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def failing_import(name, *args, **kwargs):
            if name == "youtube_transcript_api":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", failing_import)

        # Tier 2 and Tier 3 also disabled → expect None
        out = fetch_transcript("abc", "https://yt.com/watch?v=abc")
        assert out is None

    def test_tier1_failure_falls_through_to_whisper(self, monkeypatch):
        # Tier 1 raises (rate-limited / blocked)
        import sys
        monkeypatch.setitem(
            sys.modules,
            "youtube_transcript_api",
            MagicMock(
                YouTubeTranscriptApi=MagicMock(
                    fetch=MagicMock(side_effect=RuntimeError("rate limited"))
                )
            ),
        )
        # Tier 2 (Whisper) succeeds
        monkeypatch.setattr(
            tr_module,
            "_try_whisper",
            MagicMock(
                return_value={
                    "video_id": "v",
                    "source": "whisper",
                    "text": "whisper text",
                }
            ),
        )
        out = fetch_transcript("v", "https://y.t/v", openai_key="k")
        assert out is not None
        assert out["source"] == "whisper"

    def test_all_tiers_fail_returns_none(self, monkeypatch):
        import sys
        monkeypatch.setitem(
            sys.modules,
            "youtube_transcript_api",
            MagicMock(
                YouTubeTranscriptApi=MagicMock(
                    fetch=MagicMock(side_effect=RuntimeError)
                )
            ),
        )
        monkeypatch.setattr(
            tr_module, "_try_whisper", MagicMock(return_value=None)
        )
        out = fetch_transcript("v", "https://y.t/v", openai_key="k")
        assert out is None

    def test_tier3_skipped_without_openai_key(self, monkeypatch):
        import sys
        monkeypatch.setitem(
            sys.modules,
            "youtube_transcript_api",
            MagicMock(
                YouTubeTranscriptApi=MagicMock(
                    get_transcript=MagicMock(side_effect=RuntimeError)
                )
            ),
        )
        whisper_mock = MagicMock()
        monkeypatch.setattr(tr_module, "_try_whisper", whisper_mock)
        # No openai_key → tier 2 (Whisper) skipped.
        out = fetch_transcript("v", "https://y.t/v")
        assert out is None
        whisper_mock.assert_not_called()


class TestTierImplementations:
    """Unit tests for the private tier functions."""

    def test_try_transcript_api_empty_segments(self, monkeypatch):
        import sys
        monkeypatch.setitem(
            sys.modules,
            "youtube_transcript_api",
            MagicMock(
                YouTubeTranscriptApi=MagicMock(
                    get_transcript=MagicMock(return_value=[])
                )
            ),
        )
        assert tr_module._try_transcript_api("v") is None

    def test_try_transcript_api_whitespace_only(self, monkeypatch):
        import sys
        monkeypatch.setitem(
            sys.modules,
            "youtube_transcript_api",
            MagicMock(
                YouTubeTranscriptApi=MagicMock(
                    get_transcript=MagicMock(
                        return_value=[{"text": "   "}, {"text": ""}]
                    )
                )
            ),
        )
        assert tr_module._try_transcript_api("v") is None

    def test_try_whisper_yt_dlp_missing(self, monkeypatch):
        import sys
        # Simulate yt_dlp not installed
        monkeypatch.setitem(sys.modules, "yt_dlp", None)
        # Make sure the import inside _download_audio_yt_dlp raises
        original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def failing(name, *a, **kw):
            if name == "yt_dlp":
                raise ImportError
            return original_import(name, *a, **kw)

        monkeypatch.setattr("builtins.__import__", failing)
        assert tr_module._try_whisper("v", "url", "k") is None

    def test_try_whisper_download_fails(self, monkeypatch):
        monkeypatch.setattr(
            tr_module, "_download_audio_yt_dlp",
            MagicMock(return_value=None),
        )
        assert tr_module._try_whisper("v", "url", "k") is None

    def test_try_whisper_transcribe_returns_none(self, monkeypatch, tmp_path):
        fake_audio = tmp_path / "abc.m4a"
        fake_audio.write_bytes(b"fake audio")
        monkeypatch.setattr(
            tr_module, "_download_audio_yt_dlp",
            MagicMock(return_value=str(fake_audio)),
        )
        monkeypatch.setattr(
            tr_module, "_whisper_transcribe",
            MagicMock(return_value=None),
        )
        assert tr_module._try_whisper("v", "url", "k") is None

    def test_try_whisper_happy_path(self, monkeypatch, tmp_path):
        fake_audio = tmp_path / "abc.m4a"
        fake_audio.write_bytes(b"audio")
        monkeypatch.setattr(
            tr_module, "_download_audio_yt_dlp",
            MagicMock(return_value=str(fake_audio)),
        )
        monkeypatch.setattr(
            tr_module, "_whisper_transcribe",
            MagicMock(return_value="transcribed text"),
        )
        out = tr_module._try_whisper("v", "url", "k")
        assert out == {"video_id": "v", "source": "whisper", "text": "transcribed text"}
        # Audio file should be deleted
        assert not fake_audio.exists()


class TestDownloadAudioYtDlp:
    def test_subprocess_failure_returns_none(self, monkeypatch):
        """yt-dlp exits non-zero → _download_audio_yt_dlp returns None."""
        import sys
        # Ensure `import yt_dlp` succeeds (even if stub)
        monkeypatch.setitem(sys.modules, "yt_dlp", MagicMock())

        import subprocess
        monkeypatch.setattr(
            subprocess, "run",
            MagicMock(side_effect=subprocess.CalledProcessError(1, "yt-dlp")),
        )
        assert tr_module._download_audio_yt_dlp("https://y.t/v") is None

    def test_subprocess_timeout_returns_none(self, monkeypatch):
        import sys, subprocess
        monkeypatch.setitem(sys.modules, "yt_dlp", MagicMock())
        monkeypatch.setattr(
            subprocess, "run",
            MagicMock(side_effect=subprocess.TimeoutExpired("yt-dlp", 180)),
        )
        assert tr_module._download_audio_yt_dlp("url") is None

    def test_binary_not_found_returns_none(self, monkeypatch):
        import sys, subprocess
        monkeypatch.setitem(sys.modules, "yt_dlp", MagicMock())
        monkeypatch.setattr(
            subprocess, "run",
            MagicMock(side_effect=FileNotFoundError("yt-dlp binary missing")),
        )
        assert tr_module._download_audio_yt_dlp("url") is None

    def test_returns_first_produced_file(self, monkeypatch, tmp_path):
        import sys, subprocess
        monkeypatch.setitem(sys.modules, "yt_dlp", MagicMock())

        # Fake that subprocess "produced" a file in a predictable tmpdir.
        produced = None

        def fake_run(cmd, check, timeout, capture_output):
            # Find the -o out_template arg to know where to write the file.
            out_idx = cmd.index("-o") + 1
            template = cmd[out_idx]
            parent = Path(template).parent
            # Write a fake audio file at parent/abc.m4a
            f = parent / "abc.m4a"
            f.write_bytes(b"fake")
            nonlocal produced
            produced = f
            return MagicMock(returncode=0)

        from pathlib import Path
        import tempfile
        # Monkeypatch mkdtemp to return a known path under tmp_path
        tmpdir = tmp_path / "yt_transcript"
        tmpdir.mkdir()
        monkeypatch.setattr(
            tempfile, "mkdtemp", MagicMock(return_value=str(tmpdir))
        )
        monkeypatch.setattr(subprocess, "run", fake_run)

        out = tr_module._download_audio_yt_dlp("https://y.t/v")
        assert out is not None
        assert Path(out).exists()
        assert Path(out).name == "abc.m4a"

    def test_no_file_produced_returns_none(self, monkeypatch, tmp_path):
        import sys, subprocess, tempfile
        monkeypatch.setitem(sys.modules, "yt_dlp", MagicMock())
        tmpdir = tmp_path / "empty"
        tmpdir.mkdir()
        monkeypatch.setattr(
            tempfile, "mkdtemp", MagicMock(return_value=str(tmpdir))
        )
        # subprocess.run succeeds but writes no file
        monkeypatch.setattr(
            subprocess, "run", MagicMock(return_value=MagicMock())
        )
        assert tr_module._download_audio_yt_dlp("url") is None


class TestWhisperTranscribe:
    def test_openai_missing_returns_none(self, monkeypatch, tmp_path):
        import sys
        import builtins
        real_import = builtins.__import__

        def failing(name, *a, **kw):
            if name == "openai":
                raise ImportError("not installed")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", failing)
        f = tmp_path / "a.m4a"
        f.write_bytes(b"x")
        assert tr_module._whisper_transcribe(str(f), "k") is None

    def test_happy_path(self, monkeypatch, tmp_path):
        import sys
        fake_resp = MagicMock(text="hello world")
        fake_client = MagicMock()
        fake_client.audio.transcriptions.create.return_value = fake_resp
        fake_openai = MagicMock()
        fake_openai.OpenAI = MagicMock(return_value=fake_client)
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        f = tmp_path / "a.m4a"
        f.write_bytes(b"audio")
        out = tr_module._whisper_transcribe(str(f), "k")
        assert out == "hello world"

    def test_empty_text_returns_none(self, monkeypatch, tmp_path):
        import sys
        fake_resp = MagicMock(text="   ")  # whitespace → None
        fake_client = MagicMock()
        fake_client.audio.transcriptions.create.return_value = fake_resp
        fake_openai = MagicMock()
        fake_openai.OpenAI = MagicMock(return_value=fake_client)
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        f = tmp_path / "a.m4a"
        f.write_bytes(b"x")
        assert tr_module._whisper_transcribe(str(f), "k") is None

    def test_api_error_returns_none(self, monkeypatch, tmp_path):
        import sys
        fake_client = MagicMock()
        fake_client.audio.transcriptions.create.side_effect = RuntimeError(
            "api down"
        )
        fake_openai = MagicMock()
        fake_openai.OpenAI = MagicMock(return_value=fake_client)
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        f = tmp_path / "a.m4a"
        f.write_bytes(b"x")
        assert tr_module._whisper_transcribe(str(f), "k") is None
