"""Tests for scripts/bench_latency.py.

Uses a mocked voice (no ONNX session, no model download) so these run in CI
without network access or a real model file.
"""
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import bench_latency  # noqa: E402
from phoonnx.config import SynthesisConfig  # noqa: E402
from phoonnx.voice import AudioChunk  # noqa: E402


def _make_mock_voice(sample_rate=16000, add_diacritics=False):
    voice = MagicMock()
    voice.config.sample_rate = sample_rate
    voice.config.add_diacritics = add_diacritics
    voice.config.diacritizer_model = None
    voice.config.lang_code = "en-US"
    voice.phonetic_spellings = None
    voice.phonemize.return_value = [["h", "i"], ["b", "y", "e"]]
    voice.phonemes_to_ids.side_effect = lambda phonemes: list(range(len(phonemes)))

    def _synthesize(text, syn_config=None):
        for _ in range(2):
            yield AudioChunk(
                sample_rate=sample_rate, sample_width=2, sample_channels=1,
                audio_float_array=np.zeros(sample_rate // 4, dtype=np.float32),
            )

    voice.synthesize.side_effect = _synthesize

    def _phoneme_ids_to_audio(phoneme_ids, syn_config=None, language_ids=None):
        return np.zeros(sample_rate // 4, dtype=np.float32)

    voice.phoneme_ids_to_audio.side_effect = _phoneme_ids_to_audio
    return voice


class TestModuleImports:
    def test_bench_module_imports(self):
        assert hasattr(bench_latency, "main")
        assert hasattr(bench_latency, "bench_stages")
        assert hasattr(bench_latency, "bench_ttfa")


class TestTimedUtility:
    def test_timed_returns_result_and_nonnegative_elapsed(self):
        result, elapsed = bench_latency.timed(lambda: 1 + 1)
        assert result == 2
        assert elapsed >= 0.0

    def test_timed_elapsed_is_monotonic_and_reflects_sleep(self):
        _, elapsed = bench_latency.timed(lambda: time.sleep(0.01))
        assert elapsed >= 0.01

    def test_timed_propagates_exceptions(self):
        def _boom():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            bench_latency.timed(_boom)


class TestSummarize:
    def test_summarize_empty_list_is_nan_not_crash(self):
        stats = bench_latency.summarize([])
        assert stats["median"] != stats["median"]  # NaN != NaN

    def test_summarize_single_value(self):
        stats = bench_latency.summarize([0.5])
        assert stats["median"] == 0.5
        assert stats["p95"] == 0.5
        assert stats["min"] == 0.5
        assert stats["max"] == 0.5

    def test_summarize_orders_min_max_correctly(self):
        stats = bench_latency.summarize([0.3, 0.1, 0.2])
        assert stats["min"] == 0.1
        assert stats["max"] == 0.3


class TestBenchStagesWithMockedVoice:
    def test_bench_stages_returns_expected_keys(self):
        voice = _make_mock_voice()
        stages = bench_latency.bench_stages(voice, "hello there", SynthesisConfig())
        for key in ("text_normalize", "diacritics", "phonemize", "tokenize",
                    "session_run_total", "postprocess_total", "audio_seconds"):
            assert key in stages
        assert stages["audio_seconds"] > 0

    def test_bench_stages_skips_diacritics_when_not_enabled(self):
        voice = _make_mock_voice(add_diacritics=False)
        stages = bench_latency.bench_stages(voice, "hello", SynthesisConfig())
        assert stages["diacritics"] == 0.0
        voice.phonemizer.add_diacritics.assert_not_called()


class TestBenchTTFAWithMockedVoice:
    def test_ttfa_is_less_than_or_equal_to_total(self):
        voice = _make_mock_voice()
        result = bench_latency.bench_ttfa(voice, "hello there", SynthesisConfig())
        assert result["ttfa"] <= result["total"]
        assert result["audio_seconds"] > 0
        assert result["rtf"] >= 0

    def test_ttfa_handles_zero_audio_without_crashing(self):
        voice = _make_mock_voice()
        voice.synthesize.side_effect = lambda text, syn_config=None: iter(
            [AudioChunk(sample_rate=16000, sample_width=2, sample_channels=1,
                        audio_float_array=np.zeros(0, dtype=np.float32))]
        )
        result = bench_latency.bench_ttfa(voice, "x", SynthesisConfig())
        assert result["rtf"] != result["rtf"] or result["rtf"] == 0  # nan or 0, never a crash


class TestCLI:
    def test_help_exits_zero(self):
        runner = CliRunner()
        result = runner.invoke(bench_latency.main, ["--help"])
        assert result.exit_code == 0
        assert "MODEL" in result.output

    def test_missing_model_argument_fails_cleanly(self):
        runner = CliRunner()
        result = runner.invoke(bench_latency.main, [])
        assert result.exit_code != 0


class TestJSONSchema:
    """The JSON schema written by --json-out must stay stable: downstream
    tooling reads these keys."""

    def _run_with_mocked_voice(self, tmp_path, monkeypatch):
        voice = _make_mock_voice()

        monkeypatch.setattr(bench_latency, "resolve_model", lambda model, config: ("mock.onnx", None))
        monkeypatch.setattr(
            bench_latency, "load_voice_timed",
            lambda model_path, config_path: {"voice": voice, "t_config_parse": 0.001, "t_session_create": 0.01},
        )
        monkeypatch.setattr(bench_latency, "warmup_session", lambda voice, text="hi": 0.02)

        json_out = tmp_path / "results.json"
        runner = CliRunner()
        result = runner.invoke(
            bench_latency.main,
            ["mock-voice", "--runs", "2", "--json-out", str(json_out)],
        )
        assert result.exit_code == 0, result.output
        return json.loads(json_out.read_text())

    def test_top_level_keys_present(self, tmp_path, monkeypatch):
        data = self._run_with_mocked_voice(tmp_path, monkeypatch)
        for key in ("model", "model_path", "sample_rate", "onnxruntime_providers_available",
                    "load", "cold_warmup_run_s", "runs", "1sentence", "5sentence"):
            assert key in data

    def test_load_keys_present(self, tmp_path, monkeypatch):
        data = self._run_with_mocked_voice(tmp_path, monkeypatch)
        assert "config_parse_s" in data["load"]
        assert "session_create_s" in data["load"]

    def test_per_text_section_keys_present(self, tmp_path, monkeypatch):
        data = self._run_with_mocked_voice(tmp_path, monkeypatch)
        for label in ("1sentence", "5sentence"):
            section = data[label]
            for key in ("stages", "ttfa", "total", "rtf", "audio_seconds"):
                assert key in section
            for stat_key in ("median", "p95", "min", "max"):
                assert stat_key in section["ttfa"]

    def test_stage_names_present_in_stages(self, tmp_path, monkeypatch):
        data = self._run_with_mocked_voice(tmp_path, monkeypatch)
        stage_names = data["1sentence"]["stages"].keys()
        for expected in ("text_normalize", "diacritics", "phonemize", "tokenize",
                         "session_run_total", "postprocess_total"):
            assert expected in stage_names
