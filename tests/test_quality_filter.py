"""Tests for generic on-demand quality-metric filtering."""
import unittest
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from phoonnx_train.quality_filter import (FilterSpec, apply_quality_filters,
                                           parse_filter_spec, register_scorer)


@dataclass
class _Sample:
    """Minimal stand-in for Utterance: only audio_path/text matter here."""
    name: str
    text: str = "hello world"
    audio_path: str = ""


def _stub_loader(scores: Dict[str, Dict[str, float]]):
    """Fake audio_loader: returns (audio, sr, duration) where 'audio' is the
    sample name itself, so scorer stubs can key off it."""
    def loader(path):
        return path, 16000, scores.get(path, {}).get("duration", 1.0)
    return loader


class TestParseFilterSpec(unittest.TestCase):
    def test_both_bounds(self):
        spec = parse_filter_spec("utmos:3.0:4.5")
        self.assertEqual(spec, FilterSpec("utmos", 3.0, 4.5))

    def test_unbounded_max(self):
        spec = parse_filter_spec("utmos:3.0:")
        self.assertEqual(spec, FilterSpec("utmos", 3.0, None))

    def test_unbounded_min(self):
        spec = parse_filter_spec("wpm::400")
        self.assertEqual(spec, FilterSpec("wpm", None, 400.0))

    def test_boolean_like_bounds(self):
        spec = parse_filter_spec("is_music_like:0:0")
        self.assertEqual(spec, FilterSpec("is_music_like", 0.0, 0.0))

    def test_malformed_spec_raises(self):
        with self.assertRaises(ValueError):
            parse_filter_spec("utmos:3.0")
        with self.assertRaises(ValueError):
            parse_filter_spec("utmos:3.0:4.0:5.0")

    def test_empty_column_raises(self):
        with self.assertRaises(ValueError):
            parse_filter_spec(":1:2")

    def test_non_numeric_bound_raises(self):
        with self.assertRaises(ValueError):
            parse_filter_spec("utmos:low:high")


class TestApplyQualityFilters(unittest.TestCase):
    def setUp(self):
        # Register lightweight test-only scorers so this suite never needs
        # librosa/torch/speechmos: each stub reads its value straight off
        # the sample name from a per-test score table.
        self._orig_registry = {}

    def _register(self, name, table):
        def scorer(audio, sr, text, duration):
            return table[audio]
        register_scorer(name, scorer)
        return scorer

    def test_drops_samples_outside_bounds(self):
        samples = [_Sample("a", audio_path="a"), _Sample("b", audio_path="b"),
                   _Sample("c", audio_path="c")]
        self._register("metric_a", {"a": 4.0, "b": 2.0, "c": 3.5})
        specs = [parse_filter_spec("metric_a:3.0:")]
        kept, dropped = apply_quality_filters(
            samples, specs,
            audio_path_fn=lambda s: s.audio_path,
            text_fn=lambda s: s.text,
            audio_loader=_stub_loader({}),
        )
        self.assertEqual([s.name for s in kept], ["a", "c"])
        self.assertEqual(dropped, {"metric_a": 1})

    def test_unbounded_min_keeps_everything_below_max(self):
        samples = [_Sample("a", audio_path="a"), _Sample("b", audio_path="b")]
        self._register("metric_b", {"a": 1.0, "b": 999.0})
        specs = [parse_filter_spec("metric_b::100")]
        kept, dropped = apply_quality_filters(
            samples, specs,
            audio_path_fn=lambda s: s.audio_path,
            text_fn=lambda s: s.text,
            audio_loader=_stub_loader({}),
        )
        self.assertEqual([s.name for s in kept], ["a"])
        self.assertEqual(dropped, {"metric_b": 1})

    def test_unbounded_max_keeps_everything_above_min(self):
        samples = [_Sample("a", audio_path="a"), _Sample("b", audio_path="b")]
        self._register("metric_c", {"a": 1.0, "b": 999.0})
        specs = [parse_filter_spec("metric_c:100:")]
        kept, dropped = apply_quality_filters(
            samples, specs,
            audio_path_fn=lambda s: s.audio_path,
            text_fn=lambda s: s.text,
            audio_loader=_stub_loader({}),
        )
        self.assertEqual([s.name for s in kept], ["b"])
        self.assertEqual(dropped, {"metric_c": 1})

    def test_unknown_column_warns_and_is_skipped(self):
        samples = [_Sample("a", audio_path="a"), _Sample("b", audio_path="b")]
        specs = [parse_filter_spec("totally_made_up_metric:1:2")]
        with self.assertLogs("phoonnx_train.quality_filter", level="WARNING") as log:
            kept, dropped = apply_quality_filters(
                samples, specs,
                audio_path_fn=lambda s: s.audio_path,
                text_fn=lambda s: s.text,
                audio_loader=_stub_loader({}),
            )
        self.assertEqual([s.name for s in kept], ["a", "b"])
        self.assertEqual(dropped, {})
        self.assertTrue(any("unknown quality filter column" in m for m in log.output))

    def test_multiple_filters_combine_with_and_semantics(self):
        samples = [_Sample("a", audio_path="a"), _Sample("b", audio_path="b"),
                   _Sample("c", audio_path="c")]
        # a: passes both, b: fails first, c: passes first, fails second
        self._register("metric_d", {"a": 4.0, "b": 1.0, "c": 4.0})
        self._register("metric_e", {"a": 4.0, "b": 4.0, "c": 1.0})
        specs = [parse_filter_spec("metric_d:3.0:"), parse_filter_spec("metric_e:3.0:")]
        kept, dropped = apply_quality_filters(
            samples, specs,
            audio_path_fn=lambda s: s.audio_path,
            text_fn=lambda s: s.text,
            audio_loader=_stub_loader({}),
        )
        self.assertEqual([s.name for s in kept], ["a"])
        self.assertEqual(dropped["metric_d"], 1)
        self.assertEqual(dropped["metric_e"], 1)

    def test_short_circuits_before_expensive_filter(self):
        # 'b' fails the cheap filter registered first in scorer order; the
        # second (expensive) scorer must never be evaluated for it.
        calls = []

        def cheap(audio, sr, text, duration):
            return {"a": 5.0, "b": 0.0}[audio]

        def expensive(audio, sr, text, duration):
            calls.append(audio)
            return 5.0

        register_scorer("cheap_metric", cheap)
        register_scorer("expensive_metric", expensive)
        samples = [_Sample("a", audio_path="a"), _Sample("b", audio_path="b")]
        specs = [parse_filter_spec("cheap_metric:1:"), parse_filter_spec("expensive_metric:1:")]
        kept, dropped = apply_quality_filters(
            samples, specs,
            audio_path_fn=lambda s: s.audio_path,
            text_fn=lambda s: s.text,
            audio_loader=_stub_loader({}),
        )
        self.assertEqual([s.name for s in kept], ["a"])
        self.assertNotIn("b", calls)
        self.assertIn("a", calls)

    def test_is_music_like_boolean_filter_keeps_only_non_music(self):
        samples = [_Sample("speech", audio_path="speech"),
                   _Sample("music", audio_path="music")]
        self._register("is_music_like", {"speech": 0.0, "music": 1.0})
        specs = [parse_filter_spec("is_music_like:0:0")]
        kept, dropped = apply_quality_filters(
            samples, specs,
            audio_path_fn=lambda s: s.audio_path,
            text_fn=lambda s: s.text,
            audio_loader=_stub_loader({}),
        )
        self.assertEqual([s.name for s in kept], ["speech"])
        self.assertEqual(dropped, {"is_music_like": 1})

    def test_wpm_needs_no_audio_load(self):
        # wpm is arithmetic-only: it must never touch the audio loader,
        # only the (cheap, header-only) duration lookup.
        from unittest.mock import patch

        from phoonnx_train.quality_filter import wpm_score
        register_scorer("wpm", wpm_score)

        def failing_loader(path):
            raise AssertionError("audio_loader should not be called for wpm-only filters")

        samples = [_Sample("a", text="one two three four", audio_path="a")]
        specs = [parse_filter_spec("wpm:1:")]
        with patch("phoonnx_train.quality_filter._duration_only", return_value=60.0):
            kept, dropped = apply_quality_filters(
                samples, specs,
                audio_path_fn=lambda s: s.audio_path,
                text_fn=lambda s: s.text,
                audio_loader=failing_loader,
            )
        self.assertEqual(len(kept), 1)

    def test_no_filters_returns_all_samples_unchanged(self):
        samples = [_Sample("a", audio_path="a"), _Sample("b", audio_path="b")]
        kept, dropped = apply_quality_filters(
            samples, [],
            audio_path_fn=lambda s: s.audio_path,
            text_fn=lambda s: s.text,
            audio_loader=_stub_loader({}),
        )
        self.assertEqual(kept, samples)
        self.assertEqual(dropped, {})

    def test_scorer_exception_drops_sample_without_crashing(self):
        def boom(audio, sr, text, duration):
            raise RuntimeError("scorer blew up")

        register_scorer("flaky_metric", boom)
        samples = [_Sample("a", audio_path="a"), _Sample("b", audio_path="b")]
        specs = [parse_filter_spec("flaky_metric:1:")]
        kept, dropped = apply_quality_filters(
            samples, specs,
            audio_path_fn=lambda s: s.audio_path,
            text_fn=lambda s: s.text,
            audio_loader=_stub_loader({}),
        )
        self.assertEqual(kept, [])


class TestWpmScore(unittest.TestCase):
    def test_wpm_arithmetic(self):
        from phoonnx_train.quality_filter import wpm_score
        # 6 words over 30s = 12 wpm
        self.assertAlmostEqual(wpm_score(None, 16000, "one two three four five six", 30.0), 12.0)

    def test_wpm_zero_duration_is_safe(self):
        from phoonnx_train.quality_filter import wpm_score
        self.assertEqual(wpm_score(None, 16000, "hello", 0.0), 0.0)


class TestRhythmicityScore(unittest.TestCase):
    def test_silence_has_no_rhythmicity(self):
        import numpy as np
        from phoonnx_train.quality_filter import rhythmicity_score
        silence = np.zeros(16000 * 3, dtype=np.float32)
        self.assertEqual(rhythmicity_score(silence, 16000), 0.0)

    def test_periodic_click_train_scores_high(self):
        import numpy as np
        from phoonnx_train.quality_filter import rhythmicity_score
        sr = 16000
        clicks = np.zeros(sr * 3, dtype=np.float32)
        clicks[:: sr // 2] = 1.0  # a click every 0.5s: a clean periodic pulse
        self.assertGreater(rhythmicity_score(clicks, sr), 0.5)

    def test_empty_audio_is_safe(self):
        import numpy as np
        from phoonnx_train.quality_filter import rhythmicity_score
        self.assertEqual(rhythmicity_score(np.array([]), 16000), 0.0)
        self.assertEqual(rhythmicity_score(None, 16000), 0.0)


class TestIsMusicLikeScore(unittest.TestCase):
    def test_thresholds_rhythmicity_below(self):
        from unittest.mock import patch
        from phoonnx_train.quality_filter import is_music_like_score
        with patch("phoonnx_train.quality_filter.rhythmicity_score", return_value=0.2):
            self.assertEqual(is_music_like_score(object(), 16000, "", 1.0), 0.0)

    def test_thresholds_rhythmicity_above(self):
        from unittest.mock import patch
        from phoonnx_train.quality_filter import is_music_like_score
        with patch("phoonnx_train.quality_filter.rhythmicity_score", return_value=0.9):
            self.assertEqual(is_music_like_score(object(), 16000, "", 1.0), 1.0)

    def test_empty_audio_is_not_music(self):
        import numpy as np
        from phoonnx_train.quality_filter import is_music_like_score
        self.assertEqual(is_music_like_score(np.array([]), 16000, "", 0.0), 0.0)


class TestPlcmosScore(unittest.TestCase):
    def test_reads_plcmos_column_from_df(self):
        import numpy as np
        import pandas as pd
        from unittest.mock import MagicMock, patch
        from phoonnx_train.quality_filter import plcmos_score

        fake_run = MagicMock(return_value=pd.DataFrame([{"filename": "clip", "plcmos": 3.7}]))
        with patch("phoonnx_train.quality_filter._load_speechmos_run", return_value=fake_run):
            score = plcmos_score(np.zeros(16000), 16000, "", 1.0)
        self.assertAlmostEqual(score, 3.7)
        fake_run.assert_called_once()
        args, kwargs = fake_run.call_args
        self.assertEqual(kwargs.get("return_df"), True)

    def test_resamples_to_16k(self):
        import numpy as np
        import pandas as pd
        from unittest.mock import MagicMock, patch
        from phoonnx_train.quality_filter import plcmos_score

        fake_run = MagicMock(return_value=pd.DataFrame([{"plcmos": 3.0}]))
        with patch("phoonnx_train.quality_filter._load_speechmos_run", return_value=fake_run):
            plcmos_score(np.zeros(48000), 48000, "", 1.0)
        args, kwargs = fake_run.call_args
        self.assertEqual(args[1] if len(args) > 1 else kwargs.get("sr"), 16000)


class TestAecmosScore(unittest.TestCase):
    def test_reads_aecmos_column_from_df_scenarioless(self):
        import numpy as np
        import pandas as pd
        from unittest.mock import MagicMock, patch
        from phoonnx_train.quality_filter import aecmos_score

        fake_run = MagicMock(return_value=pd.DataFrame([{"aecmos": 4.1}]))
        with patch("phoonnx_train.quality_filter._load_speechmos_run", return_value=fake_run):
            score = aecmos_score(np.zeros(16000), 16000, "", 1.0)
        self.assertAlmostEqual(score, 4.1)
        args, kwargs = fake_run.call_args
        # scenarioless mode: talk_type must stay None so no far-end
        # reference signal is required to score a single clip
        self.assertIsNone(kwargs.get("talk_type"))


class TestSpeechmosDfValue(unittest.TestCase):
    def test_falls_back_to_last_numeric_column_on_name_mismatch(self):
        import pandas as pd
        from phoonnx_train.quality_filter import _speechmos_df_value
        df = pd.DataFrame([{"filename": "clip.wav", "some_other_mos_name": 2.5}])
        self.assertAlmostEqual(_speechmos_df_value(df, ["plcmos"]), 2.5)

    def test_raises_when_no_numeric_columns(self):
        import pandas as pd
        from phoonnx_train.quality_filter import _speechmos_df_value
        df = pd.DataFrame([{"filename": "clip.wav"}])
        with self.assertRaises(ValueError):
            _speechmos_df_value(df, ["plcmos"])


class TestSnrScore(unittest.TestCase):
    def test_loud_signal_over_silence_scores_high(self):
        import numpy as np
        from phoonnx_train.quality_filter import snr_score
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        # half the clip near-silent, half a strong tone: clear separation
        # between the noise-floor tier and the signal tier.
        loud = 0.9 * np.sin(2 * np.pi * 440 * t)
        quiet = 1e-4 * np.sin(2 * np.pi * 440 * t)
        audio = np.concatenate([quiet, loud]).astype(np.float32)
        score = snr_score(audio, sr, "", 2.0)
        self.assertGreater(score, 20.0)

    def test_uniform_signal_scores_near_zero_db(self):
        import numpy as np
        from phoonnx_train.quality_filter import snr_score
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        score = snr_score(audio, sr, "", 1.0)
        self.assertLess(score, 5.0)

    def test_empty_audio_is_safe(self):
        import numpy as np
        from phoonnx_train.quality_filter import snr_score
        self.assertEqual(snr_score(np.array([]), 16000, "", 0.0), 0.0)
        self.assertEqual(snr_score(None, 16000, "", 0.0), 0.0)
        self.assertEqual(snr_score(np.zeros(100), 0, "", 0.0), 0.0)


class TestClippingScore(unittest.TestCase):
    def test_fraction_of_near_full_scale_samples(self):
        import numpy as np
        from phoonnx_train.quality_filter import clipping_score
        audio = np.array([0.0, 0.5, 1.0, -1.0, 0.995, -0.2] * 10, dtype=np.float32)
        score = clipping_score(audio, 16000, "", 1.0)
        # 3 of 6 repeated values (1.0, -1.0, 0.995) exceed the 0.99 threshold
        self.assertAlmostEqual(score, 3 / 6)

    def test_no_clipping(self):
        import numpy as np
        from phoonnx_train.quality_filter import clipping_score
        audio = np.full(1000, 0.1, dtype=np.float32)
        self.assertEqual(clipping_score(audio, 16000, "", 1.0), 0.0)

    def test_empty_audio_is_safe(self):
        import numpy as np
        from phoonnx_train.quality_filter import clipping_score
        self.assertEqual(clipping_score(np.array([]), 16000, "", 0.0), 0.0)
        self.assertEqual(clipping_score(None, 16000, "", 0.0), 0.0)


class TestVadRatioScore(unittest.TestCase):
    def test_ratio_from_mocked_segments(self):
        import numpy as np
        from unittest.mock import MagicMock, patch
        from phoonnx_train.quality_filter import configure_vad_model, vad_ratio_score

        seg_a = MagicMock(duration=1.0)
        seg_b = MagicMock(duration=1.5)
        fake_vad = MagicMock()
        fake_vad.get_speech_segments.return_value = [seg_a, seg_b]

        configure_vad_model("fake-model")
        with patch("phoonnx_train.quality_filter._get_vad_model", return_value=fake_vad):
            score = vad_ratio_score(np.zeros(16000 * 5), 16000, "", 5.0)
        self.assertAlmostEqual(score, 2.5 / 5.0)
        fake_vad.get_speech_segments.assert_called_once()

    def test_empty_audio_is_safe(self):
        import numpy as np
        from phoonnx_train.quality_filter import vad_ratio_score
        self.assertEqual(vad_ratio_score(np.array([]), 16000, "", 0.0), 0.0)
        self.assertEqual(vad_ratio_score(np.zeros(100), 16000, "", 0.0), 0.0)

    def test_configure_vad_model_invalidates_cache(self):
        from phoonnx_train import quality_filter as qf
        qf.configure_vad_model("model-a")
        self.assertIsNone(qf._vad_model)
        qf._vad_model = object()
        qf._vad_model_cache_key = "model-a"
        qf.configure_vad_model("model-b")
        self.assertIsNone(qf._vad_model)
        self.assertIsNone(qf._vad_model_cache_key)


class TestSpeakerConsistencyScore(unittest.TestCase):
    def test_similar_windows_score_high(self):
        import numpy as np
        from unittest.mock import MagicMock, patch
        from phoonnx_train.quality_filter import (configure_speaker_model,
                                                   speaker_consistency_score)

        same_vec = np.array([1.0, 0.0, 0.0])
        fake_embedder = MagicMock()
        fake_embedder.embed.return_value = same_vec

        configure_speaker_model("fake-model")
        audio = np.zeros(16000 * 6, dtype=np.float32)
        with patch("phoonnx_train.quality_filter._get_speaker_embedder", return_value=fake_embedder):
            score = speaker_consistency_score(audio, 16000, "", 6.0, num_windows=3)
        self.assertAlmostEqual(score, 1.0)

    def test_dissimilar_windows_score_low(self):
        import numpy as np
        from unittest.mock import MagicMock, patch
        from phoonnx_train.quality_filter import (configure_speaker_model,
                                                   speaker_consistency_score)

        vectors = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])]
        fake_embedder = MagicMock()
        fake_embedder.embed.side_effect = vectors

        configure_speaker_model("fake-model")
        audio = np.zeros(16000 * 6, dtype=np.float32)
        with patch("phoonnx_train.quality_filter._get_speaker_embedder", return_value=fake_embedder):
            score = speaker_consistency_score(audio, 16000, "", 6.0, num_windows=3)
        # minimum pairwise similarity is between orthogonal windows 0 and 1
        self.assertAlmostEqual(score, 0.0, places=6)

    def test_too_short_clip_is_trivially_consistent(self):
        import numpy as np
        from phoonnx_train.quality_filter import speaker_consistency_score
        audio = np.zeros(100, dtype=np.float32)  # far too short to window
        score = speaker_consistency_score(audio, 16000, "", 100 / 16000, num_windows=3)
        self.assertEqual(score, 1.0)

    def test_zero_duration_is_trivially_consistent(self):
        import numpy as np
        from phoonnx_train.quality_filter import speaker_consistency_score
        self.assertEqual(speaker_consistency_score(np.array([]), 16000, "", 0.0), 1.0)


class TestWerScore(unittest.TestCase):
    def test_word_error_rate_arithmetic(self):
        from phoonnx_train.quality_filter import _word_error_rate
        self.assertEqual(_word_error_rate("hello world", "hello world"), 0.0)
        # one substitution out of two reference words
        self.assertAlmostEqual(_word_error_rate("hello world", "hello there"), 0.5)
        # one deletion out of two reference words
        self.assertAlmostEqual(_word_error_rate("hello world", "hello"), 0.5)
        # one insertion: reference has 2 words, hypothesis adds one
        self.assertAlmostEqual(_word_error_rate("hello world", "hello big world"), 0.5)

    def test_empty_reference_edge_cases(self):
        from phoonnx_train.quality_filter import _word_error_rate
        self.assertEqual(_word_error_rate("", ""), 0.0)
        self.assertEqual(_word_error_rate("", "unexpected words"), 1.0)

    def test_wer_score_uses_mocked_asr_model(self):
        import numpy as np
        from unittest.mock import MagicMock, patch
        from phoonnx_train.quality_filter import configure_asr_model, wer_score

        fake_model = MagicMock()
        fake_model.recognize.return_value = "hello world"

        configure_asr_model("fake-asr-model")
        with patch("phoonnx_train.quality_filter._get_asr_model", return_value=fake_model):
            score = wer_score(np.zeros(16000), 16000, "hello world", 1.0)
        self.assertEqual(score, 0.0)

    def test_bad_asr_model_fails_loudly_not_silently(self):
        # onnx_asr is a real dependency here; a model name it does not
        # recognize must raise, not silently fall back to another backend
        # or let the 'wer' filter behave as if it were skipped.
        from phoonnx_train.quality_filter import _get_asr_model, configure_asr_model
        configure_asr_model("definitely-not-a-real-onnx-asr-model-xyz")
        with self.assertRaises(RuntimeError) as ctx:
            _get_asr_model()
        self.assertIn("onnx_asr", str(ctx.exception))

    def test_wer_requires_asr_model_to_be_configured(self):
        from phoonnx_train.quality_filter import _get_asr_model, configure_asr_model
        configure_asr_model(None)
        with self.assertRaises(RuntimeError):
            _get_asr_model()


class TestScorerOrdering(unittest.TestCase):
    def test_arithmetic_scorers_precede_model_based_ones(self):
        from phoonnx_train.quality_filter import known_scorers
        order = known_scorers()
        cheap_tier = {"wpm", "snr", "clipping"}
        model_tier = {"utmos", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl",
                     "plcmos", "aecmos", "vad_ratio", "speaker_consistency"}
        max_cheap_index = max(order.index(name) for name in cheap_tier)
        min_model_index = min(order.index(name) for name in model_tier)
        self.assertLess(max_cheap_index, min_model_index)

    def test_wer_sorts_last(self):
        # Other tests in this module register their own throwaway scorer
        # names into the same module-level registry, so assert wer's
        # position relative to the real production scorers only, not
        # against the literal tail of the (possibly test-polluted) list.
        from phoonnx_train.quality_filter import known_scorers
        order = known_scorers()
        production_scorers = {"wpm", "snr", "clipping", "is_music_like",
                              "vad_ratio", "dnsmos_sig", "dnsmos_bak",
                              "dnsmos_ovrl", "plcmos", "aecmos", "utmos",
                              "speaker_consistency", "wer"}
        wer_index = order.index("wer")
        for name in production_scorers - {"wer"}:
            self.assertLess(order.index(name), wer_index)

    def test_short_circuits_before_wer(self):
        # A sample failing a cheap filter must never reach the wer scorer.
        from unittest.mock import MagicMock, patch
        from phoonnx_train.quality_filter import configure_asr_model

        def exploding_asr(*a, **k):
            raise AssertionError("wer scorer should not run for a sample "
                                 "already dropped by a cheaper filter")

        register_scorer("wer", exploding_asr)
        configure_asr_model("unused")

        samples = [_Sample("bad", audio_path="bad", text="x")]
        self._register_wpm_fail("bad")
        specs = [parse_filter_spec("wpm:1000:"), parse_filter_spec("wer:0:0.1")]
        kept, dropped = apply_quality_filters(
            samples, specs,
            audio_path_fn=lambda s: s.audio_path,
            text_fn=lambda s: s.text,
            audio_loader=_stub_loader({}),
        )
        self.assertEqual(kept, [])
        self.assertEqual(dropped["wpm"], 1)

        # restore the real scorers for any subsequent tests in this run
        from phoonnx_train.quality_filter import wer_score, wpm_score
        register_scorer("wer", wer_score)
        register_scorer("wpm", wpm_score)

    def _register_wpm_fail(self, name):
        def low_wpm(audio, sr, text, duration):
            return 1.0
        register_scorer("wpm", low_wpm)


if __name__ == "__main__":
    unittest.main()
