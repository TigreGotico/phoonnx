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


class TestIsMusicLikeScore(unittest.TestCase):
    def test_silence_is_not_music(self):
        import numpy as np
        from phoonnx_train.quality_filter import is_music_like_score
        silence = np.zeros(16000, dtype=np.float32)
        self.assertEqual(is_music_like_score(silence, 16000, "", 1.0), 0.0)

    def test_empty_audio_is_not_music(self):
        import numpy as np
        from phoonnx_train.quality_filter import is_music_like_score
        self.assertEqual(is_music_like_score(np.array([]), 16000, "", 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
