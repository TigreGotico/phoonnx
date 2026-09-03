"""End-to-end CLI tests for phoonnx_train/preprocess.py, focused on the
manifest-writing / quality-filter / finetune wiring not already exercised by
tests/test_dataset_loaders.py (multi-source namespacing, --resume,
--corpus-only-map, precomputed-phoneme mismatch). See that file first to
avoid duplicating coverage.

Uses ``--phonemes-column`` with a stub phonemizer (like test_dataset_loaders)
so no real phonemizer backend is needed, and real tiny WAV files on disk so
the multiprocessing worker's audio path and the quality-filter scorers run
for real, in seconds.
"""
import io
import json
import queue
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from phoonnx.config import Alphabet, PhonemeType
from phoonnx_train import preprocess
from phoonnx_train.dataset_loaders import PreprocessorConfig, Utterance

# This module both runs real torch ops in-process (via direct
# phonemize_worker() calls) AND forks multiprocessing.Process workers
# repeatedly (via the CLI). Forking after a library has spun up its own
# OS thread pool is a known deadlock hazard (see the "process is
# multi-threaded, use of fork()" warning torch/numba already emit in this
# suite) -- keeping torch single-threaded here avoids accumulating extra
# threads across tests that a later fork could inherit mid-operation.
torch.set_num_threads(1)


def _wav_bytes(seconds: float = 0.5, sr: int = 16000) -> bytes:
    tone = 0.2 * np.sin(2 * np.pi * 220 * np.linspace(0, seconds, int(sr * seconds), endpoint=False))
    buf = io.BytesIO()
    sf.write(buf, tone.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


class _DummyPhonemizer:
    """Mirrors test_dataset_loaders' stub: no real phonemizer backend needed
    since every row supplies precomputed phonemes via --phonemes-column."""
    alphabet = Alphabet.IPA

    def phonemize_to_list(self, text, lang):
        return list(text.replace(" ", ""))

    def add_diacritics(self, text, lang):
        return text


def _jsonl(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _invoke(args, catch_exceptions=True):
    import click.testing
    # -w 1: these fixtures are 1-2 rows; the CLI's os.cpu_count() worker
    # default would fork a full pool of processes per test for no benefit.
    args = [*args, "--max-workers", "1"]
    with patch.object(preprocess, "get_phonemizer", return_value=_DummyPhonemizer()):
        return click.testing.CliRunner().invoke(preprocess.cli, args, catch_exceptions=catch_exceptions)


class TestMalformedAndMissingAudio(unittest.TestCase):
    def test_missing_audio_file_is_dropped_not_fatal(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "a.jsonl"
            _jsonl(src, [{"text": "hi", "audio": str(tmp / "does_not_exist.wav"), "phon": "h i"}])
            out = tmp / "out"
            result = _invoke([
                "-i", str(src), "-o", str(out), "-l", "en", "--phonemes-column", "phon",
            ])
            # the row loads fine (text + a path string); only the audio
            # caching inside the worker fails, so this is NOT the load-time
            # "No valid utterances found" early-return -- it reaches the
            # writer with zero surviving rows.
            self.assertEqual(result.exit_code, 0, result.output)
            lines = [x for x in (out / "dataset.jsonl").read_text().splitlines() if x]
            self.assertEqual(lines, [])

    def test_malformed_audio_bytes_dropped_others_kept(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            good_wav = tmp / "good.wav"
            good_wav.write_bytes(_wav_bytes())
            bad_wav = tmp / "bad.wav"
            bad_wav.write_bytes(b"not a real wav file at all")

            src = tmp / "a.jsonl"
            _jsonl(src, [
                {"text": "hello there", "audio": str(good_wav), "phon": "h e l l o"},
                {"text": "garbled clip", "audio": str(bad_wav), "phon": "g a r"},
            ])
            out = tmp / "out"
            result = _invoke([
                "-i", str(src), "-o", str(out), "-l", "en", "--phonemes-column", "phon",
            ])
            self.assertEqual(result.exit_code, 0, result.output)
            lines = [json.loads(x) for x in (out / "dataset.jsonl").read_text().splitlines() if x]
            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0]["text"], "hello there")


class TestEmptyTranscript(unittest.TestCase):
    def test_empty_precomputed_phonemes_dropped(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "a.jsonl"
            _jsonl(src, [{"text": "", "audio": "a.wav", "phon": ""}])
            out = tmp / "out"
            with self.assertLogs("preprocess", level="ERROR") as logs:
                result = _invoke([
                    "-i", str(src), "-o", str(out), "-l", "en", "--skip-audio",
                    "--phonemes-column", "phon",
                ])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(any("No valid utterances found" in m for m in logs.output))

    def test_empty_transcript_without_precomputed_phonemes_dropped(self):
        # no phonemes-column -> normalize()+phonemizer runs; an empty
        # normalized transcript yields no phonemes and must be dropped, not
        # crash the worker.
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "a.jsonl"
            _jsonl(src, [{"text": "   ", "audio": "a.wav"}])
            out = tmp / "out"
            result = _invoke([
                "-i", str(src), "-o", str(out), "-l", "en", "--skip-audio",
            ])
            # loads fine (non-empty raw text field); only phonemization of
            # the normalized (blank) text fails inside the worker.
            self.assertEqual(result.exit_code, 0, result.output)
            lines = [x for x in (out / "dataset.jsonl").read_text().splitlines() if x]
            self.assertEqual(lines, [])


class TestQualityFilterWiring(unittest.TestCase):
    """Covers the --filter CLI wiring block in preprocess.cli (configure_*
    model setters, FilterSpec parsing, metrics sidecar, empty-result handling)."""

    def _dataset(self, tmp: Path):
        wav = tmp / "clip.wav"
        wav.write_bytes(_wav_bytes(seconds=1.0))
        src = tmp / "a.jsonl"
        _jsonl(src, [{"text": "hello world this is a test", "audio": str(wav), "phon": "h e l l o"}])
        return src

    def test_filter_dropping_every_utterance_is_handled_gracefully(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = self._dataset(tmp)
            out = tmp / "out"
            # wpm (words per minute) of a real clip is nowhere near 1e6
            with self.assertLogs("preprocess", level="ERROR") as logs:
                result = _invoke([
                    "-i", str(src), "-o", str(out), "-l", "en", "--phonemes-column", "phon",
                    "--filter", "wpm:1000000:",
                ])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(any("No utterances left after quality filtering" in m for m in logs.output))
            self.assertFalse((out / "dataset.jsonl").exists())

    def test_filter_keeping_utterances_writes_metrics_sidecar(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = self._dataset(tmp)
            out = tmp / "out"
            metrics_out = tmp / "metrics.parquet"
            result = _invoke([
                "-i", str(src), "-o", str(out), "-l", "en", "--phonemes-column", "phon",
                "--filter", "wpm:0:", "--metrics-out", str(metrics_out),
            ])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue((out / "dataset.jsonl").exists())
            self.assertTrue(metrics_out.exists())

            from phoonnx_train.preprocess import _read_metrics_sidecar
            sidecar = _read_metrics_sidecar(metrics_out)
            self.assertEqual(len(sidecar), 1)
            self.assertIn("wpm", next(iter(sidecar.values())))

    def test_unknown_filter_column_warns_and_keeps_everything(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = self._dataset(tmp)
            out = tmp / "out"
            result = _invoke([
                "-i", str(src), "-o", str(out), "-l", "en", "--phonemes-column", "phon",
                "--filter", "not_a_real_metric:0:1",
            ])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue((out / "dataset.jsonl").exists())

    def test_invalid_filter_spec_syntax_rejected(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = self._dataset(tmp)
            out = tmp / "out"
            result = _invoke([
                "-i", str(src), "-o", str(out), "-l", "en", "--phonemes-column", "phon",
                "--filter", "wpm:only-one-colon",
            ], catch_exceptions=True)
            self.assertNotEqual(result.exit_code, 0)


class TestFinetunePhonemeMap(unittest.TestCase):
    """--prev-config/--drop-extra-phonemes only applies to *phonemized* text
    (real phonemizer, no --phonemes-column): a mismatched precomputed-column
    phoneme always fails loudly regardless of --drop-extra-phonemes, by
    design (see the docstring above the check in preprocess.cli)."""

    def _prev_config(self, tmp: Path) -> Path:
        # graphemes/unicode: only the letters actually in this prev map are
        # "known"; anything else the real phonemizer emits for new text
        # counts as a new/extra phoneme.
        cfg = {
            "phoneme_id_map": {"h": 4, "i": 5, "<pad>": 0, "<bos>": 1, "<eos>": 2, "<blank>": 3},
            "num_symbols": 6,
        }
        p = tmp / "prev_config.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")
        return p

    def _invoke_real_phonemizer(self, args):
        import click.testing
        # no get_phonemizer patch here: graphemes/unicode needs no backend
        args = [*args, "--max-workers", "1"]
        return click.testing.CliRunner().invoke(preprocess.cli, args, catch_exceptions=False)

    def test_drop_extra_phonemes_true_discards_and_succeeds(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            prev = self._prev_config(tmp)
            src = tmp / "a.jsonl"
            # 'z' graphemizes to itself, which is not in the prev map
            _jsonl(src, [{"text": "hiz", "audio": "a.wav"}])
            out = tmp / "out"
            result = self._invoke_real_phonemizer([
                "-i", str(src), "-o", str(out), "-l", "en", "--skip-audio",
                "--phoneme-type", "graphemes", "--alphabet", "unicode",
                "--prev-config", str(prev), "--drop-extra-phonemes", "true",
            ])
            self.assertEqual(result.exit_code, 0, result.output)
            config = json.loads((out / "config.json").read_text())
            self.assertEqual(config["num_symbols"], 6)  # unchanged from prev
            self.assertNotIn("z", config["phoneme_id_map"])

    def test_drop_extra_phonemes_false_raises(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            prev = self._prev_config(tmp)
            src = tmp / "a.jsonl"
            _jsonl(src, [{"text": "hiz", "audio": "a.wav"}])
            out = tmp / "out"
            with self.assertRaises(ValueError):
                self._invoke_real_phonemizer([
                    "-i", str(src), "-o", str(out), "-l", "en", "--skip-audio",
                    "--phoneme-type", "graphemes", "--alphabet", "unicode",
                    "--prev-config", str(prev), "--drop-extra-phonemes", "false",
                ])

    def test_precomputed_phonemes_mismatch_ignores_drop_extra_phonemes(self):
        # a --phonemes-column mismatch always raises, even with
        # --drop-extra-phonemes true, since it is never re-derivable by
        # re-phonemizing (the whole point of supplying it precomputed).
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            prev = self._prev_config(tmp)
            src = tmp / "a.jsonl"
            _jsonl(src, [{"text": "hiz", "audio": "a.wav", "phon": "h i z"}])
            out = tmp / "out"
            with self.assertRaises(ValueError) as ctx:
                _invoke([
                    "-i", str(src), "-o", str(out), "-l", "en", "--skip-audio",
                    "--phonemes-column", "phon", "--prev-config", str(prev),
                    "--drop-extra-phonemes", "true",
                ], catch_exceptions=False)
            self.assertIn("absent from the final phoneme map", str(ctx.exception))


class TestJsonlPathOverrides(unittest.TestCase):
    def test_jsonl_audio_path_override_rewrites_wav_path(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            wav_dir = tmp / "orig" / "wav"
            wav_dir.mkdir(parents=True)
            wav_path = wav_dir / "clip1.wav"
            wav_path.write_bytes(_wav_bytes())

            src = tmp / "a.jsonl"
            _jsonl(src, [{"text": "hi", "audio": str(wav_path), "phon": "h i"}])
            out = tmp / "out"
            result = _invoke([
                "-i", str(src), "-o", str(out), "-l", "en", "--phonemes-column", "phon",
                "--jsonl-audio-path", "/override/base",
            ])
            self.assertEqual(result.exit_code, 0, result.output)
            lines = [json.loads(x) for x in (out / "dataset.jsonl").read_text().splitlines() if x]
            self.assertEqual(lines[0]["audio_path"], "/override/base/wav/clip1.wav")


class TestEngineExtraPreprocessWiring(unittest.TestCase):
    def test_engine_flag_invokes_extra_preprocess_and_merges_fields(self):
        calls = []

        class _FakeEngine:
            def extra_preprocess(self, audio_path, cache_dir, sample_rate, **kwargs):
                calls.append((str(audio_path), kwargs))
                return {"language_id": 7}

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            wav = tmp / "clip.wav"
            wav.write_bytes(_wav_bytes())
            src = tmp / "a.jsonl"
            _jsonl(src, [{"text": "hi", "audio": str(wav), "phon": "h i"}])
            out = tmp / "out"
            with patch("phoonnx_train.engines.get_engine", return_value=_FakeEngine()):
                result = _invoke([
                    "-i", str(src), "-o", str(out), "-l", "en", "--phonemes-column", "phon",
                    "--engine", "vits", "--language-id", "7",
                ])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(len(calls), 1)
            lines = [json.loads(x) for x in (out / "dataset.jsonl").read_text().splitlines() if x]
            self.assertEqual(lines[0]["language_id"], 7)


class TestSpectrogramTooShortSkip(unittest.TestCase):
    def test_utterance_shorter_than_its_phonemes_is_skipped(self):
        # forge an audio_spec_path whose spectrogram has fewer frames than
        # phoneme ids, bypassing real audio caching to hit the guard directly
        import torch as _torch

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            wav = tmp / "clip.wav"
            wav.write_bytes(_wav_bytes(seconds=0.05))  # very short clip
            spec_path = tmp / "clip.spec.pt"
            _torch.save(_torch.zeros(80, 1), spec_path)  # 1 frame only
            norm_path = tmp / "clip.norm.pt"
            _torch.save(_torch.zeros(1, 800), norm_path)

            src = tmp / "a.jsonl"
            # many phonemes -> more than the single spectrogram frame
            _jsonl(src, [{"text": "hi", "audio": str(wav), "phon": "h i h i h i h i h i h i"}])
            out = tmp / "out"

            def _fake_cache_norm_audio(audio_path, cache_dir, detector, sample_rate):
                return norm_path, spec_path

            with patch.object(preprocess, "cache_norm_audio", _fake_cache_norm_audio):
                with self.assertLogs("preprocess", level="WARNING") as logs:
                    result = _invoke([
                        "-i", str(src), "-o", str(out), "-l", "en", "--phonemes-column", "phon",
                    ])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(any("Skipping utterance with more phonemes" in m for m in logs.output))
            lines = [x for x in (out / "dataset.jsonl").read_text().splitlines() if x]
            self.assertEqual(lines, [])


class _FakeSilenceDetector:
    """A make_silence_detector() stand-in for in-process phonemize_worker
    calls. The real detector opens a real onnxruntime session; creating one
    in the *parent* pytest process (instead of inside an already-forked
    worker, as production does) leaves live OS threads behind that a later
    ``multiprocessing.Process``-based test can fork mid-operation and
    deadlock in the child (the classic fork-after-threads hazard -- see the
    "process is multi-threaded, use of fork()" warning torch/numba already
    emit in this suite). Since none of these tests assert on real VAD
    trimming, a trivial "always speech" stand-in keeps them fast, safe to
    fork after, and no less meaningful."""

    def reset(self):
        # Mirrors the real detector's per-utterance state reset (a no-op here).
        pass

    def __call__(self, audio_array, sample_rate=16000):
        return 1.0


class TestPhonemizeWorkerDirect(unittest.TestCase):
    """Drives phonemize_worker() directly in-process (a plain queue.Queue
    instead of multiprocessing's) so both its control flow AND test coverage
    see every branch -- forking it into a real Process, as the CLI does,
    hides its body from coverage.py entirely."""

    def _config(self, tmp: Path, **overrides) -> PreprocessorConfig:
        base = dict(
            input_dir=tmp, output_dir=tmp, language="en", sample_rate=16000,
            cache_dir=tmp / "cache", max_workers=1, single_speaker=False,
            speaker_id=None, phoneme_type=PhonemeType.ESPEAK, alphabet=Alphabet.IPA,
            phonemizer_model="", text_casing="ignore", dataset_name=None,
            audio_quality=None, skip_audio=True, debug=False, add_diacritics=False,
        )
        base.update(overrides)
        return PreprocessorConfig(**base)

    def _run_worker(self, config, utterances):
        task_q = queue.Queue()
        result_q = queue.Queue()
        task_q.put(list(utterances))
        task_q.put(None)

        class _Phonemizer:
            def phonemize_to_list(self, text, lang):
                return list(text.replace(" ", "")) or []

            def add_diacritics(self, text, lang):
                return "D" + text

        with patch.object(preprocess, "make_silence_detector",
                           return_value=_FakeSilenceDetector()):
            preprocess.phonemize_worker(config, task_q, result_q, _Phonemizer())
        results = []
        while not result_q.empty():
            results.append(result_q.get())
        return results

    def test_successful_phonemization_and_skip_audio(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = self._config(tmp, skip_audio=True)
            utt = Utterance(text="hello", audio_path=tmp / "a.wav")
            results = self._run_worker(config, [utt])
        self.assertEqual(len(results), 1)
        processed, phonemes = results[0]
        self.assertIsNotNone(processed)
        self.assertEqual(processed.phonemes, list("hello"))
        self.assertEqual(phonemes, set("hello"))

    def test_add_diacritics_applied_before_phonemizing(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = self._config(tmp, skip_audio=True, add_diacritics=True)
            utt = Utterance(text="hi", audio_path=tmp / "a.wav")
            results = self._run_worker(config, [utt])
        processed, _ = results[0]
        self.assertEqual(processed.phonemes[0], "D")  # add_diacritics prefix survived

    def test_empty_precomputed_phonemes_raises_and_is_dropped(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = self._config(tmp, skip_audio=True)
            utt = Utterance(text="x", audio_path=tmp / "a.wav",
                            phonemes=[], phonemes_precomputed=True)
            results = self._run_worker(config, [utt])
        processed, phonemes = results[0]
        self.assertIsNone(processed)
        self.assertEqual(phonemes, set())

    def test_precomputed_phonemes_used_verbatim_never_renormalized(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = self._config(tmp, skip_audio=True, text_casing="upper")
            utt = Utterance(text="ignored", audio_path=tmp / "a.wav",
                            phonemes=["h", "i"], phonemes_precomputed=True)
            results = self._run_worker(config, [utt])
        processed, phonemes = results[0]
        self.assertEqual(processed.phonemes, ["h", "i"])  # casing never touched precomputed
        self.assertEqual(phonemes, {"h", "i"})

    def test_phonemization_yields_nothing_is_dropped(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = self._config(tmp, skip_audio=True)
            # whitespace-only normalizes to empty text -> empty phoneme list
            utt = Utterance(text="   ", audio_path=tmp / "a.wav")
            results = self._run_worker(config, [utt])
        processed, phonemes = results[0]
        self.assertIsNone(processed)
        self.assertEqual(phonemes, set())

    def test_missing_audio_file_dropped_with_audio_enabled(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = self._config(tmp, skip_audio=False)
            utt = Utterance(text="hello", audio_path=tmp / "does_not_exist.wav")
            results = self._run_worker(config, [utt])
        processed, phonemes = results[0]
        self.assertIsNone(processed)
        self.assertEqual(phonemes, set())

    def test_real_short_wav_is_normalized_and_cached(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            wav = tmp / "clip.wav"
            wav.write_bytes(_wav_bytes())
            config = self._config(tmp, skip_audio=False)
            config.cache_dir.mkdir(parents=True, exist_ok=True)
            utt = Utterance(text="hello", audio_path=wav)
            results = self._run_worker(config, [utt])
            processed, phonemes = results[0]
            self.assertIsNotNone(processed)
            self.assertTrue(Path(processed.audio_norm_path).is_file())
            self.assertTrue(Path(processed.audio_spec_path).is_file())

    def test_worker_process_level_failure_logged_not_raised(self):
        # a failure setting up the worker (e.g. the VAD model) must not
        # propagate out of phonemize_worker and kill the whole pool silently
        # -- it is caught by the outer try/except and merely logged.
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = self._config(tmp, skip_audio=True)
            task_q = queue.Queue()
            result_q = queue.Queue()
            task_q.put([Utterance(text="hi", audio_path=tmp / "a.wav")])
            task_q.put(None)

            with patch.object(preprocess, "make_silence_detector",
                               side_effect=RuntimeError("boom")):
                with self.assertLogs("preprocess", level="ERROR") as logs:
                    preprocess.phonemize_worker(config, task_q, result_q, object())
            self.assertTrue(any("Worker process failed" in m for m in logs.output))
            self.assertTrue(result_q.empty())  # nothing was ever processed


if __name__ == "__main__":
    unittest.main()
