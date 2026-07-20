"""Tests for multi-format dataset loading and its preprocessing integration."""
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import soundfile as sf

from phoonnx_train.dataset_loaders import (PreprocessorConfig, Utterance,
                                           _audio_from_value, _jsonable,
                                           detect_format, ensure_audio_path,
                                           jsonl_loader, known_loaders,
                                           load_source, parquet_loader)
from phoonnx.config import Alphabet, PhonemeType


def _wav_bytes(seconds: float = 0.2, sr: int = 16000) -> bytes:
    tone = 0.1 * np.sin(2 * np.pi * 220 * np.linspace(0, seconds, int(sr * seconds), endpoint=False))
    buf = io.BytesIO()
    sf.write(buf, tone.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def _config(tmp: str, **overrides) -> PreprocessorConfig:
    base = dict(
        input_dir=Path(tmp), output_dir=Path(tmp), language="en", sample_rate=16000,
        cache_dir=Path(tmp) / "cache", max_workers=1, single_speaker=False,
        speaker_id=None, phoneme_type=PhonemeType.ESPEAK, alphabet=Alphabet.IPA,
        phonemizer_model="", text_casing="ignore", dataset_name=None,
        audio_quality=None, skip_audio=False, debug=False, add_diacritics=False,
    )
    base.update(overrides)
    return PreprocessorConfig(**base)


class TestDetectFormat(unittest.TestCase):
    def test_ljspeech_dir(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "metadata.csv").write_text("a|hi\n")
            self.assertEqual(detect_format(tmp), "ljspeech")

    def test_jsonl_file(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.jsonl"
            p.write_text("{}\n")
            self.assertEqual(detect_format(str(p)), "jsonl")

    def test_parquet_file_and_dir_and_glob(self):
        import pandas as pd
        with TemporaryDirectory() as tmp:
            shard = Path(tmp) / "part-0.parquet"
            pd.DataFrame({"text": ["hi"]}).to_parquet(shard)
            self.assertEqual(detect_format(str(shard)), "parquet")
            self.assertEqual(detect_format(tmp), "parquet")
            self.assertEqual(detect_format(str(Path(tmp) / "*.parquet")), "parquet")

    def test_hf_repo_id(self):
        self.assertEqual(detect_format("org/some-dataset"), "hf")

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            detect_format("not a path or repo")
        with TemporaryDirectory() as tmp:
            # existing dir with neither metadata.csv nor parquet shards
            with self.assertRaises(ValueError):
                detect_format(tmp)

    def test_all_formats_registered(self):
        self.assertEqual(set(known_loaders()), {"ljspeech", "jsonl", "parquet", "hf"})


class TestAudioFromValue(unittest.TestCase):
    def test_path_string(self):
        self.assertEqual(_audio_from_value("a.wav"), ("a.wav", None))

    def test_raw_bytes(self):
        self.assertEqual(_audio_from_value(b"xyz"), (None, b"xyz"))

    def test_hf_mapping_with_null_path(self):
        # the embedded-bytes case: path is None, bytes carry the audio
        self.assertEqual(_audio_from_value({"bytes": b"xyz", "path": None}), (None, b"xyz"))

    def test_hf_mapping_with_path_only(self):
        self.assertEqual(_audio_from_value({"bytes": None, "path": "a.wav"}), ("a.wav", None))

    def test_none(self):
        self.assertEqual(_audio_from_value(None), (None, None))


class TestJsonable(unittest.TestCase):
    def test_numpy_scalar_unwrapped(self):
        self.assertEqual(_jsonable(np.int64(5)), 5)

    def test_bytes_dropped(self):
        self.assertIsNone(_jsonable(b"raw"))

    def test_primitives_passthrough(self):
        self.assertEqual(_jsonable("x"), "x")
        self.assertEqual(_jsonable([1, np.float32(2.0)]), [1, 2.0])


class TestUtteranceAsdict(unittest.TestCase):
    def test_drops_bytes_and_flag_keeps_extras(self):
        utt = Utterance(text="hi", audio_path=Path("a.wav"), audio_bytes=b"raw",
                        phonemes_precomputed=True, row_id="r1", extras={"lang": "en"})
        data = utt.asdict()
        self.assertNotIn("audio_bytes", data)
        self.assertNotIn("phonemes_precomputed", data)
        self.assertEqual(data["row_id"], "r1")
        self.assertEqual(data["extras"], {"lang": "en"})
        self.assertEqual(data["audio_path"], "a.wav")
        # the whole thing must be JSON-serializable
        json.dumps(data)


class TestJsonlLoader(unittest.TestCase):
    def _write(self, tmp, rows):
        p = Path(tmp) / "data.jsonl"
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return str(p)

    def test_reads_rows_with_default_columns(self):
        with TemporaryDirectory() as tmp:
            src = self._write(tmp, [{"text": "hello", "audio": "a.wav", "speaker": "s1"}])
            utts = list(jsonl_loader(src, _config(tmp)))
            self.assertEqual(len(utts), 1)
            self.assertEqual(utts[0].text, "hello")
            self.assertEqual(utts[0].speaker, "s1")
            self.assertEqual(str(utts[0].audio_path), "a.wav")

    def test_alternate_text_column_fallback(self):
        with TemporaryDirectory() as tmp:
            src = self._write(tmp, [{"sentence": "hi there", "audio": "a.wav"}])
            utts = list(jsonl_loader(src, _config(tmp)))
            self.assertEqual(utts[0].text, "hi there")

    def test_explicit_missing_column_raises(self):
        with TemporaryDirectory() as tmp:
            src = self._write(tmp, [{"text": "hi", "audio": "a.wav"}])
            with self.assertRaises(ValueError):
                list(jsonl_loader(src, _config(tmp, text_column="nope")))

    def test_missing_text_skips_row(self):
        with TemporaryDirectory() as tmp:
            src = self._write(tmp, [{"text": "", "audio": "a.wav"},
                                    {"text": "ok", "audio": "b.wav"}])
            utts = list(jsonl_loader(src, _config(tmp)))
            self.assertEqual([u.text for u in utts], ["ok"])

    def test_missing_audio_skips_unless_skip_audio(self):
        with TemporaryDirectory() as tmp:
            src = self._write(tmp, [{"text": "hi"}])
            self.assertEqual(list(jsonl_loader(src, _config(tmp))), [])
            utts = list(jsonl_loader(src, _config(tmp, skip_audio=True)))
            self.assertEqual(len(utts), 1)

    def test_malformed_json_line_skipped(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.jsonl"
            p.write_text('{"text": "ok", "audio": "a.wav"}\nnot json\n{bad}\n')
            utts = list(jsonl_loader(str(p), _config(tmp)))
            self.assertEqual([u.text for u in utts], ["ok"])

    def test_precomputed_phonemes_split_on_whitespace(self):
        with TemporaryDirectory() as tmp:
            src = self._write(tmp, [{"text": "hi", "audio": "a.wav", "phon": "h a i"},
                                    {"text": "bye", "audio": "b.wav", "phon": ""}])
            utts = list(jsonl_loader(src, _config(tmp, phonemes_column="phon")))
            self.assertEqual(utts[0].phonemes, ["h", "a", "i"])
            self.assertTrue(utts[0].phonemes_precomputed)
            # empty phonemes cell falls back to the phonemizer
            self.assertIsNone(utts[1].phonemes)
            self.assertFalse(utts[1].phonemes_precomputed)

    def test_unmapped_and_lang_columns_go_to_extras(self):
        with TemporaryDirectory() as tmp:
            src = self._write(tmp, [{"text": "hi", "audio": "a.wav",
                                     "lang": "en", "dataset": "x"}])
            utts = list(jsonl_loader(src, _config(tmp, lang_column="lang")))
            self.assertEqual(utts[0].extras.get("lang"), "en")
            self.assertEqual(utts[0].extras.get("dataset"), "x")

    def test_single_speaker_ignores_speaker_column(self):
        with TemporaryDirectory() as tmp:
            src = self._write(tmp, [{"text": "hi", "audio": "a.wav", "speaker": "s1"}])
            utts = list(jsonl_loader(src, _config(tmp, single_speaker=True)))
            self.assertIsNone(utts[0].speaker)


class TestParquetLoader(unittest.TestCase):
    def test_reads_shards_in_directory(self):
        import pandas as pd
        with TemporaryDirectory() as tmp:
            pd.DataFrame({"text": ["a", "b"], "audio": ["a.wav", "b.wav"]}).to_parquet(
                Path(tmp) / "part-0.parquet")
            pd.DataFrame({"text": ["c"], "audio": ["c.wav"]}).to_parquet(
                Path(tmp) / "part-1.parquet")
            utts = list(parquet_loader(tmp, _config(tmp)))
            self.assertEqual(sorted(u.text for u in utts), ["a", "b", "c"])

    def test_embedded_bytes_row(self):
        import pandas as pd
        with TemporaryDirectory() as tmp:
            wav = _wav_bytes()
            df = pd.DataFrame({"text": ["hi"], "audio": [{"bytes": wav, "path": None}]})
            df.to_parquet(Path(tmp) / "d.parquet")
            utts = list(parquet_loader(str(Path(tmp) / "d.parquet"), _config(tmp)))
            self.assertEqual(len(utts), 1)
            self.assertEqual(utts[0].audio_bytes, wav)
            self.assertIn(str(utts[0].audio_path), ("", "."))
            self.assertTrue(utts[0].row_id)

    def test_missing_audio_column_in_shard(self):
        import pandas as pd
        with TemporaryDirectory() as tmp:
            # no audio column at all; without skip_audio every row drops
            pd.DataFrame({"text": ["a", "b"]}).to_parquet(Path(tmp) / "d.parquet")
            self.assertEqual(list(parquet_loader(str(Path(tmp) / "d.parquet"), _config(tmp))), [])
            kept = list(parquet_loader(str(Path(tmp) / "d.parquet"), _config(tmp, skip_audio=True)))
            self.assertEqual(len(kept), 2)


class TestEnsureAudioPath(unittest.TestCase):
    def test_path_backed_returns_path_unchanged(self):
        utt = Utterance(text="x", audio_path=Path("/some/a.wav"))
        with TemporaryDirectory() as tmp:
            self.assertEqual(ensure_audio_path(utt, tmp), Path("/some/a.wav"))

    def test_bytes_materialized_to_wav(self):
        wav = _wav_bytes()
        utt = Utterance(text="x", audio_path=Path(""), audio_bytes=wav, row_id="r1")
        with TemporaryDirectory() as tmp:
            out = ensure_audio_path(utt, tmp)
            self.assertTrue(out.exists())
            self.assertEqual(out.suffix, ".wav")
            data, sr = sf.read(str(out))
            self.assertGreater(len(data), 0)
            # second call is cached on the utterance
            self.assertEqual(ensure_audio_path(utt, tmp), out)

    def test_corrupt_bytes_raise(self):
        utt = Utterance(text="x", audio_path=Path(""), audio_bytes=b"not audio at all", row_id="r2")
        with TemporaryDirectory() as tmp:
            with self.assertRaises(Exception):
                ensure_audio_path(utt, tmp)

    def test_no_path_and_no_bytes_raises(self):
        utt = Utterance(text="x", audio_path=Path(""), row_id="r3")
        with TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                ensure_audio_path(utt, tmp)


class TestHfLoader(unittest.TestCase):
    def test_reads_embedded_audio_bytes(self):
        import datasets

        wav = _wav_bytes()
        # A plain struct column carrying embedded bytes (our parquet shards look
        # like this); no Audio feature so datasets never tries to encode/decode.
        ds = datasets.Dataset.from_dict(
            {"audio": [{"bytes": wav, "path": None}], "text": ["hello"]},
        )
        with TemporaryDirectory() as tmp:
            with patch("datasets.load_dataset", return_value=ds):
                from phoonnx_train.dataset_loaders import hf_loader
                utts = list(hf_loader("org/name", _config(tmp)))
        self.assertEqual(len(utts), 1)
        self.assertEqual(utts[0].text, "hello")
        self.assertIsNotNone(utts[0].audio_bytes)


class TestQualityFilterExtensions(unittest.TestCase):
    def test_metrics_sink_records_and_value_source_avoids_compute(self):
        from phoonnx_train.quality_filter import (apply_quality_filters,
                                                   parse_filter_spec, register_scorer)

        computed = []

        def scorer(audio, sr, text, duration):
            computed.append(text)
            return 9.9

        register_scorer("metric_x", scorer)
        s1 = Utterance(text="a", audio_path=Path("a.wav"), row_id="a", extras={"metric_x": 5.0})
        s2 = Utterance(text="b", audio_path=Path("b.wav"), row_id="b", extras={})

        recorded = {}

        def sink(row_id, column, value):
            recorded.setdefault(row_id, {})[column] = value

        def value_source(sample, column):
            v = sample.extras.get(column)
            return float(v) if v is not None else None

        kept, dropped = apply_quality_filters(
            [s1, s2], [parse_filter_spec("metric_x:3.0:")],
            audio_path_fn=lambda u: u.audio_path,
            text_fn=lambda u: u.text,
            audio_loader=lambda p: (None, 16000, 1.0),
            id_fn=lambda u: u.row_id,
            value_source=value_source,
            metrics_sink=sink,
        )
        # s1 sourced from column (never computed); s2 computed via scorer
        self.assertEqual(computed, ["b"])
        self.assertEqual([u.row_id for u in kept], ["a", "b"])
        self.assertEqual(recorded["a"]["metric_x"], 5.0)
        self.assertEqual(recorded["b"]["metric_x"], 9.9)


class TestMetricsSidecarRoundtrip(unittest.TestCase):
    def test_write_then_read(self):
        from phoonnx_train.preprocess import (_read_metrics_sidecar,
                                              _write_metrics_sidecar)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.parquet"
            _write_metrics_sidecar(path, {"r1": {"wpm": 120.0}, "r2": {"wpm": 80.0}})
            back = _read_metrics_sidecar(path)
            self.assertEqual(back["r1"]["wpm"], 120.0)
            self.assertEqual(back["r2"]["wpm"], 80.0)


class TestResumeHelpers(unittest.TestCase):
    def test_read_jsonl_tolerates_truncated_final_line(self):
        from phoonnx_train.preprocess import _read_jsonl, _row_key, _utt_key
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "dataset.jsonl"
            # a valid row, then a half-written (truncated) final line
            p.write_text('{"row_id": "r1", "audio_path": "a.wav"}\n{"row_id": "r2", "aud')
            rows = _read_jsonl(p)
            self.assertEqual(len(rows), 1)
            self.assertEqual(_row_key(rows[0]), "r1")
            self.assertEqual(_utt_key(Utterance(text="x", audio_path=Path("a.wav"), row_id="r2")), "r2")


class TestPreprocessIntegration(unittest.TestCase):
    """End-to-end cli over jsonl sources with precomputed phonemes + skip_audio,
    covering multi-source speaker namespacing and --resume."""

    class _DummyPhonemizer:
        alphabet = Alphabet.IPA

        def phonemize_to_list(self, text, lang):
            return list(text.replace(" ", ""))

        def add_diacritics(self, text, lang):
            return text

    def _jsonl(self, path, rows):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def _invoke(self, args):
        import click.testing
        from phoonnx_train import preprocess
        with patch.object(preprocess, "get_phonemizer", return_value=self._DummyPhonemizer()):
            return click.testing.CliRunner().invoke(preprocess.cli, args, catch_exceptions=False)

    def test_multi_source_speaker_namespacing_and_resume(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            a = tmp / "a.jsonl"
            b = tmp / "b.jsonl"
            # both sources reuse speaker id "0" -> must be namespaced apart
            self._jsonl(a, [{"text": "hi", "audio": "a.wav", "speaker": "0", "phon": "h i"}])
            self._jsonl(b, [{"text": "no", "audio": "b.wav", "speaker": "0", "phon": "n o"}])
            out = tmp / "out"
            args = ["-i", str(a), "-i", str(b), "-o", str(out), "-l", "en",
                    "--skip-audio", "--phonemes-column", "phon"]
            result = self._invoke(args)
            self.assertEqual(result.exit_code, 0, result.output)

            config = json.loads((out / "config.json").read_text())
            self.assertEqual(config["num_speakers"], 2)
            self.assertEqual(set(config["speaker_id_map"]), {"a:0", "b:0"})

            lines = [json.loads(x) for x in (out / "dataset.jsonl").read_text().splitlines() if x]
            self.assertEqual(len(lines), 2)

            # resume: add a third row in a new source; only it is processed/appended
            c = tmp / "c.jsonl"
            self._jsonl(c, [{"text": "yo", "audio": "c.wav", "speaker": "0", "phon": "i o"}])
            result2 = self._invoke(args + ["-i", str(c), "--resume"])
            self.assertEqual(result2.exit_code, 0, result2.output)
            lines2 = [json.loads(x) for x in (out / "dataset.jsonl").read_text().splitlines() if x]
            self.assertEqual(len(lines2), 3)

    def test_mismatched_precomputed_phonemes_fail_loudly(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            a = tmp / "a.jsonl"
            # 'ZZZ' is not a valid IPA symbol in the default map
            self._jsonl(a, [{"text": "hi", "audio": "a.wav", "phon": "h ZZZ"}])
            out = tmp / "out"
            args = ["-i", str(a), "-o", str(out), "-l", "en", "--corpus-only-map",
                    "--skip-audio", "--phonemes-column", "phon"]
            with self.assertRaises(ValueError) as ctx:
                self._invoke(args)
            self.assertIn("ZZZ", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
