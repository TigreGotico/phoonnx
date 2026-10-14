"""Clause punctuation reaches a model only when it was trained with it.

Piper voices were trained with ``, . ? ! : ;`` after each clause, so a Piper
config synthesizes with them. A phoonnx config records the choice in
``keep_punctuation``; a config without the key was trained without punctuation
and synthesizes exactly as before. Preprocessing keeps punctuation for new
datasets and writes the key.

These tests run the real espeak-ng phonemizer (the binary in CI, the espyak
port where the binary is missing).
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import numpy as np
import pytest

from phoonnx.config import Alphabet, SynthesisConfig, VoiceConfig
from phoonnx.tokenizer import DEFAULT_IPA_PHONEME_ID_MAP
from phoonnx.voice import TTSVoice

SONGS = "Songs near this quiet river every morning, number 3."
QUESTION = "Is it ready? Yes! We leave at 16:30; bring the bags: all of them."

PIPER_IDS = {k: v[0] for k, v in DEFAULT_IPA_PHONEME_ID_MAP.items()}
COMMA, FULL_STOP, QUESTION_MARK, EXCLAMATION = (
    PIPER_IDS[","], PIPER_IDS["."], PIPER_IDS["?"], PIPER_IDS["!"])
PUNCTUATION_IDS = {PIPER_IDS[p] for p in ",.?!:;"}


def phoonnx_config(**extra):
    return {"phoonnx_version": "1.0.0", "phoneme_type": "espeak", "alphabet": "ipa",
            "lang_code": "en-us", "audio": {"sample_rate": 22050},
            "phoneme_id_map": dict(PIPER_IDS), "num_symbols": 256, **extra}


def piper_config(**extra):
    return {"phoneme_type": "espeak", "espeak": {"voice": "en-us"},
            "audio": {"sample_rate": 22050}, "num_symbols": 256,
            "phoneme_id_map": {k: list(v) for k, v in DEFAULT_IPA_PHONEME_ID_MAP.items()},
            **extra}


def synthesis_ids(config: dict, text: str):
    voice_config = VoiceConfig.from_dict(json.loads(json.dumps(config)))
    voice = TTSVoice(session=MagicMock(), config=voice_config, adapter=MagicMock())
    return voice_config, [ids for _, ids in voice._iter_synthesis_ids(
        text, SynthesisConfig(), Alphabet.GRAPHEMES, voice_config.alphabet)]


def flat(sentences):
    return [i for ids in sentences for i in ids]


class TestSynthesis:
    def test_phoonnx_config_without_the_key_has_no_punctuation(self):
        voice_config, ids = synthesis_ids(phoonnx_config(), SONGS)
        assert voice_config.keep_punctuation is False
        assert not PUNCTUATION_IDS & set(flat(ids))

    def test_phoonnx_config_without_the_key_is_unchanged(self):
        # the ids a published voice receives are those of scriptconv's default
        # sentence phonemization, tokenized; nothing is added
        from scriptconv.phonemizers import get_phonemizer
        from phoonnx.util import normalize
        voice_config, ids = synthesis_ids(phoonnx_config(), QUESTION)
        phonemizer = get_phonemizer("espeak", normalizer=normalize)
        expected = [voice_config.tokenizer.tokenize(p)
                    for p in phonemizer.phonemize(QUESTION, "en-us") if p]
        assert ids == expected

    def test_piper_config_keeps_punctuation(self):
        voice_config, ids = synthesis_ids(piper_config(), SONGS)
        assert voice_config.keep_punctuation is True
        assert flat(ids).count(COMMA) == 1 and flat(ids).count(FULL_STOP) == 1
        _, ids = synthesis_ids(piper_config(), QUESTION)
        assert ids[0].count(QUESTION_MARK) == 1 and ids[1].count(EXCLAMATION) == 1

    def test_phoonnx_config_with_the_key_keeps_punctuation(self):
        voice_config, ids = synthesis_ids(phoonnx_config(keep_punctuation=True), SONGS)
        assert voice_config.keep_punctuation is True
        assert {COMMA, FULL_STOP} <= set(flat(ids))

    def test_explicit_false_wins_over_the_piper_default(self):
        # a phoonnx voice exported with --piper says it was trained without
        voice_config, ids = synthesis_ids(piper_config(keep_punctuation=False), SONGS)
        assert voice_config.keep_punctuation is False
        assert not PUNCTUATION_IDS & set(flat(ids))

    def test_native_dict_round_trips_the_flag(self):
        for config in (piper_config(), phoonnx_config()):
            voice_config = VoiceConfig.from_dict(config)
            native = voice_config.to_native_dict()
            assert native["keep_punctuation"] is voice_config.keep_punctuation
            assert VoiceConfig.from_dict(native).keep_punctuation is voice_config.keep_punctuation


class TestExport:
    def test_piper_sidecar_records_the_flag(self):
        from phoonnx_train.engines.vits import _write_piper_json
        with TemporaryDirectory() as tmp:
            for model_config, expected in ((phoonnx_config(), False),
                                           (phoonnx_config(keep_punctuation=True), True)):
                path = Path(tmp) / "voice.piper.json"
                _write_piper_json(model_config, path)
                assert json.loads(path.read_text())["keep_punctuation"] is expected


class TestScorer:
    def test_scorer_phonemizes_like_preprocess(self):
        from phoonnx_train.evaluation.scorer import build_encoder, text_to_ids
        for config, has_punctuation in ((phoonnx_config(), False),
                                        (phoonnx_config(keep_punctuation=True), True)):
            ph, tokenizer, lang, _, _ = build_encoder(config)
            phonemes, _ = text_to_ids(SONGS, ph, tokenizer, lang)
            assert "\n" not in phonemes
            assert ("," in phonemes and "." in phonemes) is has_punctuation


def _ljspeech(root: Path, rows):
    import soundfile as sf
    (root / "wavs").mkdir(parents=True)
    sr = 22050
    # long enough that no row is dropped for having more phonemes than frames
    tone = (0.2 * np.sin(2 * np.pi * 220 * np.arange(5 * sr) / sr)).astype(np.float32)
    lines = []
    for name, text in rows:
        sf.write(root / "wavs" / f"{name}.wav", tone, sr)
        lines.append(f"{name}|{text}")
    (root / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _preprocess(src: Path, out: Path, *extra, rows: int = 1):
    import click.testing
    from phoonnx_train import preprocess
    result = click.testing.CliRunner().invoke(
        preprocess.cli,
        ["-i", str(src), "-o", str(out), "-l", "en-us", "--single-speaker",
         "--max-workers", "1", *extra],
        catch_exceptions=False)
    assert result.exit_code == 0, result.output
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    written = [json.loads(line) for line in
               (out / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(written) == rows
    return config, {row["text"]: row for row in written}


class TestPreprocess:
    def test_new_dataset_keeps_punctuation(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _ljspeech(tmp / "src", [("a", SONGS), ("b", QUESTION)])
            config, rows = _preprocess(tmp / "src", tmp / "out", rows=2)
            assert config["keep_punctuation"] is True
            songs = rows[SONGS]
            assert songs["phonemes"].count(",") == 1 and songs["phonemes"][-1] == "."
            assert "\n" not in songs["phonemes"]
            assert songs["phoneme_ids"].count(config["phoneme_id_map"][","]) == 1
            assert songs["phoneme_ids"].count(config["phoneme_id_map"]["."]) == 1
            assert {"?", "!", ";", ":"} <= set(rows[QUESTION]["phonemes"])

    def test_training_matches_synthesis(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _ljspeech(tmp / "src", [("a", SONGS), ("b", QUESTION)])
            config, rows = _preprocess(tmp / "src", tmp / "out", rows=2)
            for text, row in rows.items():
                voice_config = VoiceConfig.from_dict(json.loads(json.dumps(config)))
                voice = TTSVoice(session=MagicMock(), config=voice_config, adapter=MagicMock())
                synth = [p for s in voice._iter_phonemes(
                    text, Alphabet.GRAPHEMES, voice_config.alphabet,
                    *_route(voice, voice_config)) for p in s]
                assert row["phonemes"] == synth, text

    def test_opt_out_writes_no_punctuation(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _ljspeech(tmp / "src", [("a", SONGS)])
            config, rows = _preprocess(tmp / "src", tmp / "out", "--no-keep-punctuation")
            assert config["keep_punctuation"] is False
            ids = set(rows[SONGS]["phoneme_ids"])
            assert not {config["phoneme_id_map"][p] for p in ",.?!:;"} & ids
            assert "\n" not in rows[SONGS]["phonemes"]

    def test_resume_keeps_the_existing_choice(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _ljspeech(tmp / "src", [("a", SONGS)])
            _preprocess(tmp / "src", tmp / "out", "--no-keep-punctuation")
            config_path = tmp / "out" / "config.json"
            old = json.loads(config_path.read_text())
            del old["keep_punctuation"]  # a config written before the key existed
            config_path.write_text(json.dumps(old))
            with open(tmp / "src" / "metadata.csv", "a", encoding="utf-8") as f:
                f.write(f"b|{QUESTION}\n")
            import soundfile as sf
            sf.write(tmp / "src" / "wavs" / "b.wav", sf.read(tmp / "src" / "wavs" / "a.wav")[0], 22050)
            config, rows = _preprocess(tmp / "src", tmp / "out", "--resume", rows=2)
            assert config["keep_punctuation"] is False
            assert not set(",.?!:;") & set(rows[QUESTION]["phonemes"])


def _route(voice, voice_config):
    from phoonnx.config import get_conversion
    return get_conversion(voice.phonemizer, voice_config, SynthesisConfig(), voice_config.alphabet)


SIDE_CAR = Path(__file__).parent / "fixtures" / "miro_sv-SE.piper.json"


class TestPhoonnxPiperSideCars:
    """phoonnx's ``--piper`` export side-cars are piper-shaped but were trained
    by phoonnx, without punctuation."""

    def test_published_side_car_has_no_punctuation(self):
        # the published OpenVoiceOS phoonnx sv-SE voice, loaded through its side-car
        config = json.loads(SIDE_CAR.read_text(encoding="utf-8"))
        assert VoiceConfig.is_piper(config)
        voice_config, ids = synthesis_ids(config, "Hej, hur mår du? Bra! Tack.")
        assert voice_config.keep_punctuation is False
        punctuation = {config["phoneme_id_map"][p][0] for p in ",.?!"}
        assert ids and not punctuation & set(flat(ids))

    @pytest.mark.parametrize("marker, expected", [
        ({"piper_version": "phoonnx-0.2.6"}, False),
        ({"alphabet": "ipa"}, False),
        ({"piper_version": "1.0.0"}, True),
        ({}, True),
    ])
    def test_markers(self, marker, expected):
        assert VoiceConfig.from_dict(piper_config(**marker)).keep_punctuation is expected


class TestModelManagerKeepsTheFlag:
    """The voice manager rebuilds a voice's phonemizer when the index overrides
    the phoneme type or names a phonemizer model; the rebuilt one must keep the
    voice's punctuation setting."""

    def _load(self, override_phoneme_type: bool, **info_kwargs):
        from types import SimpleNamespace
        from unittest.mock import PropertyMock, patch
        from phoonnx import model_manager
        from phoonnx.config import PhonemeType

        voice_config = VoiceConfig.from_dict(piper_config())
        assert voice_config.keep_punctuation is True
        if override_phoneme_type:
            voice_config.phoneme_type = PhonemeType.GRAPHEMES  # the index says espeak
        voice = SimpleNamespace(config=voice_config, phonemizer=None)
        info = model_manager.TTSModelInfo(voice_id="test/voice", lang="en-us",
                                          model_url="model.onnx", config_url="config.json",
                                          **info_kwargs)
        index_config = SimpleNamespace(lang_code="en-us", phoneme_type=PhonemeType.ESPEAK,
                                       alphabet=Alphabet.IPA)
        with patch.object(model_manager.TTSModelInfo, "config", new_callable=PropertyMock,
                          return_value=index_config), \
                patch.object(model_manager, "check_lang_supported"), \
                patch.object(model_manager.TTSModelInfo, "download_model", return_value="model.onnx"), \
                patch.object(model_manager.TTSModelInfo, "hub_path", return_value=None), \
                patch.object(model_manager.TTSModelInfo, "engine_params", return_value={}), \
                patch.object(model_manager.TTSVoice, "load", return_value=voice):
            return info.load()

    def test_phonemizer_model_from_the_index(self):
        from phoonnx.config import PhonemeType
        voice = self._load(False, phonemizer_model="variant",
                           phoneme_type=PhonemeType.ESPEAK, alphabet=Alphabet.IPA)
        assert voice.config.phonemizer_model == "variant"
        assert voice.phonemizer.keep_punctuation is True

    def test_phoneme_type_override_from_the_index(self):
        from phoonnx.config import PhonemeType
        voice = self._load(True, phoneme_type=PhonemeType.ESPEAK, alphabet=Alphabet.IPA)
        assert voice.config.phoneme_type == PhonemeType.ESPEAK
        assert voice.phonemizer.keep_punctuation is True
