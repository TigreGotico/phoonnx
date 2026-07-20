"""Tests for the OptiSpeech inference adapter + config bridge."""
import json
import numpy as np
import pytest

from phoonnx.engines import get_adapter, detect_engine
from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.optispeech import OptiSpeechAdapter
from phoonnx.engines.optispeech_config import voice_config_from_optispeech_meta
from phoonnx.config import Engine, VoiceConfig


class _Named:
    def __init__(self, name): self.name = name


class _Meta:
    def __init__(self, m): self.custom_metadata_map = m


class DummySession:
    def __init__(self, input_names, output_names, meta=None):
        self._inputs = [_Named(n) for n in input_names]
        self._outputs = [_Named(n) for n in output_names]
        self._meta = meta or {}
    def get_inputs(self): return self._inputs
    def get_outputs(self): return self._outputs
    def get_modelmeta(self): return _Meta(self._meta)


META = {
    "name": "test", "sample_rate": 24000, "languages": ["en-us"], "speakers": [],
    "inference_args": {"d_factor": 1.0, "p_factor": 1.0, "e_factor": 1.0},
    "input_symbols": {"_": 0, "^": 1, "$": 2, "a": 3, "b": 4, "c": 5, " ": 6},
    "special_symbols": {"pad": "_", "bos": "^", "eos": "$"},
    "text_processor": {"tokenizer_name": "ipa", "add_blank": False, "add_bos_eos": False},
}
OPTISPEECH_SESSION = DummySession(
    ["x", "x_lengths", "scales"], ["wav", "wav_lengths", "durations"],
    meta={"inference": json.dumps(META)})


def _req(n=4, spk=None, lang=None, **params):
    return AdapterSynthesisRequest(
        phoneme_ids=np.arange(1, n + 1, dtype=np.int64)[None, :],
        phoneme_lengths=np.array([n], dtype=np.int64),
        speaker_id=spk, language_id=lang, params=params)


def test_registered():
    assert isinstance(get_adapter("optispeech"), OptiSpeechAdapter)


def test_detect_by_metadata():
    assert OptiSpeechAdapter.detect(session=OPTISPEECH_SESSION) is True


def test_detect_priority_beats_matcha():
    # OptiSpeech shares x/x_lengths/scales with Matcha but must win via metadata
    assert isinstance(detect_engine(session=OPTISPEECH_SESSION), OptiSpeechAdapter)


def test_detect_does_not_match_plain_vits():
    s = DummySession(["input", "input_lengths", "scales"], ["output"])
    assert OptiSpeechAdapter.detect(session=s) is False


def test_build_feed_dict_dpe_scales():
    feed = OptiSpeechAdapter().build_feed_dict(_req(spk=2), OPTISPEECH_SESSION)
    assert set(feed) == {"x", "x_lengths", "scales"}  # no sids input on this session
    assert feed["scales"] == pytest.approx([1.0, 1.0, 1.0])


def test_build_feed_dict_factors_and_speaker():
    s = DummySession(["x", "x_lengths", "scales", "sids", "lids"], ["wav"])
    feed = OptiSpeechAdapter().build_feed_dict(
        _req(spk=3, lang=1, d_factor=0.8, p_factor=1.2, e_factor=0.9), s)
    assert feed["scales"] == pytest.approx([0.8, 1.2, 0.9])
    assert feed["sids"].tolist() == [3]
    assert feed["lids"].tolist() == [1]


def test_parse_outputs_wav_and_extras():
    res = OptiSpeechAdapter().parse_outputs(
        [np.zeros((1, 1, 512), np.float32), np.array([512]), np.array([[3, 4]])], _req())
    assert res.audio.shape == (512,)
    assert "durations" in res.extras and "wav_lengths" in res.extras
    # also exposed under the uniform key TTSVoice.phoneme_ids_to_audio reads
    assert "phoneme_id_samples" in res.extras
    np.testing.assert_array_equal(res.extras["phoneme_id_samples"], [3, 4])


def test_parse_outputs_durations_by_name():
    res = OptiSpeechAdapter().parse_outputs(
        [np.zeros((1, 1, 512), np.float32), np.array([512]), np.array([[3, 4]])],
        _req(), output_names=["wav", "wav_lengths", "durations"],
    )
    np.testing.assert_array_equal(res.extras["phoneme_id_samples"], [3, 4])


def test_default_params():
    assert OptiSpeechAdapter().default_params() == {"d_factor": 1.0, "p_factor": 1.0, "e_factor": 1.0}


def test_config_bridge():
    vc = voice_config_from_optispeech_meta(META)
    assert vc.engine == Engine.OPTISPEECH
    assert vc.sample_rate == 24000
    assert vc.lang_code == "en-US"
    assert vc.tokenizer.add_blank_char is False  # metadata add_blank=False
    assert vc.tokenizer.use_eos_bos is False
    assert len(vc.tokenizer.vocabulary.char2idx) == len(META["input_symbols"])


def test_config_bridge_native_roundtrip():
    vc = voice_config_from_optispeech_meta(META)
    native = vc.to_native_dict()
    assert native["engine"] == "optispeech"
    vc2 = VoiceConfig.from_dict(dict(native))
    assert vc2.engine == Engine.OPTISPEECH


# --- IPATokenizer phonemization routes through phoonnx's espeak layer (the
#     espeak-ng subprocess wrapper with an espyak fallback), never the
#     GPL-linked piper_phonemize. Verifies the reroute and that the tokenizer's
#     id-assembly contract (symbol->id, blank interspersing, bos/eos) is
#     unchanged. ---

def test_ipa_tokenizer_routes_through_phoonnx_espeak():
    from phoonnx_train.optispeech.text.tokenizers import IPATokenizer
    from phoonnx_train.optispeech.text import symbols
    from phoonnx_train.optispeech.text.normalization import (
        collapse_whitespace, intersperse)
    from phoonnx.phonemizers.mul import EspeakPhonemizer

    text = "hello world"
    tok = IPATokenizer(add_blank=False, add_bos_eos=False, normalize_text=True)

    # (1) reroute: phonemize_text yields exactly phoonnx's EspeakPhonemizer output
    phonemes, norm = tok.phonemize_text(text, "en-us")
    assert phonemes == EspeakPhonemizer().phonemize(norm, "en-us")
    assert phonemes and isinstance(phonemes[0], list)  # nested list[list[str]]

    # (2) plain id contract: flat symbol->id mapping of those phonemes
    flat = list(collapse_whitespace("".join(p for sent in phonemes for p in sent)))
    expected = symbols.phonemes_to_ids(flat)
    ids, _ = tok(text, "en-us", split_sentences=False)
    assert ids == expected and ids

    # (3) blank + bos/eos contract preserved
    tok2 = IPATokenizer(add_blank=True, add_bos_eos=True, normalize_text=True)
    ids2, _ = tok2(text, "en-us", split_sentences=False)
    assert ids2 == [symbols.BOS_ID, *intersperse(expected, 0), symbols.EOS_ID]
