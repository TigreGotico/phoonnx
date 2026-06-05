"""Tests for the FastPitch adapter (shares the Mixer-TTS FastSpeech2 contract)."""
import numpy as np
import pytest

from phoonnx.engines import get_adapter, detect_engine
from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.fastpitch import FastPitchAdapter
from phoonnx.engines.mixertts import MixerTTSAdapter
from phoonnx.engines.mixertts_config import voice_config_from_mixer
from phoonnx.config import Engine, PhonemeType, Alphabet, VoiceConfig


class _Named:
    def __init__(self, name): self.name = name
    @property
    def shape(self): return getattr(self, "_shape", None)
    @shape.setter
    def shape(self, v): self._shape = v


class _Sess:
    def __init__(self, names): self._i = [_Named(n) for n in names]
    def get_inputs(self): return self._i
    def get_outputs(self):
        o = _Named("mel_spec"); o.shape = ["b", 80, "t"]; return [o]


def _req(n=4, spk=None, **p):
    return AdapterSynthesisRequest(phoneme_ids=np.arange(1, n + 1, dtype=np.int64)[None, :],
                                   phoneme_lengths=np.array([n], np.int64),
                                   speaker_id=spk, language_id=None, params=p)


def test_registered_and_is_mixer_subclass():
    a = get_adapter("fastpitch")
    assert isinstance(a, FastPitchAdapter) and isinstance(a, MixerTTSAdapter)


def test_detect_by_config_only():
    assert FastPitchAdapter.detect(config={"engine": "fastpitch"}) is True
    assert isinstance(detect_engine(config={"engine": "fastpitch"}), FastPitchAdapter)
    # FastPitch/Mixer share I/O, so FastPitch never claims a bare session
    assert FastPitchAdapter.detect(session=_Sess(["token_ids", "pace", "pitch_mul"])) is False


def test_build_feed_dict_inherited():
    sess = _Sess(["token_ids", "pace", "speaker", "pitch_mul", "pitch_add"])  # arabic = no emotion
    feed = FastPitchAdapter().build_feed_dict(_req(pace=0.9), sess)
    assert set(feed) == {"token_ids", "pace", "speaker", "pitch_mul", "pitch_add"}
    assert feed["pace"][0] == pytest.approx(0.9)


def test_config_bridge_fastpitch_engine():
    vc = voice_config_from_mixer(["_pad_", "_+_", "<", "b"], lang_code="ar",
                                 phoneme_type=PhonemeType.MANTOQ, alphabet=Alphabet.BUCKWALTER,
                                 num_speakers=4, word_sep_token="_+_", engine=Engine.FASTPITCH)
    assert vc.engine == Engine.FASTPITCH
    native = vc.to_native_dict()
    assert native["engine"] == "fastpitch"
    assert VoiceConfig.from_dict(dict(native)).engine == Engine.FASTPITCH
