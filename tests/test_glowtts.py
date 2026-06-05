"""Tests for the GlowTTS (Larynx) inference adapter + config bridge."""
import numpy as np
import pytest

from phoonnx.engines import get_adapter, detect_engine
from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.glowtts import GlowTTSAdapter
from phoonnx.engines.glowtts_config import voice_config_from_larynx
from phoonnx.engines.vocoders.base import BaseVocoder
from phoonnx.config import Engine, VoiceConfig, PhonemeType


class _Named:
    def __init__(self, name): self.name = name
    @property
    def shape(self): return self._shape
    @shape.setter
    def shape(self, v): self._shape = v


class DummySession:
    def __init__(self, input_names, out_shapes):
        self._inputs = [_Named(n) for n in input_names]
        self._outputs = []
        for n, sh in out_shapes:
            o = _Named(n); o.shape = sh; self._outputs.append(o)
    def get_inputs(self): return self._inputs
    def get_outputs(self): return self._outputs


class FakeVocoder(BaseVocoder):
    def __init__(self): super().__init__(); self.calls = []
    def mel_to_audio(self, mel, denoise=False):
        self.calls.append(mel.shape); return np.ones(mel.shape[-1] * 256, dtype=np.float32)


GLOW_SESSION = DummySession(["input", "input_lengths", "scales"],
                            [("output", ["b", "t", 322]), ("3453", [1, 80, 322])])


def _req(n=5, spk=None, **params):
    return AdapterSynthesisRequest(
        phoneme_ids=np.arange(1, n + 1, dtype=np.int64)[None, :],
        phoneme_lengths=np.array([n], dtype=np.int64),
        speaker_id=spk, language_id=None, params=params)


def test_registered():
    assert isinstance(get_adapter("glowtts"), GlowTTSAdapter)


def test_detect_by_session_mel_output():
    assert GlowTTSAdapter.detect(session=GLOW_SESSION) is True
    assert isinstance(detect_engine(session=GLOW_SESSION), GlowTTSAdapter)


def test_detect_not_plain_vits():
    s = DummySession(["input", "input_lengths", "scales"], [("output", ["b", 1, "t"])])
    assert GlowTTSAdapter.detect(session=s) is False  # waveform output, not mel


def test_build_feed_dict_two_scales():
    feed = GlowTTSAdapter().build_feed_dict(_req(noise_scale=0.6, length_scale=1.1), GLOW_SESSION)
    assert set(feed) == {"input", "input_lengths", "scales"}
    assert feed["scales"] == pytest.approx([0.6, 1.1])  # noise, length (no noise_w)


def test_parse_outputs_picks_mel_by_shape_and_vocodes():
    adapter = GlowTTSAdapter(vocoder=FakeVocoder())
    # Larynx emits an extra non-mel output; the [1,80,T] one must be chosen
    mel = np.zeros((1, 80, 7), np.float32)
    other = np.zeros((1, 7, 322), np.float32)
    adapter.parse_outputs([other, mel], _req())
    assert adapter.vocoder.calls[0] == (1, 80, 7)


def test_without_vocoder_raises():
    with pytest.raises(RuntimeError, match="vocoder"):
        GlowTTSAdapter().parse_outputs([np.zeros((1, 80, 4), np.float32)], _req())


def test_default_params():
    assert GlowTTSAdapter().default_params() == {"noise_scale": 0.667, "length_scale": 1.0}


def test_config_bridge():
    phonemes = "0 _\n1 a\n2 b\n3 c\n4 d\n5 ˈ\n"
    cfg = {"audio": {"sample_rate": 22050}, "model": {"num_symbols": 6, "n_speakers": 1}}
    vc = voice_config_from_larynx(cfg, phonemes, lang_code="en-us")
    assert vc.engine == Engine.GLOWTTS
    assert vc.phoneme_type == PhonemeType.GRUUT
    assert vc.sample_rate == 22050
    assert vc.tokenizer.add_blank_char is True and vc.tokenizer.use_eos_bos is False
    assert len(vc.tokenizer.vocabulary.char2idx) == 6


def test_config_bridge_native_roundtrip():
    phonemes = "0 _\n1 a\n2 b\n"
    vc = voice_config_from_larynx({"audio": {"sample_rate": 22050}, "model": {"num_symbols": 3}},
                                  phonemes, lang_code="en-us")
    native = vc.to_native_dict()
    assert native["engine"] == "glowtts"
    assert VoiceConfig.from_dict(dict(native)).engine == Engine.GLOWTTS
