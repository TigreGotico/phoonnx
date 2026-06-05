"""Tests for the Mixer-TTS inference adapter + config bridge."""
import numpy as np
import pytest

from phoonnx.engines import get_adapter, detect_engine
from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.mixertts import MixerTTSAdapter
from phoonnx.engines.mixertts_config import voice_config_from_mixer
from phoonnx.engines.vocoders.base import BaseVocoder
from phoonnx.config import Engine, VoiceConfig, PhonemeType, Alphabet


class _Named:
    def __init__(self, name): self.name = name
    @property
    def shape(self): return getattr(self, "_shape", None)
    @shape.setter
    def shape(self, v): self._shape = v


class DummySession:
    def __init__(self, input_names, out_specs=(("mel_spec", ["b", 80, "t"]),)):
        self._inputs = [_Named(n) for n in input_names]
        self._outputs = []
        for n, sh in out_specs:
            o = _Named(n); o.shape = sh; self._outputs.append(o)
    def get_inputs(self): return self._inputs
    def get_outputs(self): return self._outputs


class FakeVocoder(BaseVocoder):
    def __init__(self): super().__init__(); self.calls = []
    def mel_to_audio(self, mel, denoise=False):
        self.calls.append(mel.shape); return np.ones(mel.shape[-1] * 256, dtype=np.float32)


MIXER_SESSION = DummySession(["token_ids", "pace", "speaker", "emotion", "pitch_mul", "pitch_add"])


def _req(n=4, spk=None, **params):
    return AdapterSynthesisRequest(
        phoneme_ids=np.arange(1, n + 1, dtype=np.int64)[None, :],
        phoneme_lengths=np.array([n], dtype=np.int64),
        speaker_id=spk, language_id=None, params=params)


def test_registered():
    assert isinstance(get_adapter("mixertts"), MixerTTSAdapter)


def test_detect_by_control_inputs():
    assert MixerTTSAdapter.detect(session=MIXER_SESSION) is True
    assert isinstance(detect_engine(session=MIXER_SESSION), MixerTTSAdapter)


def test_detect_by_config():
    assert MixerTTSAdapter.detect(config={"engine": "mixertts"}) is True


def test_build_feed_dict_controls():
    feed = MixerTTSAdapter().build_feed_dict(_req(pace=0.9, pitch_mul=1.2, pitch_add=2.0), MIXER_SESSION)
    assert set(feed) == {"token_ids", "pace", "speaker", "emotion", "pitch_mul", "pitch_add"}
    assert feed["pace"][0] == pytest.approx(0.9)
    assert feed["pitch_mul"][0] == pytest.approx(1.2)
    assert feed["pitch_add"][0] == pytest.approx(2.0)
    assert feed["token_ids"].dtype == np.int64 and feed["speaker"].dtype == np.int32


def test_parse_outputs_mel_to_vocoder():
    adapter = MixerTTSAdapter(vocoder=FakeVocoder())
    mel = np.zeros((1, 80, 9), np.float32)
    adapter.parse_outputs([mel], _req())
    assert adapter.vocoder.calls[0] == (1, 80, 9)


def test_without_vocoder_raises():
    with pytest.raises(RuntimeError, match="vocoder"):
        MixerTTSAdapter().parse_outputs([np.zeros((1, 80, 4), np.float32)], _req())


def test_default_params():
    assert MixerTTSAdapter().default_params() == {"pace": 1.0, "pitch_mul": 1.0, "pitch_add": 0.0, "emotion": 0}


def test_config_bridge():
    # mirror of models/symbols.py order: [pad] + punctuation + letters + ipa
    symbols = ["$", ";", ":", "a", "b", "ɑ", "ɛ", "ˈ"]
    vc = voice_config_from_mixer(symbols, sample_rate=22050)
    assert vc.engine == Engine.MIXERTTS
    assert vc.phoneme_type == PhonemeType.ESPEAK and vc.alphabet == Alphabet.IPA
    assert list(vc.tokenizer.vocabulary.char2idx) == symbols
    assert vc.tokenizer.add_blank_char is False


def test_config_bridge_native_roundtrip():
    vc = voice_config_from_mixer(["$", "a", "ɑ", "ˈ"])
    native = vc.to_native_dict()
    assert native["engine"] == "mixertts"
    assert VoiceConfig.from_dict(dict(native)).engine == Engine.MIXERTTS


# --- Arabic (tts_arabic) variant: mantoq / buckwalter, multi-speaker ---

def test_config_bridge_arabic():
    from phoonnx.config import PhonemeType, Alphabet
    syms = ["_pad_", "_eos_", "_sil_", "_dbl_", "_+_", ".", "<", "b", "t"]
    vc = voice_config_from_mixer(syms, lang_code="ar", phoneme_type=PhonemeType.MANTOQ,
                                 alphabet=Alphabet.BUCKWALTER, num_speakers=4, word_sep_token="_+_")
    assert vc.engine == Engine.MIXERTTS
    assert vc.phoneme_type == PhonemeType.MANTOQ and vc.alphabet == Alphabet.BUCKWALTER
    assert vc.num_speakers == 4 and vc.word_sep_token == "_+_"
    assert list(vc.tokenizer.vocabulary.char2idx) == syms  # 44-symbol buckwalter table order


def test_raw_vocoder_feeds_extra_scalar_inputs():
    # tts_arabic's baked Vocos takes an extra 'denoise' input alongside the mel
    from phoonnx.engines.vocoders.raw import RawWaveformVocoder

    class _In:
        def __init__(self, name): self.name = name

    class _Sess:
        def __init__(self): self.feeds = []
        def get_inputs(self): return [_In("mel_spec"), _In("denoise")]
        def run(self, _, feed):
            self.feeds.append(feed); return [np.zeros((1, 64), np.float32)]

    voc = RawWaveformVocoder(session=_Sess())
    voc.mel_to_audio(np.zeros((1, 80, 4), np.float32), denoise=False)
    feed = voc.session.feeds[0]
    assert "mel_spec" in feed and "denoise" in feed and float(feed["denoise"][0]) == 0.0


def test_tokenization_arabic_mantoq_matches_tts_arabic():
    # golden: phoonnx mantoq + the 44-symbol buckwalter vocab must equal
    # tts_arabic's own phonetise_buckwalter ids. Guarded on the lib.
    tts_arabic = pytest.importorskip("tts_arabic")
    from tts_arabic.text import symbols as ar_symbols, phonetise_buckwalter
    from phoonnx.phonemizers.ar import MantoqPhonemizer
    sid = {s: i for i, s in enumerate(ar_symbols.symbols)}
    text = "السَّلامُ عَلَيكُم"
    orig = [sid[p] for p in phonetise_buckwalter.process_utterance(text) if p in sid]
    mantoq = MantoqPhonemizer()
    ours = [sid[p] for chunk in [mantoq.phonemize(text, "ar")] for w in chunk for p in w if p in sid]
    assert ours == orig
