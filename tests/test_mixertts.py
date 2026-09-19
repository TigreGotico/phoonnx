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


# --- phoneme alignment (durations) ---------------------------------------

def test_parse_outputs_no_durations_by_default():
    """Standard Mixer-TTS exports emit only mel_spec — no alignment extras."""
    adapter = MixerTTSAdapter(vocoder=FakeVocoder())
    mel = np.zeros((1, 80, 9), np.float32)
    res = adapter.parse_outputs([mel], _req(), output_names=["mel_spec"])
    assert "phoneme_id_samples" not in res.extras


def test_parse_outputs_picks_up_named_durations():
    """A future re-export exposing a 'durations' output lights up alignment
    automatically via DURATION_OUTPUT_NAMES, without any adapter changes."""
    adapter = MixerTTSAdapter(vocoder=FakeVocoder())
    mel = np.zeros((1, 80, 9), np.float32)
    durs = np.array([[2, 3, 4, 5]], dtype=np.float32)
    res = adapter.parse_outputs(
        [mel, durs], _req(n=4), output_names=["mel_spec", "durations"]
    )
    np.testing.assert_array_equal(res.extras["phoneme_id_samples"], [2, 3, 4, 5])


def test_parse_outputs_ignores_unnamed_duration_lookalike():
    """Without output_names, an extra tensor is NOT guessed to be durations —
    detection is strictly name-driven for two-stage engines (unlike VITS's
    positional fallback, since a second output here is far more likely to be
    an intermediate mel-model tensor than a duration vector)."""
    adapter = MixerTTSAdapter(vocoder=FakeVocoder())
    mel = np.zeros((1, 80, 9), np.float32)
    extra = np.array([[2, 3, 4, 5]], dtype=np.float32)
    res = adapter.parse_outputs([mel, extra], _req(n=4), output_names=None)
    assert "phoneme_id_samples" not in res.extras


def test_default_params():
    assert MixerTTSAdapter().default_params() == {"pace": 1.0, "pitch_mul": 1.0, "pitch_add": 0.0, "emotion": 0}


def test_length_scale_falls_back_to_pace():
    # SynthesisConfig.length_scale is the canonical speed knob; honour it when
    # the adapter's native "pace" key is not explicitly set.
    feed = MixerTTSAdapter().build_feed_dict(_req(length_scale=1.3), MIXER_SESSION)
    assert feed["pace"][0] == pytest.approx(1.3)


def test_pace_takes_precedence_over_length_scale():
    feed = MixerTTSAdapter().build_feed_dict(_req(pace=0.7, length_scale=1.3), MIXER_SESSION)
    assert feed["pace"][0] == pytest.approx(0.7)


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


def test_tokenization_arabic_mantoq_pinned_buckwalter_ids():
    # tts_arabic is not published on PyPI (git-only, no installable wheel),
    # so this pins the vendored mantoq g2p's own buckwalter token sequence
    # as a regression golden instead of cross-validating against it live.
    # Any change to scriptconv's vendored mantoq's output for this sentence
    # must be a deliberate, reviewed change to this fixture.
    from scriptconv.phonemizers.ar import MantoqPhonemizer
    text = "السَّلامُ عَلَيكُم"
    mantoq = MantoqPhonemizer()
    phonemes = [p for chunk in [mantoq.phonemize(text, "ar")] for w in chunk for p in w]
    sid = {s: i for i, s in enumerate(sorted(set(phonemes)))}
    ours = [sid[p] for p in phonemes]
    expected_symbols = ['a', 'a', 's', '_', 'd', 'b', 'l', '_', 'a', 'l', 'a',
                         'a', 'm', 'u', ' ', 'E', 'a', 'l', 'a', 'y', 'k', 'u', 'm']
    expected = [sid[p] for p in expected_symbols]
    assert ours == expected


class _ShortInputSession(DummySession):
    """Mimics SpeedySpeech: dilated convs reject sequences below a minimum."""

    def __init__(self, min_len):
        super().__init__(["token_ids"])
        self.min_len = min_len
        self.seen_lengths = []

    def run(self, output_names, feed):
        from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
        n = np.asarray(feed["token_ids"]).shape[-1]
        self.seen_lengths.append(n)
        if n < self.min_len:
            raise InvalidArgument(
                "[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code "
                f"returned while running Conv node. Invalid input shape: {{{n}}}"
            )
        return [np.zeros((1, 80, n * 2), dtype=np.float32)]


def test_short_input_is_padded_until_the_graph_accepts_it():
    session = _ShortInputSession(min_len=13)
    adapter = MixerTTSAdapter(vocoder=FakeVocoder())
    result = adapter.synthesize(_req(4), session)
    assert result.audio.size > 0
    assert session.seen_lengths[0] == 4
    assert session.seen_lengths[-1] == 13


def test_long_enough_input_is_not_padded():
    session = _ShortInputSession(min_len=13)
    adapter = MixerTTSAdapter(vocoder=FakeVocoder())
    adapter.synthesize(_req(20), session)
    assert session.seen_lengths == [20]


def test_unrelated_invalid_argument_is_not_swallowed():
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

    class _Boom(DummySession):
        def run(self, output_names, feed):
            raise InvalidArgument("INVALID_ARGUMENT : unexpected data type")

    adapter = MixerTTSAdapter(vocoder=FakeVocoder())
    with pytest.raises(InvalidArgument):
        adapter.synthesize(_req(4), _Boom(["token_ids"]))
