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


def test_length_scale_falls_back_to_pace():
    # FastPitch inherits build_feed_dict from Mixer-TTS: length_scale must be
    # honoured here too.
    sess = _Sess(["token_ids", "pace", "speaker", "pitch_mul", "pitch_add"])
    feed = FastPitchAdapter().build_feed_dict(_req(length_scale=1.3), sess)
    assert feed["pace"][0] == pytest.approx(1.3)


def test_inherits_duration_output_names():
    """FastPitch reuses Mixer-TTS's parsing, including alignment detection."""
    assert FastPitchAdapter.DURATION_OUTPUT_NAMES == MixerTTSAdapter.DURATION_OUTPUT_NAMES


def test_config_bridge_fastpitch_engine():
    vc = voice_config_from_mixer(["_pad_", "_+_", "<", "b"], lang_code="ar",
                                 phoneme_type=PhonemeType.MANTOQ, alphabet=Alphabet.BUCKWALTER,
                                 num_speakers=4, word_sep_token="_+_", engine=Engine.FASTPITCH)
    assert vc.engine == Engine.FASTPITCH
    native = vc.to_native_dict()
    assert native["engine"] == "fastpitch"
    assert VoiceConfig.from_dict(dict(native)).engine == Engine.FASTPITCH


def test_coqui_bridge_targets_fastpitch_engine():
    # the shared coqui bridge (also used by GlowTTS) can build a FastPitch config
    from phoonnx.engines.coqui_config import voice_config_from_coqui
    cfg = {"use_phonemes": True, "phoneme_language": "en-us", "add_blank": False,
           "characters": {"characters": "abc", "punctuations": "!?.", "pad": "_",
                          "eos": "&", "bos": "*", "blank": "<BLNK>"}}
    vc = voice_config_from_coqui(cfg, lang_code="en-us", engine=Engine.FASTPITCH)
    assert vc.engine == Engine.FASTPITCH
    assert VoiceConfig.from_dict(dict(vc.to_native_dict())).engine == Engine.FASTPITCH


def test_coqui_vocab_is_sorted_like_coqui():
    # coqui's Graphemes/IPAPhonemes default is_sorted=True: the symbol set is
    # sorted alphabetically before ids are assigned. Regression guard for the
    # bug where the bridge kept config order -> right voice, wrong words.
    from phoonnx.engines.coqui_config import voice_config_from_coqui
    cfg = {"use_phonemes": True, "phoneme_language": "en-us", "phonemizer": "gruut",
           "add_blank": False, "characters": {"phonemes": "zyxbac", "punctuations": "!?",
                                               "pad": "_", "eos": "~", "bos": "^"}}
    c2i = voice_config_from_coqui(cfg, lang_code="en-us").tokenizer.vocabulary.char2idx
    syms = list(c2i)
    # specials first, then SORTED phonemes (a,b,c,x,y,z), then punctuation (config order)
    assert syms[:3] == ["_", "~", "^"]
    assert syms[3:9] == ["a", "b", "c", "x", "y", "z"]  # sorted, not config order zyxbac
    assert syms[9:] == ["!", "?"]


def test_coqui_phonemizer_honors_config_field():
    from phoonnx.engines.coqui_config import voice_config_from_coqui
    from phoonnx.config import PhonemeType
    base = {"use_phonemes": True, "characters": {"phonemes": "abc", "pad": "_", "eos": "~", "bos": "^"}}
    assert voice_config_from_coqui({**base, "phonemizer": "gruut"}, lang_code="en").phoneme_type == PhonemeType.GRUUT
    assert voice_config_from_coqui({**base, "phonemizer": "espeak"}, lang_code="en").phoneme_type == PhonemeType.ESPEAK


def test_coqui_vits_characters_vocab():
    # VITS uses VitsCharacters: [pad] + punct + graphemes + ipa + [blank], NOT
    # sorted, is_unique=False (blank id = full-list length, not deduped length).
    from phoonnx.engines.coqui_config import voice_config_from_coqui
    from phoonnx.config import Engine
    cfg = {"use_phonemes": True, "phonemizer": "espeak",
           "characters": {"characters_class": "TTS.tts.models.vits.VitsCharacters",
                          "characters": "ab", "phonemes": "ɑɐ", "punctuations": "!", "pad": "_"}}
    c2i = voice_config_from_coqui(cfg, lang_code="en", engine=Engine.COQUI).tokenizer.vocabulary.char2idx
    # _ ! a b ɑ ɐ <BLNK>  -> blank at 6 (full length 7), unsorted
    assert list(c2i) == ["_", "!", "a", "b", "ɑ", "ɐ", "<BLNK>"]
    assert c2i["<BLNK>"] == 6
