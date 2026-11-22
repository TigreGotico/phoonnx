"""Inline ``[[...]]`` blocks reach a voice in the voice's own alphabet, or not at all.

A block holds IPA. A voice that reads another alphabet gets the block converted. A voice with
no symbol input, tiny-voca or a raw-text engine, gets one warning and no block: the model
input never holds the brackets.
"""
import json
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock

from scriptconv.phonemizers.base import BasePhonemizer

from phoonnx.config import Alphabet, PhonemeType, SynthesisConfig, VoiceConfig
from phoonnx.engines.chatterbox import ChatterboxAdapter
from phoonnx.engines.tinyvoca import TinyVocaAdapter
from phoonnx.voice import TTSVoice

TINYVOCA = json.loads((Path(__file__).parent / "fixtures" / "tinyvoca_config.json").read_text())


class _Echo(BasePhonemizer):
    """Reads each letter of a text as one symbol."""

    def phonemize_string(self, text, lang):
        return text


def _voice(alphabet, adapter):
    voice = TTSVoice.__new__(TTSVoice)
    voice.phonetic_spellings = None
    voice.phonemizer = _Echo()
    voice.model_path = "/voices/test-voice.onnx"
    voice.adapter = adapter
    voice.config = types.SimpleNamespace(
        text_casing="ignore",
        phoneme_type=PhonemeType.UNICODE, alphabet=alphabet, lang_code="en", phonemizer_lang=None, lang_tokens={},
        add_diacritics=False, diacritizer_model=None, sample_rate=22050,
        tokenizer=types.SimpleNamespace(tokenize=lambda symbols: [1] * len(symbols)))
    return voice


def _warnings(monkeypatch):
    """The warnings that ``phoonnx.voice`` logs on this thread, as one list of formatted messages.

    A voice built by an earlier test can still run its provenance check in a daemon
    thread and log through the same ``LOG``, so warnings from other threads are left out.
    """
    logged = []
    test_thread = threading.current_thread()

    def warning(message, *args):
        if threading.current_thread() is test_thread:
            logged.append(message % args)

    log = MagicMock()
    log.warning.side_effect = warning
    monkeypatch.setattr("phoonnx.voice.LOG", log)
    return logged


def test_an_ipa_block_on_an_xsampa_voice_gives_xsampa_symbols():
    voice = _voice(Alphabet.XSAMPA, MagicMock())
    (utterance,) = voice.prepare("[[ʃə]]", SynthesisConfig())
    # X-SAMPA chart: ʃ is S, ə is @
    assert utterance.symbols == ("S", "@")


def test_a_block_on_tinyvoca_warns_and_sends_no_brackets(monkeypatch):
    warnings = _warnings(monkeypatch)
    adapter = TinyVocaAdapter()
    voice = TTSVoice(session=MagicMock(), config=VoiceConfig.from_dict(json.loads(json.dumps(TINYVOCA))),
                     adapter=adapter, model_path="/voices/tiny-voca.onnx")
    plain = [u.ids for u in voice.prepare("hello world", SynthesisConfig())]
    blocked = [u.ids for u in voice.prepare("hello [[wɜːld]] world", SynthesisConfig())]
    assert blocked == plain
    assert len(warnings) == 1, warnings
    (warning,) = warnings
    assert "tiny-voca" in warning and "[[wɜːld]]" in warning


def test_a_block_on_a_raw_text_engine_warns_and_sends_no_brackets(monkeypatch):
    warnings = _warnings(monkeypatch)
    sent = []
    voice = _voice(Alphabet.GRAPHEMES, ChatterboxAdapter())
    voice.config.tokenizer = types.SimpleNamespace(
        tokenize=lambda text, **_: sent.append(text) or [1])
    list(voice.prepare("hello [[wɜːld]] world.", SynthesisConfig()))
    assert sent == ["hello world."]
    assert len(warnings) == 1, warnings
    (warning,) = warnings
    assert "test-voice" in warning and "[[wɜːld]]" in warning
