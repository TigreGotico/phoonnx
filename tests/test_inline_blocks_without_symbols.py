"""Inline blocks on voices with no symbol input: spacing, SSML, NeuTTS and phonetic input."""
import types
from unittest.mock import MagicMock

from scriptconv.phonemizers.base import BasePhonemizer

from phoonnx.config import Alphabet, PhonemeType, SynthesisConfig
from phoonnx.engines.chatterbox import ChatterboxAdapter
from phoonnx.engines.neutts import NeuTTSAdapter
from phoonnx.ssml import parse_ssml
from phoonnx.voice import TTSVoice
from scriptconv.frontend import Stages


class _Echo(BasePhonemizer):
    def phonemize_string(self, text, lang):
        return text


def _voice(adapter, sent):
    voice = TTSVoice.__new__(TTSVoice)
    voice.phonetic_spellings = None
    voice.phonemizer = _Echo()
    voice.model_path = "/voices/test-voice.onnx"
    voice.adapter = adapter
    voice.config = types.SimpleNamespace(
        text_casing="ignore",
        phoneme_type=PhonemeType.UNICODE, alphabet=Alphabet.GRAPHEMES, lang_code="en", phonemizer_lang=None, lang_tokens={},
        add_diacritics=False, diacritizer_model=None, sample_rate=22050,
        tokenizer=types.SimpleNamespace(tokenize=lambda text, **_: sent.append(text) or [1]))
    return voice


def _warnings(monkeypatch, module="phoonnx.voice"):
    logged = []
    log = MagicMock()
    log.warning.side_effect = lambda message, *args: logged.append(message % args)
    monkeypatch.setattr(f"{module}.LOG", log)
    return logged


def _read(text, config=None):
    sent = []
    list(_voice(ChatterboxAdapter(), sent).prepare(text, config or SynthesisConfig()))
    return sent


def test_a_skipped_block_leaves_one_space():
    assert _read("foo [[x]]bar") == ["foo bar"]
    assert _read("foo[[x]]bar") == ["foo bar"]
    assert _read("a [[x]] [[y]] b") == ["a b"]


def test_a_skipped_block_before_a_mark_leaves_no_space():
    assert _read("hello [[x]].") == ["hello."]
    assert _read("hello [[x]], world") == ["hello, world"]
    assert _read("hello [[x]] world") == ["hello world"]


def test_a_skipped_block_keeps_a_paragraph_break():
    # a paragraph chunk ends only at a blank line, so a space in its place gives one chunk
    by_paragraph = SynthesisConfig(stages=Stages(chunk="paragraph"))
    assert _read("para one.\n\n[[x]] para two.", by_paragraph) == ["para one.", "para two."]
    assert _read("para one. [[x]] para two.", by_paragraph) == ["para one. para two."]


def test_a_line_break_between_skipped_blocks_stays():
    assert _read("a [[x]]\n[[y]] b") == ["a\nb"]
    assert _read("a\n[[x]]\n\n[[y]] b", SynthesisConfig(stages=Stages(chunk="paragraph"))) == ["a", "b"]


def test_a_mark_after_the_last_skipped_block_keeps_its_word():
    assert _read("one [[x]]\n[[y]], two") == ["one, two"]
    assert _read("hello [[x]]\n\n[[y]].") == ["hello."]


def test_ssml_phoneme_on_a_voice_with_no_symbols_reads_its_word(monkeypatch):
    warnings = _warnings(monkeypatch, "phoonnx.ssml")
    ssml = 'I like <phoneme alphabet="ipa" ph="təˈmeɪtoʊ">tomato</phoneme> soup.'
    (piece,) = parse_ssml(ssml, "en", symbols=False)
    assert piece.text == "I like tomato soup."
    assert len(warnings) == 1 and "təˈmeɪtoʊ" in warnings[0]
    (piece,) = parse_ssml(ssml, "en")
    assert piece.text == "I like [[ təˈmeɪtoʊ ]] soup."


def test_neutts_takes_symbols_and_a_block_reaches_the_phone_string():
    adapter = NeuTTSAdapter()
    adapter._phonemizer = types.SimpleNamespace(
        phonemize_string=lambda text, lang: f"<{text.strip()}>")
    assert adapter.takes_symbols
    assert adapter._espeak("say [[ ʃə ]] now") == "<say> ʃə <now>"
    assert adapter._espeak("say now") == "<say now>"


def test_phonetic_input_on_a_voice_with_no_symbols_warns_once(monkeypatch):
    warnings = _warnings(monkeypatch)
    sent = []
    voice = _voice(ChatterboxAdapter(), sent)
    voice.config.tokenizer = types.SimpleNamespace(tokenize=lambda symbols, **_: [1] * len(symbols))
    list(voice.prepare("ʃə", SynthesisConfig(input_alphabet=Alphabet.IPA)))
    assert len(warnings) == 1
    assert "test-voice" in warnings[0] and "phonetic input" in warnings[0]
