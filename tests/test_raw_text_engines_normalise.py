"""The engines that read raw text get the text2spoken reading of a sentence with digits.

``TTSVoice.prepare`` reads the text through the frontend, which normalises it, and then
calls the adapter's ``encode_text``. Each adapter here gets a tokenizer that writes one id
per character, so the ids of the model input decode back to the text the model reads.
The expected reading comes from ``text2spoken.normalize`` called alone. The comparison is
on lower-cased words, because each adapter keeps its own model-side rules (supertonic's
``<lang>`` wrap, pockettts capitalisation, outetts quote folding and lower-casing). A
prompt template may hold digits of its own, so the digits of the model input must equal
those of the input for a sentence with none.
"""
import re
import types
import unicodedata
from unittest.mock import MagicMock

import pytest
from text2spoken import normalize

from phoonnx.config import SynthesisConfig, VoiceConfig
from phoonnx.engines.arktts import ArkTTSAdapter
from phoonnx.engines.indic_parler import IndicParlerAdapter
from phoonnx.engines.llasa import LlasaAdapter
from phoonnx.engines.magpie import MagpieAdapter
from phoonnx.engines.mosstts import MossTTSNanoAdapter
from phoonnx.engines.neutts import NeuTTSAdapter
from phoonnx.engines.omnivoice import OmniVoiceAdapter
from phoonnx.engines.orpheus import OrpheusAdapter
from phoonnx.engines.outetts import OuteTTSAdapter
from phoonnx.engines.pockettts import PocketTTSAdapter
from phoonnx.engines.qwen3tts import Qwen3TTSAdapter
from phoonnx.engines.sparktts import SparkTTSAdapter
from phoonnx.engines.supertonic import SuperTonicAdapter
from phoonnx.voice import TTSVoice

SENTENCES = {
    "en": "I have 23 apples.",
    "es": "Tengo 23 manzanas.",
    "de": "Ich habe 23 Äpfel.",
    "hi": "मेरे पास 23 सेब हैं।",
    "eu": "Kaixo, 23 lagun daude.",
}


class CharTokenizer:
    """One id per character, under every method name the adapters call."""

    kinds = {"char": {}}

    def tokenize(self, text):
        return [ord(c) for c in text]

    def encode(self, text, *args, **kwargs):
        ids = self.tokenize(text)
        return types.SimpleNamespace(ids=ids) if not args else ids

    def Encode(self, text):
        return self.tokenize(text)

    def Decode(self, ids):
        return "".join(chr(i) for i in ids)

    def encodable_kind(self, name):
        return self.kinds[name]


class CharSentencePiece:
    def encode(self, text, out_type=int):
        return [ord(c) for c in text]


def _arktts(a):
    a.tokenizer = CharTokenizer()


def _qwen3tts(a):
    a.tokenizer = CharTokenizer()


def _sparktts(a):
    a.tokenizer = CharTokenizer()


def _supertonic(a):
    a.tokenizer = CharTokenizer()


def _llasa(a):
    a.tokenizer = CharTokenizer()


def _orpheus(a):
    a.tokenizer = CharTokenizer()


def _neutts(a):
    a.tokenizer = CharTokenizer()
    a._espeak = lambda text: text


def _magpie(a):
    a.tokenizer = CharTokenizer()
    a.config = {"language_to_tokenizer": {"en": ["char"], "es": ["char"]}}


def _mosstts(a):
    a.sp = CharSentencePiece()


def _outetts(a):
    a.tokenizer = CharTokenizer()


def _pockettts(a):
    a.tokenizer = CharTokenizer()


def _indic_parler(a):
    a.prompt_tokenizer = CharTokenizer()


def _omnivoice(a):
    a.tokenizer = CharTokenizer()


ENGINES = {
    "arktts": (ArkTTSAdapter, _arktts, ("en", "eu")),
    "qwen3tts": (Qwen3TTSAdapter, _qwen3tts, ("en", "es")),
    "sparktts": (SparkTTSAdapter, _sparktts, ("en",)),
    "supertonic": (SuperTonicAdapter, _supertonic, ("en", "es")),
    "llasa": (LlasaAdapter, _llasa, ("en",)),
    "orpheus": (OrpheusAdapter, _orpheus, ("en", "de")),
    "neutts": (NeuTTSAdapter, _neutts, ("en", "de")),
    "magpie": (MagpieAdapter, _magpie, ("en", "es")),
    "mosstts": (MossTTSNanoAdapter, _mosstts, ("en", "es")),
    "outetts": (OuteTTSAdapter, _outetts, ("en",)),
    "pockettts": (PocketTTSAdapter, _pockettts, ("en", "es")),
    "indic_parler": (IndicParlerAdapter, _indic_parler, ("hi",)),
    "omnivoice": (OmniVoiceAdapter, _omnivoice, ("en", "hi")),
}
CASES = [(engine, lang) for engine, (_, _, langs) in ENGINES.items() for lang in langs]


def _voice(engine, lang):
    adapter_class, wire, _ = ENGINES[engine]
    adapter = adapter_class()
    wire(adapter)
    config = VoiceConfig(num_symbols=1, num_speakers=1, num_langs=1, sample_rate=24000,
                         lang_code=lang, phoneme_type="graphemes", alphabet="graphemes",
                         phonemizer_model=None)
    return TTSVoice(config=config, session=MagicMock(), adapter=adapter)


def _model_input(voice, text):
    """What the model reads, one string per engine call, as the characters of its ids."""
    syn_config = SynthesisConfig(enable_phonetic_spellings=False)
    return " ".join("".join(chr(i) for i in utterance.ids) for utterance in voice.prepare(text, syn_config))


def _words(text):
    """Lower-cased words without accents, because supertonic decomposes them (NFKD)."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return re.findall(r"\w+", "".join(c for c in decomposed if not unicodedata.combining(c)))


def _digits(text):
    return [c for c in text if c.isdigit()]


def _contains(haystack, needle):
    return any(haystack[i:i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1))


@pytest.mark.parametrize("engine,lang", CASES)
def test_model_input_holds_the_text2spoken_reading(engine, lang):
    sentence = SENTENCES[lang]
    reading = normalize(sentence, lang)
    assert not any(c.isdigit() for c in reading)

    voice = _voice(engine, lang)
    model_input = _model_input(voice, sentence)

    assert _contains(_words(model_input), _words(reading)), model_input
    assert _digits(model_input) == _digits(_model_input(voice, "Hello."))


@pytest.mark.parametrize("engine,lang", CASES)
def test_the_check_sees_digits_when_the_frontend_does_not_normalise(engine, lang):
    voice = _voice(engine, lang)
    voice.phonemizer.normalizer = lambda text, lang: text

    model_input = _model_input(voice, SENTENCES[lang])

    assert "23" in model_input
