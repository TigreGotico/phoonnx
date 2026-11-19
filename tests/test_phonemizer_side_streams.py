"""A phonemizer's per-symbol side stream reaches the engine request.

The Shami phonemizer tags every symbol with its language: 1 for Arabic, 2 for
English, 3 for neutral symbols such as the word gap, 0 for the <bos> and <eos>
pads. The ShamiVITS graph reads those tags as its required ``language_ids``
input, so a request without them speaks code-switched text with the pad tag on
every symbol.
"""
import json
from unittest.mock import MagicMock

import numpy as np
from scriptconv.phonemizers.shami import ShamiPhonemizer

from phoonnx.config import SynthesisConfig, VoiceConfig
from phoonnx.engines.base import AdapterSynthesisResult
from phoonnx.voice import TTSVoice

TEXT = "مرحبا كيفك؟ I am fine, شكرا."


def _shami_voice():
    sentences = list(ShamiPhonemizer().phonemize_with_language_ids_lazy(TEXT, "ar"))
    symbols = sorted({s for phonemes, _ in sentences for s in phonemes} - {"<bos>", "<eos>"})
    id_map = {"<pad>": 0, "<bos>": 1, "<eos>": 2, **{s: i + 3 for i, s in enumerate(symbols)}}
    config = {
        "engine": "shami", "phoneme_type": "shami", "alphabet": "ipa", "lang_code": "ar",
        "audio": {"sample_rate": 24000}, "num_symbols": len(id_map), "num_speakers": 1,
        "phoneme_id_map": id_map, "pad": "<pad>", "blank": "<pad>", "bos": "<bos>",
        "eos": "<eos>", "add_blank_char": False, "add_blank_word": False,
        "use_eos_bos": False, "blank_at_start": False, "blank_at_end": False,
        "add_diacritics": False,
    }
    adapter = MagicMock()
    adapter.default_params.return_value = {}
    adapter.synthesize.return_value = AdapterSynthesisResult(audio=np.zeros(8, dtype=np.float32))
    voice = TTSVoice(session=MagicMock(), config=VoiceConfig.from_dict(json.loads(json.dumps(config))),
                     adapter=adapter)
    return voice, sentences, id_map


def _requests(voice, text):
    list(voice.synthesize(text, SynthesisConfig()))
    return [c.args[0] for c in voice.adapter.synthesize.call_args_list]


def test_shami_language_ids_reach_the_request():
    voice, sentences, id_map = _shami_voice()
    requests = _requests(voice, TEXT)
    assert len(requests) == len(sentences)
    for request, (phonemes, language_ids) in zip(requests, sentences):
        assert request.phoneme_ids.tolist() == [[id_map[s] for s in phonemes]]
        assert request.language_ids is not None
        assert request.language_ids.dtype == np.int64
        assert request.language_ids.tolist() == [language_ids]
    tags = {t for r in requests for t in r.language_ids[0].tolist()}
    assert {1, 2} <= tags, "the Arabic and the English words carry their own tags"


def test_ids_added_by_the_tokenizer_get_the_neutral_value():
    """With blanks and BOS/EOS the ids are more than the symbols; every added id
    carries 0 and each symbol's id keeps its own tag."""
    voice, sentences, id_map = _shami_voice()
    tokenizer = voice.tokenizer
    tokenizer.add_blank_char = tokenizer.blank_at_start = tokenizer.blank_at_end = tokenizer.use_eos_bos = True
    utterances = list(voice.prepare(TEXT))
    assert len(utterances) == len(sentences)
    for utterance, (phonemes, language_ids) in zip(utterances, sentences):
        # the blank is <pad>, id 0: blank, s1, blank, s2, ..., blank, then BOS and EOS around
        expected_ids = [id_map["<bos>"], 0]
        expected_tags = [0, 0]
        for symbol, tag in zip(phonemes, language_ids):
            expected_ids += [id_map[symbol], 0]
            expected_tags += [tag, 0]
        expected_ids.append(id_map["<eos>"])
        expected_tags.append(0)
        assert list(utterance.ids) == expected_ids
        assert list(utterance.streams["language_ids"]) == expected_tags


def test_aligned_stream_is_kept():
    voice, _, _ = _shami_voice()
    voice.phoneme_ids_to_audio([1, 4, 2], SynthesisConfig(), streams={"language_ids": [0, 2, 0]})
    assert voice.adapter.synthesize.call_args.args[0].language_ids.tolist() == [[0, 2, 0]]


def test_inline_override_carries_no_stream():
    voice, _, _ = _shami_voice()
    requests = _requests(voice, "[[m a]]")
    assert requests and all(r.language_ids is None for r in requests)
