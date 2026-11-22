"""Packed sentences keep the gap the source text has between them."""
import pytest

from phoonnx.engines import base, llasa, neutts, orpheus
from phoonnx.engines.arktts import chunk_text as arktts_chunk
from phoonnx.engines.base import pack_sentences
from phoonnx.engines.llasa import LlasaAdapter
from phoonnx.engines.neutts import NeuTTSAdapter
from phoonnx.engines.orpheus import OrpheusAdapter
from phoonnx.engines.qwen3tts import chunk_text as qwen3tts_chunk
from phoonnx.engines.sparktts import chunk_text as sparktts_chunk
from phoonnx.engines.supertonic import chunk_text as supertonic_chunk

CHINESE = "我有三本书。你有几本？太好了！"
LATIN = "Hello there. How are you? Fine."

FUNCTION_PACKERS = [qwen3tts_chunk, arktts_chunk, sparktts_chunk, supertonic_chunk]
METHOD_PACKERS = [
    lambda text, n: LlasaAdapter().chunk_text(text, n),
    lambda text, n: NeuTTSAdapter().chunk_text(text, n),
    lambda text, n: OrpheusAdapter().chunk_text(text, n),
]
PACKERS = FUNCTION_PACKERS + METHOD_PACKERS


@pytest.mark.parametrize("pack", PACKERS)
def test_chinese_under_budget_is_one_chunk_equal_to_the_input(pack):
    assert pack(CHINESE, 100) == [CHINESE]


@pytest.mark.parametrize("pack", PACKERS)
def test_chinese_over_budget_joins_back_to_the_input(pack):
    chunks = pack(CHINESE, 11)
    assert "".join(chunks) == CHINESE
    assert all(chunk == chunk.strip() for chunk in chunks)
    assert all(chunk[-1] in "。？！" for chunk in chunks)


@pytest.mark.parametrize("pack", PACKERS)
def test_latin_keeps_its_single_space(pack):
    assert pack(LATIN, 100) == [LATIN]
    assert pack(LATIN, 28) == ["Hello there. How are you?", "Fine."]


@pytest.mark.parametrize("pack", FUNCTION_PACKERS)
def test_paragraphs_stay_separate_chunks(pack):
    assert pack("你好。\n\n再见。", 100) == ["你好。", "再见。"]


def test_pack_sentences_splits_only_at_sentence_ends(monkeypatch):
    # sentence spans of "我有三本书。你有几本？太好了！" as a splitter that breaks after 。？！ reports them
    spans = [(0, 6, "我有三本书。"), (6, 11, "你有几本？"), (11, 15, "太好了！")]
    monkeypatch.setattr(base, "span_indexed_sentence_tokenize", lambda text: spans)
    assert pack_sentences(CHINESE, 100) == [CHINESE]
    assert pack_sentences(CHINESE, 11) == ["我有三本书。你有几本？", "太好了！"]
    assert pack_sentences(CHINESE, 10) == ["我有三本书。", "你有几本？太好了！"]
    assert pack_sentences(CHINESE, 3) == ["我有三本书。", "你有几本？", "太好了！"]


@pytest.mark.parametrize("module, pack", [(llasa, METHOD_PACKERS[0]), (neutts, METHOD_PACKERS[1]),
                                          (orpheus, METHOD_PACKERS[2])])
def test_the_method_packers_split_only_at_sentence_ends(monkeypatch, module, pack):
    # the same spans as above: the sentences touch, so the chunk budget counts no space between them
    spans = [(0, 6, "我有三本书。"), (6, 11, "你有几本？"), (11, 15, "太好了！")]
    monkeypatch.setattr(module, "span_indexed_sentence_tokenize", lambda text: spans)
    assert pack(CHINESE, 11) == ["我有三本书。你有几本？", "太好了！"]
    assert pack(CHINESE, 10) == ["我有三本书。", "你有几本？太好了！"]
    assert pack(CHINESE, 3) == ["我有三本书。", "你有几本？", "太好了！"]
