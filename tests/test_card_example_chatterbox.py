"""The Chatterbox card example: it compiles for each variant, writes phoonnx's multilingual front end,
and refuses the languages whose text phoonnx changes with a tool the example cannot use."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import card_examples  # noqa: E402

from phoonnx.tokenizer import ChatterboxMTLTokenizer  # noqa: E402


def files(repo):
    return {key: {"repo": repo, "file": name} for key, name in (
        ("model", "language_model_q4.onnx"), ("speech_encoder", "speech_encoder.onnx"),
        ("embed_tokens", "embed_tokens.onnx"), ("conditional_decoder", "conditional_decoder.onnx"),
        ("tokenizer_config", "tokenizer.json"))}


def code(voice_id, lang, text, repo="Phoonnx/chatterbox-multi-unicode", graph_files=None):
    entry = {"voice_id": voice_id, "engine": "chatterbox", "lang": lang}
    block = card_examples.standalone_example(entry, graph_files or files(repo), repo_id=repo, text=text)
    return block.removeprefix("```python\n").removesuffix("\n```")


@pytest.mark.parametrize("voice_id, lang", [("chatterbox/base/en", "en-US"), ("chatterbox/turbo/en", "en-US"),
                                            ("chatterbox/multilingual/de", "de-DE")])
def test_compiles(voice_id, lang):
    compile(code(voice_id, lang, "Guten Tag."), "example", "exec")


def test_turbo_feeds_positions_from_the_graph():
    graph_files = files("Phoonnx/chatterbox-en_us-turbo-unicode")
    graph_files["embed_tokens"]["inputs"] = {"input_ids": 2}
    graph_files["model"]["inputs"] = {"inputs_embeds": 3, "attention_mask": 2, "position_ids": 2,
                                      "past_key_values.0.key": 4}
    source = code("chatterbox/turbo/en", "en-US", "Hi.", repo="Phoonnx/chatterbox-en_us-turbo-unicode", graph_files=graph_files)
    assert '"position_ids": np.arange(length - embeds.shape[1], length)[None]' in source
    assert "exaggeration\": np.array" not in source


class StubTokenizer:
    """Ids by code point, with the bracketed tokens of the multilingual vocabulary as single ids."""

    SPECIAL = {"[SPACE]": 1, "[de]": 2, "[ko]": 3, "[pt]": 4}

    def encode(self, text):
        ids, i = [], 0
        while i < len(text):
            token = next((t for t in self.SPECIAL if text.startswith(t, i)), None)
            ids.append(self.SPECIAL[token] if token else ord(text[i]) + 100)
            i += len(token) if token else 1
        return type("Encoding", (), {"ids": ids})()

    def token_to_id(self, token):
        return self.SPECIAL.get(token)


@pytest.mark.parametrize("voice_id, lang, text", [
    ("chatterbox/multilingual/de", "de-DE", "Der Zug nach Berlin fährt um acht Uhr ab."),
    ("chatterbox/multilingual/ko", "ko-KR", "안녕하세요 반갑습니다"),
    ("chatterbox/multilingual/pt", "pt-PT", "Olá, Ação É Já"),
])
def test_multilingual_ids_equal_phoonnx(voice_id, lang, text):
    source = code(voice_id, lang, text)
    start = source.index("text = ")
    names = {"tokenizer": StubTokenizer()}
    exec("import unicodedata\n" + source[start:source.index("\n\n", start)], names)
    phoonnx = ChatterboxMTLTokenizer.__new__(ChatterboxMTLTokenizer)
    phoonnx._tok = StubTokenizer()
    assert names["ids"] == phoonnx.tokenize(text, language=lang)


@pytest.mark.parametrize("voice_id, lang", [("chatterbox/multilingual/ar", "ar"), ("chatterbox/lahgtna/eg", "ar-EG"),
                                            ("chatterbox/multilingual/he", "he-IL"),
                                            ("chatterbox/multilingual/ja", "ja-JP"),
                                            ("chatterbox/multilingual/ru", "ru-RU"),
                                            ("chatterbox/multilingual/zh", "zh-CN")])
def test_refuses_text_phoonnx_changes_with_other_tools(voice_id, lang):
    with pytest.raises(card_examples.UnsupportedExample, match="phoonnx"):
        code(voice_id, lang, "...")
