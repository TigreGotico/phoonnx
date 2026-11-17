"""The Spark-TTS card example: it compiles, and its prompt equals the one phoonnx builds."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import card_examples  # noqa: E402

from phoonnx.engines.sparktts import SparkTTSAdapter, _SPECIAL_TOKENS  # noqa: E402

REPO = "Phoonnx/sparktts-multi-0.5b-unicode"
ENTRY = {"voice_id": "sparktts/female/en", "engine": "sparktts", "lang": "en-US"}
FILES = {key: {"repo": REPO, "file": name} for key, name in (
    ("model", "model.onnx"), ("vocoder_path", "bicodec_vocoder.onnx"), ("bpe_tokenizer_path", "tokenizer.json"),
    ("speaker_tokens_path", "voices/female_en.json"))}


class StubTokenizer:
    """Stands in for tokenizers.Tokenizer: one id per character, special tokens from a table."""
    special = {token: 151000 + n for n, token in enumerate(_SPECIAL_TOKENS)}

    def encode(self, text):
        return type("Encoding", (), {"ids": [ord(c) for c in text]})()

    def token_to_id(self, token):
        return self.special[token]


def code(text="Hello there."):
    block = card_examples.standalone_example(ENTRY, FILES, repo_id=REPO, text=text)
    return block.removeprefix("```python\n").removesuffix("\n```")


def test_compiles_and_downloads_the_graph_weights():
    source = code()
    compile(source, "example", "exec")
    assert 'hf_hub_download(repo, "model.onnx_data")' in source
    assert 'hf_hub_download(repo, "voices/female_en.json")' in source


def test_prompt_equals_phoonnx():
    source = code("Hello there.")
    start = source.index('text = "')
    prompt_lines = source[start:source.index("\n\n", start)]
    names = {"tokenizer": StubTokenizer(), "token": StubTokenizer().token_to_id, "global_tokens": list(range(7, 39))}
    exec(prompt_lines, names)
    adapter = SparkTTSAdapter()
    adapter.special = dict(StubTokenizer.special)
    expected = adapter.build_prompt([ord(c) for c in "Hello there."], np.arange(7, 39))[0].tolist()
    assert names["ids"] == expected


def test_graph_names_are_checked():
    files = {**FILES, "vocoder_path": {**FILES["vocoder_path"], "inputs": {"tokens": 2}, "outputs": {"wav": 3}}}
    try:
        card_examples.standalone_example(ENTRY, files, repo_id=REPO, text="Hi.")
    except card_examples.UnsupportedExample:
        return
    raise AssertionError("a vocoder without semantic_tokens was accepted")
