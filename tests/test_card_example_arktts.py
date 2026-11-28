"""The ArkTTS card example: it compiles, and its prompt equals the one phoonnx builds."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import card_examples  # noqa: E402

from phoonnx.engines.arktts import ArkTTSAdapter  # noqa: E402

REPO = "Phoonnx/arktts-eu_es-zortzi-unicode"
ENTRY = {"voice_id": "arktts/zortzi-maider/eu", "engine": "arktts", "lang": "eu-ES"}
FILES = {key: {"repo": REPO, "file": name} for key, name in (
    ("model", "slow_ar_fp16.onnx"), ("fast_ar_path", "fast_ar_fp16.onnx"),
    ("codec_decoder_path", "codec_decoder_fp16.onnx"), ("bpe_tokenizer_path", "tokenizer.json"),
    ("voice_codes_path", "voices/maider.json"))}


class StubTokenizer:
    """Stands in for both tokenizers: one id per character, so a wrong split or order shows."""

    def encode(self, text):
        return type("Encoding", (), {"ids": [ord(c) for c in text]})()

    def tokenize(self, text):
        return self.encode(text).ids


def code(text="Kaixo, zer moduz?"):
    block = card_examples.standalone_example(ENTRY, FILES, repo_id=REPO, text=text)
    return block.removeprefix("```python\n").removesuffix("\n```")


def test_compiles():
    compile(code(), "example", "exec")


def test_prompt_equals_phoonnx():
    source = code("Kaixo,  zer moduz?")
    voice = {"codes": np.arange(30).reshape(10, 3) % 7, "reference_text": " Kaixo denoi. "}
    start = source.index('reference, speaker = ')
    names = {"np": np, "tokenizer": StubTokenizer(), "voice": voice}
    exec(source[start:source.index("\n\ntemperature", start)], names)
    adapter = ArkTTSAdapter()
    adapter.tokenizer = StubTokenizer()
    adapter.reference_codes, adapter.reference_text = np.asarray(voice["codes"]), voice["reference_text"].strip()
    suffix = adapter.encode_text("Kaixo,  zer moduz?", None, None)[0]
    expected = adapter.build_prompt(suffix)
    assert names["ids"] == expected[0, 0].tolist()
    assert np.array_equal(names["step"], expected)
