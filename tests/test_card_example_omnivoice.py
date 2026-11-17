"""The OmniVoice card example: its output length, prompt, unmasking loop and post-processing are phoonnx's."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import card_examples  # noqa: E402

from phoonnx.engines.omnivoice import OmniVoiceAdapter  # noqa: E402

REPO = "Phoonnx/omnivoice-multi-unicode"
ENTRIES = [e for e in json.loads((ROOT / "phoonnx/voice_index/omnivoice.json").read_text()).values()
           if isinstance(e, dict)]
FILES = {key: {"repo": REPO, "file": name} for key, name in (
    ("model", "omnivoice_backbone.onnx"), ("decoder_path", "higgs_decoder.onnx"), ("bpe_tokenizer_path", "tokenizer.json"))}


def code(entry, text="The quick brown fox jumps over the lazy dog."):
    block = card_examples.standalone_example(entry, FILES, repo_id=REPO, text=text)
    return block.removeprefix("```python\n").removesuffix("\n```")


def part(source, first, last):
    start = source.index(first)
    return source[start:source.index(last, start)]


ENGLISH = next(e for e in ENTRIES if e["voice_id"] == "omnivoice/en")


class StubTokenizer:
    """Ids by code point, as phoonnx's BPE reader and the tokenizers package both see it."""

    def encode(self, text):
        return type("Encoding", (), {"ids": [ord(c) for c in text]})()

    def tokenize(self, text):
        return [ord(c) for c in text]


def test_every_entry_compiles_with_the_language_phoonnx_resolves():
    for entry in ENTRIES:
        source = code(entry)
        compile(source, entry["voice_id"], "exec")
        lang = re.search(r'^lang, text = "([^"]+)"', source, re.M).group(1)
        assert lang == OmniVoiceAdapter.resolve_language(entry["engine_options"]["lang"]), entry["voice_id"]


@pytest.mark.parametrize("text", ["Nice to meet you.", "The quick brown fox jumps over the lazy dog.",
                                  "Olá, ação é já! 123", "Привет, как дела?", "مرحبا بك", "नमस्ते दुनिया",
                                  "你好，世界。", "こんにちは", "안녕하세요", "ሰላም", "ॐ ✓ … 𠀀𠀁", "Hi"])
def test_length_is_phoonnx_estimate(text):
    names = {}
    source = code(ENGLISH, text)
    exec("import bisect\nimport unicodedata\n" + part(source, "lang, text = ", "\ntext = re.sub"), names)
    assert names["length"] == OmniVoiceAdapter().estimate_target_len(text, None, None)


def run_example_loop(source, text, length):
    """The example's prompt and unmasking loop, on a stand-in backbone, with the generator seeded to 0."""
    names = {"tokenizer": StubTokenizer(), "backbone": StubBackbone(), "length": length}
    exec("import math\nimport re\nimport numpy as np\n"
         + part(source, "lang, text = ", "\n# OmniVoice writes")
         + part(source, "\ntext = re.sub", "\naudio = decoder.run").replace(
             "rng = np.random.default_rng()", "rng = np.random.default_rng(0)"), names)
    return names


class StubBackbone:
    """Logits that depend on each slot's token and position and on the codes filled so far,
    so the order in which the loop fills the slots changes the codes."""

    table = np.random.default_rng(7).standard_normal((8, 97, 1025)).astype(np.float32) * 4

    def run(self, _, feed):
        ids = feed["input_ids"][0]
        filled = np.where(ids < 1024, ids, 0).sum(axis=1, keepdims=True)
        positions = (ids + np.arange(ids.shape[1]) * 3 + feed["audio_mask"][0] * 11 + filled) % 97
        return [self.table[np.arange(8)[:, None], positions][None]]


@pytest.mark.parametrize("text, length", [("Hello there [laughter] friend.", 40), ("你好 世界", 9)])
def test_prompt_and_codes_are_phoonnx(text, length):
    names = run_example_loop(code(ENGLISH, text), text, length)
    adapter = OmniVoiceAdapter()
    adapter.tokenizer = StubTokenizer()
    prompt, _ = adapter.build_prompt(text, length, lang="en")
    assert names["ids"] == prompt[0, 0, :prompt.shape[2] - length].tolist()
    expected = adapter.generate_codes(StubBackbone(), text, length, lang="en", seed=0)
    assert np.array_equal(names["tokens"], expected)


def test_post_processing_is_phoonnx():
    rng = np.random.default_rng(3)
    speech = lambda seconds: (rng.standard_normal(int(24000 * seconds)) * 0.2).astype(np.float32)  # noqa: E731
    quiet = lambda seconds: np.zeros(int(24000 * seconds), np.float32)  # noqa: E731
    raw = np.concatenate([quiet(0.4), speech(0.7), quiet(1.3), speech(0.5), quiet(0.2), speech(0.3), quiet(0.6)])
    names = {"audio": raw.copy()}
    exec("import numpy as np\nfrom pydub import AudioSegment\nfrom pydub.silence import detect_leading_silence, "
         "split_on_silence\n" + part(code(ENGLISH), "segment = AudioSegment", "\nwith wave.open"), names)
    expected = OmniVoiceAdapter().post_process(raw, None)
    assert names["audio"].size < raw.size and np.array_equal(names["audio"], expected)


def test_refuses_a_language_that_is_not_a_model_id():
    with pytest.raises(card_examples.UnsupportedExample):
        code(dict(ENGLISH, lang="en-US", engine_options={"lang": "en-US"}))
