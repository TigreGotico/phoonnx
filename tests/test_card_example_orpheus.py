"""The Orpheus card example: its prompt, sampling loop and SNAC streams are phoonnx's."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import card_examples  # noqa: E402

from phoonnx.engines import orpheus as engine  # noqa: E402

ENTRIES = [e for e in json.loads((ROOT / "phoonnx/voice_index/orpheus.json").read_text()).values()
           if isinstance(e, dict)]
TARA = next(e for e in ENTRIES if e["voice_id"] == "orpheus/canopylabs/en/tara")


def code(entry=TARA, text="The quick brown fox jumps over the lazy dog."):
    files = card_examples.index_files(entry)
    block = card_examples.standalone_example(entry, files, repo_id=files["model"]["repo"], text=text)
    return block.removeprefix("```python\n").removesuffix("\n```")


def part(source, first, last):
    start = source.index(first)
    return source[start:source.index(last, start)]


class StubTokenizer:
    """Llama's BOS when special tokens are added, then ids by code point."""

    def encode(self, text, add_special_tokens=True):
        return type("Encoding", (), {"ids": [128000] * add_special_tokens + [ord(c) for c in text]})()


def test_every_entry_compiles_with_its_voice():
    for entry in ENTRIES:
        source = code(entry)
        compile(source, entry["voice_id"], "exec")
        assert f'voice, text = "{entry["engine_options"]["voice"]}", ' in source


def test_prompt_ids_are_phoonnx_ids():
    text = "Hello there, how are you?"
    names = {"tokenizer": StubTokenizer()}
    exec(part(code(text=text), "voice, text = ", "\n\ntemperature"), names)
    adapter = engine.OrpheusAdapter()
    adapter.tokenizer = StubTokenizer()
    assert names["ids"] == adapter.build_prompt_ids(text, "tara")


class Spec:
    def __init__(self, name, shape=None):
        self.name, self.shape = name, shape


class StubLM:
    """Logits that depend on the last token and the length; end of speech after 40 tokens. A text token
    of the prompt scores above every audio token, and only the penalty over the prompt keeps it out."""

    rows = np.random.default_rng(11).standard_normal((64, 156940)).astype(np.float32) * 2

    def __init__(self, prompt_length):
        self.prompt_length = prompt_length

    def get_inputs(self):
        return [Spec("input_ids"), Spec("attention_mask"), Spec("past_key_values.0.key", [1, 2, "p", 4]),
                Spec("past_key_values.0.value", [1, 2, "p", 4])]

    def get_outputs(self):
        return [Spec("logits"), Spec("present.0.key"), Spec("present.0.value")]

    def run(self, _, feed):
        length = feed["attention_mask"].shape[1]
        row = self.rows[(feed["input_ids"][0, -1] + length) % 64].copy()
        row[engine.AUDIO_TOKEN_BASE:engine.AUDIO_TOKEN_LAST + 1] += 6.0
        row[5] = row[engine.AUDIO_TOKEN_BASE:engine.AUDIO_TOKEN_LAST + 1].max() * 1.1
        if length > self.prompt_length + 40:
            row[engine.END_OF_SPEECH] = 80.0
        cache = np.zeros((1, 2, length, 4), np.float32)
        return [row[None, None], cache, cache]


@pytest.mark.parametrize("temperature", [0.0, 0.6])
def test_loop_samples_as_phoonnx(temperature):
    ids = [128000, 128259, 128000, 5, 6, 7, 128009, 128260, 128261, 128257]
    loop = part(code(), "temperature, top_p", "\n# 7 tokens a frame")
    loop = loop.replace("temperature, top_p, repetition_penalty = 0.6,", f"temperature, top_p, repetition_penalty = {temperature},")
    loop = loop.replace("rng = np.random.default_rng()", "rng = np.random.default_rng(0)")
    names = {"np": np, "lm": StubLM(len(ids)), "ids": ids}
    exec(loop, names)
    adapter = engine.OrpheusAdapter()
    params = {**adapter.default_params(), "temperature": temperature}
    expected = adapter.generate(StubLM(len(ids)), ids, params, np.random.default_rng(0))
    assert len(expected) > 7 and names["tokens"] == expected


def test_sampling_settings_are_phoonnx_defaults():
    defaults = engine.OrpheusAdapter().default_params()
    assert (f"temperature, top_p, repetition_penalty = {defaults['temperature']}, {defaults['top_p']}, "
            f"{defaults['repetition_penalty']}") in code()
    assert f"range({engine.OrpheusAdapter.MAX_NEW_TOKENS})" in code()


def test_streams_are_phoonnx_streams():
    rng = np.random.default_rng(2)
    codes = rng.integers(0, 4096, 7 * 6 + 3)
    codes[7 * 2 + 4] = 4100  # one frame with a code outside its table
    tokens = [int(c) + 128266 + i % 7 * 4096 for i, c in enumerate(codes)]
    names = {"tokens": tokens}
    exec(part(code(), "codes = [t - 128266", "\naudio = snac.run"), names)
    assert names["streams"] == engine.OrpheusAdapter.token_ids_to_codes(tokens)
