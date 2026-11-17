"""The Qwen3-TTS card example: its speaker, language, prompt and sampler are the ones phoonnx uses."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from card_examples import UnsupportedExample, index_files, standalone_example  # noqa: E402

from phoonnx.engines import qwen3tts  # noqa: E402

REPO = "Phoonnx/phoonnx-qwen3-tts"
ENTRIES = [e for e in json.loads((ROOT / "phoonnx/voice_index/qwen3tts.json").read_text()).values()
           if isinstance(e, dict)]


def _code(entry, text="Hello there."):
    block = standalone_example(entry, index_files(entry), repo_id=REPO, text=text)
    return block.removeprefix("```python\n").removesuffix("\n```")


def test_qwen3tts_example_compiles():
    for entry in ENTRIES:
        compile(_code(entry), entry["voice_id"], "exec")
    compile(_code(ENTRIES[0], None), "qwen3tts", "exec")


@pytest.mark.parametrize("entry", ENTRIES, ids=[e["voice_id"] for e in ENTRIES])
def test_qwen3tts_speaker_and_language_tokens_are_phoonnx_tokens(entry):
    options = entry["engine_options"]
    code = _code(entry)
    speaker = int(re.search(r"^speaker = (\d+)", code, re.M).group(1))
    think = json.loads(re.search(r"^think = (\[[\d, ]+\])", code, re.M).group(1))
    language = qwen3tts.resolve_language_id(qwen3tts.resolve_language_name(options["language"]), options["speaker"])
    assert speaker == qwen3tts.SPEAKER_IDS[options["speaker"]]
    expected = ([qwen3tts.CODEC_NOTHINK, qwen3tts.CODEC_THINK_BOS, qwen3tts.CODEC_THINK_EOS] if language is None else
                [qwen3tts.CODEC_THINK, qwen3tts.CODEC_THINK_BOS, language, qwen3tts.CODEC_THINK_EOS])
    assert think == expected


class _Table:
    """A stand-in embedding graph: a fixed random table, looked up by id."""

    def __init__(self, seed, rows=160000):
        self.table = np.random.default_rng(seed).standard_normal((rows, 4)).astype(np.float32)

    def run(self, _, feed):
        return [self.table[feed["input_ids"]]]


def test_qwen3tts_prompt_is_phoonnx_prompt():
    """The example's prompt lines give the embeddings that phoonnx's build_prompt gives."""
    entry = next(e for e in ENTRIES if e["voice_id"] == "qwen3tts/ryan/en")
    code = _code(entry)
    ids = [151644, 77091, 198, 9707, 1052, 13, 151645, 198, 151644, 77091, 198]
    names = {"np": np, "text_embed": _Table(1), "codec_embed": _Table(2), "ids": ids,
             "speaker": qwen3tts.SPEAKER_IDS["ryan"]}
    exec(code[code.index("def embed"):code.index("def sub_embed")]
         + code[code.index("# The talker reads embeddings"):code.index("temperature, top_k")], names)
    adapter = qwen3tts.Qwen3TTSAdapter()
    adapter.text_embed, adapter.codec_embed = names["text_embed"], names["codec_embed"]
    prompt, tts_pad = adapter.build_prompt(np.array(ids), "ryan", "english")
    assert np.array_equal(names["step_embeds"], prompt)
    assert np.array_equal(names["tts_pad"], tts_pad)


def test_qwen3tts_sampler_draws_as_phoonnx():
    code = _code(ENTRIES[0])
    names = {"np": np}
    exec(code[code.index("temperature, top_k"):code.index("def next_past")], names)
    names["rng"] = np.random.default_rng(0)  # the example's generator, seeded for the comparison
    logits = np.random.default_rng(5).standard_normal((20, 3072)).astype(np.float32)
    rng = np.random.default_rng(0)
    assert ([names["sample"](row) for row in logits]
            == [qwen3tts.sample_token(row, 0.9, 50, 1.0, rng) for row in logits])


def test_qwen3tts_refuses_an_unknown_speaker():
    entry = dict(ENTRIES[0], engine_options={"speaker": "nobody", "language": "en"})
    with pytest.raises(UnsupportedExample):
        _code(entry)
