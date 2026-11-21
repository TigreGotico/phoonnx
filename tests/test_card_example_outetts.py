"""The OuteTTS card example builds phoonnx's prompt and samples with phoonnx's settings."""
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from card_examples import UnsupportedExample, index_files, standalone_example  # noqa: E402
from phoonnx.engines import outetts as outetts_engine  # noqa: E402
from phoonnx.engines.outetts import OuteTTSAdapter  # noqa: E402


OUTETTS = {"voice_id": "outetts/0.6B/en", "engine": "outetts", "lang": "en-US",
           "model_url": "https://huggingface.co/Phoonnx/outetts-0.6b/resolve/main/model.onnx",
           "aux_model_urls": {
               "codec_decoder_path": "https://huggingface.co/Phoonnx/outetts-dac/resolve/main/decoder_model.onnx",
               "tokenizer_path": "https://huggingface.co/Phoonnx/outetts-0.6b/resolve/main/tokenizer.json",
               "speakers_path": "https://huggingface.co/Phoonnx/outetts-0.6b/resolve/main/speakers/en.json"}}
SPEAKER = {"text": "The cat sat on the mat.",
           "words": [{"word": "The", "duration": 0.2, "c1": [720, 5], "c2": [658, 6],
                      "features": {"energy": 10, "spectral_centroid": 15, "pitch": 45}},
                     {"word": "mat", "duration": 0.31, "c1": [1], "c2": [2], "features": {"energy": 3}}]}


def outetts_code(text, entry=OUTETTS, **files):
    block = standalone_example(entry, {**index_files(entry), **files}, repo_id="Phoonnx/outetts-0.6b", text=text)
    code = block.removeprefix("```python\n").removesuffix("\n```")
    compile(code, "example.py", "exec")
    return code


def example_prompt(text, speaker):
    code = outetts_code(text)
    start, end = code.index("\ntext = "), code.index("\nids = tokenizer.encode")
    names = {"speaker": speaker, "unicodedata": unicodedata}
    exec(code[start:end], names)
    return names["prompt"]


@pytest.mark.parametrize("text", ["The quick brown fox jumps over the lazy dog.",
                                  'She said  "hello"   to me, twice!', "Ｆｕｌｌ width ﬁ ligature."])
@pytest.mark.parametrize("speaker_text", [SPEAKER["text"], "The cat sat on the mat", "Is it the mat?"])
def test_prompt_is_phoonnx_prompt(text, speaker_text):
    speaker = {**SPEAKER, "text": speaker_text}
    assert example_prompt(text, speaker) == OuteTTSAdapter().build_prompt(text, speaker)


def test_sampling_settings_are_phoonnx_defaults():
    line = next(line for line in outetts_code("Hi.").splitlines() if line.startswith("temperature, "))
    names, values = line.split("#")[0].split("=")
    settings = dict(zip((n.strip() for n in names.split(",")), (float(v) for v in values.split(","))))
    defaults = OuteTTSAdapter().default_params()
    assert settings == {k: float(defaults[k]) for k in settings}
    assert re.search(r"history\[-64:\]", outetts_code("Hi."))


def test_names_the_files_of_the_entry():
    code = outetts_code(None)
    assert 'repo = "Phoonnx/outetts-0.6b"' in code
    assert 'hf_hub_download(repo, "model.onnx_data")' in code
    assert 'hf_hub_download("Phoonnx/outetts-dac", "decoder_model.onnx")' in code
    assert 'text = "..."  # a sentence in en-US' in code


def test_refuses_a_graph_without_its_feeds():
    files = index_files(OUTETTS)
    files["model"]["inputs"] = {"input_ids": 2, "attention_mask": 2}
    with pytest.raises(UnsupportedExample):
        standalone_example(OUTETTS, files, repo_id="Phoonnx/outetts-0.6b")


def run_loop(code, logits_per_step, ids, temperature):
    """Run the example's generation loop on a stand-in session that returns fixed logits."""
    steps = iter(logits_per_step)

    class Session:
        def get_inputs(self):
            return [type("Input", (), {"name": "past_key_values.0.key", "shape": [1, 2, 0, 4]})()]

        def get_outputs(self):
            return [type("Output", (), {"name": n})() for n in ("logits", "present.0.key")]

        def run(self, _, feed):
            return [next(steps)[None, None], np.zeros((1, 2, 1, 4), np.float32)]

    class Tokens:
        def token_to_id(self, token):
            return {"<|audio_end|>": 2, "<|im_end|>": 3}[token]

    loop = code[code.index("\ncache = "):code.index("\n# each frame")]
    loop = re.sub(r"^temperature, (.*) = [\d.]+, ", rf"temperature, \1 = {temperature}, ", loop, flags=re.M)
    loop = loop.replace("rng = np.random.default_rng()", "rng = np.random.default_rng(0)")
    names = {"np": np, "session": Session(), "tokenizer": Tokens(), "ids": list(ids)}
    exec(loop, names)
    return names["generated"]


@pytest.mark.parametrize("temperature", [0.0, 0.4, 1.3])
def test_loop_samples_as_phoonnx(temperature):
    rng = np.random.default_rng(5)
    logits = [rng.standard_normal(300).astype(np.float32) * 3 for _ in range(60)]
    for step, row in enumerate(logits):
        row[[2, 3]] = -50.0 if step < 40 else 50.0  # the run ends at the 40th step at the latest
        row[10:14] = 12.0  # tokens the repetition penalty has to push down
        if step % 2:  # one strong token and a tail that top-p keeps and min-p drops
            row[100 + step] = 30.0
            row[200:230] = 30.0 - 3.9 * max(temperature, 0.1)  # each 2 % of the strong one
    ids = [10, 11, 12, 13, 20, 21]
    defaults = OuteTTSAdapter().default_params()
    expected, history, rng = [], list(ids), np.random.default_rng(0)
    for row in logits:
        scores = outetts_engine._apply_repetition_penalty(row, np.asarray(history[-64:]), defaults["repetition_penalty"])
        token = outetts_engine._sample(scores, temperature, int(defaults["top_k"]), defaults["top_p"],
                                       defaults["min_p"], rng)
        if token in (2, 3):
            break
        expected.append(token)
        history.append(token)
    assert len(expected) > 10 and run_loop(outetts_code("Hi."), logits, ids, temperature) == expected
