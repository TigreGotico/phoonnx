"""The Llasa card example builds phoonnx's prompt ids and samples as phoonnx does."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from card_examples import UnsupportedExample, index_files, standalone_example  # noqa: E402

from phoonnx.engines import llasa as engine  # noqa: E402


HUB = "https://huggingface.co/Phoonnx/llasa-1b/resolve/main/"
LLASA = {"voice_id": "llasa/HKUST/en/en_female_a", "engine": "llasa", "lang": "en-US",
         "engine_options": {"voice": "en_female_a"}, "model_url": HUB + "model.onnx",
         "aux_model_urls": {"lm_weights_path": HUB + "model.onnx_data", "codec_decoder_path": HUB + "xcodec2.onnx",
                            "tokenizer_path": HUB + "tokenizer.json", "voices_path": HUB + "voices.json"}}
PRESET = {"text": " A short reference clip. ", "codes": list(range(600))}
SPEECH, END = 1000, 7


class FakeTokenizer:
    """Ids by code point; the speech block and the end marker at fixed ids."""

    def encode(self, text, add_special_tokens=True):
        return type("Encoding", (), {"ids": [ord(c) for c in text]})()

    def token_to_id(self, token):
        return {"<|s_0|>": SPEECH, "<|SPEECH_GENERATION_END|>": END}[token]


def llasa_code(text, entry=LLASA):
    block = standalone_example(entry, index_files(entry), repo_id="Phoonnx/llasa-1b", text=text)
    code = block.removeprefix("```python\n").removesuffix("\n```")
    compile(code, "example.py", "exec")
    return code


def test_prompt_ids_are_phoonnx_ids():
    text = "The quick brown fox jumps over the lazy dog."
    code = llasa_code(text)
    start, end = code.index("\ntext = "), code.index("\n\nhf_hub_download")
    names = {"tokenizer": FakeTokenizer(), "preset": PRESET}
    exec(code[start:end], names)
    adapter = engine.LlasaAdapter()
    adapter.tokenizer, adapter.speech_token_base = FakeTokenizer(), SPEECH
    prompt = adapter.build_prompt(text, PRESET["text"].strip())
    expected = [ord(c) for c in prompt] + adapter.codes_to_token_ids(PRESET["codes"][:engine.MAX_REFERENCE_CODES])
    assert names["ids"] == expected


def run_loop(code, logits_per_step, temperature):
    """Run the example's generation loop on a stand-in session that returns fixed logits."""
    steps = iter(logits_per_step)

    class Session:
        def get_inputs(self):
            return [type("Input", (), {"name": "past_key_values.0.key", "shape": [1, 2, 0, 4]})()]

        def get_outputs(self):
            return [type("Output", (), {"name": n})() for n in ("logits", "present.0.key")]

        def run(self, _, feed):
            return [next(steps)[None, None], np.zeros((1, 2, 1, 4), np.float32)]

    start = code.index("\ncache = ")
    loop = code[start:code.index("\ncodes = ")]
    loop = loop.replace("temperature, top_p = 0.9, 0.95", f"temperature, top_p = {temperature}, 0.95")
    loop = loop.replace("rng = np.random.default_rng()", "rng = np.random.default_rng(0)")
    names = {"np": np, "session": Session(), "tokenizer": FakeTokenizer(), "ids": [1, 2, 3], "speech": SPEECH}
    exec(loop, names)
    return names["generated"]


@pytest.mark.parametrize("temperature", [0.0, 0.9])
def test_loop_samples_as_phoonnx(temperature):
    rng = np.random.default_rng(5)
    logits = [rng.standard_normal(SPEECH + 65536 + 10).astype(np.float32) * 3 for _ in range(40)]
    for step, row in enumerate(logits):
        row[END] = 9.0 if step < 20 else 50.0  # the run ends at the 20th step at the latest
        row[3] = 80.0  # a text token, which must never be chosen
    logits[0][END] = 90.0  # phoonnx masks the end marker at the first step
    adapter = engine.LlasaAdapter()
    adapter.speech_token_base, adapter.speech_end_id = SPEECH, END
    expected, rng = [], np.random.default_rng(0)
    for step, row in enumerate(logits):
        token = engine._sample(adapter._mask_logits(row, allow_end=step > 0), temperature, 0.95, rng)
        if token == END:
            break
        expected.append(token)
    assert expected and run_loop(llasa_code("Hi."), logits, temperature) == expected


def test_sampling_settings_are_phoonnx_defaults():
    code = llasa_code("Hi.")
    defaults = engine.LlasaAdapter().default_params()
    assert f"temperature, top_p = {defaults['temperature']}, {defaults['top_p']}" in code
    assert f"range({engine.LlasaAdapter.MAX_NEW_TOKENS})" in code


def test_uses_the_preset_of_the_entry_and_refuses_another_codec():
    assert 'preset = voices["presets"]["en_female_a"]' in llasa_code(None)
    files = index_files(LLASA)
    files["codec_decoder_path"]["inputs"] = {"audio_codes": 3}
    with pytest.raises(UnsupportedExample):
        standalone_example(LLASA, files, repo_id="Phoonnx/llasa-1b")
