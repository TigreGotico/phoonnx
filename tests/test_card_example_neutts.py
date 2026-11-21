"""The NeuTTS card example builds phoonnx's prompt and samples with phoonnx's settings."""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from card_examples import UnsupportedExample, index_files, standalone_example  # noqa: E402
from phoonnx.engines import neutts as neutts_engine  # noqa: E402
from phoonnx.engines._sampling import apply_repetition_penalty  # noqa: E402
from phoonnx.engines.neutts import NeuTTSAdapter  # noqa: E402


HUB = "https://huggingface.co/Phoonnx/neutts-tw/resolve/main/"
NEUTTS = {"voice_id": "neutts/akiti/twi/abena", "engine": "neutts", "lang": "tw", "engine_options": {"voice": "abena"},
          "model_url": HUB + "neutts_lm.onnx",
          "aux_model_urls": {"lm_weights_path": HUB + "neutts_lm.onnx.data", "codec_decoder_path": HUB + "neucodec.onnx",
                             "tokenizer_path": HUB + "tokenizer.json", "voices_path": HUB + "voices.json"}}
PRESET = {"text": "nti ɔyɛ nipa, a mmaa ɛda n'akoma so yie.", "codes": list(range(300))}


class FakeEspeak:
    """Stands in for espeak-ng: a visible, lossless change, so the test sees what reached it."""

    def phonemize_string(self, text, lang):
        return f" <{lang}:{text.upper()}> "


def neutts_code(text, entry=NEUTTS):
    block = standalone_example(entry, index_files(entry), repo_id="Phoonnx/neutts-tw", text=text)
    code = block.removeprefix("```python\n").removesuffix("\n```")
    compile(code, "example.py", "exec")
    return code


def example_prompt(text):
    code = neutts_code(text)
    start, end = code.index("\nmarks = "), code.index("\nids = tokenizer.encode")
    names = {"re": re, "phonemizer": FakeEspeak(), "preset": PRESET}
    exec(code[start:end], names)
    return names["prompt"]


def phoonnx_prompt(text):
    adapter = NeuTTSAdapter()
    adapter._phonemizer = FakeEspeak()
    return adapter.build_prompt(adapter._espeak(text).strip(), adapter.reference_phones("abena", PRESET),
                                PRESET["codes"])


@pytest.mark.parametrize("text", ["Maakye, wo ho te sɛn?", "Me din de Kwame!  Mefiri Kumase…", "“Aane” ɔkaa sɛ."])
def test_prompt_is_phoonnx_prompt(text):
    assert example_prompt(text) == phoonnx_prompt(text)


def test_sampling_settings_are_phoonnx_defaults():
    code = neutts_code("Maakye.")
    line = next(line for line in code.splitlines() if line.startswith("temperature, "))
    names, values = line.split("#")[0].split("=")
    settings = dict(zip((n.strip() for n in names.split(",")), (float(v) for v in values.split(","))))
    defaults = NeuTTSAdapter().default_params()
    assert settings == {k: float(defaults[k]) for k in settings}
    assert f"range({NeuTTSAdapter.MAX_NEW_TOKENS})" in code


def test_uses_the_preset_of_the_entry():
    assert 'preset = voices["presets"]["abena"]' in neutts_code(None)
    entry = {**NEUTTS, "engine_options": {}}
    assert 'preset = voices["presets"][voices["default_voice"]]' in neutts_code(None, entry)


def test_refuses_a_language_lfn_does_not_read():
    with pytest.raises(UnsupportedExample):
        neutts_code("Hello.", {**NEUTTS, "lang": "en-US"})


def run_loop(code, logits_per_step, ids, temperature):
    """Run the example's generation loop on a stand-in session that returns fixed logits."""
    steps = iter(logits_per_step)

    class Session:
        def get_inputs(self):
            return [type("Input", (), {"name": "past_key_0", "shape": [1, 2, 0, 4]})()]

        def get_outputs(self):
            return [type("Output", (), {"name": n})() for n in ("logits", "present_key_0")]

        def run(self, _, feed):
            return [next(steps)[None], np.zeros((1, 2, 1, 4), np.float32)]

    class Tokens:
        def token_to_id(self, token):
            return {"<|SPEECH_GENERATION_END|>": 2, "<|endoftext|>": 3}[token]

    loop = code[code.index("\ncache = "):code.index("\ncodes = ")]
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
        row[30] = -12.0  # a negative score the penalty makes lower still
    ids = [10, 11, 12, 13, 30]
    defaults = neutts_engine.NeuTTSAdapter().default_params()
    expected, history, rng = [], list(ids), np.random.default_rng(0)
    for row in logits:
        scores = apply_repetition_penalty(row, np.asarray(history[-64:]), defaults["repetition_penalty"])
        token = neutts_engine._sample(scores, temperature, int(defaults["top_k"]), defaults["top_p"], rng)
        if token in (2, 3):
            break
        expected.append(token)
        history.append(token)
    assert len(expected) > 10 and run_loop(neutts_code("Maakye."), logits, ids, temperature) == expected
