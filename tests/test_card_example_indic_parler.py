"""The Indic Parler card example: it compiles, and its delay pattern equals phoonnx's."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import card_examples  # noqa: E402

from phoonnx.engines.indic_parler import apply_delay_pattern_mask, build_delay_pattern_mask, strip_delay_pattern  # noqa: E402

REPO = "Phoonnx/parlertts-multi-indic-unicode"
ENTRY = {"voice_id": "indic_parler/rohit/hi", "engine": "indic_parler", "lang": "hi",
         "engine_options": {"description": "Rohit's voice is clear.", "speaker": "Rohit"}}
FILES = {key: {"repo": REPO, "file": name} for key, name in (
    ("model", "decoder_prefill.onnx"), ("text_encoder_path", "text_encoder.onnx"),
    ("decoder_decode_path", "decoder_decode.onnx"), ("dac_decoder_path", "dac_decoder.onnx"),
    ("prompt_tokenizer_path", "tokenizer.json"), ("description_tokenizer_path", "description_tokenizer.json"))}


def code(text="नमस्ते, आप कैसे हैं?"):
    block = card_examples.standalone_example(ENTRY, FILES, repo_id=REPO, text=text)
    return block.removeprefix("```python\n").removesuffix("\n```")


def lines(source, first, last):
    start = source.index(first)
    return source[start:source.index(last, start) + len(last)]


def test_compiles_and_names_the_description():
    source = code()
    compile(source, "example", "exec")
    assert "description = \"Rohit's voice is clear.\"" in source


def test_delay_pattern_equals_phoonnx():
    names = {"np": np}
    exec(lines(code(), "max_length, column", "-1))  # BOS, PAD, predict"), names)
    assert np.array_equal(names["pattern"], build_delay_pattern_mask(1, 1501)[1])


def test_undelay_equals_phoonnx():
    rng = np.random.default_rng(3)
    source = lines(code(), "codes = np.stack", "len(end) else None]")
    _, pattern = build_delay_pattern_mask(1, 1501)
    for n, stop in ((40, None), (60, 30), (1501, None)):
        raw = rng.integers(0, 1024, (9, n))
        raw[:, 0] = 1025
        if stop:
            raw[:, stop:] = 1024  # every codebook ends; phoonnx cuts at the first end of speech
        raw = apply_delay_pattern_mask(raw, pattern) if n == 1501 else raw
        names = {"np": np, "raw": raw}
        exec(source, names)
        assert np.array_equal(names["codes"], strip_delay_pattern(raw, pattern)), n
