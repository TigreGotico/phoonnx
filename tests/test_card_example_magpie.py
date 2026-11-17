import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from card_examples import UnsupportedExample, index_files, standalone_example  # noqa: E402
from phoonnx.engines import magpie  # noqa: E402

REPO = "Phoonnx/phoonnx-magpie-tts"
ENTRIES = [e for e in json.loads((ROOT / "phoonnx/voice_index/magpie.json").read_text()).values()
           if isinstance(e, dict)]
CONFIG = {"num_audio_codebooks": 2, "frame_stacking_factor": 2, "codebook_size": 92, "num_all_tokens_per_codebook": 100,
          "sa_n_heads": 1, "sa_d_head": 2, "decoder_n_layers": 3, "local_transformer_n_layers": 1,
          "estimate_alignment_from_layers": [0, 2], "transcript_decoder_layers": [0, 1, 2],
          "language_to_tokenizer": {"fr": ["french_bytes"], "ar": ["arabic_chars"]},
          "inference": {"max_decoder_steps": 80, "min_generated_frames": 2}}
ARABIC = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي ،."
TOKENS = {"tokens": [f"t{i}" for i in range(10)] + ["<pad>", "</s>", "<unk>"] + [f"b{i}" for i in range(256)]
          + list(ARABIC) + ["x"] * 4,
          "tokenizer_offsets": {"french_bytes": 10, "arabic_chars": 269}, "eos_id": 2,
          "num_tokens_per_tokenizer": {"french_bytes": 259, "arabic_chars": len(ARABIC)}}
TOKENS["vocab_size"] = len(TOKENS["tokens"])


def _code(entry, text="Bonjour."):
    block = standalone_example(entry, index_files(entry), repo_id=REPO, text=text)
    return block.removeprefix("```python\n").removesuffix("\n```")


def _entry(voice):
    return next(e for e in ENTRIES if e["voice_id"] == voice)


def test_magpie_example_compiles():
    for entry in ENTRIES:
        compile(_code(entry), entry["voice_id"], "exec")
    compile(_code(ENTRIES[0], None), "magpie", "exec")


def test_magpie_refuses_a_language_that_needs_ipa():
    with pytest.raises(UnsupportedExample):
        _code(dict(ENTRIES[0], lang="en", engine_options={"speaker_id": 0, "lang": "en"}))


def _rng(*arrays):
    return np.random.default_rng(zlib.crc32(b"".join(np.ascontiguousarray(a).tobytes() for a in arrays)))


class _Graph:
    """A stand-in graph: deterministic outputs drawn from a hash of the inputs."""

    def __init__(self, run):
        self._run = run

    def run(self, _, feed):
        return self._run(feed)


def _decoder(feed):
    x, prior = feed["x"], feed["attn_prior"]
    rng = _rng(x, prior)
    steps = x.shape[1]
    logits = rng.standard_normal((2, steps, 4 * 100)).astype(np.float32)
    logits[:, :, 93::100] += max(0, steps - 40) * 3.0  # the end of speech grows likely after some steps
    attention = rng.random((3, 2, 1, steps, feed["cond_mask"].shape[1])).astype(np.float32)
    return [logits, rng.standard_normal((2, steps, 4)).astype(np.float32), None, None, attention]


GRAPHS = {
    "decoder_step.onnx": _decoder,
    "text_encoder.onnx": lambda f: [_rng(f["text"]).standard_normal((1, f["text"].shape[1], 4)).astype(np.float32)],
    "cross_kv.onnx": lambda f: [f["cond"] * 2, f["cond"] * 3],
    "local_step.onnx": lambda f: [_rng(f["h"], f["pos"]).standard_normal((2, 100)).astype(np.float32),
                                  f["cache_k"], f["cache_v"]],
    "audio_embed.onnx": lambda f: [_rng(f["codes"]).standard_normal((f["codes"].shape[0], 1, 4)).astype(np.float32)],
    "lt_embed.onnx": lambda f: [_rng(f["tok"], f["cb"]).standard_normal((2, 1, 4)).astype(np.float32)],
    "codec_decoder.onnx": lambda f: [f["codes"].astype(np.float32).reshape(1, -1) / 100],
}


@pytest.mark.parametrize("voice,text", [("magpie/Jason/fr", "Bonjour, ça va?"), ("magpie/Leo/ar", "مرحبا، كيف حالك.")])
def test_magpie_loop_is_phoonnx_loop(voice, text, tmp_path, monkeypatch):
    """The example's tokens, codes and audio equal phoonnx's, both run on the same stand-in graphs and seed."""
    import huggingface_hub
    import onnxruntime
    files = {"assets/magpie_config.json": json.dumps(CONFIG), "assets/tokenizer.json": json.dumps(TOKENS)}
    for name, content in files.items():
        (tmp_path / Path(name).name).write_text(content, encoding="utf-8")
    context = np.random.default_rng(3).standard_normal((5, 6, 4)).astype(np.float32)
    np.save(tmp_path / "context_embeddings.npy", context)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name: str(tmp_path / Path(name).name))
    monkeypatch.setattr(onnxruntime, "InferenceSession", lambda path: _Graph(GRAPHS[Path(path).name]))
    monkeypatch.chdir(tmp_path)
    names = {}
    exec(_code(_entry(voice), text), names)

    adapter = magpie.MagpieAdapter()
    for attribute, name in (("decoder", "decoder_step.onnx"), ("encoder", "text_encoder.onnx"), ("cross_kv", "cross_kv.onnx"),
                            ("local", "local_step.onnx"), ("audio_embed", "audio_embed.onnx"),
                            ("lt_embed", "lt_embed.onnx"), ("codec", "codec_decoder.onnx")):
        setattr(adapter, attribute, _Graph(GRAPHS[name]))
    adapter.config, adapter.context_embeddings = CONFIG, context
    adapter.tokenizer = magpie.MagpieTokenizer(str(tmp_path / "tokenizer.json"))
    lang = _entry(voice)["lang"]
    ids = adapter.tokenizer.encode(text, lang, adapter.tokenizer_name_for(lang))
    codes = adapter.generate_codes(np.array(ids), _entry(voice)["engine_options"]["speaker_id"],
                                   {**adapter.default_params(), "seed": 0})
    assert names["ids"] == ids
    assert codes.shape[1] > 4 and np.array_equal(names["codes"], codes)
    assert np.array_equal(names["audio"], codes.reshape(1, -1).astype(np.float32).reshape(-1) / 100)
