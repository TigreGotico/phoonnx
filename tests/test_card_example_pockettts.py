"""The Pocket TTS card example builds phoonnx's token ids, voice state, latents and audio."""
import json
import struct
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from card_examples import UnsupportedExample, index_files, standalone_example  # noqa: E402
from phoonnx.engines.base import AdapterSynthesisRequest  # noqa: E402
from phoonnx.engines.pockettts import PocketTTSAdapter  # noqa: E402

POCKET_HUB = "https://huggingface.co/Phoonnx/pockettts-x/resolve/main/"
POCKET_ENTRY = {"voice_id": "pockettts/en/alba", "engine": "pockettts", "lang": "en",
                "model_url": POCKET_HUB + "flow_lm_main_int8.onnx",
                "aux_model_urls": {k: POCKET_HUB + v for k, v in {
                    "bundle_path": "bundle.json", "tokenizer_path": "tokenizer.model",
                    "text_conditioner_path": "text_conditioner.onnx", "flow_lm_flow_path": "flow_lm_flow_int8.onnx",
                    "mimi_decoder_path": "mimi_decoder_int8.onnx", "mimi_encoder_path": "mimi_encoder.onnx",
                    "voice_state_path": "voices/alba.safetensors"}.items()}}
POCKET_BUNDLE = {"sample_rate": 24000, "frame_rate": 12.5, "latent_dim": 2, "conditioning_dim": 3,
                 "flow_lm_state_manifest": [
                     {"input_name": "state_0", "index": 0, "shape": [2, 3], "dtype": "float32", "fill": "nan",
                      "module": "transformer", "key": "cache"},
                     {"input_name": "state_1", "index": 1, "shape": [1], "dtype": "int64", "fill": "zeros",
                      "module": "transformer", "key": "step"}],
                 "mimi_state_manifest": [{"input_name": "state_0", "index": 0, "shape": [1], "dtype": "float32",
                                          "fill": "zeros"}]}
POCKET_PIECES = [("<unk>", 0.0, 2), ("<s>", 0.0, 3), ("</s>", 0.0, 3), ("▁", -2.0, 1), ("▁the", -3.0, 1),
                 ("▁qu", -4.0, 1), ("ick", -4.5, 1), ("▁quick", -7.5, 1), (".", -2.5, 1), ("T", -5.0, 1),
                 ("▁T", -4.0, 1), ("he", -3.5, 1), ("▁fox", -3.0, 1), ("o", -6.0, 1), ("x", -6.0, 1), ("f", -6.0, 1)]
POCKET_TEXT = "the quick fox the quick fox"


def _varint(n):
    out = b""
    while n >= 128:
        out, n = out + bytes([n & 127 | 128]), n >> 7
    return out + bytes([n])


def _field(number, payload):
    if isinstance(payload, float):
        return _varint(number << 3 | 5) + struct.pack("<f", payload)
    if isinstance(payload, int):
        return _varint(number << 3) + _varint(payload)
    return _varint(number << 3 | 2) + _varint(len(payload)) + payload


def write_pocket_files(folder):
    model = b"".join(_field(1, _field(1, p.encode()) + _field(2, s) + _field(3, t)) for p, s, t in POCKET_PIECES)
    model += _field(2, _field(3, 1)) + _field(3, _field(1, b"identity") + _field(3, 1) + _field(5, 1))
    (folder / "tokenizer.model").write_bytes(model)
    (folder / "bundle.json").write_text(json.dumps(POCKET_BUNDLE))
    tensors = {"transformer/cache": np.array([[0.5, 1.5, 2.5]], dtype=np.float32),
               "transformer/offset": np.array([4], dtype=np.int64)}
    header, offset, blob = {}, 0, b""
    for name, array in tensors.items():
        data = array.tobytes()
        header[name] = {"dtype": {"float32": "F32", "int64": "I64"}[array.dtype.name], "shape": list(array.shape),
                        "data_offsets": [offset, offset + len(data)]}
        offset, blob = offset + len(data), blob + data
    head = json.dumps(header).encode()
    (folder / "alba.safetensors").write_bytes(len(head).to_bytes(8, "little") + head + blob)


class FakePocketGraph:
    """Stands in for one Pocket TTS graph: a fixed function of its feeds."""

    def run(self, _, feeds):
        if "token_ids" in feeds:
            return [feeds["token_ids"][..., None].astype(np.float32) * np.array([0.1, 0.2, 0.3], np.float32)]
        if "sequence" in feeds:
            cache, count = feeds["state_0"], feeds["state_1"]
            seen = np.nansum(feeds["sequence"]) + feeds["text_embeddings"].sum()
            cache = np.where(np.isnan(cache), 0.0, cache).astype(np.float32) + np.float32(seen)
            conditioning = np.full((1, 3), cache.mean() + count[0], dtype=np.float32)
            return [conditioning, np.array([[float(count[0]) - 9.0]], np.float32), cache, count + 1]
        if "c" in feeds:
            return [(feeds["c"][:, :2] * 0.01 + feeds["x"] * 0.5 + feeds["t"]).astype(np.float32)]
        latent = feeds["latent"]
        return [latent.reshape(1, 1, -1) * 0.1 + feeds["state_0"][0], feeds["state_0"] + latent.shape[1]]


def test_pockettts_example_matches_the_adapter(tmp_path, monkeypatch):
    write_pocket_files(tmp_path)
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda repo, name: str(tmp_path / Path(name).name)
    ort = types.ModuleType("onnxruntime")
    ort.InferenceSession = lambda path: FakePocketGraph()
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.chdir(tmp_path)
    block = standalone_example(POCKET_ENTRY, index_files(POCKET_ENTRY), repo_id="Phoonnx/pockettts-x", text=POCKET_TEXT)
    names = {}
    exec(compile(block.removeprefix("```python\n").removesuffix("\n```"), "example", "exec"), names)

    adapter = PocketTTSAdapter(seed=0)
    adapter.configure(types.SimpleNamespace(engine_params={
        "bundle_path": str(tmp_path / "bundle.json"), "tokenizer_path": str(tmp_path / "tokenizer.model"),
        "voice_state_path": str(tmp_path / "alba.safetensors")}))
    adapter.text_conditioner = adapter.flow_lm_flow = adapter.mimi_decoder = FakePocketGraph()
    [ids] = adapter.encode_text(POCKET_TEXT, None, None)
    assert names["ids"] == ids
    result = adapter.synthesize(AdapterSynthesisRequest(np.array([ids]), np.array([len(ids)])), FakePocketGraph())
    assert names["codes"].shape[1] == result.extras["frames"]
    np.testing.assert_array_equal(names["audio"], result.audio)


def test_pockettts_refuses_an_untrained_language():
    with pytest.raises(UnsupportedExample):
        standalone_example({**POCKET_ENTRY, "lang": "nl"}, index_files(POCKET_ENTRY), repo_id="Phoonnx/pockettts-x")
