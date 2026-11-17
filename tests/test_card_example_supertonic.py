"""The SuperTonic card example builds phoonnx's ids, noise and flow-matching steps."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from card_examples import UnsupportedExample, index_files, standalone_example  # noqa: E402
import phoonnx.engines.supertonic as supertonic_engine  # noqa: E402
from phoonnx.engines.base import AdapterSynthesisRequest  # noqa: E402

SUPERTONIC_HUB = "https://huggingface.co/Phoonnx/supertonic-x/resolve/main/"
SUPERTONIC_ENTRY = {"voice_id": "supertonic/F1/en", "engine": "supertonic", "lang": "en",
                    "model_url": SUPERTONIC_HUB + "onnx/vector_estimator.onnx",
                    "aux_model_urls": {k: SUPERTONIC_HUB + v for k, v in {
                        "duration_predictor_path": "onnx/duration_predictor.onnx",
                        "text_encoder_path": "onnx/text_encoder.onnx", "vocoder_path": "onnx/vocoder.onnx",
                        "unicode_indexer_path": "onnx/unicode_indexer.json", "tts_config_path": "onnx/tts.json",
                        "style_path": "voice_styles/F1.json"}.items()}}
SUPERTONIC_TTS = {"ae": {"sample_rate": 44100, "base_chunk_size": 512}, "ttl": {"chunk_compress_factor": 6, "latent_dim": 24}}
SUPERTONIC_TEXT = "Hello there,   my old friend"  # no final mark: the example adds one


class FakeSupertonicGraph:
    """Stands in for one SuperTonic graph: a fixed function of its feeds."""

    def run(self, _, feeds):
        if "style_dp" in feeds:
            return [np.array([feeds["text_ids"].sum() / 10.0 + 1.0], dtype=np.float32)]  # seconds
        if "noisy_latent" in feeds:
            return [(feeds["noisy_latent"] * 0.9 + feeds["text_emb"].mean() + feeds["current_step"][0]
                     / feeds["total_step"][0]).astype(np.float32)]
        if "style_ttl" in feeds:
            return [np.repeat(feeds["text_ids"][..., None].astype(np.float32), 3, -1) + feeds["style_ttl"].mean()]
        return [feeds["latent"].reshape(1, -1)[:, ::7] * 0.01]


def supertonic_files(tmp_path):
    indexer = [-1] * 65536
    for n, c in enumerate("<>/ ,.!?abcdefghijklmnopqrstuvwxyzHḾ"):
        indexer[ord(c)] = n
    style = {k: {"dims": [1, 2, 4], "data": [[[0.1 * i + j for i in range(4)] for j in range(2)]]}
             for k in ("style_ttl", "style_dp")}
    for name, data in (("tts.json", SUPERTONIC_TTS), ("unicode_indexer.json", indexer), ("F1.json", style)):
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    return indexer, style


def test_supertonic_example_matches_the_adapter(tmp_path, monkeypatch):
    indexer, style = supertonic_files(tmp_path)
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda repo, name: str(tmp_path / Path(name).name)
    ort = types.ModuleType("onnxruntime")
    ort.InferenceSession = lambda path: FakeSupertonicGraph()
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.chdir(tmp_path)
    block = standalone_example(SUPERTONIC_ENTRY, index_files(SUPERTONIC_ENTRY), repo_id="Phoonnx/supertonic-x",
                               text=SUPERTONIC_TEXT)
    names = {}
    exec(compile(block.removeprefix("```python\n").removesuffix("\n```"), "example", "exec"), names)

    adapter = supertonic_engine.SuperTonicAdapter()
    adapter.configure(types.SimpleNamespace(engine_params={
        "unicode_indexer_path": str(tmp_path / "unicode_indexer.json"), "tts_config_path": str(tmp_path / "tts.json"),
        "style_path": str(tmp_path / "F1.json")}))
    adapter.duration_predictor = adapter.text_encoder = adapter.vocoder = FakeSupertonicGraph()
    voice = types.SimpleNamespace(config=types.SimpleNamespace(lang_code="en"))
    [ids] = adapter.encode_text(SUPERTONIC_TEXT, voice, None)
    assert names["ids"] == list(ids)
    monkeypatch.setattr(supertonic_engine.np.random, "default_rng",
                        lambda seed=None, rng=np.random.default_rng: rng(0))
    result = adapter.synthesize(AdapterSynthesisRequest(np.array([ids]), np.array([len(ids)])), FakeSupertonicGraph())
    np.testing.assert_array_equal(names["audio"], result.audio)


def test_supertonic_refuses_an_untrained_language():
    with pytest.raises(UnsupportedExample):
        standalone_example({**SUPERTONIC_ENTRY, "lang": "gl"}, index_files(SUPERTONIC_ENTRY), repo_id="Phoonnx/supertonic-x")
