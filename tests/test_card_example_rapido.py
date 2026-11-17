"""The Rápido card example builds phoonnx's ids, word and frame layout, and flow-matching latents."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from card_examples import UnsupportedExample, index_files, standalone_example  # noqa: E402
from phoonnx.config import VoiceConfig  # noqa: E402
from phoonnx.engines.base import AdapterSynthesisRequest  # noqa: E402
from phoonnx.engines.rapido import RapidoAdapter, layout_boundary_ids  # noqa: E402

HUB = "https://huggingface.co/Phoonnx/rapido-x/resolve/main/"
RAPIDO_ENTRY = {"voice_id": "edwixx/pt-BR-rapido", "engine": "rapido", "lang": "pt-BR",
                "phoneme_type": "graphemes", "alphabet": "unicode",
                "model_url": HUB + "model.onnx", "config_url": HUB + "config.json",
                "aux_model_urls": {"text_encoder_path": HUB + "text_encoder.onnx", "aligner_path": HUB + "aligner.onnx",
                                   "decoder_path": HUB + "decoder.onnx"}}
LETTERS = " abcdefghijklmnopqrstuvwxyzáàâãçéêíóôõúü.,?!;:-'"
RAPIDO_CONFIG = {"phoonnx_version": "1.0", "engine": "rapido", "phoneme_type": "graphemes", "alphabet": "unicode",
                 "lang_code": "pt-BR", "audio": {"sample_rate": 48000}, "num_speakers": 42,
                 "pad": "<pad>", "blank": "_", "bos": "^", "eos": "$", "add_blank_char": False,
                 "add_blank_word": False, "use_eos_bos": False, "blank_at_start": False, "blank_at_end": False,
                 "keep_punctuation": True, "word_sep_token": " ",
                 "phoneme_id_map": {"<pad>": 0, "<unk>": 1, **{c: i + 2 for i, c in enumerate(LETTERS)}},
                 "engine_params": {"latent_mean": [0.1 * i for i in range(64)], "latent_std": [1 + 0.01 * i for i in range(64)]}}
TEXT = "Olá, como você está hoje? Bem;obrigado!"  # a clause mark with no space after it still ends a word


class FakeGraph:
    """Stands in for one Rápido graph: a fixed function of its feeds, recording each call."""

    def __init__(self, name):
        self.name, self.calls = name, []

    def run(self, _, feeds):
        self.calls.append(feeds)
        if "ids" in feeds:
            ids = feeds["ids"].astype(np.float32)
            return [np.repeat(ids[..., None], 4, -1), (ids % 7) * 0.9 + 0.4]
        if "frame_word" in feeds:
            base = feeds["frame_word"] * 2.0 + feeds["frame_pos"] + feeds["word"].sum() * 0.01 + feeds["speaker"][0]
            return [np.repeat(base[..., None], 4, -1).astype(np.float32)]
        if "velocity" in self.name or "condition" in feeds:
            return [(-feeds["x"] * feeds["t"][0] + feeds["condition"].mean(-1, keepdims=True)
                     * feeds["guidance"][0] * 0.1).astype(np.float32)]
        return [feeds["latents"].reshape(1, 1, -1) * 0.01]


def run_example(block, tmp_path, monkeypatch):
    """Run the rendered example with fake graphs and a local config.json; return its names and graphs."""
    (tmp_path / "config.json").write_text(json.dumps(RAPIDO_CONFIG), encoding="utf-8")
    graphs = {}
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda repo, name: str(tmp_path / name)
    ort = types.ModuleType("onnxruntime")
    ort.InferenceSession = lambda path: graphs.setdefault(Path(path).name, FakeGraph(Path(path).name))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.chdir(tmp_path)
    names = {}
    exec(compile(block.removeprefix("```python\n").removesuffix("\n```"), "example", "exec"), names)
    return names, graphs


def test_rapido_example_matches_the_adapter(tmp_path, monkeypatch):
    block = standalone_example(RAPIDO_ENTRY, index_files(RAPIDO_ENTRY), repo_id="Phoonnx/rapido-x", text=TEXT)
    names, graphs = run_example(block, tmp_path, monkeypatch)
    config = VoiceConfig.from_dict(RAPIDO_CONFIG)
    assert names["ids"] == config.tokenizer.tokenize(TEXT.lower())

    adapter = RapidoAdapter()
    adapter.text_encoder, adapter.aligner, adapter.decoder = (FakeGraph(n) for n in ("te", "al", "de"))
    adapter.latent_mean = np.array(RAPIDO_CONFIG["engine_params"]["latent_mean"], dtype=np.float32)
    adapter.latent_std = np.array(RAPIDO_CONFIG["engine_params"]["latent_std"], dtype=np.float32)
    adapter.boundary_ids = layout_boundary_ids(RAPIDO_CONFIG["phoneme_id_map"], " ", "^", "$")
    adapter.num_speakers = 42
    ids = np.array([names["ids"]], dtype=np.int64)
    result = adapter.synthesize(AdapterSynthesisRequest(ids, np.array([ids.shape[1]]), speaker_id=0),
                                FakeGraph("velocity"))
    for key in ("word", "word_start", "frame_word", "frame_pos", "speaker"):
        np.testing.assert_array_equal(graphs["aligner.onnx"].calls[0][key], adapter.aligner.calls[0][key])
    assert names["audio"].shape == result.audio.shape
    np.testing.assert_array_equal(names["audio"], result.audio)
    assert (tmp_path / "output.wav").stat().st_size > 44


def test_rapido_refuses_a_graph_with_other_names():
    files = index_files(RAPIDO_ENTRY)
    files["aligner_path"].update(inputs={"features": 3, "durations": 2}, outputs={"condition": 3})
    with pytest.raises(UnsupportedExample):
        standalone_example(RAPIDO_ENTRY, files, repo_id="Phoonnx/rapido-x")
