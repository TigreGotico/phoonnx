"""The F5-TTS card example builds phoonnx's text ids, prompt and Euler loop."""
import json
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from card_examples import UnsupportedExample, index_files, standalone_example  # noqa: E402
from phoonnx.config import VoiceConfig  # noqa: E402
from phoonnx.engines.base import AdapterSynthesisRequest  # noqa: E402
from phoonnx.engines.f5tts import F5TTSAdapter  # noqa: E402
from phoonnx.voice import TTSVoice  # noqa: E402

F5_HUB = "https://huggingface.co/Phoonnx/f5tts-x/resolve/main/"
F5_ENTRY = {"voice_id": "habibi/ar-unified", "engine": "f5tts", "lang": "ar", "phoneme_type": "graphemes",
            "alphabet": "unicode", "model_url": F5_HUB + "model.onnx", "config_url": F5_HUB + "config.json",
            "aux_model_urls": {"preprocess_path": F5_HUB + "F5_Preprocess.onnx", "decode_path": F5_HUB + "F5_Decode.onnx"}}
F5_CONFIG = {"phoonnx_version": "1.0", "engine": "f5tts", "phoneme_type": "graphemes", "alphabet": "unicode",
             "lang_code": "en", "audio": {"sample_rate": 24000}, "add_blank_char": False, "blank_at_start": False,
             "blank_at_end": False, "use_eos_bos": False,
             "engine_params": {"nfe": 6, "speed": 1.0, "dialect": "EGY"},
             "phoneme_id_map": {c: i for i, c in enumerate([*" !,.?abcdefghijklmnopqrstuvwxyzABC⓪①②③④⑤⑥⑦〈〉", "er", "le",
                                                             "ang4"])}}
F5_REFERENCE_TEXT = "Some call me nature, others call me mother nature!"
F5_TEXT = "Hello there, my old friend."


class FakeF5Graph:
    """Stands in for one F5-TTS graph: a fixed function of its feeds."""

    def __init__(self, inputs=()):
        self._inputs = [types.SimpleNamespace(name=n, type="tensor(int16)") for n in inputs]

    def get_inputs(self):
        return self._inputs

    def run(self, _, feeds):
        if "max_duration" in feeds:
            n = int(feeds["max_duration"][0])
            base = feeds["audio"].astype(np.float32).mean() + feeds["text_ids"].sum()
            rope = [np.full((1, n, 2), base + k, dtype=np.float32) for k in range(4)]
            return [np.full((1, n, 3), base, dtype=np.float32), *rope, np.full((1, n, 3), 2.0, np.float32),
                    np.zeros((1, n, 3), np.float32), np.array(n // 3, dtype=np.int64)]
        if "denoised" in feeds:
            return [np.round(feeds["denoised"][:, int(feeds["ref_signal_len"]):] * 10).astype(np.int16)]
        values = list(feeds.values())
        return [values[0] * 0.5 + values[1].mean() + values[5] + values[7][0], values[7] + 1]


def test_f5tts_example_matches_the_adapter(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps(F5_CONFIG), encoding="utf-8")
    (tmp_path / "reference.txt").write_text(F5_REFERENCE_TEXT, encoding="utf-8")
    pcm = (np.sin(np.arange(24000) / 7.0) * 12000).astype(np.int16)  # 1 s: the length estimate shows a 3-id end
    with wave.open(str(tmp_path / "reference.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm.tobytes())
    names_in = ["noise", "rope_cos_q", "rope_sin_q", "rope_cos_k", "rope_sin_k", "cat_mel_text",
                "cat_mel_text_drop", "time_step.1"]
    graphs = {"F5_Preprocess.onnx": FakeF5Graph(["audio"]), "model.onnx": FakeF5Graph(names_in),
              "F5_Decode.onnx": FakeF5Graph()}
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda repo, name: str(tmp_path / name)
    ort = types.ModuleType("onnxruntime")
    ort.InferenceSession = lambda path: graphs[Path(path).name]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.chdir(tmp_path)
    block = standalone_example(F5_ENTRY, index_files(F5_ENTRY), repo_id="Phoonnx/f5tts-x", text=F5_TEXT)
    names = {}
    exec(compile(block.removeprefix("```python\n").removesuffix("\n```"), "example", "exec"), names)

    voice = TTSVoice(session=graphs["model.onnx"], config=VoiceConfig.from_dict(F5_CONFIG))
    prompt = voice._prompt_token_ids(F5_REFERENCE_TEXT)
    text_ids = voice._prompt_token_ids(F5_TEXT)
    adapter = F5TTSAdapter()
    adapter.preprocess, adapter.decode = graphs["F5_Preprocess.onnx"], graphs["F5_Decode.onnx"]
    adapter.configure(types.SimpleNamespace(engine_params=F5_CONFIG["engine_params"], tokenizer=voice.config.tokenizer))
    exact = ((pcm.astype(np.float64) + 0.5 * np.sign(pcm)) / 32767).astype(np.float32)
    result = adapter.synthesize(AdapterSynthesisRequest(
        np.array([text_ids]), np.array([len(text_ids)]),
        params={"reference_audio": (exact, 24000), "prompt_tokens": prompt, "nfe": 6,
                "speaker_reference_text": F5_REFERENCE_TEXT}), graphs["model.onnx"])
    id_map = F5_CONFIG["phoneme_id_map"]
    # ⑥〈 reference, its end ".  " (it ends in "!"), text 〉
    assert names["ids"] == [id_map["⑥"], id_map["〈"], *prompt, id_map["."], id_map[" "], id_map[" "],
                            *text_ids, id_map["〉"]]
    np.testing.assert_array_equal(names["audio"], result.audio)


def test_f5tts_refuses_a_preprocess_graph_with_other_outputs():
    files = index_files(F5_ENTRY)
    files["preprocess_path"].update(inputs={"audio": 3, "text_ids": 2, "max_duration": 1},
                                    outputs={"noise": 3, "cat_mel_text": 3, "ref_signal_len": 0})
    with pytest.raises(UnsupportedExample):
        standalone_example(F5_ENTRY, files, repo_id="Phoonnx/f5tts-x")
