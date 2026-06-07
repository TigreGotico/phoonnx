"""Tests for the StyleTTS2 / Kokoro adapter."""
import numpy as np
import pytest

from phoonnx.engines import get_adapter, detect_engine
from phoonnx.engines.styletts2 import StyleTTS2Adapter
from phoonnx.engines.base import AdapterSynthesisRequest


class _In:
    def __init__(self, name): self.name = name


class _Sess:
    def __init__(self, names): self._i = [_In(n) for n in names]
    def get_inputs(self): return self._i


def _req(n=5, **p):
    return AdapterSynthesisRequest(phoneme_ids=np.arange(1, n + 1, dtype=np.int64)[None, :],
                                   phoneme_lengths=np.array([n], np.int64), params=p)


def test_registered_and_detect():
    assert isinstance(get_adapter("styletts2"), StyleTTS2Adapter)
    assert StyleTTS2Adapter.detect(config={"engine": "styletts2"}) is True
    assert StyleTTS2Adapter.detect(config={"engine": "kokoro"}) is True  # same family
    assert isinstance(detect_engine(config={"engine": "kokoro"}), StyleTTS2Adapter)


def test_styletts2_feed_pads_and_filters():
    # baked-ref StyleTTS2: input_ids + attention_mask + speed (no style)
    sess = _Sess(["input_ids", "attention_mask", "speed"])
    feed = StyleTTS2Adapter().build_feed_dict(_req(5, speed=1.2), sess)
    assert set(feed) == {"input_ids", "attention_mask", "speed"}
    assert feed["input_ids"].shape == (1, 7)        # $-padded both ends
    assert feed["input_ids"][0, 0] == 0 and feed["input_ids"][0, -1] == 0
    assert feed["speed"][0] == pytest.approx(1.2)
    assert feed["attention_mask"].shape == (1, 7)


def test_kokoro_style_pack_length_indexed():
    # Kokoro: a [510, 256] style pack, indexed by token length
    pack = np.arange(510 * 256, dtype=np.float32).reshape(510, 256)
    sess = _Sess(["input_ids", "style", "speed"])
    feed = StyleTTS2Adapter(style_pack=pack).build_feed_dict(_req(5), sess)
    assert "style" in feed and feed["style"].shape == (1, 256)
    # 5 tokens -> +2 pad = 7 -> style_pack[7]
    assert np.allclose(feed["style"][0], pack[7])
    assert "attention_mask" not in feed   # filtered (not a model input)


def test_parse_outputs_picks_waveform():
    r = StyleTTS2Adapter().parse_outputs([np.zeros((1, 512, 8), np.float32), np.ones(20000, np.float32)], _req())
    assert r.audio.ndim == 1 and r.audio.size == 20000


def test_configure_loads_style_pack_from_engine_params(tmp_path):
    """Kokoro: the manager downloads a style blob; configure() reshapes it to [N,256]."""
    import numpy as np
    blob = tmp_path / "style.bin"
    np.arange(510 * 256, dtype=np.float32).tofile(blob)

    class _Cfg:
        engine_params = {"style_path": str(blob)}

    a = StyleTTS2Adapter()
    a.configure(_Cfg())
    assert a.style_pack.shape == (510, 256)
