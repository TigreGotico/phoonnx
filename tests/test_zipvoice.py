import numpy as np
import pytest

from phoonnx.engines.base import (AdapterSynthesisRequest, AdapterSynthesisResult,
                                   BaseOnnxAdapter)
from phoonnx.engines.zipvoice import ZipVoiceAdapter, _resample


def _req(**params):
    return AdapterSynthesisRequest(phoneme_ids=np.array([[1, 2, 3]], np.int64),
                                   phoneme_lengths=np.array([3], np.int64),
                                   speaker_id=0, language_id=0, params=params)


def test_zipvoice_registered():
    from phoonnx.engines import list_engines
    assert "zipvoice" in list_engines()


def test_zipvoice_detect():
    assert ZipVoiceAdapter.detect({"engine": "zipvoice"})
    assert not ZipVoiceAdapter.detect({"engine": "vits"})
    assert not ZipVoiceAdapter.detect(None)


def test_zipvoice_requires_aux_graphs():
    # not configured -> clear error rather than an opaque None.run crash
    with pytest.raises(RuntimeError):
        ZipVoiceAdapter().synthesize(_req(reference_audio=(np.zeros(100, np.float32), 24000),
                                          prompt_tokens=[1, 2]), None)


def test_zipvoice_requires_reference():
    ad = ZipVoiceAdapter()
    ad.text_encoder = ad.mel = ad.vocoder = object()   # pretend configured
    with pytest.raises(RuntimeError):
        ad.synthesize(_req(), None)                    # no reference_audio / prompt_tokens


def test_synthesize_hook_default_is_single_pass():
    """The base synthesize() hook = build_feed_dict -> run -> parse_outputs."""
    class _A(BaseOnnxAdapter):
        def build_feed_dict(self, request, session): return {"x": np.array([3.0])}
        def parse_outputs(self, outputs, request): return AdapterSynthesisResult(audio=outputs[0])
        def default_params(self): return {}

    class _S:
        def run(self, names, feed): return [feed["x"] * 2]

    res = _A().synthesize(_req(), _S())
    assert res.audio[0] == 6.0


def test_resample():
    a = np.zeros(1000, np.float32)
    assert _resample(a, 24000, 24000) is a                          # no-op
    assert abs(len(_resample(a, 22050, 24000)) - 1000 * 24000 / 22050) <= 2


def test_load_reference_audio(tmp_path):
    """The cloning UX: speaker_reference (wav path or tuple) -> (audio, sr)."""
    import wave
    from phoonnx.voice import _load_reference_audio
    a, sr = _load_reference_audio((np.zeros(100, np.float32), 16000))   # tuple passthrough
    assert sr == 16000 and a.shape == (100,) and a.dtype == np.float32
    p = tmp_path / "r.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050)
        w.writeframes(np.zeros(2205, np.int16).tobytes())
    a, sr = _load_reference_audio(str(p))                              # wav path
    assert sr == 22050 and a.ndim == 1
