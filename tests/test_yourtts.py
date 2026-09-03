"""Tests for the YourTTS adapter (multilingual VITS + d-vector)."""
import numpy as np
from phoonnx.engines import get_adapter
from phoonnx.engines.yourtts import YourTTSAdapter
from phoonnx.engines.base import AdapterSynthesisRequest


class _In:
    def __init__(self, name): self.name = name
class _Sess:
    def __init__(self, names): self._i = [_In(n) for n in names]
    def get_inputs(self): return self._i

def _req(**p):
    return AdapterSynthesisRequest(phoneme_ids=np.arange(1, 6, dtype=np.int64)[None, :],
                                   phoneme_lengths=np.array([5], np.int64), language_id=None, params=p)

SESS = _Sess(["input", "input_lengths", "scales", "d_vector", "langid"])


def test_registered_and_detect():
    assert isinstance(get_adapter("yourtts"), YourTTSAdapter)
    assert YourTTSAdapter.detect(config={"engine": "yourtts"}) is True


def test_bundled_dvector_and_langid_fed():
    a = YourTTSAdapter(d_vector=np.ones(512, np.float32), langid=2)
    feed = a.build_feed_dict(_req(), SESS)
    assert feed["d_vector"].shape == (1, 512)
    assert int(feed["langid"][0]) == 2
    assert feed["scales"].shape == (3,)


def test_request_dvector_overrides_bundled():
    a = YourTTSAdapter(d_vector=np.zeros(512, np.float32), langid=0)
    feed = a.build_feed_dict(_req(d_vector=np.full(512, 0.5, np.float32), langid=1), SESS)
    assert np.allclose(feed["d_vector"][0], 0.5)   # clone d-vector wins
    assert int(feed["langid"][0]) == 1


def test_configure_loads_from_engine_params():
    class _Cfg:
        engine_params = {"d_vector": [0.1] * 512, "langid": 2}
    a = YourTTSAdapter()
    a.configure(_Cfg())
    assert a.d_vector.shape == (1, 512) and a.langid == 2


def test_clone_from_reference_uses_encoder():
    class _Enc:
        def encode(self, audio, sr): return np.full(512, 0.3, np.float32)
    a = YourTTSAdapter(d_vector=np.zeros(512, np.float32), speaker_encoder=_Enc())
    feed = a.build_feed_dict(_req(reference_audio=(np.zeros(16000, np.float32), 16000)), SESS)
    assert np.allclose(feed["d_vector"][0], 0.3)   # cloned vector, not the bundled zeros


def test_speaker_encoder_registry():
    from phoonnx.engines.speaker_encoders import list_speaker_encoders, get_speaker_encoder
    from phoonnx.engines.speaker_encoders.coqui_resnet import CoquiResNetSpeakerEncoder
    assert "coqui_resnet" in list_speaker_encoders()
    assert get_speaker_encoder("coqui_resnet") is CoquiResNetSpeakerEncoder


def test_speaker_reference_loads_and_flows(tmp_path):
    """SynthesisConfig.speaker_reference -> a wav is loaded to (audio, sr)."""
    import wave, numpy as np
    from phoonnx.reference_audio import load_reference_audio
    p = tmp_path / "ref.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050)
        w.writeframes((np.zeros(2205, np.float32) * 32767).astype("<i2").tobytes())
    audio, sr = load_reference_audio(str(p))
    assert sr == 22050 and audio.dtype == np.float32 and audio.ndim == 1
    # tuple passthrough
    a2, s2 = load_reference_audio((np.zeros(100, np.float32), 16000))
    assert s2 == 16000 and a2.shape == (100,)


# --- phoneme alignment (durations) ---------------------------------------

def test_parse_outputs_no_durations_single_output():
    a = YourTTSAdapter()
    audio = np.zeros((1, 1, 400), np.float32)
    res = a.parse_outputs([audio], _req(), output_names=["output"])
    assert "phoneme_id_samples" not in res.extras


def test_parse_outputs_picks_up_named_durations():
    a = YourTTSAdapter()
    audio = np.zeros((1, 1, 400), np.float32)
    durs = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
    res = a.parse_outputs(
        [audio, durs], _req(), output_names=["output", "w_ceil"]
    )
    np.testing.assert_array_equal(res.extras["phoneme_id_samples"], [1, 2, 3, 4, 5])


def test_reference_audio_outranks_bundled_dvector():
    class _Enc:
        def encode(self, audio, sr): return np.full(512, 0.9, np.float32)
    a = YourTTSAdapter(d_vector=np.zeros(512, np.float32), speaker_encoder=_Enc())
    # both a bundled/explicit d_vector AND a reference present -> reference wins (clone)
    feed = a.build_feed_dict(_req(d_vector=np.full(512, 0.1, np.float32),
                                  reference_audio=(np.zeros(16000, np.float32), 16000)), SESS)
    assert np.allclose(feed["d_vector"][0], 0.9)


def test_missing_d_vector_raises_a_clear_error():
    """d_vector is YourTTS's only speaker conditioning and is always a declared
    input; omitting it left onnxruntime to report a missing required input,
    which says nothing about what the voice actually lacks."""
    import types, numpy as np, pytest
    from phoonnx.engines.yourtts import YourTTSAdapter
    from phoonnx.engines.base import AdapterSynthesisRequest
    sess = type("S", (), {"get_inputs": lambda self: [
        types.SimpleNamespace(name=n) for n in ("input", "input_lengths", "scales", "langid", "d_vector")]})()
    req = AdapterSynthesisRequest(phoneme_ids=np.array([[1, 2]], dtype=np.int64),
                                  phoneme_lengths=np.array([2], dtype=np.int64), params={})
    with pytest.raises(ValueError, match="d-vector"):
        YourTTSAdapter().build_feed_dict(req, sess)
