"""Unit tests for the Matcha adapter's vocoder wiring + e2e/two-stage paths."""
import numpy as np
import pytest

from phoonnx.engines import get_adapter, detect_engine
from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.matcha import MatchaAdapter
from phoonnx.engines.vocoders.base import BaseVocoder


class _Named:
    def __init__(self, name):
        self.name = name


class DummySession:
    def __init__(self, input_names, output_count=1):
        self._inputs = [_Named(n) for n in input_names]
        self._outputs = [_Named(f"out{i}") for i in range(output_count)]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs


class FakeVocoder(BaseVocoder):
    name = "fake"

    def __init__(self):
        super().__init__()
        self.calls = []

    def mel_to_audio(self, mel, denoise=False):
        self.calls.append((mel.shape, denoise))
        # one sample per mel frame
        return np.ones(mel.shape[-1], dtype=np.float32)


def _request(n=6):
    ids = np.arange(n, dtype=np.int64)[None, :]
    return AdapterSynthesisRequest(
        phoneme_ids=ids,
        phoneme_lengths=np.array([n], dtype=np.int64),
        speaker_id=0,
        params={},
    )


def test_registered_as_matcha_engine():
    assert isinstance(get_adapter("matcha"), MatchaAdapter)


def test_detect_by_session_inputs():
    sess = DummySession(["x", "x_lengths", "scales"])
    assert MatchaAdapter.detect(session=sess) is True
    assert isinstance(detect_engine(session=sess), MatchaAdapter)


def test_detect_by_engine_params_vocoder():
    cfg = {"engine_params": {"vocoder_path": "v.onnx"}}
    assert MatchaAdapter.detect(config=cfg) is True


def test_build_feed_dict_filters_to_model_inputs():
    adapter = MatchaAdapter()
    sess = DummySession(["x", "x_lengths", "scales", "spks"])
    feed = adapter.build_feed_dict(_request(), sess)
    assert set(feed) == {"x", "x_lengths", "scales", "spks"}
    assert feed["scales"] == pytest.approx([0.667, 1.0], abs=1e-4)
    # a model without spks input drops it
    feed2 = adapter.build_feed_dict(_request(), DummySession(["x", "x_lengths", "scales"]))
    assert "spks" not in feed2


def test_two_stage_uses_vocoder():
    adapter = MatchaAdapter(vocoder=FakeVocoder())
    mel = np.zeros((1, 80, 7), dtype=np.float32)
    mel_lengths = np.array([5], dtype=np.int64)
    result = adapter.parse_outputs([mel, mel_lengths], _request())
    # vocoder ran on the trimmed mel (length 5, not 7)
    assert adapter.vocoder.calls[0][0] == (1, 80, 5)
    assert result.audio.shape[0] == 5


def test_end_to_end_needs_no_vocoder():
    adapter = MatchaAdapter()  # no vocoder configured
    waveform = np.sin(np.linspace(0, 6.28, 100)).astype(np.float32)[None, :]
    result = adapter.parse_outputs([waveform], _request())
    assert result.audio.ndim == 1
    assert result.audio.shape[0] == 100


def test_end_to_end_lengths_first_ordering():
    # BSC fused models emit [mel_lengths, waveform] — audio is NOT outputs[0].
    adapter = MatchaAdapter()
    mel_lengths = np.array([200], dtype=np.int64)
    waveform = np.sin(np.linspace(0, 6.28, 200)).astype(np.float32)[None, :]
    result = adapter.parse_outputs([mel_lengths, waveform], _request())
    assert result.audio.shape[0] == 200


def test_end_to_end_rank3_audio():
    # HiFi-GAN fused export puts audio in a rank-3 tensor (e.g. [1, 1, N]).
    adapter = MatchaAdapter()
    mel_lengths = np.array([200], dtype=np.int64)
    audio = np.ones((1, 1, 200), dtype=np.float32)
    result = adapter.parse_outputs([mel_lengths, audio], _request())
    assert result.audio.ndim == 1
    assert result.audio.shape[0] == 200


def test_two_stage_mel_identified_by_n_mels_axis():
    # Even if mel comes after mel_lengths, it's found by its n_mels axis.
    adapter = MatchaAdapter(vocoder=FakeVocoder())
    mel = np.zeros((1, 80, 9), dtype=np.float32)
    mel_lengths = np.array([9], dtype=np.int64)
    adapter.parse_outputs([mel_lengths, mel], _request())
    assert adapter.vocoder.calls[0][0] == (1, 80, 9)


def test_two_stage_without_vocoder_raises():
    adapter = MatchaAdapter()
    mel = np.zeros((1, 80, 4), dtype=np.float32)
    with pytest.raises(RuntimeError, match="vocoder"):
        adapter.parse_outputs([mel], _request())


# --- phoneme alignment (durations) ---------------------------------------

def test_two_stage_no_durations_by_default():
    """Standard Matcha exports emit [mel, mel_lengths] — no durations."""
    adapter = MatchaAdapter(vocoder=FakeVocoder())
    mel = np.zeros((1, 80, 7), dtype=np.float32)
    mel_lengths = np.array([5], dtype=np.int64)
    result = adapter.parse_outputs(
        [mel, mel_lengths], _request(), output_names=["mel", "mel_lengths"]
    )
    assert "phoneme_id_samples" not in result.extras


def test_two_stage_picks_up_named_durations():
    adapter = MatchaAdapter(vocoder=FakeVocoder())
    mel = np.zeros((1, 80, 7), dtype=np.float32)
    mel_lengths = np.array([7], dtype=np.int64)
    durs = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32)
    result = adapter.parse_outputs(
        [mel, mel_lengths, durs], _request(n=6),
        output_names=["mel", "mel_lengths", "durations"],
    )
    np.testing.assert_array_equal(
        result.extras["phoneme_id_samples"], [1, 2, 3, 4, 5, 6]
    )


def test_end_to_end_has_no_durations_extra():
    """The fused end-to-end path returns early without inspecting durations."""
    adapter = MatchaAdapter()
    waveform = np.sin(np.linspace(0, 6.28, 100)).astype(np.float32)[None, :]
    result = adapter.parse_outputs([waveform], _request(), output_names=["waveform"])
    assert result.extras == {}


def test_configure_from_voice_config_engine_params(monkeypatch):
    built = {}

    def fake_build(model_path=None, vocoder_type=None, config=None, session=None):
        built["args"] = (model_path, vocoder_type, config)
        return FakeVocoder()

    monkeypatch.setattr("phoonnx.engines.matcha.build_vocoder", fake_build)

    class Cfg:
        engine_params = {"vocoder_path": "v.onnx", "vocoder_type": "wavenext",
                         "vocoder_config": {"n_fft": 1024}}

    adapter = MatchaAdapter()
    adapter.configure(Cfg())
    assert built["args"] == ("v.onnx", "wavenext", {"n_fft": 1024})
    assert isinstance(adapter.vocoder, FakeVocoder)
