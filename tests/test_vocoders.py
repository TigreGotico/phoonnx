"""Unit tests for the pluggable vocoder registry (no model downloads)."""
import numpy as np
import pytest

from phoonnx.engines.vocoders import (
    build_vocoder,
    list_vocoders,
    get_vocoder_cls,
)
from phoonnx.engines.vocoders.base import BaseVocoder
from phoonnx.engines.vocoders.vocos import VocosVocoder
from phoonnx.engines.vocoders.raw import (
    RawWaveformVocoder,
    WavenextVocoder,
    HiFiGANVocoder,
)


class _Named:
    def __init__(self, name):
        self.name = name


class DummySession:
    """Minimal stand-in for onnxruntime.InferenceSession."""

    def __init__(self, input_names, output_count, run_fn):
        self._inputs = [_Named(n) for n in input_names]
        self._outputs = [_Named(f"out{i}") for i in range(output_count)]
        self._run_fn = run_fn

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, _none, feed):
        return self._run_fn(feed)


N_FFT, HOP, T = 16, 4, 5
FREQ = N_FFT // 2 + 1


def _vocos_session():
    def run_fn(feed):
        mel = feed["mels"]
        frames = mel.shape[-1]
        shape = (1, FREQ, frames)
        rng = np.linspace(0.1, 0.5, FREQ * frames).reshape(shape).astype(np.float32)
        return [rng, rng * 0.5, rng * 0.25]  # mag, real, imag
    return DummySession(["mels"], 3, run_fn)


def _raw_session():
    def run_fn(feed):
        frames = feed["mels"].shape[-1]
        return [np.sin(np.linspace(0, 6.28, frames * HOP)).astype(np.float32)[None, :]]
    return DummySession(["mels"], 1, run_fn)


def test_registry_has_builtins():
    names = list_vocoders()
    assert {"vocos", "wavenext", "hifigan", "raw"}.issubset(set(names))


def test_get_vocoder_cls():
    assert get_vocoder_cls("vocos") is VocosVocoder
    assert get_vocoder_cls("wavenext") is WavenextVocoder
    with pytest.raises(KeyError):
        get_vocoder_cls("does-not-exist")


def test_build_by_type_no_session_needed():
    assert isinstance(build_vocoder(vocoder_type="vocos"), VocosVocoder)
    assert isinstance(build_vocoder(vocoder_type="wavenext"), WavenextVocoder)
    assert isinstance(build_vocoder(vocoder_type="hifigan"), HiFiGANVocoder)


def test_unknown_type_falls_back_to_detection_then_vocos():
    # No session, unknown type -> auto-detect with nothing -> Vocos fallback
    voc = build_vocoder(vocoder_type="bogus")
    assert isinstance(voc, VocosVocoder)


def test_detect_vocos_vs_raw_by_output_count():
    assert VocosVocoder.detect(session=_vocos_session()) is True
    assert VocosVocoder.detect(session=_raw_session()) is False
    assert RawWaveformVocoder.detect(session=_raw_session()) is True
    assert RawWaveformVocoder.detect(session=_vocos_session()) is False


def test_build_autodetect_from_session():
    voc = build_vocoder(session=_vocos_session())
    assert isinstance(voc, VocosVocoder)
    voc = build_vocoder(session=_raw_session())
    assert isinstance(voc, RawWaveformVocoder)


def test_vocos_mel_to_audio_shape_and_finite():
    voc = VocosVocoder(session=_vocos_session(),
                       config={"n_fft": N_FFT, "hop_length": HOP})
    mel = np.zeros((1, 80, T), dtype=np.float32)
    audio = voc.mel_to_audio(mel, denoise=False)
    assert audio.ndim == 1
    # center=True padding (n_fft//2 each side) is trimmed on the inverse
    assert audio.shape[0] == (T - 1) * HOP + N_FFT - 2 * (N_FFT // 2)
    assert np.all(np.isfinite(audio))
    assert voc.supports_denoise is True


def test_vocos_denoise_runs():
    voc = VocosVocoder(session=_vocos_session(),
                       config={"n_fft": N_FFT, "hop_length": HOP})
    mel = np.zeros((1, 80, T), dtype=np.float32)
    audio = voc.mel_to_audio(mel, denoise=True)
    assert np.all(np.isfinite(audio))


def test_raw_mel_to_audio_passthrough():
    voc = WavenextVocoder(session=_raw_session())
    mel = np.zeros((1, 80, T), dtype=np.float32)
    audio = voc.mel_to_audio(mel)
    assert audio.ndim == 1
    assert audio.shape[0] == T * HOP
    assert voc.supports_denoise is False


def test_vocos_nested_config_layout():
    voc = VocosVocoder(
        session=_vocos_session(),
        config={"feature_extractor": {"init_args": {"n_fft": N_FFT, "hop_length": HOP}}},
    )
    params = voc._stft_params()
    assert params == {"n_fft": N_FFT, "hop_length": HOP}
