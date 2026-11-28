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


def test_unknown_type_names_the_type_and_the_known_types():
    with pytest.raises(ValueError) as err:
        build_vocoder(vocoder_type="griffin_lim")
    message = str(err.value)
    assert "'griffin_lim'" in message
    for name in ("griffinlim", "hifigan", "melgan", "raw", "vocos", "wavenext"):
        assert f"'{name}'" in message


def test_unknown_type_in_the_config_is_an_error():
    with pytest.raises(ValueError, match="'bogus'"):
        build_vocoder(config={"vocoder_type": "bogus"})


def test_alvocat_selects_vocos():
    assert isinstance(build_vocoder(vocoder_type="alvocat"), VocosVocoder)


def test_no_type_and_no_session_falls_back_to_vocos():
    assert isinstance(build_vocoder(), VocosVocoder)


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


def _seeded_spectrogram(seed, batch, n_fft, frames):
    """mag * (x + 1j*y) from float32 tensors, the way ``mel_to_audio`` builds it."""
    rng = np.random.default_rng(seed)
    mag, x, y = (rng.standard_normal((batch, n_fft // 2 + 1, frames)).astype(np.float32)
                 for _ in range(3))
    return mag * (x + 1j * y)


@pytest.mark.parametrize("n_fft,hop,batch,frames", [
    (16, 4, 1, 5),
    (1024, 256, 2, 7),     # the shipped Vocos geometry
    (16, 5, 1, 6),         # hop does not divide n_fft
    (12, 12, 1, 3),        # no overlap
    (8, 3, 2, 1),          # a single frame
    (10, 16, 1, 4),        # hop longer than the window leaves gaps
])
def test_vocos_overlap_add_is_bit_identical_to_the_per_sample_sum(n_fft, hop, batch, frames):
    voc = VocosVocoder(session=_vocos_session(), config={"n_fft": n_fft, "hop_length": hop})
    spectrogram = _seeded_spectrogram(n_fft + hop + frames, batch, n_fft, frames)
    got = voc._istft_overlap_add(spectrogram)

    # windowed inverse FFT, then each sample summed in Python floats in frame order
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)
    ifft = np.fft.irfft(spectrogram, n=n_fft, axis=1) * window[None, :, None]
    want = np.empty((batch, (frames - 1) * hop + n_fft))
    for b in range(batch):
        for i in range(want.shape[1]):
            total, envelope = 0.0, 0.0
            for t in range(frames):
                k = i - t * hop
                if 0 <= k < n_fft:
                    total += float(ifft[b, k, t])
                    envelope += float(window[k]) * float(window[k])
            want[b, i] = total / max(envelope, 1e-11)
    pad = n_fft // 2
    if want.shape[1] > 2 * pad:
        want = want[:, pad:-pad]
    want = want.astype(np.float32)
    assert got.dtype == np.float32 and got.shape == want.shape
    assert np.array_equal(got, want)
    assert got.tobytes() == want.tobytes()


@pytest.mark.parametrize("win,hop,frames", [(1024, 256, 9), (16, 5, 6), (10, 16, 4)])
def test_overlap_add_keeps_the_float64_sums_of_a_frame_by_frame_loop(win, hop, frames):
    """Checked before the float32 cast, where a changed summation order would show."""
    from phoonnx.engines.vocoders.vocos import _overlap_add
    stack = np.random.default_rng(win * hop).standard_normal((2, frames, win)) * 1e3
    got = _overlap_add(stack, hop)
    want = np.empty((2, (frames - 1) * hop + win))
    for b in range(2):
        for i in range(want.shape[1]):
            total = 0.0
            for t in range(frames):
                if 0 <= i - t * hop < win:
                    total += float(stack[b, t, i - t * hop])
            want[b, i] = total
    assert got.tobytes() == want.tobytes()
