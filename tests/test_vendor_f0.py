"""Tests for the shared pyin-based F0 extractor
(phoonnx_train.vendor.f0.extract_f0), the in-repo ground-truth pitch path
used by the FastPitch/Mixer-TTS/StyleTTS2/OptiSpeech training engines.
"""
import numpy as np

from phoonnx_train.vendor.f0 import extract_f0


def _sine(freq_hz, sample_rate=22050, duration_s=1.0):
    t = np.arange(int(sample_rate * duration_s)) / sample_rate
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float64)


def test_extract_f0_sine_wave_matches_frequency():
    sample_rate = 22050
    hop_length = 256
    wav = _sine(220.0, sample_rate=sample_rate, duration_s=1.0)
    f0 = extract_f0(wav, sample_rate, hop_length)
    assert f0.dtype == np.float64
    voiced = f0[f0 > 0]
    assert voiced.size > 0
    assert abs(np.median(voiced) - 220.0) <= 10.0


def test_extract_f0_silence_is_all_zero():
    sample_rate = 22050
    hop_length = 256
    wav = np.zeros(sample_rate, dtype=np.float64)
    f0 = extract_f0(wav, sample_rate, hop_length)
    assert np.all(f0 == 0.0)
    assert not np.any(np.isnan(f0))


def test_extract_f0_frame_count_matches_hop_grid():
    sample_rate = 22050
    hop_length = 256
    wav = _sine(220.0, sample_rate=sample_rate, duration_s=1.0)
    f0 = extract_f0(wav, sample_rate, hop_length)
    # librosa.pyin frames the signal on the same hop grid as other
    # librosa frame-level features (STFT-equivalent frame count)
    expected_frames = 1 + len(wav) // hop_length
    assert abs(len(f0) - expected_frames) <= 1
