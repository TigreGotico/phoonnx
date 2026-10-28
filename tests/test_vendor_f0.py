"""Tests for the shared F0 extractors (phoonnx_train.vendor.f0), the
in-repo ground-truth pitch path used by the FastPitch/Mixer-TTS/StyleTTS2/
OptiSpeech training engines: ``extract_f0`` (librosa.pyin, default) and
``extract_f0_world`` (pyworld dio/harvest, opt-in via the train-pyworld
extra).
"""
import numpy as np
import pytest

from phoonnx_train.vendor.f0 import (
    EXTRACTOR_TAG,
    extract_f0,
    extract_f0_world,
    get_extractor_tag,
)


def _sine(freq_hz, sample_rate=22050, duration_s=1.0):
    t = np.arange(int(sample_rate * duration_s)) / sample_rate
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float64)


def _noisy_sine(freq_hz, sample_rate=22050, duration_s=1.0, seed=0):
    """WORLD's harvest returns all-unvoiced on a pure synthetic sine (its
    aperiodicity estimator reads a perfectly clean tone as unnatural); a
    lightly amplitude/noise-modulated tone is tracked normally, so this is
    used for harvest-specific tests."""
    rng = np.random.RandomState(seed)
    n = int(sample_rate * duration_s)
    t = np.arange(n) / sample_rate
    sine = np.sin(2 * np.pi * freq_hz * t)
    return (sine * (1.0 + 0.05 * rng.randn(n)) + 0.01 * rng.randn(n)).astype(np.float64)


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


# --------------------------------------------------------- get_extractor_tag
def test_get_extractor_tag_default_is_pyin():
    assert get_extractor_tag() == "pyin" == EXTRACTOR_TAG


@pytest.mark.parametrize("method", ["pyin", "dio", "harvest"])
def test_get_extractor_tag_known_methods(method):
    assert get_extractor_tag(method) == method


def test_get_extractor_tag_unknown_method_raises():
    with pytest.raises(ValueError):
        get_extractor_tag("crepe")


# -------------------------------------------------------- extract_f0_world
def test_extract_f0_world_dio_matches_frequency():
    sample_rate = 22050
    hop_length = 256
    wav = _sine(220.0, sample_rate=sample_rate, duration_s=1.0)
    f0 = extract_f0_world(wav, sample_rate, hop_length, method="dio")
    assert f0.dtype == np.float64
    voiced = f0[f0 > 0]
    assert voiced.size > 0
    assert abs(np.median(voiced) - 220.0) <= 10.0


def test_extract_f0_world_harvest_matches_frequency():
    # harvest reads a perfectly clean synthetic sine as unnatural and
    # returns all-unvoiced; a lightly modulated tone is tracked normally
    sample_rate = 22050
    hop_length = 256
    wav = _noisy_sine(220.0, sample_rate=sample_rate, duration_s=1.0)
    f0 = extract_f0_world(wav, sample_rate, hop_length, method="harvest")
    voiced = f0[f0 > 0]
    assert voiced.size > 0
    assert abs(np.median(voiced) - 220.0) <= 10.0


@pytest.mark.parametrize("freq", [110.0, 220.0, 440.0])
def test_pyin_and_world_agree_on_known_frequency(freq):
    """The default pyin backend and the WORLD (dio + stonemask) backend must
    track the same pitch on a tone of known frequency — a silently wrong F0
    target would corrupt every model trained with it."""
    sample_rate = 22050
    hop_length = 256
    wav = _sine(freq, sample_rate=sample_rate, duration_s=1.0)

    pyin = extract_f0(wav, sample_rate, hop_length)
    world = extract_f0_world(wav, sample_rate, hop_length, method="dio")

    pyin_median = np.median(pyin[pyin > 0])
    world_median = np.median(world[world > 0])

    # each backend is within 1% of the true frequency ...
    assert abs(pyin_median - freq) / freq < 0.01
    assert abs(world_median - freq) / freq < 0.01
    # ... and within 1% of each other
    assert abs(pyin_median - world_median) / freq < 0.01
    # both produce one value per hop, so the F0 track lines up with the mel
    assert abs(len(pyin) - len(world)) <= 1


def test_extract_f0_world_silence_is_all_zero():
    sample_rate = 22050
    hop_length = 256
    wav = np.zeros(sample_rate, dtype=np.float64)
    f0 = extract_f0_world(wav, sample_rate, hop_length, method="dio")
    assert np.all(f0 == 0.0)


def test_extract_f0_world_invalid_method_raises():
    wav = _sine(220.0)
    with pytest.raises(ValueError):
        extract_f0_world(wav, 22050, 256, method="pyin")


def test_extract_f0_world_missing_pyworld_raises_named_import_error(monkeypatch):
    """When pyworld isn't importable, the error message names the
    train-pyworld extra instead of a bare ModuleNotFoundError."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "pyworld":
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    wav = _sine(220.0)
    with pytest.raises(ImportError, match="train-pyworld"):
        extract_f0_world(wav, 22050, 256, method="dio")
