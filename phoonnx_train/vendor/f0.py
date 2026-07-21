"""Frame-level fundamental-frequency (F0/pitch) extraction shared by every
training engine that needs a ground-truth pitch target (FastPitch, Mixer-TTS,
the StyleTTS2 aligner/pitch-extractor auxiliaries, OptiSpeech).

Built on ``librosa.pyin`` (probabilistic YIN) so no extra native dependency
is required beyond ``librosa``, which the ``train`` extras already pull in.
"""
import numpy as np
import librosa

_DEFAULT_FMIN = 65.0
_DEFAULT_FMAX = 2093.0


def extract_f0(wav: np.ndarray, sample_rate: int, hop_length: int,
                f_min: float = _DEFAULT_FMIN, f_max: float = _DEFAULT_FMAX) -> np.ndarray:
    """Frame-level fundamental frequency via probabilistic YIN.

    Returns a float64 array of F0 in Hz, one value per hop, 0.0 for
    unvoiced frames — the same shape/convention the previous WORLD
    (dio/harvest + stonemask) path produced.
    """
    if not f_min:
        f_min = _DEFAULT_FMIN
    if not f_max:
        f_max = _DEFAULT_FMAX
    wav = np.asarray(wav, dtype=np.float64)
    f0, _voiced_flag, _voiced_prob = librosa.pyin(
        wav,
        fmin=f_min,
        fmax=f_max,
        sr=sample_rate,
        hop_length=hop_length,
        fill_na=0.0,
    )
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float64)
    return f0
