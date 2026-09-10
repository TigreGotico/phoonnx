"""Frame-level fundamental-frequency (F0/pitch) extraction shared by every
training engine that needs a ground-truth pitch target (FastPitch, Mixer-TTS,
the StyleTTS2 aligner/pitch-extractor auxiliaries, OptiSpeech).

Two backends:

- ``extract_f0`` (default): ``librosa.pyin`` (probabilistic YIN). No extra
  native dependency beyond ``librosa``, which the ``train`` extras already
  pull in.
- ``extract_f0_world`` (opt-in): WORLD ``dio``/``harvest`` + ``stonemask``
  via ``pyworld``. ~50x faster than pyin at dataset-preprocessing time, at
  the cost of an extra native dependency (the ``train-pyworld`` extra).

``get_extractor_tag(method)`` identifies the extraction method for
cache-key purposes: every module that persists an F0 sidecar (``.npy``
cache keyed by audio content hash / mel-cache path) must fold this tag
into the cache filename, so a change of extraction method naturally
invalidates every previously-cached F0 track instead of silently mixing
tracks computed by different methods. Derive the tag from this function
rather than hardcoding the string.
"""
import typing

if typing.TYPE_CHECKING:
    import numpy as np

_METHODS = ("pyin", "dio", "harvest")

_DEFAULT_FMIN = 65.0
_DEFAULT_FMAX = 2093.0


def get_extractor_tag(method: str = "pyin") -> str:
    """Cache-key tag for an F0 extraction method (``"pyin"``, ``"dio"`` or
    ``"harvest"``). The single source of truth every F0 cache-key site
    derives from — bump/extend ``_METHODS`` here, not at each call site."""
    if method not in _METHODS:
        raise ValueError(f"unknown F0 extraction method {method!r} — expected one of {_METHODS}")
    return method


# Default-method tag, kept for call sites/tests that don't need to be
# method-aware — equivalent to ``get_extractor_tag()``.
EXTRACTOR_TAG = get_extractor_tag()


def extract_f0(wav: "np.ndarray", sample_rate: int, hop_length: int,
                f_min: float = _DEFAULT_FMIN, f_max: float = _DEFAULT_FMAX) -> "np.ndarray":
    """Frame-level fundamental frequency via probabilistic YIN.

    Returns a float64 array of F0 in Hz, one value per hop, 0.0 for
    unvoiced frames — the same shape/convention the WORLD (dio/harvest +
    stonemask) path produces.
    """
    import numpy as np
    import librosa

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


def extract_f0_world(wav: "np.ndarray", sample_rate: int, hop_length: int,
                      f_min: "typing.Optional[float]" = None,
                      f_max: "typing.Optional[float]" = None,
                      method: str = "dio") -> "np.ndarray":
    """Frame-level fundamental frequency via WORLD (``pyworld``) — opt-in,
    ~50x faster than :func:`extract_f0` at dataset-preprocessing time.

    ``method`` selects ``pyworld.dio`` (fast) or ``pyworld.harvest`` (slower,
    generally more robust); either way the raw estimate is refined with
    ``pyworld.stonemask``, at ``frame_period = hop_length / sample_rate *
    1000`` ms (matched to the mel hop), exactly as the previous pyworld-based
    pipeline did. ``f_min``/``f_max`` are accepted for call-site symmetry
    with :func:`extract_f0` but are not passed to WORLD, which the original
    pipeline never bounded by frequency.

    Requires the ``train-pyworld`` extra; raises ``ImportError`` naming it
    when ``pyworld`` isn't installed.
    """
    if method not in ("dio", "harvest"):
        raise ValueError(f"extract_f0_world method must be 'dio' or 'harvest', got {method!r}")

    try:
        import pyworld as pw
    except ImportError as e:
        raise ImportError(
            "pyworld is required for WORLD F0 extraction (f0_method="
            f"{method!r}) — install it via the 'train-pyworld' extra "
            "(pip install phoonnx[train-pyworld])."
        ) from e

    import numpy as np

    wav = np.asarray(wav, dtype=np.float64)
    frame_period = hop_length / sample_rate * 1000.0
    extraction_func = getattr(pw, method)
    f0, timeaxis = extraction_func(wav, sample_rate, frame_period=frame_period)
    f0 = pw.stonemask(wav, f0, timeaxis, sample_rate)
    return f0.astype(np.float64)
