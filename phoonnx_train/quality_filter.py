"""Generic per-sample quality-metric filtering for training data preprocessing.

Datasets fed into phoonnx do not reliably carry precomputed quality columns
(UTMOS, DNSMOS, ...), and even where an upstream pipeline happens to have
populated them we don't want preprocessing to silently depend on that. So
every metric named on the CLI is instead computed on demand, once per
sample, from its raw audio/text, by a small registry of named "scorers".

New metrics are added by registering a new scorer function here; the
``--filter`` CLI flag in ``phoonnx_train/preprocess.py`` stays generic and
never needs a dedicated flag per metric.

Scorers are evaluated cheapest-first (arithmetic, then heuristic, then
model-based) and short-circuit per sample: once a sample fails one filter it
is dropped without paying for any remaining (possibly expensive) scorers.
"""
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

_LOGGER = logging.getLogger(__name__)

# A scorer computes one metric from a sample's raw audio/text.
# Signature: (audio: np.ndarray, sr: int, text: str, duration: float) -> float
Scorer = Callable[[object, int, str, float], float]

_SCORER_REGISTRY: Dict[str, Scorer] = {}
# Registration order doubles as cheapest-first evaluation order.
_SCORER_ORDER: List[str] = []


def register_scorer(name: str, fn: Scorer) -> None:
    """Register a named on-demand quality scorer.

    Args:
        name: filter column name (e.g. "utmos", "wpm").
        fn: callable computing the metric from (audio, sr, text, duration).
    """
    _SCORER_REGISTRY[name] = fn
    if name not in _SCORER_ORDER:
        _SCORER_ORDER.append(name)


def known_scorers() -> List[str]:
    """Names of all registered scorers, cheapest-first."""
    return list(_SCORER_ORDER)


@dataclass
class FilterSpec:
    """One `column:min:max` filter. min/max of None means unbounded."""
    column: str
    min: Optional[float] = None
    max: Optional[float] = None

    def passes(self, value: float) -> bool:
        if self.min is not None and value < self.min:
            return False
        if self.max is not None and value > self.max:
            return False
        return True


def parse_filter_spec(spec: str) -> FilterSpec:
    """Parse a `column:min:max` CLI value into a FilterSpec.

    Either bound may be empty (unbounded on that side), e.g. "utmos:3.0:"
    keeps utmos >= 3.0 with no upper bound.

    Raises:
        ValueError: if spec is not exactly three colon-separated fields, or
            a non-empty bound is not a number.
    """
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"invalid --filter spec {spec!r}: expected 'column:min:max' "
            "(min/max may be empty for unbounded)"
        )
    column, raw_min, raw_max = (p.strip() for p in parts)
    if not column:
        raise ValueError(f"invalid --filter spec {spec!r}: column name is empty")
    min_val = float(raw_min) if raw_min else None
    max_val = float(raw_max) if raw_max else None
    return FilterSpec(column=column, min=min_val, max=max_val)


def _ordered_specs(specs: List[FilterSpec]) -> List[FilterSpec]:
    """Specs sorted cheapest-scorer-first, unknown columns last (they warn
    once up-front and are skipped for every sample anyway)."""
    order = {name: i for i, name in enumerate(_SCORER_ORDER)}
    return sorted(specs, key=lambda s: order.get(s.column, len(order)))


def apply_quality_filters(
    samples: List[object],
    specs: List[FilterSpec],
    audio_path_fn: Callable[[object], Union[str, Path]],
    text_fn: Callable[[object], str],
    audio_loader: Optional[Callable[[Union[str, Path]], Tuple[object, int, float]]] = None,
) -> Tuple[List[object], Dict[str, int]]:
    """Filter samples by on-demand-computed quality scorers.

    Args:
        samples: opaque per-sample objects (e.g. Utterance instances).
        specs: filter specs to apply (AND semantics: a sample must pass all
            of them to be kept).
        audio_path_fn: samples[i] -> path to its audio file.
        text_fn: samples[i] -> its transcript text.
        audio_loader: path -> (audio, sample_rate, duration_seconds). Only
            called lazily, and at most once per sample, when a filter that
            actually needs audio is evaluated. Defaults to a librosa-backed
            16kHz mono loader.

    Returns:
        (kept_samples, dropped_counts) where dropped_counts maps each
        applied filter's column name to how many samples it dropped.
        Unknown/unregistered column names are warned about once and
        excluded from dropped_counts entirely (they are skipped, not
        applied).
    """
    known: List[FilterSpec] = []
    for spec in specs:
        if spec.column not in _SCORER_REGISTRY:
            _LOGGER.warning(
                "unknown quality filter column %r (known: %s); skipping this filter",
                spec.column, ", ".join(_SCORER_ORDER) or "<none registered>",
            )
            continue
        known.append(spec)

    dropped: Dict[str, int] = {spec.column: 0 for spec in known}
    if not known:
        return list(samples), dropped

    ordered = _ordered_specs(known)
    loader = audio_loader or _default_audio_loader
    kept: List[object] = []

    for sample in samples:
        audio = None
        sr = 0
        duration = 0.0
        audio_loaded = False
        text = text_fn(sample)
        sample_dropped = False

        try:
            for spec in ordered:
                scorer = _SCORER_REGISTRY[spec.column]
                if spec.column != "wpm" and not audio_loaded:
                    audio, sr, duration = loader(audio_path_fn(sample))
                    audio_loaded = True
                elif spec.column == "wpm" and not audio_loaded:
                    # wpm only needs duration; fetch it without decoding audio
                    # unless a later filter already forced a full decode.
                    duration = _duration_only(audio_path_fn(sample))

                value = scorer(audio, sr, text, duration)
                if not spec.passes(value):
                    dropped[spec.column] += 1
                    sample_dropped = True
                    break
        except Exception:
            _LOGGER.exception(
                "quality filtering failed for %s; dropping sample",
                audio_path_fn(sample),
            )
            sample_dropped = True

        if not sample_dropped:
            kept.append(sample)

    return kept, dropped


def _duration_only(path: Union[str, Path]) -> float:
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        _LOGGER.exception("failed to read duration for %s", path)
        return 0.0


def _default_audio_loader(path: Union[str, Path]) -> Tuple[object, int, float]:
    """Loads audio mono at 16kHz for scoring. Lazily imports librosa."""
    import librosa
    audio, sr = librosa.load(str(path), sr=16000, mono=True)
    duration = float(len(audio)) / float(sr)
    return audio, sr, duration


def log_filter_summary(total: int, dropped: Dict[str, int], remaining: int) -> None:
    """Logs a standard before/dropped-per-filter/after summary."""
    _LOGGER.info("Quality filtering: %d samples before filtering.", total)
    for column, count in dropped.items():
        _LOGGER.info("  --filter %s dropped %d samples.", column, count)
    _LOGGER.info("Quality filtering: %d samples remaining.", remaining)


# -----------------------------------------------------------------------------
# Built-in scorers
# -----------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(text.split())


def wpm_score(audio: object, sr: int, text: str, duration: float) -> float:
    """Words per minute = word_count / (duration_seconds / 60).

    Flags partial transcriptions (very low wpm, more audio than text
    accounts for) and unnaturally fast speech (very high wpm).
    """
    if duration <= 0:
        return 0.0
    return _word_count(text) / (duration / 60.0)


def is_music_like_score(audio: object, sr: int, text: str, duration: float) -> float:
    """Cheap heuristic proxy for music contamination: 1.0 if music-like,
    else 0.0. NOT a trained classifier — spectral flatness (music tends
    noisier/flatter spectra than voiced speech) combined with the harmonic
    energy ratio from HPSS (music leans more harmonic-dominant than typical
    conversational speech). Tune thresholds per-corpus if this proves noisy.
    """
    import numpy as np
    import librosa

    if audio is None or len(audio) == 0:
        return 0.0
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio)))
    harmonic, _percussive = librosa.effects.hpss(audio)
    total_energy = float(np.sum(np.asarray(audio, dtype=np.float64) ** 2)) + 1e-9
    harmonic_ratio = float(np.sum(harmonic.astype(np.float64) ** 2)) / total_energy
    return 1.0 if (flatness > 0.25 and harmonic_ratio > 0.6) else 0.0


_utmos_predictor = None


def _load_utmos():
    global _utmos_predictor
    if _utmos_predictor is None:
        import torch
        _LOGGER.info("loading UTMOS (SpeechMOS utmos22_strong)...")
        _utmos_predictor = torch.hub.load(
            "tarepan/SpeechMOS", "utmos22_strong", trust_repo=True
        )
        _utmos_predictor.eval()
    return _utmos_predictor


def utmos_score(audio: object, sr: int, text: str, duration: float) -> float:
    """UTMOS1 naturalness MOS via SpeechMOS utmos22_strong. Resamples to
    16kHz mono, expects roughly peak-normalized audio."""
    import torch
    import torchaudio

    predictor = _load_utmos()
    t = torch.from_numpy(audio).float() if hasattr(audio, "dtype") else torch.tensor(audio).float()
    if sr != 16000:
        t = torchaudio.functional.resample(t, sr, 16000)
    with torch.no_grad():
        score = predictor(t.unsqueeze(0), 16000)
    return float(score.squeeze().item())


_dnsmos_run = None
_dnsmos_cache_key = None
_dnsmos_cache_value: Optional[Dict[str, float]] = None


def _load_dnsmos_run():
    """Imports the real (pip) speechmos.dnsmos.run and purges `speechmos`
    from sys.modules afterward.

    torch.hub.load(...) for UTMOS caches a github repo that bundles its own
    package also named `speechmos` (UTMOS-only). Importing that cached repo
    after the real pip package has already been imported (or vice versa)
    lets one shadow the other in sys.modules. Grabbing a bound reference to
    the real `run` function first and then purging `speechmos*` from
    sys.modules keeps both usable regardless of import order.
    """
    global _dnsmos_run
    if _dnsmos_run is None:
        import speechmos.dnsmos as _real_dnsmos
        _dnsmos_run = _real_dnsmos.run
        for mod_name in list(sys.modules):
            if mod_name == "speechmos" or mod_name.startswith("speechmos."):
                del sys.modules[mod_name]
    return _dnsmos_run


def _dnsmos_all(audio: object, sr: int) -> Dict[str, float]:
    global _dnsmos_cache_key, _dnsmos_cache_value
    key = id(audio)
    if _dnsmos_cache_key != key or _dnsmos_cache_value is None:
        run = _load_dnsmos_run()
        target_sr = 16000
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        _dnsmos_cache_value = run(audio, sr=target_sr)
        _dnsmos_cache_key = key
    return _dnsmos_cache_value


def dnsmos_sig_score(audio: object, sr: int, text: str, duration: float) -> float:
    """DNSMOS P.835 signal quality MOS, via the `speechmos` pip package."""
    return float(_dnsmos_all(audio, sr)["sig_mos"])


def dnsmos_bak_score(audio: object, sr: int, text: str, duration: float) -> float:
    """DNSMOS P.835 background-noise quality MOS."""
    return float(_dnsmos_all(audio, sr)["bak_mos"])


def dnsmos_ovrl_score(audio: object, sr: int, text: str, duration: float) -> float:
    """DNSMOS P.835 overall quality MOS."""
    return float(_dnsmos_all(audio, sr)["ovrl_mos"])


# Cheapest first: pure arithmetic, then a lightweight heuristic, then
# model-based scorers, roughly cheapest-to-most-expensive within that tier.
register_scorer("wpm", wpm_score)
register_scorer("is_music_like", is_music_like_score)
register_scorer("dnsmos_sig", dnsmos_sig_score)
register_scorer("dnsmos_bak", dnsmos_bak_score)
register_scorer("dnsmos_ovrl", dnsmos_ovrl_score)
register_scorer("utmos", utmos_score)
