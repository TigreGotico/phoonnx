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
    id_fn: Optional[Callable[[object], str]] = None,
    value_source: Optional[Callable[[object, str], Optional[float]]] = None,
    metrics_sink: Optional[Callable[[str, str, float], None]] = None,
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
        id_fn: samples[i] -> stable row id, used only to key values passed to
            ``metrics_sink``. Defaults to ``str(audio_path_fn(sample))``.
        value_source: (sample, column) -> a precomputed metric value to use
            instead of running the scorer, or ``None`` to compute on demand.
            A source that satisfies every needed metric avoids decoding audio.
        metrics_sink: called ``(row_id, column, value)`` for every metric value
            evaluated for a sample (computed or sourced), before the pass/fail
            decision. Used to persist a metrics sidecar.

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
        row_id = id_fn(sample) if id_fn else str(audio_path_fn(sample))

        try:
            for spec in ordered:
                value = value_source(sample, spec.column) if value_source else None
                if value is None:
                    scorer = _SCORER_REGISTRY[spec.column]
                    if spec.column != "wpm" and not audio_loaded:
                        audio, sr, duration = loader(audio_path_fn(sample))
                        audio_loaded = True
                    elif spec.column == "wpm" and not audio_loaded:
                        # wpm only needs duration; fetch it without decoding
                        # audio unless a later filter forced a full decode.
                        duration = _duration_only(audio_path_fn(sample))
                    value = scorer(audio, sr, text, duration)

                if metrics_sink is not None:
                    metrics_sink(row_id, spec.column, float(value))

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


IS_MUSIC_LIKE_RHYTHMICITY_THRESHOLD = 0.5


def rhythmicity_score(audio: object, sr: int) -> float:
    """Onset-strength autocorrelation peak (beyond the trivial lag-0/near-0
    peak), in roughly [0, 1]: how strongly the clip pulses at a periodic
    tempo. Music tends to score high even under vocals (a spoken-over beat
    still pulses); conversational speech has no such periodicity."""
    import numpy as np
    import librosa

    if audio is None or len(audio) == 0:
        return 0.0
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    if len(onset_env) <= 8:
        return 0.0
    ac = librosa.autocorrelate(onset_env, max_size=len(onset_env) // 2)
    ac = ac / (ac[0] + 1e-9)
    return float(np.max(ac[4:])) if len(ac) > 4 else 0.0


def is_music_like_score(audio: object, sr: int, text: str, duration: float) -> float:
    """Cheap heuristic proxy for music contamination: 1.0 if music-like,
    else 0.0. NOT a trained classifier, and NOT reliable enough to treat as
    ground truth: it thresholds `rhythmicity_score` (onset-strength
    autocorrelation), which on a 200-clip labeled validation set (100
    music-tagged / 100 clean-speech-tagged, real broadcast audio) scored
    ROC-AUC 0.744 — clearly the best of several candidate features tried
    (spectral flatness AUC 0.282, harmonic ratio AUC 0.544, tempo AUC 0.481,
    chroma variants AUC ~0.48-0.50, all near or below chance), but AUC 0.744
    still implies a real false-positive/false-negative rate on the order of
    25-30% at any single threshold. Treat this as a coarse, cheap pre-filter
    to catch the more obvious music contamination, not a substitute for a
    trained speech/music/silence classifier; verify a sample of what it
    drops and keeps before trusting it on a new corpus.
    """
    return 1.0 if rhythmicity_score(audio, sr) > IS_MUSIC_LIKE_RHYTHMICITY_THRESHOLD else 0.0


def _som_score(audio: object, sr: int, metric: str) -> Dict[str, float]:
    """Score `audio` (sr Hz) with a single speechonnxmetrics MOS metric,
    returning its flat result dict (e.g. {'utmos': 4.4} or
    {'dnsmos.sig': ..., 'dnsmos.bak': ..., 'dnsmos.ovrl': ...}).

    speechonnxmetrics resamples to each model's native rate internally and
    downloads its ONNX weights lazily on first use from public HF repos.
    Per-item failures are surfaced (score() records them under '_errors')
    and re-raised here so a failing sample is dropped, matching the old
    behaviour where a scorer exception dropped the sample.
    """
    import speechonnxmetrics as som
    result = som.score(audio, [metric], sr=sr)
    errors = result.get("_errors")
    if errors:
        raise RuntimeError(f"speechonnxmetrics {metric} failed: {errors}")
    return result


def utmos_score(audio: object, sr: int, text: str, duration: float) -> float:
    """UTMOS naturalness MOS via speechonnxmetrics (UTMOS22 strong, ONNX).
    Audio is scored as-is; the library resamples to 16kHz mono internally."""
    return float(_som_score(audio, sr, "utmos")["utmos"])


_dnsmos_cache_audio = None
_dnsmos_cache_value: Optional[Dict[str, float]] = None


def _resample_16k(audio: object, sr: int) -> object:
    target_sr = 16000
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def _dnsmos_all(audio: object, sr: int) -> Dict[str, float]:
    """All three DNSMOS P.835 heads for a clip, via speechonnxmetrics.
    Cached on the audio object so the three dnsmos_* scorers share one
    forward pass per sample."""
    global _dnsmos_cache_audio, _dnsmos_cache_value
    if _dnsmos_cache_audio is not audio or _dnsmos_cache_value is None:
        result = _som_score(audio, sr, "dnsmos")
        _dnsmos_cache_value = {
            "sig_mos": float(result["dnsmos.sig"]),
            "bak_mos": float(result["dnsmos.bak"]),
            "ovrl_mos": float(result["dnsmos.ovrl"]),
        }
        _dnsmos_cache_audio = audio
    return _dnsmos_cache_value


def _load_speechmos_run(submodule: str) -> Callable:
    """Bound `run` of the pip `speechmos.<submodule>` (plcmos/aecmos).

    UTMOS no longer comes from torch.hub, so the cached SpeechMOS repo that
    bundled a shadowing `speechmos` package is never imported and the real
    pip package can be imported plainly."""
    import importlib
    return importlib.import_module(f"speechmos.{submodule}").run


def _speechmos_df_value(df, preferred_columns: List[str]) -> float:
    """Pulls a scalar metric out of a speechmos result, preferring an exact
    expected column name but falling back to the last numeric column so a
    harmless naming difference across speechmos versions doesn't hard-crash
    preprocessing. Accepts either a single-row DataFrame (older speechmos
    `return_df=True`) or a plain dict (newer speechmos returns a dict even
    with `return_df=True`)."""
    row = dict(df) if isinstance(df, dict) else df.iloc[0].to_dict()
    for col in preferred_columns:
        if col in row:
            return float(row[col])
    numeric = [v for v in row.values()
               if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not numeric:
        raise ValueError(f"speechmos result has no numeric columns: {list(row)}")
    return float(numeric[-1])


def dnsmos_sig_score(audio: object, sr: int, text: str, duration: float) -> float:
    """DNSMOS P.835 signal quality MOS, via speechonnxmetrics (ONNX)."""
    return float(_dnsmos_all(audio, sr)["sig_mos"])


def dnsmos_bak_score(audio: object, sr: int, text: str, duration: float) -> float:
    """DNSMOS P.835 background-noise quality MOS."""
    return float(_dnsmos_all(audio, sr)["bak_mos"])


def dnsmos_ovrl_score(audio: object, sr: int, text: str, duration: float) -> float:
    """DNSMOS P.835 overall quality MOS."""
    return float(_dnsmos_all(audio, sr)["ovrl_mos"])


def plcmos_score(audio: object, sr: int, text: str, duration: float) -> float:
    """Packet-loss-concealment quality MOS, via `speechmos.plcmos`. Flags
    VoIP/call audio with dropped-packet artifacts that DNSMOS doesn't
    specifically target; most relevant to call-based corpora."""
    run = _load_speechmos_run("plcmos")
    audio = _resample_16k(audio, sr)
    df = run(audio, 16000, return_df=True, verbose=False)
    return _speechmos_df_value(df, ["plcmos", "PLCMOS", "plcmos_score"])


def aecmos_score(audio: object, sr: int, text: str, duration: float) -> float:
    """Echo-cancellation quality MOS, via `speechmos.aecmos`. Flags
    speakerphone/echo artifacts that DNSMOS doesn't specifically target;
    most relevant to call-based corpora. Runs in scenarioless mode
    (talk_type=None): at 16kHz that selects aecmos_scenarioless_16kHz,
    which needs no far-end reference signal, so it can score one clip at a
    time like every other scorer here (at 48kHz talk_type would be
    mandatory instead, which is why audio is always resampled to 16kHz
    first)."""
    run = _load_speechmos_run("aecmos")
    audio = _resample_16k(audio, sr)
    df = run(audio, 16000, talk_type=None, return_df=True, verbose=False)
    return _speechmos_df_value(df, ["aecmos", "AECMOS", "aecmos_score"])


def snr_score(audio: object, sr: int, text: str, duration: float) -> float:
    """Energy-based SNR estimate in dB, no model.

    Frames the clip (20ms), treats the quietest 10% of frame energies as the
    noise floor and the loudest half as signal, and returns
    10*log10(signal/noise). A cheap proxy, not a WADA-SNR-grade estimate;
    tune the corpus threshold empirically.
    """
    import numpy as np

    if audio is None or len(audio) == 0 or not sr:
        return 0.0
    frame_len = max(1, int(0.02 * sr))
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return 0.0
    frames = np.asarray(audio[: n_frames * frame_len], dtype=np.float64).reshape(n_frames, frame_len)
    energies = np.mean(frames ** 2, axis=1)
    sorted_energies = np.sort(energies)
    noise_floor = float(np.mean(sorted_energies[: max(1, n_frames // 10)])) + 1e-12
    signal_power = float(np.mean(sorted_energies[max(1, n_frames // 2):])) + 1e-12
    return float(10.0 * np.log10(signal_power / noise_floor))


def clipping_score(audio: object, sr: int, text: str, duration: float) -> float:
    """Fraction of samples at or near full-scale (|x| > 0.99) on a
    [-1, 1]-normalized waveform. No model."""
    import numpy as np

    if audio is None or len(audio) == 0:
        return 0.0
    return float(np.mean(np.abs(np.asarray(audio, dtype=np.float64)) > 0.99))


_vad_model_name = "silero"
_vad_model = None
_vad_model_cache_key: Optional[str] = None


def configure_vad_model(name: str) -> None:
    """Sets the vadonnx model used by the vad_ratio scorer. Takes effect on
    the next call (the cached model, if any, is dropped)."""
    global _vad_model_name, _vad_model, _vad_model_cache_key
    _vad_model_name = name
    _vad_model = None
    _vad_model_cache_key = None


def _get_vad_model():
    global _vad_model, _vad_model_cache_key
    if _vad_model is None or _vad_model_cache_key != _vad_model_name:
        from vadonnx import load_vad
        _LOGGER.info("loading VAD model %r (vadonnx) for vad_ratio scoring...", _vad_model_name)
        _vad_model = load_vad(_vad_model_name)
        _vad_model_cache_key = _vad_model_name
    return _vad_model


def vad_ratio_score(audio: object, sr: int, text: str, duration: float) -> float:
    """Fraction of the clip vadonnx detects as speech. Model set via
    configure_vad_model()/--vad-model (default: vadonnx's bundled 'silero',
    offline with no extra download)."""
    if audio is None or len(audio) == 0 or duration <= 0:
        return 0.0
    vad = _get_vad_model()
    segments = vad.get_speech_segments(audio, sample_rate=sr)
    speech_seconds = sum(seg.duration for seg in segments)
    return float(speech_seconds / duration)


_speaker_model_name = "wespeaker-resnet34"
_speaker_embedder = None
_speaker_embedder_cache_key: Optional[str] = None


def configure_speaker_model(name: str) -> None:
    """Sets the speakeronnx model used by the speaker_consistency scorer.
    Takes effect on the next call (the cached embedder, if any, is dropped)."""
    global _speaker_model_name, _speaker_embedder, _speaker_embedder_cache_key
    _speaker_model_name = name
    _speaker_embedder = None
    _speaker_embedder_cache_key = None


def _get_speaker_embedder():
    global _speaker_embedder, _speaker_embedder_cache_key
    if _speaker_embedder is None or _speaker_embedder_cache_key != _speaker_model_name:
        from speakeronnx import SpeakerEmbedder
        _LOGGER.info("loading speaker embedder %r (speakeronnx) for "
                     "speaker_consistency scoring...", _speaker_model_name)
        _speaker_embedder = SpeakerEmbedder(model=_speaker_model_name)
        _speaker_embedder_cache_key = _speaker_model_name
    return _speaker_embedder


def _cosine_similarity(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def speaker_consistency_score(audio: object, sr: int, text: str, duration: float,
                              num_windows: int = 3, min_window_seconds: float = 0.5) -> float:
    """Minimum pairwise cosine similarity between speaker embeddings of
    N evenly-spaced windows of the clip; low similarity between windows of
    the same nominal clip suggests a speaker change or bleed-in. Clips too
    short to split into `num_windows` windows of at least
    `min_window_seconds` are treated as trivially consistent (1.0) since
    there's nothing to compare. Model set via configure_speaker_model()/
    --speaker-model (default: speakeronnx's own default,
    'wespeaker-resnet34')."""
    import numpy as np

    if audio is None or duration <= 0:
        return 1.0
    window_len = len(audio) // num_windows
    if sr and window_len < int(min_window_seconds * sr):
        return 1.0
    embedder = _get_speaker_embedder()
    embeddings = []
    for i in range(num_windows):
        start = i * window_len
        end = start + window_len if i < num_windows - 1 else len(audio)
        window = np.asarray(audio[start:end])
        embeddings.append(embedder.embed(window))
    similarities = [
        _cosine_similarity(embeddings[i], embeddings[j])
        for i in range(len(embeddings))
        for j in range(i + 1, len(embeddings))
    ]
    return float(min(similarities)) if similarities else 1.0


_asr_model_name: Optional[str] = "whisper-base"
_asr_model = None
_asr_model_cache_key: Optional[str] = None


def configure_asr_model(name: str) -> None:
    """Sets the onnx_asr model used by the wer scorer. Takes effect on the
    next call (the cached model, if any, is dropped)."""
    global _asr_model_name, _asr_model, _asr_model_cache_key
    _asr_model_name = name
    _asr_model = None
    _asr_model_cache_key = None


def _get_asr_model():
    global _asr_model, _asr_model_cache_key
    if _asr_model_name is None:
        raise RuntimeError(
            "the 'wer' filter requires an ASR model; set --asr-model to a "
            "model identifier or path loadable via onnx_asr.load_model()"
        )
    if _asr_model is None or _asr_model_cache_key != _asr_model_name:
        try:
            import onnx_asr
        except ImportError as exc:
            raise RuntimeError(
                "the 'wer' filter requires the 'onnx_asr' package "
                "(pip install onnx-asr, or the phoonnx 'train-eval' extra); "
                "it does not fall back to any other ASR backend"
            ) from exc
        _LOGGER.info("loading ASR model %r (onnx_asr) for wer scoring...", _asr_model_name)
        try:
            _asr_model = onnx_asr.load_model(_asr_model_name)
        except Exception as exc:
            raise RuntimeError(
                f"--asr-model {_asr_model_name!r} is not loadable via "
                "onnx_asr.load_model(); the 'wer' filter only supports "
                "onnx-asr-compatible model identifiers or paths, and does "
                "not fall back to any other ASR backend"
            ) from exc
        _asr_model_cache_key = _asr_model_name
    return _asr_model


def _word_error_rate(reference: str, hypothesis: str) -> float:
    """Word error rate: word-level Levenshtein edit distance / reference
    word count. An empty reference with a non-empty hypothesis scores 1.0
    (fully wrong); an empty reference with an empty hypothesis scores 0.0."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    n, m = len(ref_words), len(hyp_words)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[m] / n


def wer_score(audio: object, sr: int, text: str, duration: float) -> float:
    """Word error rate of an onnx_asr transcription against the sample's own
    transcript. The most expensive scorer here (a full ASR pass per clip),
    so it is evaluated last. Model set via configure_asr_model()/
    --asr-model; loading a model that onnx_asr.load_model() rejects raises
    immediately rather than silently skipping this filter."""
    model = _get_asr_model()
    hypothesis = model.recognize(audio, sample_rate=sr or 16000)
    if isinstance(hypothesis, list):
        hypothesis = hypothesis[0] if hypothesis else ""
    return _word_error_rate(text, str(hypothesis))


# Cheapest first: pure arithmetic, then a lightweight heuristic, then
# model-based scorers, roughly cheapest-to-most-expensive within that tier.
# wer is a full ASR pass per clip and always sorts last.
register_scorer("wpm", wpm_score)
register_scorer("snr", snr_score)
register_scorer("clipping", clipping_score)
register_scorer("is_music_like", is_music_like_score)
register_scorer("vad_ratio", vad_ratio_score)
register_scorer("dnsmos_sig", dnsmos_sig_score)
register_scorer("dnsmos_bak", dnsmos_bak_score)
register_scorer("dnsmos_ovrl", dnsmos_ovrl_score)
register_scorer("plcmos", plcmos_score)
register_scorer("aecmos", aecmos_score)
register_scorer("utmos", utmos_score)
register_scorer("speaker_consistency", speaker_consistency_score)
register_scorer("wer", wer_score)
