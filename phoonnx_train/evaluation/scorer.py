"""Checkpoint scoring: checkpoint -> synthesized held-out wavs -> EvalRow.

:class:`CheckpointScorer` loads a checkpoint through the training-engine
registry (``engine.eval_synthesize``), synthesizes a fixed set of held-out
sentences on CPU with **per-utterance deterministic seeding** (``manual_seed``
is re-applied immediately before each synthesis so a checkpoint scored twice
produces identical wavs, independent of any RNG drift between epochs), scores
each clip with a small registry of metric functions that reuse
``phoonnx_train.quality_filter`` where they fit (UTMOS at least), optionally adds
speaker similarity against a reference-speaker centroid, and returns a structured
:class:`EvalRow`.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from phoonnx_train import quality_filter
from phoonnx_train.eval_utils import largest_wavs, parse_step

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric registry (reuses quality_filter scorers where they fit)
# ---------------------------------------------------------------------------
# A metric maps a synthesized clip to a scalar. Signature:
#   (wav: np.ndarray, sr: int, text: str) -> float
Metric = Callable[[np.ndarray, int, str], float]

_METRIC_REGISTRY: Dict[str, Metric] = {}


def register_metric(name: str, fn: Metric) -> None:
    """Register a named held-out clip metric."""
    _METRIC_REGISTRY[name] = fn


def known_metrics() -> List[str]:
    return list(_METRIC_REGISTRY)


def _utmos_metric(wav: np.ndarray, sr: int, text: str) -> float:
    # Reuse quality_filter's UTMOS scorer (SpeechMOS utmos22_strong).
    duration = float(len(wav)) / float(sr) if sr else 0.0
    return quality_filter.utmos_score(wav, sr, text, duration)


register_metric("utmos", _utmos_metric)


# ---------------------------------------------------------------------------
# Structured result row
# ---------------------------------------------------------------------------
@dataclass
class EvalRow:
    """One checkpoint's evaluation result.

    ``aggregates`` holds ``<metric>_mean/std/min/max`` for every scored metric
    plus, when speaker scoring is active, ``spk_sim_mean/std/min/max``.
    """

    epoch: int
    step: int
    checkpoint: str
    n_sentences: int
    aggregates: Dict[str, float] = field(default_factory=dict)
    # Per-utterance rows: (sentence, {metric: value}, spk_sim or None)
    perutt: List = field(default_factory=list)

    def value(self, metric: str) -> Optional[float]:
        """The selection value for ``metric`` (e.g. ``"utmos_mean"``)."""
        return self.aggregates.get(metric)

    @property
    def spk_sim_mean(self) -> Optional[float]:
        return self.aggregates.get("spk_sim_mean")

    @property
    def has_speaker_score(self) -> bool:
        return "spk_sim_mean" in self.aggregates

    def to_csv_row(self) -> Dict[str, object]:
        """Flat dict for metrics.csv (keys match tracker columns)."""
        row: Dict[str, object] = {
            "epoch": self.epoch,
            "step": self.step,
            "checkpoint": self.checkpoint,
            "n_sentences": self.n_sentences,
        }
        for k, v in self.aggregates.items():
            row[k] = f"{v:.4f}"
        return row


# ---------------------------------------------------------------------------
# Speaker reference
# ---------------------------------------------------------------------------
def load_speaker_reference(ref_dir: Optional[Path], num_ref_wavs: int):
    """Mean unit-norm speaker embedding over the largest reference wavs.

    Returns ``(embedder, centroid)`` or ``(None, None)`` when speaker
    similarity cannot be computed (no ref dir, speakeronnx missing, no wavs).
    """
    if ref_dir is None:
        return None, None
    try:
        from speakeronnx import SpeakerEmbedder
    except ImportError:
        _LOGGER.warning("speakeronnx not installed; scoring without speaker similarity")
        return None, None
    wavs = largest_wavs(ref_dir, num_ref_wavs)
    if not wavs:
        _LOGGER.warning("no reference wavs under %s; scoring without speaker similarity", ref_dir)
        return None, None
    emb = SpeakerEmbedder()
    vecs = []
    for w in wavs:
        try:
            v = np.asarray(emb.embed(str(w)), dtype=np.float32)
            vecs.append(v / (np.linalg.norm(v) + 1e-9))
        except Exception:
            _LOGGER.exception("failed to embed reference %s", w)
    if not vecs:
        _LOGGER.warning("no reference wavs could be embedded; scoring without speaker similarity")
        return None, None
    ref = np.mean(vecs, axis=0)
    ref /= np.linalg.norm(ref) + 1e-9
    _LOGGER.info("speaker reference centroid from %d clips", len(vecs))
    return emb, ref


def speaker_similarity(emb, ref, wav: np.ndarray, sr: int) -> float:
    # speakeronnx SpeakerEmbedder.embed accepts a waveform array (as used by
    # quality_filter.speaker_consistency_score).
    v = np.asarray(emb.embed(wav), dtype=np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return float(np.dot(v, ref))


# ---------------------------------------------------------------------------
# Text encoding (matches preprocess/train exactly)
# ---------------------------------------------------------------------------
def build_encoder(config: Dict, noise_scale=None, length_scale=None, noise_w=None):
    """Return ``(phonemizer, tokenizer, lang, sample_rate, scales)`` matching
    preprocess/train exactly."""
    from phoonnx.config import Alphabet, PhonemeType, get_phonemizer
    from phoonnx.tokenizer import TTSTokenizer

    phoneme_type = PhonemeType(config["phoneme_type"])
    alphabet = Alphabet(config.get("alphabet") or "ipa")
    ph = get_phonemizer(phoneme_type, alphabet, config.get("phonemizer_model"))
    tokenizer = TTSTokenizer.from_phoonnx_config(config)
    lang = config.get("lang_code", "")
    sample_rate = int(config.get("audio", {}).get("sample_rate", 22050))
    inf = config.get("inference", {})
    scales = [
        noise_scale if noise_scale is not None else float(inf.get("noise_scale", 0.667)),
        length_scale if length_scale is not None else float(inf.get("length_scale", 1.0)),
        noise_w if noise_w is not None else float(inf.get("noise_w", 0.8)),
    ]
    return ph, tokenizer, lang, sample_rate, scales


def text_to_ids(text: str, ph, tokenizer, lang):
    """Replicate preprocess.py: normalize -> phonemize_to_list (drop '\\n')
    -> tokenizer.tokenize (intersperse pad + BOS/EOS)."""
    from phoonnx.util import normalize

    utt = normalize(text, lang)
    phonemes = [p for p in ph.phonemize_to_list(utt, lang) if p != "\n"]
    return phonemes, tokenizer.tokenize(phonemes)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------
class CheckpointScorer:
    """Synthesize + score held-out sentences for a checkpoint.

    Args:
        engine: a ``BaseTrainingEngine`` (provides ``eval_synthesize``).
        config: the training config.json dict.
        sentences: held-out sentences (one string each).
        metrics: metric names to score (default: ``("utmos",)``).
        speaker_ref_dir: reference-speaker wav dir (enables speaker similarity).
        num_ref_wavs: largest reference wavs averaged into the centroid.
        speaker_id: sid for multi-speaker synthesis (``None`` = default voice).
        seed: base seed; utterance ``i`` is synthesized under ``seed + i``.
        vocoder_path / device: passed through to ``engine.eval_synthesize``.
        scales_override: optional ``(noise_scale, length_scale, noise_w)``.
    """

    def __init__(
        self,
        engine,
        config: Dict,
        sentences: List[str],
        *,
        metrics=("utmos",),
        speaker_ref_dir: Optional[Path] = None,
        num_ref_wavs: int = 60,
        speaker_id: Optional[int] = None,
        seed: int = 1234,
        vocoder_path: Optional[Path] = None,
        device: str = "cpu",
        scales_override=(None, None, None),
    ):
        self.engine = engine
        self.config = config
        self.sentences = list(sentences)
        for m in metrics:
            if m not in _METRIC_REGISTRY:
                raise ValueError(
                    f"unknown eval metric {m!r}; known: {', '.join(known_metrics())}"
                )
        self.metrics = list(metrics)
        self.seed = seed
        self.vocoder_path = vocoder_path
        self.device = device

        self.ph, self.tokenizer, self.lang, self.sample_rate, self.scales = build_encoder(
            config, *scales_override
        )

        num_speakers = int(config.get("num_speakers", 1))
        if num_speakers > 1 and speaker_id is None:
            _LOGGER.warning(
                "config declares num_speakers=%d but no --speaker-id given; "
                "synthesizing with sid=None (engine's default voice)",
                num_speakers,
            )
        self.speaker_id = speaker_id

        self.emb, self.ref = load_speaker_reference(speaker_ref_dir, num_ref_wavs)

    @property
    def speaker_scoring_active(self) -> bool:
        return self.emb is not None and self.ref is not None

    def score(self, checkpoint_path: Path, epoch: int, work_dir: Optional[Path] = None) -> EvalRow:
        """Synthesize + score ``checkpoint_path``. Returns an :class:`EvalRow`.

        Wavs and a per-utterance CSV are written under ``work_dir`` when given
        (used by selection to keep best-epoch samples). Raises on load failure;
        the caller decides whether to record the epoch failed.
        """
        import soundfile as sf

        checkpoint_path = Path(checkpoint_path)
        synth = self.engine.eval_synthesize(
            checkpoint_path, self.config,
            vocoder_path=self.vocoder_path, device=self.device,
        )

        if work_dir is not None:
            work_dir = Path(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

        perutt = []
        for i, text in enumerate(self.sentences):
            try:
                _, ids = text_to_ids(text, self.ph, self.tokenizer, self.lang)
                # Per-utterance deterministic reseed: fixes cross-epoch RNG
                # drift so the same checkpoint always yields the same wav.
                self._reseed(self.seed + i)
                wav = synth(ids, self.scales, self.speaker_id)
                metric_vals = {
                    name: _METRIC_REGISTRY[name](wav, self.sample_rate, text)
                    for name in self.metrics
                }
                sim = None
                if self.speaker_scoring_active:
                    sim = speaker_similarity(self.emb, self.ref, wav, self.sample_rate)
                if work_dir is not None:
                    sf.write(work_dir / f"utt{i:02d}.wav", wav, self.sample_rate)
                _LOGGER.info(
                    "  utt%02d dur=%.2fs %s spk_sim=%s  %s",
                    i, len(wav) / self.sample_rate,
                    " ".join(f"{k}={v:.3f}" for k, v in metric_vals.items()),
                    f"{sim:.3f}" if sim is not None else "-", text[:30],
                )
                perutt.append((text, metric_vals, sim))
            except Exception:
                _LOGGER.exception("  failed utt %d: %s", i, text)

        if not perutt:
            raise RuntimeError(f"no utterances scored for epoch {epoch}")

        aggregates = self._aggregate(perutt)
        if work_dir is not None:
            self._write_perutt(work_dir, perutt)

        return EvalRow(
            epoch=epoch,
            step=parse_step(checkpoint_path.name),
            checkpoint=str(checkpoint_path),
            n_sentences=len(perutt),
            aggregates=aggregates,
            perutt=perutt,
        )

    @staticmethod
    def _reseed(seed: int) -> None:
        import torch

        torch.manual_seed(seed)

    def _aggregate(self, perutt) -> Dict[str, float]:
        agg: Dict[str, float] = {}
        for name in self.metrics:
            vals = np.array([m[name] for _, m, _ in perutt], dtype=np.float64)
            agg[f"{name}_mean"] = float(vals.mean())
            agg[f"{name}_std"] = float(vals.std())
            agg[f"{name}_min"] = float(vals.min())
            agg[f"{name}_max"] = float(vals.max())
        sims = np.array([s for _, _, s in perutt if s is not None], dtype=np.float64)
        if sims.size:
            agg["spk_sim_mean"] = float(sims.mean())
            agg["spk_sim_std"] = float(sims.std())
            agg["spk_sim_min"] = float(sims.min())
            agg["spk_sim_max"] = float(sims.max())
        return agg

    def _write_perutt(self, work_dir: Path, perutt) -> None:
        import csv

        header = ["sentence"] + self.metrics + ["spk_sim"]
        with open(work_dir / "perutt.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for text, mvals, sim in perutt:
                w.writerow(
                    [text]
                    + [f"{mvals[m]:.4f}" for m in self.metrics]
                    + [f"{sim:.4f}" if sim is not None else ""]
                )
