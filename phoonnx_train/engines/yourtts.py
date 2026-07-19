"""
YourTTS training engine.

YourTTS is architecturally plain VITS conditioned on an external 512-d
speaker d-vector (instead of a learned per-speaker embedding table) plus an
optional additive language embedding, which is what enables zero-shot voice
cloning. Rather than duplicating the VITS model/lightning module, this engine
reuses ``phoonnx_train.vits`` end to end:

- ``phoonnx_train/vits/models.py`` — ``SynthesizerTrn`` grew an
  ``external_speaker_embedding`` conditioning path (a d-vector, optionally
  projected into ``gin_channels``) and an optional ``n_langs`` additive
  language embedding. Both are opt-in and default off, so plain and
  multi-speaker-by-id VITS are unaffected.
- ``phoonnx_train/vits/lightning.py`` — ``VitsModel`` grew matching
  constructor params (``external_speaker_embedding``, ``speaker_embedding_dim``,
  ``n_langs``) that it threads through to the dataloaders (a d-vector-aware
  ``UtteranceCollate``), ``forward``, and the training/validation steps.
- ``phoonnx_train/vits/dataset.py`` — ``Utterance``/``UtteranceTensors``/``Batch``
  grew optional ``d_vector`` and ``language_id`` fields, populated only when
  present in ``dataset.jsonl`` (backward compatible with plain VITS datasets).

This engine's job is therefore just: (1) drive ``VitsModel`` with the YourTTS
conditioning flags on, (2) precompute per-utterance d-vectors with the *same*
speaker encoder used at inference time so train/inference embeddings match,
and (3) export a checkpoint to the ONNX graph shape ``YourTTSAdapter`` expects
(``input, input_lengths, scales, d_vector, langid``), bundling a default
d-vector (the training-set mean) into ``engine_params`` so the voice
synthesizes with no reference clip.
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # heavy imports — only needed for type annotations
    import pytorch_lightning as pl
    import torch

from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig
from phoonnx_train.engines.vits import _write_tokens_txt

_LOG = logging.getLogger(__name__)

OPSET_VERSION = 15

#: Dimensionality of the bundled Coqui ResNet d-vector speaker encoder used
#: at inference by :class:`phoonnx.engines.yourtts.YourTTSAdapter`.
SPEAKER_EMBEDDING_DIM = 512

# Quality tier -> VITS hyper-param overrides (mirrors phoonnx_train.engines.vits).
_QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "x-low": {
        "inter_channels": 96,
        "hidden_channels": 96,
        "filter_channels": 384,
        "n_heads": 2,
        "n_layers": 4,
    },
    "medium": {
        "inter_channels": 192,
        "hidden_channels": 192,
        "filter_channels": 768,
        "n_heads": 2,
        "n_layers": 6,
    },
    "high": {
        "inter_channels": 256,
        "hidden_channels": 256,
        "filter_channels": 1024,
        "n_heads": 4,
        "n_layers": 8,
    },
}


class YourttsTrainingEngine(BaseTrainingEngine):
    """VITS + external d-vector conditioning (zero-shot voice cloning)."""

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    def create_model(
        self,
        config: TrainingEngineConfig,
        dataset_paths: List[Path],
        **kwargs: Any,
    ) -> "pl.LightningModule":
        """Build a d-vector-conditioned VitsModel LightningModule."""
        from phoonnx_train.vits.lightning import VitsModel

        extra = dict(config.extra)

        # The d-vector dimension and language count are properties of the
        # pre-processed dataset, not the quality preset; the shared train CLI
        # only threads num_symbols/num_speakers/sample_rate, so read them from
        # the dataset's config.json when the caller did not pass them in extra.
        dataset_cfg: Dict[str, Any] = {}
        for p in dataset_paths:
            cfg_path = Path(p) / "config.json" if Path(p).is_dir() else Path(p).parent / "config.json"
            if cfg_path.is_file():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    dataset_cfg = json.load(f)
                break

        n_langs = int(extra.pop("n_langs", dataset_cfg.get("n_langs", 0)))
        speaker_embedding_dim = int(
            extra.pop(
                "speaker_embedding_dim",
                dataset_cfg.get("speaker_embedding_dim", SPEAKER_EMBEDDING_DIM),
            )
        )

        return VitsModel(
            num_symbols=config.num_symbols,
            # YourTTS conditions on a d-vector, not a per-speaker id table.
            num_speakers=1,
            sample_rate=config.sample_rate,
            dataset=[
                str(Path(p) / "dataset.jsonl" if Path(p).is_dir() else Path(p))
                for p in dataset_paths
            ],
            external_speaker_embedding=True,
            speaker_embedding_dim=speaker_embedding_dim,
            n_langs=n_langs,
            **extra,
            **kwargs,
        )

    def export_onnx(
        self,
        checkpoint_path: Path,
        config_path: Path,
        output_dir: Path,
        **kwargs: Any,
    ) -> Path:
        """Export a YourTTS checkpoint to ONNX with d-vector + langid inputs."""
        import torch

        from phoonnx_train.vits.lightning import VitsModel

        with open(config_path, "r", encoding="utf-8") as f:
            model_config: Dict[str, Any] = json.load(f)

        sample_rate = model_config.get("audio", {}).get("sample_rate", 22050)
        phoneme_id_map = model_config.get("phoneme_id_map", {})
        alphabet = model_config.get("alphabet", "")
        phoneme_type = model_config.get("phoneme_type", "")
        phonemizer_model = model_config.get("phonemizer_model", "")

        model: VitsModel = VitsModel.load_from_checkpoint(checkpoint_path, dataset=None)
        model_g = model.model_g
        num_symbols = model_g.n_vocab
        speaker_embedding_dim = model_g.speaker_embedding_dim
        n_langs = max(model_g.n_langs, 1)

        model_g.eval()
        with torch.no_grad():
            model_g.dec.remove_weight_norm()

        def infer_forward(
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            scales: torch.Tensor,
            d_vector: torch.Tensor,
            langid: torch.Tensor,
        ) -> torch.Tensor:
            noise_scale = scales[0]
            length_scale = scales[1]
            noise_scale_w = scales[2]
            audio = model_g.infer(
                text, text_lengths,
                noise_scale=noise_scale,
                length_scale=length_scale,
                noise_scale_w=noise_scale_w,
                speaker_embedding=d_vector,
                lid=langid,
            )[0].unsqueeze(1)
            return audio

        model_g.forward = infer_forward

        sequences = torch.randint(low=0, high=num_symbols, size=(1, 50), dtype=torch.long)
        sequence_lengths = torch.LongTensor([sequences.size(1)])
        scales = torch.FloatTensor([0.667, 1.0, 0.8])
        d_vector = torch.randn(1, speaker_embedding_dim, dtype=torch.float32)
        langid = torch.LongTensor([0])

        input_names = ["input", "input_lengths", "scales", "d_vector", "langid"]
        output_names = ["output"]
        dynamic_axes = {
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "d_vector": {0: "batch_size"},
            "langid": {0: "batch_size"},
            "output": {0: "batch_size", 1: "channels", 2: "time"},
        }
        dummy_input = (sequences, sequence_lengths, scales, d_vector, langid)

        output_path = output_dir / f"{checkpoint_path.name}.onnx"
        export_kwargs = {}
        import inspect
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            # VITS has data-dependent control flow the dynamo exporter
            # cannot trace — force the TorchScript exporter
            export_kwargs["dynamo"] = False
        torch.onnx.export(
            model=model_g,
            args=dummy_input,
            f=str(output_path),
            verbose=False,
            opset_version=OPSET_VERSION,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            **export_kwargs,
        )

        default_d_vector = kwargs.get("default_d_vector")
        if default_d_vector is None:
            default_d_vector = _mean_dataset_d_vector(kwargs.get("dataset_paths") or [])

        try:
            import onnx as _onnx

            onnx_model = _onnx.load(str(output_path))
            del onnx_model.metadata_props[:]
            for key, value in {
                "model_type": "yourtts",
                "engine": "yourtts",
                "n_vocab": num_symbols,
                "sample_rate": sample_rate,
                "alphabet": alphabet,
                "phoneme_type": phoneme_type,
                "phonemizer_model": phonemizer_model,
                "phoneme_id_map": json.dumps(phoneme_id_map),
                "has_espeak": phoneme_type == "espeak",
                "speaker_embedding_dim": speaker_embedding_dim,
                "n_langs": n_langs,
            }.items():
                meta = onnx_model.metadata_props.add()
                meta.key = key
                meta.value = str(value)
            _onnx.save(onnx_model, str(output_path))
        except ImportError:
            _LOG.warning("onnx package not installed — skipping metadata")

        # The voice config's engine_params carries the default speaker so the
        # exported voice works with no reference clip (YourTTSAdapter.configure
        # reads engine_params["d_vector"]); a per-request reference clip / an
        # explicit d_vector param still overrides it at synthesis time.
        piper_path = output_dir / f"{checkpoint_path.stem}.json"
        piper_cfg = _yourtts_voice_json(model_config, default_d_vector)
        with open(piper_path, "w", encoding="utf-8") as f:
            json.dump(piper_cfg, f, indent=2, ensure_ascii=False)
        _LOG.info("Generated YourTTS voice JSON: %s", piper_path)

        if kwargs.get("generate_tokens", False):
            tokens_path = output_dir / "tokens.txt"
            _write_tokens_txt(phoneme_id_map, tokens_path)
            _LOG.info("Generated tokens.txt: %s", tokens_path)

        _LOG.info("Exported YourTTS ONNX model to %s", output_path)
        return output_path

    def quality_presets(self) -> Dict[str, Dict[str, Any]]:
        """Return VITS-style quality tier -> hyper-parameter overrides."""
        return _QUALITY_PRESETS

    # ------------------------------------------------------------------
    # Optional
    # ------------------------------------------------------------------

    def extra_preprocess(
        self,
        utterance_audio_path: Path,
        cache_dir: Path,
        sample_rate: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Compute a d-vector for *utterance_audio_path* with the same speaker
        encoder used at inference time (:class:`phoonnx.engines.speaker_encoders
        .coqui_resnet.CoquiResNetSpeakerEncoder`), caching it next to the mel
        features so train and inference embeddings match exactly.

        ``kwargs`` accepts ``speaker_encoder_path`` (required — path to the
        Coqui ResNet ONNX speaker encoder) and ``language_id`` (optional,
        forwarded through unchanged for multilingual datasets).
        """
        speaker_encoder_path = kwargs.get("speaker_encoder_path")
        if not speaker_encoder_path:
            raise ValueError(
                "YourttsTrainingEngine.extra_preprocess requires "
                "speaker_encoder_path (Coqui ResNet ONNX speaker encoder)"
            )

        encoder = self._get_speaker_encoder(speaker_encoder_path)

        dvec_dir = cache_dir / "dvec"
        dvec_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(str(utterance_audio_path).encode("utf-8")).hexdigest()
        dvec_path = dvec_dir / f"{digest}.pt"

        if not dvec_path.exists():
            import soundfile as sf

            audio, sr = sf.read(str(utterance_audio_path), dtype="float32", always_2d=False)
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            d_vector = encoder.encode(audio, sr)
            torch.save(torch.from_numpy(np.asarray(d_vector, dtype=np.float32)), dvec_path)

        extra: Dict[str, Any] = {"d_vector_path": str(dvec_path)}
        if "language_id" in kwargs and kwargs["language_id"] is not None:
            extra["language_id"] = int(kwargs["language_id"])
        return extra

    def _get_speaker_encoder(self, speaker_encoder_path: str):
        """Lazily build (and cache on the instance) the speaker encoder."""
        cached = getattr(self, "_speaker_encoder", None)
        if cached is not None and getattr(self, "_speaker_encoder_path", None) == speaker_encoder_path:
            return cached
        from phoonnx.engines.speaker_encoders import build_speaker_encoder

        encoder = build_speaker_encoder(speaker_encoder_path, "coqui_resnet")
        self._speaker_encoder = encoder
        self._speaker_encoder_path = speaker_encoder_path
        return encoder

    def extra_cli_options(self) -> List[Any]:
        """CLI options for driving YourTTS preprocessing/training."""
        import click

        return [
            click.option(
                "--speaker-encoder-path",
                "speaker_encoder_path",
                default=None,
                help="Path to the Coqui ResNet ONNX speaker encoder used to "
                     "compute training d-vectors (must match the encoder "
                     "bundled with the resulting voice at inference time).",
            ),
            click.option(
                "--language-id",
                "language_id",
                type=int,
                default=None,
                help="Numeric language id for this dataset run, used by the "
                     "additive language embedding in multilingual training.",
            ),
            click.option(
                "--n-langs",
                "n_langs",
                type=int,
                default=0,
                help="Total number of languages for the additive language "
                     "embedding (0 disables it).",
            ),
        ]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _renormalize(vec):
    """Every individual d-vector is L2-normalized; their mean is not
    (norm < 1, shrinking with speaker diversity) and would be
    out-of-distribution for the model — renormalize after averaging."""
    import numpy as np

    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _mean_dataset_d_vector(dataset_paths: List[Any]) -> Optional[List[float]]:
    """
    Average every cached d-vector referenced by one or more dataset.jsonl
    files, used as the packaged voice's default speaker when no reference
    clip is supplied at synthesis time. Returns ``None`` if no d-vectors are
    found (e.g. exporting without passing ``dataset_paths``, or from a
    checkpoint whose dataset cache is unavailable).
    """
    vectors: List[np.ndarray] = []
    for dataset_path in dataset_paths:
        p = Path(dataset_path)
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                dvec_path = entry.get("d_vector_path")
                if not dvec_path or not Path(dvec_path).exists():
                    continue
                try:
                    vectors.append(torch.load(dvec_path).reshape(-1).numpy())
                except Exception:
                    continue
    if not vectors:
        return None
    mean_vec = _renormalize(np.mean(np.stack(vectors, axis=0), axis=0))
    return mean_vec.astype(np.float32).tolist()


def _yourtts_voice_json(
    model_config: Dict[str, Any],
    default_d_vector: Optional[List[float]],
) -> Dict[str, Any]:
    """Build a phoonnx voice config for a YourTTS export."""
    cfg = {
        "engine": "yourtts",
        "audio": model_config.get("audio", {}),
        "inference": model_config.get("inference", {}),
        "num_symbols": model_config.get("num_symbols"),
        "phoneme_id_map": model_config.get("phoneme_id_map", {}),
        "language": model_config.get("language", {}),
        "espeak": model_config.get("espeak", {}),
        "phoneme_type": model_config.get("phoneme_type", ""),
        "phonemizer_model": model_config.get("phonemizer_model", ""),
        "alphabet": model_config.get("alphabet", ""),
        "engine_params": {
            "langid": model_config.get("language_id", 0),
            "speaker_encoder_type": "coqui_resnet",
        },
    }
    if default_d_vector is not None:
        cfg["engine_params"]["d_vector"] = default_d_vector
    if model_config.get("speaker_encoder_path"):
        cfg["engine_params"]["speaker_encoder_path"] = model_config["speaker_encoder_path"]
    cfg = {k: v for k, v in cfg.items() if v is not None}
    return cfg
