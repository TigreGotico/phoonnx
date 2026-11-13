"""Rápido training engine (``--engine rapido``).

Rápido (anurag12-webster/rapido-tts, Apache-2.0) is a 21M flow-matching DiT
over 64-dim VoxCPM2 AudioVAE latents at 25 Hz, decoded to 48 kHz by the
frozen VAE. The model code is ours (``phoonnx_train/rapido/model.py``); the
training loop (``phoonnx_train/rapido/lightning.py``) is ours too, because
upstream ships none.

Dataset: the shared ``preprocess.py`` output. The model reads the phoneme ids
without the interleaved blanks and learns letter durations by monotonic
alignment search, so no external aligner is needed. Latents are cached beside
the normalized-audio cache, by ``preprocess.py --engine rapido`` or on the
first training run.

Fine-tuning from the released weights: pass the upstream ``rapido.pt``, or a
Lightning checkpoint of this engine, as ``--resume-from-checkpoint <path>
--discard-encoder``. Tensors whose shape differs are dropped, and the letter
embedding is dropped whenever the vocabulary differs from the dataset's.
"""
import dataclasses
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple

from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig

if TYPE_CHECKING:  # heavy import — only needed for type annotations
    import pytorch_lightning as pl

LOG = logging.getLogger(__name__)

# "base" = upstream Rápido sizes (needed to fine-tune rapido.pt);
# "low" is a tiny CI/smoke tier (architecture only, not a quality choice)
_QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "base": {},
    "low": {
        "d": 64,
        "n_heads": 2,
        "n_layers": 2,
        "text_conv": 1,
        "text_attn": 1,
        "align_heads": 2,
        "dur_hidden": 32,
    },
}

EMBEDDING_KEY = "net.text.emb.weight"


def read_dataset_config(dataset_paths: List[Path]) -> Dict[str, Any]:
    first = Path(dataset_paths[0])
    config_path = (first if first.is_dir() else first.parent) / "config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def split_model_params(extra: Mapping[str, Any],
                       model_keys: set) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split the CLI ``extra`` bag into (model params, Lightning kwargs).

    Order, later wins: the quality preset, explicit ``model_params``, bare
    model keys. The input dict is not changed.
    """
    extra = dict(extra)
    quality = extra.pop("quality", "base")
    preset = _QUALITY_PRESETS.get(quality)
    if preset is None:
        LOG.warning("unknown rapido quality %r, using 'base'", quality)
        preset = _QUALITY_PRESETS["base"]
    model_params = {**preset, **extra.pop("model_params", {})}
    for key in list(extra):
        if key in model_keys:
            model_params[key] = extra.pop(key)
    return model_params, extra


def checkpoint_state(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    """Model tensors of an upstream ``rapido.pt`` or of a Lightning checkpoint,
    keyed as ``RapidoModel.state_dict()``."""
    if "ema" in checkpoint:
        return dict(checkpoint["ema"])
    state = checkpoint.get("state_dict", checkpoint)
    return {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}


def checkpoint_vocab(checkpoint: Mapping[str, Any]) -> Optional[List[str]]:
    """Symbol of every embedding row, when the checkpoint records it."""
    if "vocab" in checkpoint:
        return list(checkpoint["vocab"])
    id_map = checkpoint.get("hyper_parameters", {}).get("phoneme_id_map")
    if id_map is None:
        return None
    return [symbol for symbol, _ in sorted(id_map.items(), key=lambda kv: kv[1])]


class RapidoTrainingEngine(BaseTrainingEngine):
    """Flow-matching DiT over AudioVAE latents, aligned by MAS."""

    def __init__(self) -> None:
        self._encoder: Optional[Tuple[Any, ...]] = None
        self._vae = None

    def create_model(self, config: TrainingEngineConfig,
                     dataset_paths: List[Path],
                     **kwargs: Any) -> "pl.LightningModule":
        from phoonnx.engines.rapido import layout_boundary_ids
        from phoonnx_train.rapido.dataset import blank_id
        from phoonnx_train.rapido.lightning import RapidoModule
        from phoonnx_train.rapido.model import RapidoConfig

        dataset_config = read_dataset_config(dataset_paths)
        id_map = dataset_config["phoneme_id_map"]
        model_keys = {f.name for f in dataclasses.fields(RapidoConfig)}
        model_params, extra = split_model_params(config.extra, model_keys)
        return RapidoModule(
            num_symbols=config.num_symbols,
            num_speakers=config.num_speakers,
            boundary_ids=sorted(layout_boundary_ids(id_map)),
            blank=blank_id(dataset_config),
            phoneme_id_map={k: v[0] if isinstance(v, list) else v for k, v in id_map.items()},
            sample_rate=config.sample_rate,
            dataset=[str(p) for p in dataset_paths],
            model_params=model_params,
            **extra, **kwargs)

    def quality_presets(self) -> Dict[str, Dict[str, Any]]:
        return _QUALITY_PRESETS

    def trainer_kwargs(self) -> Dict[str, Any]:
        return {"gradient_clip_val": 1.0, "log_every_n_steps": 10}

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def extra_preprocess(self, utterance_audio_path: Path, cache_dir: Path,
                         sample_rate: int, **kwargs: Any) -> Dict[str, Any]:
        """Cache the AudioVAE latents of the trimmed, normalized audio.

        The cache sits beside ``<sha>.pt`` as ``<sha>.rapido-latent.pt``,
        where the training dataset looks for it. Nothing is added to the
        utterance record.
        """
        from phoonnx_train.rapido.audio_vae import load_audio_vae
        from phoonnx_train.rapido.dataset import cache_latents, norm_audio_path

        norm_path = norm_audio_path(utterance_audio_path, cache_dir)
        if not norm_path.is_file():
            LOG.warning("no normalized-audio cache %s; training encodes the latents "
                        "from dataset.jsonl's audio_norm_path instead", norm_path)
            return {}
        if self._vae is None:
            self._vae = load_audio_vae()
        cache_latents(self._vae, norm_path, sample_rate)
        return {}

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def load_checkpoint(self, model: "pl.LightningModule", checkpoint_path: Path,
                        discard_encoder: bool = False,
                        **kwargs: Any) -> "pl.LightningModule":
        """Warm-start from upstream ``rapido.pt`` (key ``ema``) or our checkpoint.

        Tensors with another shape are dropped (the speaker table when the
        speaker count differs, the letter embedding when the vocabulary size
        differs). The letter embedding is also dropped when the checkpoint's
        vocabulary is not the dataset's, or when ``discard_encoder`` is set.
        The prior head loads from our checkpoints only. Every drop is logged
        with counts.
        """
        import torch

        from phoonnx_train.torch_compat import trusting_torch_load

        # rapido.pt and our checkpoints carry pickled config objects
        with trusting_torch_load():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint_state(checkpoint)
        target = model.model.state_dict()

        dataset_vocab = [s for s, _ in sorted(model.hparams.phoneme_id_map.items(),
                                              key=lambda kv: kv[1])]
        same_vocab = checkpoint_vocab(checkpoint) == dataset_vocab
        dropped_shape = [k for k, v in state.items()
                         if k in target and v.shape != target[k].shape]
        unknown = [k for k in state if k not in target]
        loaded = {k: v.float() for k, v in state.items()
                  if k in target and v.shape == target[k].shape}
        if EMBEDDING_KEY in loaded and (discard_encoder or not same_vocab):
            del loaded[EMBEDDING_KEY]
            LOG.info("letter embedding not loaded: %s",
                     "--discard-encoder" if discard_encoder else "vocabulary differs")
        model.model.load_state_dict(loaded, strict=False)

        prior = {k[len("prior."):]: v for k, v in checkpoint.get("state_dict", {}).items()
                 if k.startswith("prior.")}
        if prior:
            model.prior.load_state_dict(prior)

        LOG.info("loaded %d/%d tensors from %s; dropped %d with another shape (%s); "
                 "%d unknown keys; %d kept at init",
                 len(loaded), len(target), checkpoint_path, len(dropped_shape),
                 ", ".join(dropped_shape) or "none", len(unknown),
                 len(target) - len(loaded))
        model.load_report = {"loaded": sorted(loaded), "dropped_shape": dropped_shape,
                             "unknown": unknown, "same_vocab": same_vocab}
        return model

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_onnx(self, checkpoint_path: Path, config_path: Path,
                    output_dir: Path, **kwargs: Any) -> Path:
        """Export a trained checkpoint; returns the voice ``config.json`` path.

        The voice is several graphs plus ``config.json`` in ``output_dir``,
        written by ``phoonnx_train.rapido.export_onnx.export_rapido``.
        """
        from phoonnx_train.rapido.audio_vae import load_audio_vae
        from phoonnx_train.rapido.dataset import RAPIDO_TOKENIZER_FLAGS
        from phoonnx_train.rapido.export_onnx import export_rapido
        from phoonnx_train.rapido.lightning import RapidoModule
        from phoonnx_train.torch_compat import trusting_torch_load

        with trusting_torch_load():
            module = RapidoModule.load_from_checkpoint(
                str(checkpoint_path), map_location="cpu", dataset=None)
        dataset_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        voice_config = {**dataset_config, **RAPIDO_TOKENIZER_FLAGS}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return export_rapido(module.model.eval(), voice_config, output_dir,
                             vae=load_audio_vae())

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def eval_text_to_ids(self, text: str,
                         config: Optional[Dict[str, Any]]) -> Optional[List[int]]:
        """Phonemize and tokenize as ``preprocess.py`` does, then drop the blanks
        exactly as the training dataset does."""
        from phoonnx_train.evaluation.scorer import build_encoder, text_to_ids
        from phoonnx_train.rapido.dataset import strip_blanks

        if self._encoder is None or self._encoder[0] is not config:
            self._encoder = (config, *build_encoder(config)[:3])
        _, phonemizer, tokenizer, lang = self._encoder
        _, ids = text_to_ids(text, phonemizer, tokenizer, lang)
        return strip_blanks(ids, tokenizer.blank_id)

    def eval_synthesize(self, checkpoint_path: Path, config: Optional[Dict[str, Any]], *,
                        vocoder_path: Optional[str] = None, device: str = "cpu"):
        """Checkpoint → ``synth(ids, scales, sid)``: 48 kHz VAE audio resampled
        to the dataset sample rate the scorer measures at. ``scales[1]`` is the
        length scale; ``vocoder_path`` is unused (the VAE decodes)."""
        import numpy as np
        import torch

        from phoonnx_train.rapido.audio_vae import load_audio_vae
        from phoonnx_train.rapido.lightning import RapidoModule
        from phoonnx_train.torch_compat import trusting_torch_load

        with trusting_torch_load():
            module = RapidoModule.load_from_checkpoint(
                str(checkpoint_path), map_location=device, dataset=None)
        model = module.model.eval()
        boundary_ids = module.boundary_ids
        if self._vae is None:
            self._vae = load_audio_vae(device=device)
        vae = self._vae
        target_rate = int((config or {}).get("audio", {}).get("sample_rate", 22050))

        def synth(ids: List[int], scales: Optional[List[float]],
                  sid: Optional[int]) -> np.ndarray:
            length_scale = float(scales[1]) if scales else 1.0
            with torch.inference_mode():
                latents = model.generate(ids, boundary_ids, speaker=sid or 0,
                                         length_scale=length_scale)
                audio = vae.decode(latents.T[None].to(device)).reshape(-1).float().cpu()
                if target_rate != 48000:
                    import torchaudio

                    audio = torchaudio.functional.resample(audio, 48000, target_rate)
            return audio.numpy().astype(np.float32)

        return synth
