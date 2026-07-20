"""
OptiSpeech training engine.

OptiSpeech is a non-autoregressive, FastSpeech2-style acoustic model with a
GAN-trained neural vocoder (WaveNeXt).  The model code is vendored under
``phoonnx_train.optispeech`` so training runs on the same pytorch_lightning
stack, dataset format and CLI as every other phoonnx engine.

The upstream project wired its model together with Hydra: an external YAML
tree described every sub-module and Hydra ``instantiate``-d it at runtime.
None of that is used here.  Every sub-module is constructed with plain
Python — ``functools.partial`` factories carry the preset-derived
hyper-parameters, and the nested config namespaces become explicit
dataclasses with sane defaults.  A quality preset therefore fully
determines the constructed model without any config file.

Checkpointing:
    The LightningModule uses manual optimization (GAN).  ``configure_optimizers``
    builds two optimizers (generator + discriminator) and two step-wise cosine
    schedulers.  Lightning persists optimizer state, scheduler state, epoch and
    global step in every checkpoint, so training is resumable via
    ``Trainer(...).fit(model, ckpt_path=...)``.  ``load_checkpoint`` additionally
    supports warm-starting from a *partial* checkpoint: it maps matching keys by
    name+shape and logs exactly which keys were loaded and which were skipped.
"""
import functools
import json
import logging
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # heavy imports — only needed for type annotations
    import pytorch_lightning as pl

from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig

_LOG = logging.getLogger(__name__)

# ONNX opset used for export.
OPSET_VERSION = 15


# ------------------------------------------------------------------
# Quality presets — each tier fully changes the constructed model.
# Fields map directly onto ``OptiSpeechEngineConfig``.
# ------------------------------------------------------------------
_QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "x-low": {
        "dim": 128,
        "encoder_num_layers": 2,
        "encoder_intermediate_dim": 512,
        "decoder_num_layers": 2,
        "decoder_intermediate_dim": 512,
        "vocoder_dim": 256,
        "vocoder_intermediate_dim": 768,
        "vocoder_num_layers": 4,
    },
    "medium": {
        "dim": 256,
        "encoder_num_layers": 4,
        "encoder_intermediate_dim": 1024,
        "decoder_num_layers": 4,
        "decoder_intermediate_dim": 1024,
        "vocoder_dim": 384,
        "vocoder_intermediate_dim": 1152,
        "vocoder_num_layers": 8,
    },
    "high": {
        "dim": 384,
        "encoder_num_layers": 6,
        "encoder_intermediate_dim": 1536,
        "decoder_num_layers": 6,
        "decoder_intermediate_dim": 1536,
        "vocoder_dim": 512,
        "vocoder_intermediate_dim": 1536,
        "vocoder_num_layers": 8,
    },
}


# ------------------------------------------------------------------
# Explicit dataclasses replacing the Hydra DictConfig namespaces.
# These are module-level (picklable) so they survive save_hyperparameters
# and checkpoint round-trips.
# ------------------------------------------------------------------
@dataclass
class GeneratorLossCoeffs:
    """Weights for the acoustic-model loss terms (was ``loss_coeffs``)."""

    lambda_align: float = 5.0
    lambda_duration: float = 1.0
    lambda_pitch: float = 1.0
    lambda_energy: float = 1.0


@dataclass
class DiscriminatorLossCoeffs:
    """Weights for the vocoder-discriminator loss terms."""

    lambda_mel: float = 45.0
    lambda_mr_stft: float = 1.0
    lambda_mrd: float = 0.1


@dataclass
class OptiSpeechTrainArgs:
    """Training-loop knobs (was ``train_args``)."""

    cache_generator_outputs: bool = False
    gradient_clip_val: float = 10.0
    gradient_accumulate_batches: Optional[int] = None
    pretraining_steps: int = 0
    # Perceptual validation metrics pull in optional heavy deps
    # (UTMOS / PESQ / periodicity); off by default.
    evaluate_periodicity: bool = False
    evaluate_utmos: bool = False
    evaluate_pesq: bool = False


@dataclass
class OptiSpeechInferenceArgs:
    """Default synthesis scaling factors (was ``inference_args``)."""

    d_factor: float = 1.0
    p_factor: float = 1.0
    e_factor: float = 1.0


@dataclass
class OptiSpeechDataArgs:
    """Data-side namespace passed to the LightningModule (was ``data_args``)."""

    num_speakers: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    text_processor: Any
    feature_extractor: Any
    data_statistics: Any = None


# ------------------------------------------------------------------
# Dependency-free cosine-with-warmup schedule (replaces the
# transformers.get_cosine_schedule_with_warmup import).
# ------------------------------------------------------------------
def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """LambdaLR cosine schedule with linear warmup.

    Kept API-compatible with the upstream helper so
    ``BaseLightningModule.configure_optimizers`` can rewrite
    ``num_training_steps`` via the partial's ``keywords``.
    """
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


@dataclass
class OptiSpeechEngineConfig:
    """Explicit, flat config for OptiSpeech training.

    ``from_training_config`` maps the shared :class:`TrainingEngineConfig`
    plus a quality preset onto these fields; :func:`_build_model_kwargs`
    turns them into the ``functools.partial`` factories the vendored model
    expects.
    """

    # Shared
    num_symbols: int = 178
    num_speakers: int = 1
    sample_rate: int = 22050

    # Acoustic-model hidden width (shared by text-embedding / encoder /
    # variance predictors / decoder).
    dim: int = 256

    # Feature extractor / mel
    n_feats: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    f_min: int = 0
    f_max: int = 8000

    # Text processing
    tokenizer: str = "ipa"
    add_blank: bool = False
    add_bos_eos: bool = False
    normalize_text: bool = True
    languages: List[str] = field(default_factory=lambda: ["en-us"])
    max_source_positions: int = 2000

    # Encoder (ConvNeXt backbone)
    encoder_num_layers: int = 4
    encoder_intermediate_dim: int = 1024
    encoder_drop_path: float = 0.0

    # Decoder (ConvNeXt backbone)
    decoder_num_layers: int = 4
    decoder_intermediate_dim: int = 1024
    decoder_drop_path: float = 0.0

    # Duration predictor
    duration_num_layers: int = 2
    duration_intermediate_dim: int = 384
    duration_kernel_size: int = 3
    duration_dropout: float = 0.1

    # Pitch predictor
    pitch_num_layers: int = 5
    pitch_intermediate_dim: int = 256
    pitch_kernel_size: int = 5
    pitch_dropout: float = 0.5
    pitch_embed_kernel_size: int = 9
    pitch_embed_dropout: float = 0.2

    # Energy predictor
    energy_num_layers: int = 2
    energy_intermediate_dim: int = 384
    energy_kernel_size: int = 3
    energy_dropout: float = 0.5
    energy_embed_kernel_size: int = 9
    energy_embed_dropout: float = 0.5

    # Vocoder (WaveNeXt)
    vocoder_dim: int = 384
    vocoder_intermediate_dim: int = 1152
    vocoder_num_layers: int = 8
    vocoder_drop_path: float = 0.0

    # Alignment / segmenting
    segment_size: int = 64
    text_dropout: float = 0.1

    # Loss coefficients
    lambda_align: float = 5.0
    lambda_duration: float = 1.0
    lambda_pitch: float = 1.0
    lambda_energy: float = 1.0
    lambda_mel: float = 45.0
    lambda_mr_stft: float = 1.0
    lambda_mrd: float = 0.1

    # Inference defaults
    d_factor: float = 1.0
    p_factor: float = 1.0
    e_factor: float = 1.0

    # Training loop
    batch_size: int = 16
    num_workers: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    adam_b1: float = 0.8
    adam_b2: float = 0.99
    warmup_steps: int = 1000
    pretraining_steps: int = 0
    gradient_clip_val: float = 10.0
    cache_generator_outputs: bool = False

    @classmethod
    def from_training_config(cls, cfg: TrainingEngineConfig) -> "OptiSpeechEngineConfig":
        """Build from the shared config + quality preset + engine overrides."""
        extra = dict(cfg.extra)
        preset_name = extra.pop("quality", "medium")
        if preset_name not in _QUALITY_PRESETS:
            _LOG.warning(
                "unknown quality preset %r; falling back to 'medium'", preset_name
            )
            preset_name = "medium"
        params = dict(_QUALITY_PRESETS[preset_name])
        params.update(extra)

        known = {f.name for f in fields(cls)}
        ignored = set(params) - known
        if ignored:
            _LOG.debug("ignoring non-optispeech config keys: %s", sorted(ignored))

        return cls(
            num_symbols=cfg.num_symbols,
            num_speakers=cfg.num_speakers,
            sample_rate=cfg.sample_rate,
            **{k: v for k, v in params.items() if k in known},
        )


def _build_feature_extractor(ocfg: "OptiSpeechEngineConfig"):
    from phoonnx_train.optispeech.dataset.feature_extractors import CommonFeatureExtractor
    from phoonnx_train.optispeech.dataset.feature_extractors.pitch_extractors import (
        DIOPitchExtractor,
    )

    return CommonFeatureExtractor(
        sample_rate=ocfg.sample_rate,
        n_feats=ocfg.n_feats,
        n_fft=ocfg.n_fft,
        hop_length=ocfg.hop_length,
        win_length=ocfg.win_length,
        f_min=ocfg.f_min,
        f_max=ocfg.f_max,
        center=True,
        pitch_extractor=functools.partial(DIOPitchExtractor, interpolate=True),
    )


def _build_text_processor(ocfg: "OptiSpeechEngineConfig"):
    from phoonnx_train.optispeech.text import TextProcessor

    # The tokenizer treats add_blank / add_bos_eos truthiness as the flag.
    return TextProcessor(
        tokenizer=ocfg.tokenizer,
        add_blank=bool(ocfg.add_blank),
        add_bos_eos=bool(ocfg.add_bos_eos),
        normalize_text=ocfg.normalize_text,
        languages=list(ocfg.languages),
    )


def _build_model_kwargs(
    ocfg: "OptiSpeechEngineConfig",
    feature_extractor,
    text_processor,
) -> Dict[str, Any]:
    """Assemble the explicit partial factories the vendored model expects."""
    import torch

    from phoonnx_train.optispeech.model.generator import OptiSpeechGenerator
    from phoonnx_train.optispeech.model.generator.modules import (
        ConvNeXtBackbone,
        DurationPredictor,
        EnergyPredictor,
        PitchPredictor,
        TextEmbedding,
    )
    from phoonnx_train.optispeech.model.vocoder.wavenext import WaveNeXt
    from phoonnx_train.optispeech.model.vocoder.wavenext.disc import VocosDiscriminator

    text_embedding = functools.partial(
        TextEmbedding,
        n_vocab=ocfg.num_symbols,
        dropout=ocfg.text_dropout,
        padding_idx=0,
        max_source_positions=ocfg.max_source_positions,
    )
    encoder = functools.partial(
        ConvNeXtBackbone,
        intermediate_dim=ocfg.encoder_intermediate_dim,
        num_layers=ocfg.encoder_num_layers,
        drop_path=ocfg.encoder_drop_path,
    )
    decoder = functools.partial(
        ConvNeXtBackbone,
        intermediate_dim=ocfg.decoder_intermediate_dim,
        num_layers=ocfg.decoder_num_layers,
        drop_path=ocfg.decoder_drop_path,
    )
    duration_predictor = functools.partial(
        DurationPredictor,
        num_layers=ocfg.duration_num_layers,
        intermediate_dim=ocfg.duration_intermediate_dim,
        kernel_size=ocfg.duration_kernel_size,
        dropout=ocfg.duration_dropout,
        conv_layer_class=torch.nn.Conv1d,
    )
    pitch_predictor = functools.partial(
        PitchPredictor,
        num_layers=ocfg.pitch_num_layers,
        intermediate_dim=ocfg.pitch_intermediate_dim,
        kernel_size=ocfg.pitch_kernel_size,
        dropout=ocfg.pitch_dropout,
        conv_layer_class=torch.nn.Conv1d,
        embed_kernel_size=ocfg.pitch_embed_kernel_size,
        embed_dropout=ocfg.pitch_embed_dropout,
    )
    energy_predictor = functools.partial(
        EnergyPredictor,
        num_layers=ocfg.energy_num_layers,
        intermediate_dim=ocfg.energy_intermediate_dim,
        kernel_size=ocfg.energy_kernel_size,
        dropout=ocfg.energy_dropout,
        conv_layer_class=torch.nn.Conv1d,
        embed_kernel_size=ocfg.energy_embed_kernel_size,
        embed_dropout=ocfg.energy_embed_dropout,
    )

    generator = functools.partial(
        OptiSpeechGenerator,
        segment_size=ocfg.segment_size,
        text_embedding=text_embedding,
        encoder=encoder,
        duration_predictor=duration_predictor,
        pitch_predictor=pitch_predictor,
        energy_predictor=energy_predictor,
        decoder=decoder,
        loss_coeffs=GeneratorLossCoeffs(
            lambda_align=ocfg.lambda_align,
            lambda_duration=ocfg.lambda_duration,
            lambda_pitch=ocfg.lambda_pitch,
            lambda_energy=ocfg.lambda_energy,
        ),
    )
    vocoder = functools.partial(
        WaveNeXt,
        dim=ocfg.vocoder_dim,
        intermediate_dim=ocfg.vocoder_intermediate_dim,
        num_layers=ocfg.vocoder_num_layers,
        drop_path=ocfg.vocoder_drop_path,
    )
    discriminator = functools.partial(
        VocosDiscriminator,
        loss_coeffs=DiscriminatorLossCoeffs(
            lambda_mel=ocfg.lambda_mel,
            lambda_mr_stft=ocfg.lambda_mr_stft,
            lambda_mrd=ocfg.lambda_mrd,
        ),
    )

    optimizer = functools.partial(
        torch.optim.AdamW,
        lr=ocfg.learning_rate,
        betas=(ocfg.adam_b1, ocfg.adam_b2),
        weight_decay=ocfg.weight_decay,
    )
    scheduler = functools.partial(
        get_cosine_schedule_with_warmup,
        num_warmup_steps=ocfg.warmup_steps,
        num_training_steps=1,  # rewritten in configure_optimizers
    )

    return {
        "dim": ocfg.dim,
        "generator": generator,
        "vocoder": vocoder,
        "discriminator": discriminator,
        "train_args": OptiSpeechTrainArgs(
            cache_generator_outputs=ocfg.cache_generator_outputs,
            gradient_clip_val=ocfg.gradient_clip_val,
            gradient_accumulate_batches=None,
            pretraining_steps=ocfg.pretraining_steps,
        ),
        "data_args": OptiSpeechDataArgs(
            num_speakers=ocfg.num_speakers,
            batch_size=ocfg.batch_size,
            num_workers=ocfg.num_workers,
            pin_memory=False,
            text_processor=text_processor,
            feature_extractor=feature_extractor,
            data_statistics=None,
        ),
        "inference_args": OptiSpeechInferenceArgs(
            d_factor=ocfg.d_factor,
            p_factor=ocfg.p_factor,
            e_factor=ocfg.e_factor,
        ),
        "optimizer": optimizer,
        "scheduler": scheduler,
    }


def build_optispeech(ocfg: "OptiSpeechEngineConfig") -> "pl.LightningModule":
    """Construct an :class:`OptiSpeech` LightningModule from an explicit config."""
    from phoonnx_train.optispeech.model.optispeech import OptiSpeech

    feature_extractor = _build_feature_extractor(ocfg)
    text_processor = _build_text_processor(ocfg)
    kwargs = _build_model_kwargs(ocfg, feature_extractor, text_processor)
    model = OptiSpeech(**kwargs)
    # Keep the flat config around for export/metadata.
    model._engine_config = ocfg
    return model


class OptiSpeechTrainingEngine(BaseTrainingEngine):

    def trainer_kwargs(self) -> Dict[str, Any]:
        # OptiSpeech uses manual optimization; a global gradient_clip_val is
        # rejected by Lightning in that mode. Clipping is applied inside the
        # LightningModule via train_args.gradient_clip_val.
        return {}

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    def create_model(
        self,
        config: TrainingEngineConfig,
        dataset_paths: List[Path],
        **kwargs: Any,
    ) -> "pl.LightningModule":
        ocfg = OptiSpeechEngineConfig.from_training_config(config)
        return build_optispeech(ocfg)

    def extra_preprocess(
        self,
        utterance_audio_path: Path,
        cache_dir: Path,
        sample_rate: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Pitch/energy features are extracted by the vendored feature
        # extractor inside the datamodule; nothing extra is cached per
        # utterance at this stage.
        return {}

    def export_onnx(
        self,
        checkpoint_path: Path,
        config_path: Path,
        output_dir: Path,
        **kwargs: Any,
    ) -> Path:
        """Export a trained OptiSpeech checkpoint to ONNX.

        Produces the input/output contract the phoonnx ``OptiSpeechAdapter``
        expects:

          inputs:  ``x``, ``x_lengths``, ``scales`` (=[d, p, e]),
                   optional ``sids`` / ``lids``
          outputs: ``wav``, ``wav_lengths``, ``durations``

        All runtime config is embedded under the ``inference`` metadata key
        as a JSON blob (sample_rate, inference_args, input_symbols,
        special_symbols, text_processor, speakers, languages).
        """
        import torch

        model = self.load_lightning_module(checkpoint_path)
        model.eval()

        gen = model.generator
        is_multi_speaker = model.num_speakers > 1
        is_multi_language = getattr(model.text_processor, "is_multi_language", False)

        d0 = model.inference_args.d_factor
        p0 = model.inference_args.p_factor
        e0 = model.inference_args.e_factor

        dummy_len = 20
        # phoneme ids within a safe low range of the vocab
        x = torch.randint(1, 10, (1, dummy_len), dtype=torch.long)
        x_lengths = torch.LongTensor([dummy_len])
        scales = torch.tensor([d0, p0, e0], dtype=torch.float32)

        model_inputs: List[Any] = [x, x_lengths, scales]
        input_names = ["x", "x_lengths", "scales"]
        dynamic_axes: Dict[str, Dict[int, str]] = {
            "x": {0: "batch_size", 1: "time"},
            "x_lengths": {0: "batch_size"},
            "wav": {0: "batch_size", 1: "time"},
            "wav_lengths": {0: "batch_size"},
            "durations": {0: "batch_size", 1: "time"},
        }
        if is_multi_speaker:
            model_inputs.append(torch.LongTensor([0]))
            input_names.append("sids")
            dynamic_axes["sids"] = {0: "batch_size"}
        if is_multi_language:
            model_inputs.append(torch.LongTensor([0]))
            input_names.append("lids")
            dynamic_axes["lids"] = {0: "batch_size"}

        def onnx_forward(x, x_lengths, scales, sids=None, lids=None):
            out = gen.synthesise(
                x=x,
                x_lengths=x_lengths,
                sids=sids,
                lids=lids,
                d_factor=scales[0],
                p_factor=scales[1],
                e_factor=scales[2],
            )
            return out["wav"], out["wav_lengths"], out["durations"]

        model.forward = onnx_forward

        output_path = output_dir / "model.onnx"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        from phoonnx_train.torch_compat import onnx_export_kwargs

        torch.onnx.export(
            model,
            tuple(model_inputs),
            str(output_path),
            input_names=input_names,
            output_names=["wav", "wav_lengths", "durations"],
            dynamic_axes=dynamic_axes,
            opset_version=OPSET_VERSION,
            export_params=True,
            do_constant_folding=True,
            **onnx_export_kwargs(),
        )

        self._embed_metadata(output_path, model)
        _LOG.info("ONNX model exported to %s", output_path)
        return output_path

    def quality_presets(self) -> Dict[str, Dict[str, Any]]:
        return _QUALITY_PRESETS

    # ------------------------------------------------------------------
    # Checkpoint handling
    # ------------------------------------------------------------------

    def load_lightning_module(self, checkpoint_path: Path) -> "pl.LightningModule":
        """Reconstruct a full :class:`OptiSpeech` module from a checkpoint.

        The hyper-parameters saved by ``save_hyperparameters`` include the
        explicit dataclasses and ``functools.partial`` factories built by this
        engine, so the whole module (optimizer/scheduler config included) is
        rebuilt on load.
        """
        from phoonnx_train.optispeech.model.optispeech import OptiSpeech
        from phoonnx_train.torch_compat import trusting_torch_load

        # Our checkpoints embed the engine's dataclasses and partial factories;
        # torch>=2.6 defaults torch.load(weights_only=True) and rejects them.
        # Loading our own checkpoint is trusted.
        with trusting_torch_load():
            return OptiSpeech.load_from_checkpoint(
                str(checkpoint_path), map_location="cpu"
            )

    def load_checkpoint(
        self,
        model: "pl.LightningModule",
        checkpoint_path: Path,
        **kwargs: Any,
    ) -> "pl.LightningModule":
        """Warm-start *model* from a (possibly partial) checkpoint.

        Matches keys by name and shape, loads them, and logs precisely which
        keys were loaded and which were skipped — so a mismatched or partial
        checkpoint (e.g. different vocab / speaker count / vocoder) still
        transfers what it can.
        """
        import torch

        from phoonnx_train.torch_compat import trusting_torch_load

        with trusting_torch_load():
            ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model_state = model.state_dict()

        matched = {
            k: v
            for k, v in state_dict.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        skipped_ckpt = sorted(set(state_dict) - set(matched))
        missing_model = sorted(set(model_state) - set(matched))

        model.load_state_dict(matched, strict=False)

        _LOG.info(
            "warm-start: loaded %d/%d checkpoint tensors into model (%d model "
            "tensors left at init)",
            len(matched),
            len(state_dict),
            len(missing_model),
        )
        if skipped_ckpt:
            _LOG.info("warm-start: skipped %d checkpoint keys", len(skipped_ckpt))
            _LOG.debug("warm-start skipped keys: %s", skipped_ckpt)
        if missing_model:
            _LOG.debug("warm-start missing (kept at init) keys: %s", missing_model)
        return model

    # ------------------------------------------------------------------
    # Standalone evaluation synthesis
    # ------------------------------------------------------------------

    def eval_synthesize(
        self,
        checkpoint_path: Path,
        config: Optional[Dict[str, Any]],
        *,
        vocoder_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """Return ``callable(ids, scales, sid) -> np.ndarray`` for an OptiSpeech checkpoint.

        OptiSpeech is self-vocoding — the generator emits the waveform
        directly (WaveNeXt head) — so ``vocoder_path`` never applies here
        and is ignored with a debug log if given.

        ``scales`` is ``[d_factor, p_factor, e_factor]`` (duration / pitch /
        energy), matching ``OptiSpeechGenerator.synthesise`` and the
        checkpoint's own ``inference_args`` defaults when omitted.
        """
        import numpy as np
        import torch

        if vocoder_path:
            _LOG.debug(
                "OptiSpeech is self-vocoding; ignoring vocoder_path=%s", vocoder_path
            )

        model = self.load_lightning_module(Path(checkpoint_path))
        model = model.to(device)
        model.eval()

        gen = model.generator
        is_multi_speaker = model.num_speakers > 1
        is_multi_language = getattr(model.text_processor, "is_multi_language", False)

        d0 = model.inference_args.d_factor
        p0 = model.inference_args.p_factor
        e0 = model.inference_args.e_factor

        def _synth(ids: List[int], scales: List[float], sid: Optional[int]) -> "np.ndarray":
            # scales=None/[] means the checkpoint's own inference defaults;
            # when set it is [d_factor, p_factor, e_factor] (OptiSpeech
            # semantics, NOT VITS's noise/length/noise_w).
            scales = scales or []
            x = torch.LongTensor(ids).unsqueeze(0).to(device)
            x_lengths = torch.LongTensor([len(ids)]).to(device)

            d_factor = float(scales[0]) if len(scales) > 0 else d0
            p_factor = float(scales[1]) if len(scales) > 1 else p0
            e_factor = float(scales[2]) if len(scales) > 2 else e0

            sids = None
            if is_multi_speaker:
                sids = torch.LongTensor([sid if sid is not None else 0]).to(device)
            lids = torch.LongTensor([0]).to(device) if is_multi_language else None

            with torch.no_grad():
                out = gen.synthesise(
                    x=x,
                    x_lengths=x_lengths,
                    sids=sids,
                    lids=lids,
                    d_factor=d_factor,
                    p_factor=p_factor,
                    e_factor=e_factor,
                )
            wav = out["wav"].detach().cpu().numpy().astype(np.float32).reshape(-1)
            if not np.isfinite(wav).all():
                raise RuntimeError(
                    "OptiSpeech eval_synthesize produced non-finite samples "
                    "(NaN/Inf) — checkpoint is likely broken"
                )
            return wav

        return _synth

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _embed_metadata(onnx_path: Path, model: "pl.LightningModule") -> None:
        import onnx

        tp = model.text_processor
        tokenizer = getattr(tp, "tokenizer", None)
        input_symbols = dict(getattr(tokenizer, "input_symbols", {}) or {})
        special_symbols = dict(getattr(tokenizer, "special_symbols", {}) or {})

        inference_meta = {
            "name": "optispeech",
            "sample_rate": int(model.sample_rate),
            "inference_args": {
                "d_factor": model.inference_args.d_factor,
                "p_factor": model.inference_args.p_factor,
                "e_factor": model.inference_args.e_factor,
            },
            "input_symbols": input_symbols,
            "special_symbols": special_symbols,
            "speakers": list(getattr(model, "speakers", []) or []),
            "languages": list(getattr(tp, "languages", []) or []),
            "text_processor": {
                "tokenizer": getattr(tokenizer, "name", tp.tokenizer_ref)
                if not isinstance(tp.tokenizer_ref, str)
                else tp.tokenizer_ref,
                "add_blank": bool(tp.add_blank),
                "add_bos_eos": bool(tp.add_bos_eos),
                "normalize_text": bool(tp.normalize_text),
                "languages": list(tp.languages),
            },
        }

        onnx_model = onnx.load(str(onnx_path))
        entry = onnx_model.metadata_props.add()
        entry.key = "inference"
        entry.value = json.dumps(inference_meta)
        type_entry = onnx_model.metadata_props.add()
        type_entry.key = "model_type"
        type_entry.value = "optispeech"
        onnx.save(onnx_model, str(onnx_path))
