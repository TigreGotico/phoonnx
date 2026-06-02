"""
Matcha-TTS training engine adapter.

Wraps the upstream ``matcha-tts`` package behind the
``BaseTrainingEngine`` interface.

Training a Matcha-TTS model from scratch still requires the upstream
config format (see ``configs/`` in the Matcha-TTS repo).  This adapter
provides the bridge: it accepts phoonnx's ``TrainingEngineConfig`` and
assembles the nested dicts the upstream ``MatchaTTS`` expects.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch

from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig

_LOG = logging.getLogger(__name__)

# ONNX opset used for export
OPSET_VERSION = 15

# Quality tier → Matcha hyper-param overrides
_QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "x-low": {
        "encoder_channels": 128,
        "encoder_filter_channels": 512,
        "encoder_filter_channels_dp": 192,
        "encoder_n_heads": 2,
        "encoder_n_layers": 4,
        "decoder_channels": [192, 192],
        "decoder_num_heads": 2,
        "decoder_num_mid_blocks": 2,
    },
    "medium": {
        "encoder_channels": 192,
        "encoder_filter_channels": 768,
        "encoder_filter_channels_dp": 256,
        "encoder_n_heads": 2,
        "encoder_n_layers": 6,
        "decoder_channels": [256, 256],
        "decoder_num_heads": 2,
        "decoder_num_mid_blocks": 2,
    },
    "high": {
        "encoder_channels": 256,
        "encoder_filter_channels": 1024,
        "encoder_filter_channels_dp": 384,
        "encoder_n_heads": 4,
        "encoder_n_layers": 8,
        "decoder_channels": [384, 384],
        "decoder_num_heads": 4,
        "decoder_num_mid_blocks": 2,
    },
}


def _dict_to_namespace(d: Dict[str, Any]) -> Any:
    """Recursively convert a dict to SimpleNamespace for attribute access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    return d


@dataclass
class MatchaEngineConfig:
    """Extended config for Matcha-TTS training."""

    n_vocab: int = 178
    n_spks: int = 1
    spk_emb_dim: int = 64
    n_feats: int = 80
    out_size: Optional[int] = None
    prior_loss: bool = True
    use_precomputed_durations: bool = False

    # Encoder
    encoder_type: str = "RoPE Encoder"
    encoder_channels: int = 192
    encoder_filter_channels: int = 768
    encoder_filter_channels_dp: int = 256
    encoder_n_heads: int = 2
    encoder_n_layers: int = 6
    encoder_kernel_size: int = 3
    encoder_p_dropout: float = 0.1
    encoder_prenet: bool = True

    # Decoder
    decoder_channels: List[int] = None  # type: ignore[assignment]
    decoder_dropout: float = 0.05
    decoder_attention_head_dim: int = 64
    decoder_n_blocks: int = 1
    decoder_num_mid_blocks: int = 2
    decoder_num_heads: int = 2
    decoder_act_fn: str = "snakebeta"
    decoder_down_block_type: str = "transformer"
    decoder_mid_block_type: str = "transformer"
    decoder_up_block_type: str = "transformer"

    # CFM
    cfm_name: str = "CFM"
    cfm_solver: str = "euler"
    cfm_sigma_min: float = 1e-4

    # Data
    mel_mean: float = -5.536622
    mel_std: float = 2.116101

    def __post_init__(self):
        if self.decoder_channels is None:
            self.decoder_channels = [256, 256]

    @classmethod
    def from_training_config(cls, cfg: TrainingEngineConfig) -> "MatchaEngineConfig":
        """Build from shared TrainingEngineConfig + extra overrides."""
        extra = dict(cfg.extra)
        preset_name = extra.pop("quality", "medium")
        preset = _QUALITY_PRESETS.get(preset_name, _QUALITY_PRESETS["medium"])

        # Override preset with any explicit extra params
        preset.update(extra)

        return cls(
            n_vocab=cfg.num_symbols,
            n_spks=cfg.num_speakers,
            **preset,
        )


class MatchaTrainingEngine(BaseTrainingEngine):

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    def create_model(
        self,
        config: TrainingEngineConfig,
        dataset_paths: List[Path],
        **kwargs: Any,
    ) -> pl.LightningModule:
        """Build a MatchaTTS LightningModule from *config*."""
        from matcha.models.matcha_tts import MatchaTTS

        mcfg = MatchaEngineConfig.from_training_config(config)

        encoder_cfg = _dict_to_namespace(
            {
                "encoder_type": mcfg.encoder_type,
                "encoder_params": {
                    "n_feats": mcfg.n_feats,
                    "n_channels": mcfg.encoder_channels,
                    "filter_channels": mcfg.encoder_filter_channels,
                    "filter_channels_dp": mcfg.encoder_filter_channels_dp,
                    "n_heads": mcfg.encoder_n_heads,
                    "n_layers": mcfg.encoder_n_layers,
                    "kernel_size": mcfg.encoder_kernel_size,
                    "p_dropout": mcfg.encoder_p_dropout,
                    "spk_emb_dim": mcfg.spk_emb_dim,
                    "n_spks": mcfg.n_spks,
                    "prenet": mcfg.encoder_prenet,
                },
                "duration_predictor_params": {
                    "filter_channels_dp": mcfg.encoder_filter_channels_dp,
                    "kernel_size": mcfg.encoder_kernel_size,
                    "p_dropout": mcfg.encoder_p_dropout,
                },
            }
        )

        decoder_cfg = {
            "channels": tuple(mcfg.decoder_channels),
            "dropout": mcfg.decoder_dropout,
            "attention_head_dim": mcfg.decoder_attention_head_dim,
            "n_blocks": mcfg.decoder_n_blocks,
            "num_mid_blocks": mcfg.decoder_num_mid_blocks,
            "num_heads": mcfg.decoder_num_heads,
            "act_fn": mcfg.decoder_act_fn,
            "down_block_type": mcfg.decoder_down_block_type,
            "mid_block_type": mcfg.decoder_mid_block_type,
            "up_block_type": mcfg.decoder_up_block_type,
        }

        cfm_cfg = _dict_to_namespace(
            {
                "name": mcfg.cfm_name,
                "solver": mcfg.cfm_solver,
                "sigma_min": mcfg.cfm_sigma_min,
            }
        )

        data_statistics = {"mel_mean": mcfg.mel_mean, "mel_std": mcfg.mel_std}

        model = MatchaTTS(
            n_vocab=mcfg.n_vocab,
            n_spks=mcfg.n_spks,
            spk_emb_dim=mcfg.spk_emb_dim,
            n_feats=mcfg.n_feats,
            encoder=encoder_cfg,
            decoder=decoder_cfg,
            cfm=cfm_cfg,
            data_statistics=data_statistics,
            out_size=mcfg.out_size,
            prior_loss=mcfg.prior_loss,
            use_precomputed_durations=mcfg.use_precomputed_durations,
        )
        return model

    def export_onnx(
        self,
        checkpoint_path: Path,
        config_path: Path,
        output_dir: Path,
        **kwargs: Any,
    ) -> Path:
        """Export a Matcha checkpoint to ONNX (mel model only)."""
        from matcha.cli import load_matcha

        n_timesteps = kwargs.get("n_timesteps", 5)

        _LOG.info("Loading Matcha checkpoint from %s", checkpoint_path)
        matcha = load_matcha(checkpoint_path.stem, checkpoint_path, "cpu")
        matcha.eval()

        is_multi_speaker = matcha.n_spks > 1

        # Dummy inputs
        dummy_input_length = 50
        x = torch.randint(low=0, high=20, size=(1, dummy_input_length), dtype=torch.long)
        x_lengths = torch.LongTensor([dummy_input_length])
        scales = torch.Tensor([0.667, 1.0])

        model_inputs = [x, x_lengths, scales]
        input_names = ["x", "x_lengths", "scales"]

        if is_multi_speaker:
            spks = torch.LongTensor([1])
            model_inputs.append(spks)
            input_names.append("spks")

        # Monkey-patch forward for ONNX export
        def onnx_forward_func(x, x_lengths, scales, spks=None):
            temperature = scales[0]
            length_scale = scales[1]
            output = matcha.synthesise(
                x, x_lengths, n_timesteps, temperature, spks, length_scale
            )
            return output["mel"], output["mel_lengths"]

        matcha.forward = onnx_forward_func

        dynamic_axes = {
            "x": {0: "batch_size", 1: "time"},
            "x_lengths": {0: "batch_size"},
            "mel": {0: "batch_size", 2: "time"},
            "mel_lengths": {0: "batch_size"},
        }
        if is_multi_speaker:
            dynamic_axes["spks"] = {0: "batch_size"}

        output_path = output_dir / "model.onnx"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        matcha.to_onnx(
            str(output_path),
            tuple(model_inputs),
            input_names=input_names,
            output_names=["mel", "mel_lengths"],
            dynamic_axes=dynamic_axes,
            opset_version=OPSET_VERSION,
            export_params=True,
            do_constant_folding=True,
        )
        _LOG.info("ONNX model exported to %s", output_path)
        return output_path

    def quality_presets(self) -> Dict[str, Dict[str, Any]]:
        return _QUALITY_PRESETS

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def load_checkpoint(
        self,
        model: pl.LightningModule,
        checkpoint_path: Path,
        **kwargs: Any,
    ) -> pl.LightningModule:
        """Load Matcha checkpoint with standard state dict."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        return model
