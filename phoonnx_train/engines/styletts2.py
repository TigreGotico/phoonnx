"""StyleTTS2 training engine — full two-stage from-scratch training plus fine-tuning.

Ports the yl4579/StyleTTS2 recipe (MIT) onto the phoonnx training framework,
using the vendored upstream code in ``phoonnx_train/styletts2``:

- **stage ``first``** — acoustic pre-training: mel reconstruction through the
  decoder with ground-truth F0/energy, plus transferable monotonic alignment
  (TMA) training of the text aligner (s2s CE + monotonicity L1) and
  MPD/MRSD adversarial + SLM feature-matching losses after ``tma_epoch``.
- **stage ``second``** — joint training: duration (CE + L1) and prosody
  (F0/energy) predictors driven by PL-BERT, style-diffusion denoiser (EDM
  loss + style reconstruction, after ``diff_epoch``), and joint decoder
  training with SLM adversarial loss (after ``joint_epoch``).
- **stage ``finetune``** — the second-stage recipe with diffusion and joint
  training enabled from epoch 0, starting from an existing checkpoint.

This trains new models from scratch in new languages: point ``plbert_dir``
at the PL-BERT for your language (or a multilingual one), supply the
aligner/pitch-extractor checkpoints, and run stage ``first`` then
``second``.

Auxiliary models (all configurable; set ``download_aux: true`` to fetch the
yl4579 English ones automatically, or train your own with the
``styletts2-aligner`` / ``styletts2-plbert`` / ``styletts2-pitch`` engines):

- ``asr_path``/``asr_config`` — text aligner (ASRCNN); yl4579's English one
  is language-independent enough for most Latin-script languages, and is
  *itself trained further* during TMA.
- ``f0_path`` — JDC pitch extractor checkpoint.
- ``plbert_dir`` — PL-BERT directory (config.yml + step_*.t7).
- ``slm.model`` — WavLM (HF hub id) for the SLM losses; disable with
  ``use_slm: false`` (e.g. offline CI).

When an auxiliary checkpoint is not provided the module is randomly
initialized with a warning — correct for unit tests and for users training
every component from scratch, wrong for quick fine-tunes.

Paper: Li et al., "StyleTTS 2: Towards Human-Level Text-to-Speech through
Style Diffusion and Adversarial Training with Large Speech Language Models"
(NeurIPS 2023, https://arxiv.org/abs/2306.07691). Reference implementation:
https://github.com/yl4579/StyleTTS2 (MIT), vendored in
``phoonnx_train/styletts2``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig

if TYPE_CHECKING:  # heavy imports — only for type annotations
    import pytorch_lightning as pl

_LOG = logging.getLogger(__name__)

# Upstream Configs/config.yml (LJSpeech) model_params — the reference single
# speaker recipe.
_DEFAULT_MODEL_PARAMS: Dict[str, Any] = {
    "multispeaker": False,
    "dim_in": 64,
    "hidden_dim": 512,
    "max_conv_dim": 512,
    "n_layer": 3,
    "n_mels": 80,
    "n_token": 178,
    "max_dur": 50,
    "style_dim": 128,
    "dropout": 0.2,
    "decoder": {
        "type": "istftnet",
        "resblock_kernel_sizes": [3, 7, 11],
        "upsample_rates": [10, 6],
        "upsample_initial_channel": 512,
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "upsample_kernel_sizes": [20, 12],
        "gen_istft_n_fft": 20,
        "gen_istft_hop_size": 5,
    },
    "slm": {
        "model": "microsoft/wavlm-base-plus",
        "sr": 16000,
        "hidden": 768,
        "nlayers": 13,
        "initial_channel": 64,
    },
    "diffusion": {
        "embedding_mask_proba": 0.1,
        "transformer": {
            "num_layers": 3,
            "num_heads": 8,
            "head_features": 64,
            "multiplier": 2,
        },
        "dist": {
            "sigma_data": 0.2,
            "estimate_sigma_data": True,
            "mean": -3.0,
            "std": 1.0,
        },
    },
}

# Upstream loss_params defaults (LJSpeech config).
_DEFAULT_LOSS_PARAMS: Dict[str, Any] = {
    "lambda_mel": 5.0,
    "lambda_gen": 1.0,
    "lambda_slm": 1.0,
    "lambda_mono": 1.0,
    "lambda_s2s": 1.0,
    "tma_epoch": 50,
    "lambda_F0": 1.0,
    "lambda_norm": 1.0,
    "lambda_dur": 1.0,
    "lambda_ce": 20.0,
    "lambda_sty": 1.0,
    "lambda_diff": 1.0,
    "diff_epoch": 20,
    "joint_epoch": 50,
}

_DEFAULT_SLMADV_PARAMS: Dict[str, Any] = {
    "min_len": 400,
    "max_len": 500,
    "batch_percentage": 0.5,
    "iter": 10,
    "thresh": 5,
    "scale": 0.01,
    "sig": 1.5,
}

# Quality tiers select the model size / decoder family:
#   low    — halved widths, istftnet decoder (fast, small)
#   medium — upstream LJSpeech recipe (istftnet)
#   high   — upstream LibriTTS recipe (hifigan decoder, multispeaker-ready)
_QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "low": {
        "hidden_dim": 256,
        "max_conv_dim": 256,
        "style_dim": 64,
        "decoder": {
            "type": "istftnet",
            "resblock_kernel_sizes": [3, 7, 11],
            "upsample_rates": [10, 6],
            # must stay 512: the Decoder trunk hardcodes a 512-channel output
            # into the generator's first upsampler
            "upsample_initial_channel": 512,
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_kernel_sizes": [20, 12],
            "gen_istft_n_fft": 20,
            "gen_istft_hop_size": 5,
        },
    },
    "medium": {},  # upstream LJSpeech defaults
    "high": {
        "decoder": {
            "type": "hifigan",
            "resblock_kernel_sizes": [3, 7, 11],
            "upsample_rates": [10, 5, 3, 2],
            "upsample_initial_channel": 512,
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_kernel_sizes": [20, 10, 6, 4],
        },
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class StyleTTS2Config:
    stage: str = "first"  # first | second | finetune
    model_params: Dict[str, Any] = field(default_factory=dict)
    loss_params: Dict[str, Any] = field(default_factory=dict)
    slmadv_params: Dict[str, Any] = field(default_factory=dict)

    sample_rate: int = 24000
    max_len: int = 400  # max mel frames per training clip
    hop_length: int = 300

    # optimizer (upstream OneCycle with div_factor=1 == constant LR)
    lr: float = 1e-4
    bert_lr: float = 1e-5
    ft_lr: float = 1e-5

    # data
    batch_size: int = 8
    num_workers: int = 2
    root_path: str = ""
    ood_data: Optional[str] = None
    min_length: int = 50

    # auxiliary models
    asr_config: Optional[str] = None
    asr_path: Optional[str] = None
    f0_path: Optional[str] = None
    plbert_dir: Optional[str] = None
    # auto-download the yl4579 English auxiliaries for any path left unset
    # (disable for from-scratch new-language training or offline tests)
    download_aux: bool = False

    # SLM (WavLM) losses — disable for offline/CPU-only runs
    use_slm: bool = True
    # SLM adversarial run (joint phase, stage second/finetune)
    use_slm_adv: bool = True

    # stage second/finetune: starting checkpoint (stage-1 output or full model)
    first_stage_path: Optional[str] = None

    @classmethod
    def from_training_config(cls, cfg: TrainingEngineConfig) -> "StyleTTS2Config":
        extra = dict(cfg.extra)
        extra.pop("validation_split", None)  # handled by the dataset list split
        quality = extra.pop("quality", "medium")
        preset = _QUALITY_PRESETS.get(quality, {})
        model_overrides = _deep_merge(preset, extra.pop("model_params", {}))
        # train.py flattens the quality-preset kwargs into extra — route any
        # model-level keys (hidden_dim, decoder, ...) into model_params
        for k in list(extra):
            if k in _DEFAULT_MODEL_PARAMS:
                model_overrides = _deep_merge(model_overrides, {k: extra.pop(k)})
        model_overrides.setdefault("n_token", cfg.num_symbols)
        model_overrides.setdefault("multispeaker", cfg.num_speakers > 1)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: extra[k] for k in list(extra) if k in known}
        kwargs["model_params"] = model_overrides
        kwargs.setdefault("sample_rate", cfg.sample_rate)
        return cls(**kwargs)

    def resolved_model_params(self) -> Dict[str, Any]:
        return _deep_merge(_DEFAULT_MODEL_PARAMS, self.model_params)

    def resolved_loss_params(self) -> Dict[str, Any]:
        merged = _deep_merge(_DEFAULT_LOSS_PARAMS, self.loss_params)
        if self.stage == "finetune":
            merged["diff_epoch"] = 0
            merged["joint_epoch"] = 0
        return merged

    def resolved_slmadv_params(self) -> Dict[str, Any]:
        return _deep_merge(_DEFAULT_SLMADV_PARAMS, self.slmadv_params)


# ----------------------------------------------------------------------
# Auxiliary model builders
# ----------------------------------------------------------------------

def _resolve_aux_paths(cfg: StyleTTS2Config) -> None:
    """Fill any unset aux-model path from the yl4579 English release
    (downloaded and cached) when ``download_aux`` is enabled."""
    if not cfg.download_aux:
        return
    from phoonnx_train.styletts2.downloads import download_aux_models
    paths = download_aux_models()
    cfg.asr_path = cfg.asr_path or paths["asr_path"]
    cfg.asr_config = cfg.asr_config or paths["asr_config"]
    cfg.f0_path = cfg.f0_path or paths["f0_path"]
    cfg.plbert_dir = cfg.plbert_dir or paths["plbert_dir"]


def _build_text_aligner(cfg: StyleTTS2Config, n_mels: int, n_token: int):
    from phoonnx_train.styletts2.models import load_ASR_models
    from phoonnx_train.styletts2.Utils.ASR.models import ASRCNN

    if cfg.asr_path and cfg.asr_config:
        return load_ASR_models(cfg.asr_path, cfg.asr_config)
    _LOG.warning("No ASR aligner checkpoint given — random init (train-from-scratch/test mode)")
    return ASRCNN(input_dim=n_mels, hidden_dim=256, n_token=n_token,
                  n_layers=6, token_embedding_dim=512)


def _build_pitch_extractor(cfg: StyleTTS2Config):
    from phoonnx_train.styletts2.models import load_F0_models
    from phoonnx_train.styletts2.Utils.JDC.model import JDCNet

    if cfg.f0_path:
        return load_F0_models(cfg.f0_path)
    _LOG.warning("No JDC pitch-extractor checkpoint given — random init (train-from-scratch/test mode)")
    return JDCNet(num_class=1, seq_len=192)


def _build_plbert(cfg: StyleTTS2Config, n_token: int):
    if cfg.plbert_dir:
        from phoonnx_train.styletts2.Utils.PLBERT.util import load_plbert
        return load_plbert(cfg.plbert_dir)
    _LOG.warning("No PL-BERT directory given — building a small random-init ALBERT; "
                 "for real training supply a (multilingual) PL-BERT via plbert_dir")
    from transformers import AlbertConfig, AlbertModel

    class CustomAlbert(AlbertModel):
        def forward(self, *args, **kwargs):
            return super().forward(*args, **kwargs).last_hidden_state

    return CustomAlbert(AlbertConfig(
        vocab_size=max(n_token, 178), hidden_size=256, num_attention_heads=4,
        intermediate_size=512, num_hidden_layers=2, max_position_embeddings=512,
        embedding_size=128, dropout=0.1))


# ----------------------------------------------------------------------
# LightningModule
# ----------------------------------------------------------------------

def __getattr__(name):
    """Lazily expose the torch ``LightningModule`` (kept in the vendored
    package so this engine module imports torch-free)."""
    if name == "StyleTTS2Module":
        from phoonnx_train.styletts2.tts_module import StyleTTS2Module
        return StyleTTS2Module
    raise AttributeError(name)


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------

class StyleTTS2TrainingEngine(BaseTrainingEngine):
    """Full two-stage StyleTTS2 training (from scratch or fine-tune)."""

    def create_model(
        self,
        config: TrainingEngineConfig,
        dataset_paths: List[Path],
        **kwargs: Any,
    ) -> pl.LightningModule:
        from phoonnx_train.styletts2.tts_module import StyleTTS2Module
        scfg = StyleTTS2Config.from_training_config(config)
        train_list: List[str] = []
        val_list: List[str] = []

        def _read(p: Path) -> List[str]:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            return [ln for ln in lines if ln.strip()]

        for p in dataset_paths:
            p = Path(p)
            if p.is_dir():
                # upstream layout: train_list.txt / val_list.txt + wavs/
                tl, vl = p / "train_list.txt", p / "val_list.txt"
                if tl.exists():
                    train_list.extend(_read(tl))
                if vl.exists():
                    val_list.extend(_read(vl))
                if not scfg.root_path:
                    wavs = p / "wavs"
                    scfg.root_path = str(wavs if wavs.is_dir() else p)
            else:
                (val_list if "val" in p.stem else train_list).extend(_read(p))
                if not scfg.root_path:
                    scfg.root_path = str(p.parent)
        if not train_list:
            raise ValueError(
                "StyleTTS2 needs a train_list.txt (filename|text|speaker per line) "
                f"in the dataset dir(s): {[str(p) for p in dataset_paths]}")
        return StyleTTS2Module(scfg, train_list=train_list, val_list=val_list, **kwargs)

    def export_onnx(
        self,
        checkpoint_path: Path,
        config_path: Path,
        output_dir: Path,
        **kwargs: Any,
    ) -> Path:
        from phoonnx_train.styletts2.export import export_styletts2_onnx
        return export_styletts2_onnx(Path(checkpoint_path), Path(config_path),
                                     Path(output_dir), **kwargs)

    def quality_presets(self) -> Dict[str, Dict[str, Any]]:
        return _QUALITY_PRESETS

    def load_checkpoint(
        self,
        model: pl.LightningModule,
        checkpoint_path: Path,
        **kwargs: Any,
    ) -> pl.LightningModule:
        from phoonnx_train.styletts2.tts_module import StyleTTS2Module
        assert isinstance(model, StyleTTS2Module)
        model._load_net_checkpoint(Path(checkpoint_path))
        return model
