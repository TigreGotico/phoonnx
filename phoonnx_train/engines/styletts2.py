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
import copy as _copy
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig

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

class StyleTTS2Module(pl.LightningModule):
    """Lightning port of yl4579 ``train_first.py`` / ``train_second.py`` /
    ``train_finetune.py`` with manual optimization (one optimizer per
    sub-module, exactly like upstream's MultiOptimizer)."""

    automatic_optimization = False

    def __init__(self, config: StyleTTS2Config,
                 train_list: Optional[List[str]] = None,
                 val_list: Optional[List[str]] = None,
                 **_: Any):
        super().__init__()
        self.cfg = config
        self.train_list = train_list or []
        self.val_list = val_list or []
        self.save_hyperparameters({"styletts2": config.__dict__})

        from munch import Munch

        from phoonnx_train.styletts2.models import build_model
        from phoonnx_train.styletts2.utils import recursive_munch

        mp = recursive_munch(config.resolved_model_params())
        self.model_params = mp
        self.loss_params = Munch(config.resolved_loss_params())
        self.multispeaker = bool(mp.multispeaker)

        _resolve_aux_paths(config)
        text_aligner = _build_text_aligner(config, mp.n_mels, mp.n_token)
        pitch_extractor = _build_pitch_extractor(config)
        plbert = _build_plbert(config, mp.n_token)

        model = build_model(mp, text_aligner, pitch_extractor, plbert)
        # register everything so Lightning moves/saves it
        self.nets = nn.ModuleDict({k: model[k] for k in model})
        self.model = model  # Munch view, used by the ported training code
        self.n_down = model.text_aligner.n_down

        from phoonnx_train.styletts2.losses import (DiscriminatorLoss,
                                                    GeneratorLoss,
                                                    MultiResolutionSTFTLoss)
        self.stft_loss = MultiResolutionSTFTLoss()
        self.gl = GeneratorLoss(model.mpd, model.msd)
        self.dl = DiscriminatorLoss(model.mpd, model.msd)
        self.wl = None  # WavLM — built lazily (downloads from HF)
        self._sampler = None
        self._slmadv = None
        self._loaded_first_stage = False

    # -- lazy heavy pieces --------------------------------------------

    def _wavlm_loss(self):
        if self.wl is None and self.cfg.use_slm:
            from phoonnx_train.styletts2.losses import WavLMLoss
            self.wl = WavLMLoss(self.model_params.slm.model, self.model.wd,
                                self.cfg.sample_rate,
                                self.model_params.slm.sr).to(self.device)
        return self.wl

    def _diffusion_sampler(self):
        if self._sampler is None:
            from phoonnx_train.styletts2.Modules.diffusion.sampler import (
                ADPM2Sampler, DiffusionSampler, KarrasSchedule)
            self._sampler = DiffusionSampler(
                self.model.diffusion.diffusion,
                sampler=ADPM2Sampler(),
                sigma_schedule=KarrasSchedule(sigma_min=1e-4, sigma_max=3.0, rho=9.0),
                clamp=False)
        return self._sampler

    def _slm_adv(self):
        if self._slmadv is None and self.cfg.use_slm and self.cfg.use_slm_adv:
            from munch import Munch

            from phoonnx_train.styletts2.Modules.slmadv import SLMAdversarialLoss
            p = Munch(self.cfg.resolved_slmadv_params())
            self._slmadv = SLMAdversarialLoss(
                self.model, self._wavlm_loss(), self._diffusion_sampler(),
                p.min_len, p.max_len, batch_percentage=p.batch_percentage,
                skip_update=p.iter, sig=p.sig)
        return self._slmadv

    def setup(self, stage: Optional[str] = None) -> None:
        if (self.cfg.stage in ("second", "finetune")
                and self.cfg.first_stage_path and not self._loaded_first_stage):
            self._load_net_checkpoint(Path(self.cfg.first_stage_path))
            if self.cfg.stage == "second":
                # stage-1 checkpoints have no trained prosodic style encoder
                self.model.predictor_encoder = _copy.deepcopy(self.model.style_encoder)
                self.nets["predictor_encoder"] = self.model.predictor_encoder
            self._loaded_first_stage = True

    def _load_net_checkpoint(self, path: Path) -> None:
        """Load an upstream-layout ('net' dict of per-module state_dicts,
        possibly DataParallel 'module.'-prefixed) or Lightning checkpoint."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "net" in state:
            params = state["net"]
            for key in self.model:
                if key not in params:
                    continue
                sd = {(k[7:] if k.startswith("module.") else k): v
                      for k, v in params[key].items()}
                self.model[key].load_state_dict(sd, strict=False)
                _LOG.info("StyleTTS2: loaded %s from %s", key, path)
        elif "state_dict" in state:
            self.load_state_dict(state["state_dict"], strict=False)
        else:
            _LOG.warning("Unrecognised checkpoint layout at %s", path)

    # -- optimizers ----------------------------------------------------

    def configure_optimizers(self):
        self._opt_keys = list(self.model.keys())
        opts = []
        for key in self._opt_keys:
            if key == "bert":
                opts.append(torch.optim.AdamW(
                    self.model[key].parameters(), lr=self.cfg.bert_lr,
                    weight_decay=0.01, betas=(0.9, 0.99), eps=1e-9))
            elif key in ("decoder", "style_encoder") and self.cfg.stage != "first":
                opts.append(torch.optim.AdamW(
                    self.model[key].parameters(), lr=self.cfg.ft_lr,
                    weight_decay=1e-4, betas=(0.0, 0.99), eps=1e-9))
            else:
                opts.append(torch.optim.AdamW(
                    self.model[key].parameters(), lr=self.cfg.lr,
                    weight_decay=1e-4, betas=(0.0, 0.99), eps=1e-9))
        return opts

    def _step_opts(self, keys: List[str]) -> None:
        opts = self.optimizers()
        if not isinstance(opts, list):
            opts = [opts]
        for key in keys:
            opts[self._opt_keys.index(key)].step()

    def _zero_grad(self) -> None:
        opts = self.optimizers()
        if not isinstance(opts, list):
            opts = [opts]
        for o in opts:
            o.zero_grad()

    # -- data ----------------------------------------------------------

    def _dataloader(self, path_list: List[str], validation: bool):
        from phoonnx_train.styletts2.meldataset import build_dataloader
        return build_dataloader(
            path_list, self.cfg.root_path,
            OOD_data=self.cfg.ood_data or self.cfg.root_path,
            min_length=self.cfg.min_length,
            batch_size=self.cfg.batch_size,
            validation=validation,
            num_workers=0 if validation else self.cfg.num_workers,
            device=str(self.device))

    def train_dataloader(self):
        return self._dataloader(self.train_list, validation=False)

    def val_dataloader(self):
        if not self.val_list:
            return None
        return self._dataloader(self.val_list, validation=True)

    # -- shared alignment front-end -------------------------------------

    def _align(self, mels, mel_input_length, texts, input_lengths, grad: bool):
        from phoonnx_train.styletts2.monotonic import mask_from_lens, maximum_path
        from phoonnx_train.styletts2.utils import length_to_mask

        mask = length_to_mask(mel_input_length // (2 ** self.n_down)).to(self.device)
        text_mask = length_to_mask(input_lengths).to(texts.device)

        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            ppgs, s2s_pred, s2s_attn = self.model.text_aligner(mels, mask, texts)
        s2s_attn = s2s_attn.transpose(-1, -2)
        s2s_attn = s2s_attn[..., 1:]
        s2s_attn = s2s_attn.transpose(-1, -2)

        with torch.no_grad():
            attn_mask = ((~mask).unsqueeze(-1)
                         .expand(mask.shape[0], mask.shape[1], text_mask.shape[-1])
                         .float().transpose(-1, -2))
            attn_mask = attn_mask * ((~text_mask).unsqueeze(-1)
                                     .expand(text_mask.shape[0], text_mask.shape[1],
                                             mask.shape[-1]).float())
            attn_mask = (attn_mask < 1)
        s2s_attn = s2s_attn.masked_fill(attn_mask, 0.0)

        with torch.no_grad():
            mask_ST = mask_from_lens(s2s_attn, input_lengths,
                                     mel_input_length // (2 ** self.n_down))
            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

        return s2s_pred, s2s_attn, s2s_attn_mono, text_mask

    # -- stage 1 ---------------------------------------------------------

    def _training_step_first(self, batch) -> None:
        from phoonnx_train.styletts2.utils import log_norm

        waves = batch[0]
        texts, input_lengths, _, _, mels, mel_input_length, _ = [
            b.to(self.device) if torch.is_tensor(b) else b for b in batch[1:]]
        lp = self.loss_params
        tma = self.current_epoch >= lp.tma_epoch
        hop = self.cfg.hop_length

        s2s_pred, s2s_attn, s2s_attn_mono, text_mask = self._align(
            mels, mel_input_length, texts, input_lengths, grad=True)

        t_en = self.model.text_encoder(texts, input_lengths, text_mask)
        # 50% chance of using the monotonic version
        asr = (t_en @ s2s_attn) if bool(random.getrandbits(1)) else (t_en @ s2s_attn_mono)

        mel_len = min(int(mel_input_length.min().item() / 2 - 1), self.cfg.max_len // 2)
        mel_len_st = int(mel_input_length.min().item() / 2 - 1)
        if mel_len <= 0 or mel_len_st <= 0:
            return

        en, gt, wav, st = [], [], [], []
        for bib in range(len(mel_input_length)):
            mel_length = int(mel_input_length[bib].item() / 2)
            rs = np.random.randint(0, max(1, mel_length - mel_len))
            en.append(asr[bib, :, rs:rs + mel_len])
            gt.append(mels[bib, :, rs * 2:(rs + mel_len) * 2])
            y = waves[bib][rs * 2 * hop:(rs + mel_len) * 2 * hop]
            wav.append(torch.as_tensor(y, device=self.device))
            rs = np.random.randint(0, max(1, mel_length - mel_len_st))
            st.append(mels[bib, :, rs * 2:(rs + mel_len_st) * 2])

        en = torch.stack(en)
        gt = torch.stack(gt).detach()
        st = torch.stack(st).detach()
        wav = torch.stack(wav).float().detach()

        if gt.shape[-1] < 80:  # too short for the style encoder
            return

        with torch.no_grad():
            real_norm = log_norm(gt.unsqueeze(1)).squeeze(1).detach()
            F0_real, _, _ = self.model.pitch_extractor(gt.unsqueeze(1))

        s = self.model.style_encoder(
            st.unsqueeze(1) if self.multispeaker else gt.unsqueeze(1))
        y_rec = self.model.decoder(en, F0_real, real_norm, s)

        # discriminator step
        if tma:
            self._zero_grad()
            d_loss = self.dl(wav.detach().unsqueeze(1).float(), y_rec.detach()).mean()
            self.manual_backward(d_loss)
            self._step_opts(["msd", "mpd"])
            self.log("d_loss", d_loss, prog_bar=False)

        # generator step
        self._zero_grad()
        loss_mel = self.stft_loss(y_rec.squeeze(1), wav.detach())

        if tma:
            loss_s2s = 0
            for _p, _t, _l in zip(s2s_pred, texts, input_lengths):
                loss_s2s += F.cross_entropy(_p[:_l], _t[:_l])
            loss_s2s /= texts.size(0)
            loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10
            loss_gen_all = self.gl(wav.detach().unsqueeze(1).float(), y_rec).mean()
            wl = self._wavlm_loss()
            loss_slm = wl(wav.detach(), y_rec).mean() if wl is not None else 0
            g_loss = (lp.lambda_mel * loss_mel + lp.lambda_mono * loss_mono
                      + lp.lambda_s2s * loss_s2s + lp.lambda_gen * loss_gen_all
                      + lp.lambda_slm * loss_slm)
            self.log_dict({"mono_loss": loss_mono, "s2s_loss": loss_s2s})
        else:
            g_loss = loss_mel

        self.manual_backward(g_loss)
        self._step_opts(["text_encoder", "style_encoder", "decoder"])
        if tma:
            self._step_opts(["text_aligner", "pitch_extractor"])

        self.log_dict({"train_loss": g_loss, "mel_loss": loss_mel}, prog_bar=True)

    # -- stage 2 / finetune ------------------------------------------------

    def _training_step_second(self, batch, batch_idx: int) -> None:
        from phoonnx_train.styletts2.utils import log_norm

        waves = batch[0]
        texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = [
            b.to(self.device) if torch.is_tensor(b) else b for b in batch[1:]]
        lp = self.loss_params
        start_ds = self.current_epoch >= lp.diff_epoch
        joint = self.current_epoch >= lp.joint_epoch
        hop = self.cfg.hop_length

        try:
            s2s_pred, s2s_attn, s2s_attn_mono, text_mask = self._align(
                mels, mel_input_length, texts, input_lengths, grad=False)
        except Exception:  # upstream skips misaligned batches the same way
            return

        with torch.no_grad():
            t_en = self.model.text_encoder(texts, input_lengths, text_mask)
            asr = t_en @ s2s_attn_mono
            d_gt = s2s_attn_mono.sum(axis=-1).detach()

            ref = None
            if self.multispeaker and start_ds:
                ref_ss = self.model.style_encoder(ref_mels.unsqueeze(1))
                ref_sp = self.model.predictor_encoder(ref_mels.unsqueeze(1))
                ref = torch.cat([ref_ss, ref_sp], dim=1)

        # full-utterance styles (per-item: avgpool can't be batched w/ padding)
        ss, gs = [], []
        for bib in range(len(mel_input_length)):
            mel = mels[bib, :, :mel_input_length[bib]]
            ss.append(self.model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1)))
            gs.append(self.model.style_encoder(mel.unsqueeze(0).unsqueeze(1)))
        s_dur = torch.stack(ss).squeeze(1).squeeze(1)
        gs = torch.stack(gs).squeeze(1).squeeze(1)
        s_trg = torch.cat([gs, s_dur], dim=-1).detach()

        bert_dur = self.model.bert(texts, attention_mask=(~text_mask).int())
        d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

        # style diffusion (denoiser) training
        if start_ds:
            sampler = self._diffusion_sampler()
            num_steps = np.random.randint(3, 5)
            if self.model_params.diffusion.dist.estimate_sigma_data:
                self.model.diffusion.diffusion.sigma_data = \
                    s_trg.std(axis=-1).mean().item()
            noise = torch.randn_like(s_trg).unsqueeze(1)
            if self.multispeaker:
                s_preds = sampler(noise=noise, embedding=bert_dur, embedding_scale=1,
                                  features=ref, embedding_mask_proba=0.1,
                                  num_steps=num_steps).squeeze(1)
                loss_diff = self.model.diffusion(
                    s_trg.unsqueeze(1), embedding=bert_dur, features=ref).mean()
            else:
                s_preds = sampler(noise=noise, embedding=bert_dur, embedding_scale=1,
                                  embedding_mask_proba=0.1,
                                  num_steps=num_steps).squeeze(1)
                loss_diff = self.model.diffusion.diffusion(
                    s_trg.unsqueeze(1), embedding=bert_dur).mean()
            loss_sty = F.l1_loss(s_preds, s_trg.detach())
        else:
            loss_sty = 0
            loss_diff = 0

        d, p = self.model.predictor(d_en, s_dur, input_lengths, s2s_attn_mono, text_mask)

        mel_len = min(int(mel_input_length.min().item() / 2 - 1), self.cfg.max_len // 2)
        mel_len_st = int(mel_input_length.min().item() / 2 - 1)
        if mel_len <= 0 or mel_len_st <= 0:
            return

        en, gt, st, p_en, wav = [], [], [], [], []
        for bib in range(len(mel_input_length)):
            mel_length = int(mel_input_length[bib].item() / 2)
            rs = np.random.randint(0, max(1, mel_length - mel_len))
            en.append(asr[bib, :, rs:rs + mel_len])
            p_en.append(p[bib, :, rs:rs + mel_len])
            gt.append(mels[bib, :, rs * 2:(rs + mel_len) * 2])
            y = waves[bib][rs * 2 * hop:(rs + mel_len) * 2 * hop]
            wav.append(torch.as_tensor(y, device=self.device))
            rs = np.random.randint(0, max(1, mel_length - mel_len_st))
            st.append(mels[bib, :, rs * 2:(rs + mel_len_st) * 2])

        wav = torch.stack(wav).float().detach()
        en = torch.stack(en)
        p_en = torch.stack(p_en)
        gt = torch.stack(gt).detach()
        st = torch.stack(st).detach()
        if gt.size(-1) < 80:
            return

        s_dur = self.model.predictor_encoder(
            st.unsqueeze(1) if self.multispeaker else gt.unsqueeze(1))
        s = self.model.style_encoder(
            st.unsqueeze(1) if self.multispeaker else gt.unsqueeze(1))

        with torch.no_grad():
            F0_real, _, F0 = self.model.pitch_extractor(gt.unsqueeze(1))
            N_real = log_norm(gt.unsqueeze(1)).squeeze(1)
            y_rec_gt = wav.unsqueeze(1)
            y_rec_gt_pred = self.model.decoder(en, F0_real, N_real, s)
            # decoder tuned during joint phase -> real recording as target;
            # frozen before that -> reconstruction as target
            wav_tgt = y_rec_gt if joint else y_rec_gt_pred

        F0_fake, N_fake = self.model.predictor.F0Ntrain(p_en, s_dur)
        y_rec = self.model.decoder(en, F0_fake, N_fake, s)

        loss_F0_rec = F.smooth_l1_loss(F0_real, F0_fake) / 10
        loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)

        if start_ds:
            self._zero_grad()
            d_loss = self.dl(wav_tgt.detach(), y_rec.detach()).mean()
            self.manual_backward(d_loss)
            self._step_opts(["msd", "mpd"])
            self.log("d_loss", d_loss)

        self._zero_grad()
        loss_mel = self.stft_loss(y_rec.squeeze(1), wav_tgt.squeeze(1))
        loss_gen_all = self.gl(wav_tgt, y_rec).mean() if start_ds else 0
        wl = self._wavlm_loss()
        loss_lm = wl(wav_tgt.detach().squeeze(1), y_rec.squeeze(1)).mean() if wl is not None else 0

        loss_ce = 0
        loss_dur = 0
        for _pred, _dur, _len in zip(d, d_gt, input_lengths):
            _pred = _pred[:_len, :]
            _dur = _dur[:_len].long()
            _trg = torch.zeros_like(_pred)
            for i in range(_trg.shape[0]):
                _trg[i, :_dur[i]] = 1
            _dur_pred = torch.sigmoid(_pred).sum(axis=1)
            loss_dur += F.l1_loss(_dur_pred[1:_len - 1], _dur[1:_len - 1])
            loss_ce += F.binary_cross_entropy_with_logits(_pred.flatten(), _trg.flatten())
        loss_ce /= texts.size(0)
        loss_dur /= texts.size(0)

        g_loss = (lp.lambda_mel * loss_mel + lp.lambda_F0 * loss_F0_rec
                  + lp.lambda_ce * loss_ce + lp.lambda_norm * loss_norm_rec
                  + lp.lambda_dur * loss_dur + lp.lambda_gen * loss_gen_all
                  + lp.lambda_slm * loss_lm + lp.lambda_sty * loss_sty
                  + lp.lambda_diff * loss_diff)

        if torch.isnan(g_loss):
            _LOG.warning("NaN generator loss — skipping step")
            return
        self.manual_backward(g_loss)
        self._step_opts(["bert_encoder", "bert", "predictor", "predictor_encoder"])
        if start_ds:
            self._step_opts(["diffusion"])
        if joint:
            self._step_opts(["style_encoder", "decoder"])

        self.log_dict({"train_loss": g_loss, "mel_loss": loss_mel,
                       "dur_loss": loss_dur, "ce_loss": loss_ce,
                       "F0_loss": loss_F0_rec, "norm_loss": loss_norm_rec},
                      prog_bar=True)
        if start_ds:
            self.log_dict({"sty_loss": loss_sty, "diff_loss": loss_diff})

        # SLM adversarial run (joint phase only)
        slmadv = self._slm_adv() if joint else None
        if slmadv is not None:
            sp = self.cfg.resolved_slmadv_params()
            use_ind = np.random.rand() < 0.5
            if use_ind:
                ref_lengths = input_lengths
                ref_texts = texts
            slm_out = slmadv(batch_idx, y_rec_gt, y_rec_gt_pred, waves,
                             mel_input_length, ref_texts, ref_lengths, use_ind,
                             s_trg.detach(), ref if self.multispeaker else None)
            if slm_out is None:
                return
            d_loss_slm, loss_gen_lm, _ = slm_out

            self._zero_grad()
            self.manual_backward(loss_gen_lm)

            # gradient scaling against the predictor's gradient norm
            total_norm = 0.0
            for prm in self.model.predictor.parameters():
                if prm.grad is not None:
                    total_norm += prm.grad.detach().norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            if total_norm > sp["thresh"]:
                for key in self.model.keys():
                    for prm in self.model[key].parameters():
                        if prm.grad is not None:
                            prm.grad *= (1 / total_norm)
            for mod in (self.model.predictor.duration_proj,
                        self.model.predictor.lstm, self.model.diffusion):
                for prm in mod.parameters():
                    if prm.grad is not None:
                        prm.grad *= sp["scale"]

            self._step_opts(["bert_encoder", "bert", "predictor", "diffusion"])

            if torch.is_tensor(d_loss_slm) and d_loss_slm != 0:
                self._zero_grad()
                self.manual_backward(d_loss_slm, retain_graph=True)
                self._step_opts(["wd"])
            self.log_dict({"slm_d_loss": d_loss_slm, "slm_gen_loss": loss_gen_lm})

    # -- Lightning hooks -------------------------------------------------

    def training_step(self, batch, batch_idx: int):
        if self.cfg.stage == "first":
            self._training_step_first(batch)
        else:
            self._training_step_second(batch, batch_idx)
        return None

    def validation_step(self, batch, batch_idx: int):
        from phoonnx_train.styletts2.utils import log_norm

        waves = batch[0]
        texts, input_lengths, _, _, mels, mel_input_length, _ = [
            b.to(self.device) if torch.is_tensor(b) else b for b in batch[1:]]
        hop = self.cfg.hop_length
        with torch.no_grad():
            _, s2s_attn, s2s_attn_mono, text_mask = self._align(
                mels, mel_input_length, texts, input_lengths, grad=False)
            t_en = self.model.text_encoder(texts, input_lengths, text_mask)
            asr = t_en @ (s2s_attn if self.cfg.stage == "first" else s2s_attn_mono)

            mel_len = min(int(mel_input_length.min().item() / 2 - 1),
                          self.cfg.max_len // 2)
            if mel_len <= 0:
                return None
            en, gt, wav = [], [], []
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)
                rs = np.random.randint(0, max(1, mel_length - mel_len))
                en.append(asr[bib, :, rs:rs + mel_len])
                gt.append(mels[bib, :, rs * 2:(rs + mel_len) * 2])
                y = waves[bib][rs * 2 * hop:(rs + mel_len) * 2 * hop]
                wav.append(torch.as_tensor(y, device=self.device))
            en = torch.stack(en)
            gt = torch.stack(gt).detach()
            wav = torch.stack(wav).float().detach()
            if gt.shape[-1] < 80:
                return None

            F0_real, _, _ = self.model.pitch_extractor(gt.unsqueeze(1))
            s = self.model.style_encoder(gt.unsqueeze(1))
            real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
            y_rec = self.model.decoder(en, F0_real, real_norm, s)
            loss_mel = self.stft_loss(y_rec.squeeze(1), wav)
        self.log("val_loss", loss_mel, prog_bar=True)
        return loss_mel

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # also store the upstream 'net' layout for interop with yl4579 tooling
        checkpoint["net"] = {k: self.model[k].state_dict() for k in self.model}


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
        assert isinstance(model, StyleTTS2Module)
        model._load_net_checkpoint(Path(checkpoint_path))
        return model
