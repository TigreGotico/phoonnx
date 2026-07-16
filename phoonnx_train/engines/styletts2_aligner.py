"""StyleTTS2 text-aligner training engine (``--engine styletts2-aligner``).

Lightning port of `yl4579/AuxiliaryASR <https://github.com/yl4579/AuxiliaryASR>`_
— the mel→phoneme ASR whose attention provides the TMA alignment target in
StyleTTS2 stage-1. Train one per language, or warm-start from the English one
via ``pretrained_path`` for a fine-tune.

Loss = CTC(ctc_logits) + CrossEntropy(s2s decoder), exactly upstream
``trainer.py``. Dataset: ``train_list.txt``/``val_list.txt``
(``filename|phonemes|speaker``) + ``wavs/`` — same layout as the StyleTTS2
engine; phonemize raw text lists first with
``python -m phoonnx_train.styletts2.phonemize_corpus``.

Output (``save_asr_checkpoint``): ``epoch_XXXXX.pth`` + ``config.yml`` in the
exact layout ``load_ASR_models`` consumes — point the StyleTTS2 engine's
``asr_path``/``asr_config`` at them.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml

from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig

LOG = logging.getLogger(__name__)


def _fused_kwargs() -> Dict[str, Any]:
    """``fused=True`` for AdamW needs CUDA and torch>=2."""
    if torch.cuda.is_available() and int(torch.__version__.split(".")[0]) >= 2:
        return {"fused": True}
    return {}


def load_aux_state_dict(checkpoint_path) -> Dict[str, Any]:
    """Load an aux-model checkpoint in any of its layouts: a Lightning
    ``state_dict`` (keys ``model.*``), the aux-engine ``{"model": ...}`` /
    ``{"net": ...}`` layouts, or a bare state_dict — returned with the
    Lightning ``model.`` prefix stripped so it loads into the inner module."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model", "net"):
        if isinstance(ckpt.get(key), dict):
            ckpt = ckpt[key]
            break
    return {(k[len("model."):] if k.startswith("model.") else k): v
            for k, v in ckpt.items()}


@dataclass
class AlignerConfig:
    sample_rate: int = 24000
    n_mels: int = 80
    n_token: int = 178
    hidden_dim: int = 256
    token_embedding_dim: int = 512
    n_layers: int = 6

    lr: float = 5e-4
    batch_size: int = 32
    num_workers: int = 2
    grad_clip_value: float = 5.0  # upstream clip_grad_value_(…, 5)
    compile_model: bool = False

    root_path: str = ""
    save_dir: Optional[str] = None  # where epoch_XXXXX.pth + config.yml land
    pretrained_path: Optional[str] = None  # warm-start (e.g. English aligner)

    @classmethod
    def from_training_config(cls, cfg: TrainingEngineConfig) -> "AlignerConfig":
        extra = dict(cfg.extra)
        extra.pop("quality", None)
        extra.pop("validation_split", None)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: extra[k] for k in list(extra) if k in known}
        kwargs.setdefault("n_token", cfg.num_symbols)
        kwargs.setdefault("sample_rate", cfg.sample_rate)
        return cls(**kwargs)


class AlignerModule(pl.LightningModule):

    def __init__(self, config: AlignerConfig,
                 train_list: Optional[List[str]] = None,
                 val_list: Optional[List[str]] = None):
        super().__init__()
        from phoonnx_train.styletts2.Utils.ASR.models import ASRCNN

        self.config = config
        self.train_list = train_list or []
        self.val_list = val_list or []
        self.model = ASRCNN(input_dim=config.n_mels,
                            hidden_dim=config.hidden_dim,
                            n_token=config.n_token,
                            n_layers=config.n_layers,
                            token_embedding_dim=config.token_embedding_dim)
        if config.pretrained_path:
            params = torch.load(config.pretrained_path, map_location="cpu")
            params = params.get("model", params)
            model_state = self.model.state_dict()
            filtered = {k: v for k, v in params.items()
                        if k in model_state and v.shape == model_state[k].shape}
            missing = len(model_state) - len(filtered)
            self.model.load_state_dict(filtered, strict=False)
            LOG.info("warm-started aligner from %s (%d tensors skipped)",
                     config.pretrained_path, missing)
        if config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        # the CTC blank is the space symbol — the same "silence" token the
        # dataset frames every utterance with (upstream train.py)
        from phoonnx_train.styletts2.meldataset import TextCleaner
        blank_index = TextCleaner().word_index_dictionary[" "]
        self.ctc = torch.nn.CTCLoss(blank=blank_index, zero_infinity=True)
        self.save_hyperparameters({"config": config.__dict__})

    # ------------------------------------------------------------------
    def _losses(self, batch):
        texts, text_lengths, mels, mel_lengths = batch
        mel_lengths = mel_lengths // (2 ** self.model.n_down)
        mel_mask = self.model.length_to_mask(mel_lengths)
        ctc_logit, s2s_logit, _s2s_attn = self.model(
            mels, src_key_padding_mask=mel_mask, text_input=texts)
        loss_ctc = self.ctc(ctc_logit.log_softmax(dim=2).transpose(0, 1),
                            texts, mel_lengths, text_lengths)
        loss_s2s = 0.0
        for pred, target, length in zip(s2s_logit, texts, text_lengths):
            loss_s2s += F.cross_entropy(pred[:length], target[:length],
                                        ignore_index=-1)
        loss_s2s = loss_s2s / texts.size(0)
        return loss_ctc, loss_s2s, s2s_logit

    def training_step(self, batch, batch_idx):
        loss_ctc, loss_s2s, _ = self._losses(batch)
        loss = loss_ctc + loss_s2s
        self.log_dict({"train/loss": loss, "train/ctc": loss_ctc,
                       "train/s2s": loss_s2s}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss_ctc, loss_s2s, s2s_logit = self._losses(batch)
        texts, text_lengths = batch[0], batch[1]
        preds = s2s_logit.argmax(dim=2)
        acc = torch.stack([
            torch.eq(t[:l], p[:l]).float().mean()
            for t, p, l in zip(texts, preds, text_lengths)]).mean()
        self.log_dict({"val_loss": loss_ctc + loss_s2s, "val/ctc": loss_ctc,
                       "val/s2s": loss_s2s, "val/acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        # upstream optimizers.py: AdamW(wd=5e-4, betas=(0.9, 0.98), eps=1e-9)
        # + per-step OneCycleLR(pct_start=0, final_div_factor=5)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr,
                                weight_decay=5e-4, betas=(0.9, 0.98),
                                eps=1e-9, **_fused_kwargs())
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.config.lr,
            total_steps=max(int(self.trainer.estimated_stepping_batches), 2),
            pct_start=0.0, final_div_factor=5)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None,
                                    gradient_clip_algorithm=None):
        torch.nn.utils.clip_grad_value_(self.model.parameters(),
                                        self.config.grad_clip_value)

    # ------------------------------------------------------------------
    def _dataloader(self, data_list, validation: bool):
        from phoonnx_train.styletts2.aligner_dataset import (AuxMelDataset,
                                                             build_aux_dataloader)
        ds = AuxMelDataset(data_list, root_path=self.config.root_path,
                           sr=self.config.sample_rate,
                           n_mels=self.config.n_mels)
        return build_aux_dataloader(ds, self.config.batch_size,
                                    self.config.num_workers,
                                    validation=validation)

    def train_dataloader(self):
        return self._dataloader(self.train_list, validation=False)

    def val_dataloader(self):
        if not self.val_list:
            return None
        return self._dataloader(self.val_list, validation=True)

    # ------------------------------------------------------------------
    def save_asr_checkpoint(self, out_dir: Path) -> Path:
        """Write ``epoch_XXXXX.pth`` + ``config.yml`` consumable by the
        StyleTTS2 engine's ``asr_path``/``asr_config``."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model = getattr(self.model, "_orig_mod", self.model)  # un-compile
        ckpt = out_dir / f"epoch_{self.current_epoch:05d}.pth"
        torch.save({"model": model.state_dict()}, ckpt)
        cfg = {
            "model_params": {
                "input_dim": self.config.n_mels,
                "hidden_dim": self.config.hidden_dim,
                "n_token": self.config.n_token,
                "n_layers": self.config.n_layers,
                "token_embedding_dim": self.config.token_embedding_dim,
            },
            "preprocess_params": {
                "sr": self.config.sample_rate,
                "spect_params": {"n_fft": 2048, "win_length": 1200,
                                 "hop_length": 300},
                "mel_params": {"n_mels": self.config.n_mels},
            },
        }
        (out_dir / "config.yml").write_text(yaml.safe_dump(cfg))
        return ckpt

    def on_train_epoch_end(self):
        if self.config.save_dir:
            self.save_asr_checkpoint(Path(self.config.save_dir))


def _read_lists(dataset_paths: List[Path]):
    train_list: List[str] = []
    val_list: List[str] = []
    root = ""
    for p in dataset_paths:
        p = Path(p)
        if p.is_dir():
            tl, vl = p / "train_list.txt", p / "val_list.txt"
            if tl.is_file():
                train_list.extend(tl.read_text(encoding="utf-8").splitlines())
            if vl.is_file():
                val_list.extend(vl.read_text(encoding="utf-8").splitlines())
            if (p / "wavs").is_dir():
                root = str(p / "wavs")
            elif not root:
                root = str(p)
        elif p.is_file():
            lines = p.read_text(encoding="utf-8").splitlines()
            (val_list if "val" in p.stem else train_list).extend(lines)
            if not root:
                root = str(p.parent)
    return train_list, val_list, root


class AlignerTrainingEngine(BaseTrainingEngine):
    """Trains the AuxiliaryASR text aligner consumed by ``--engine styletts2``."""

    def load_checkpoint(self, model: pl.LightningModule, checkpoint_path: Path,
                        **kwargs: Any) -> pl.LightningModule:
        state = load_aux_state_dict(checkpoint_path)
        model.model.load_state_dict(state, strict=False)
        return model

    def create_model(self, config: TrainingEngineConfig,
                     dataset_paths: List[Path], **kwargs: Any) -> pl.LightningModule:
        acfg = AlignerConfig.from_training_config(config)
        train_list, val_list, root = _read_lists(dataset_paths)
        if not train_list:
            raise FileNotFoundError(
                "styletts2-aligner needs a train_list.txt "
                "(filename|phonemes|speaker per line) in the dataset dir(s): "
                f"{[str(p) for p in dataset_paths]}")
        if not acfg.root_path:
            acfg.root_path = root
        return AlignerModule(acfg, train_list=train_list, val_list=val_list, **kwargs)

    def export_onnx(self, checkpoint_path: Path, config_path: Path,
                    output_dir: Path, **kwargs: Any) -> Path:
        raise NotImplementedError(
            "The text aligner is a training-time auxiliary model; it is "
            "consumed as a torch checkpoint via asr_path/asr_config.")

    def quality_presets(self) -> Dict[str, Dict[str, Any]]:
        return {
            "low": {"hidden_dim": 128, "token_embedding_dim": 256, "n_layers": 3},
            "medium": {},  # upstream AuxiliaryASR defaults
        }
