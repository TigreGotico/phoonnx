"""Torch ``LightningModule`` for the StyleTTS2 text-aligner engine.

Split out of :mod:`phoonnx_train.engines.styletts2_aligner` so the engine
module — and therefore the training registry — imports torch-free; the heavy
torch / pytorch_lightning class lives here and is imported lazily when a model
is actually built. See the engine module for the AuxiliaryASR recipe and
citation notes.
"""
import logging
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml

from phoonnx_train.engines.styletts2_aligner import AlignerConfig, _fused_kwargs

LOG = logging.getLogger(__name__)


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
