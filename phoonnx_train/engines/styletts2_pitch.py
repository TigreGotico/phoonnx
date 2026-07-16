"""StyleTTS2 pitch-extractor training engine (``--engine styletts2-pitch``).

Lightning port of `yl4579/PitchExtractor <https://github.com/yl4579/PitchExtractor>`_
— the JDCNet F0 estimator StyleTTS2 stage-1 uses for ground-truth pitch.
The English checkpoint transfers well across languages (BSC reused it for
Spanish/Catalan), so training one is optional; ``pretrained_path``
warm-starts a fine-tune.

Losses per upstream: ``lambda_f0 * SmoothL1(f0)`` + ``BCEWithLogits`` on the
voicing/silence labels, with pyworld harvest (dio fallback) ground-truth F0
cached next to the audio. Dataset: same ``train_list.txt``/``wavs/`` layout
as the other StyleTTS2 engines (the text field is unused).

Output (``save_f0_checkpoint``): a ``.t7`` with ``{"net": state_dict}`` —
exactly what ``load_F0_models`` consumes via the StyleTTS2 engine's
``f0_path``.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig
from phoonnx_train.engines.styletts2_aligner import _read_lists

LOG = logging.getLogger(__name__)


@dataclass
class PitchConfig:
    sample_rate: int = 24000
    n_mels: int = 80
    seq_len: int = 192  # JDCNet training segment (frames)

    lr: float = 3e-4  # upstream optimizer_params.lr
    lambda_f0: float = 0.1  # upstream loss_params.lambda_f0
    batch_size: int = 32
    num_workers: int = 2
    compile_model: bool = False

    root_path: str = ""
    save_dir: Optional[str] = None
    pretrained_path: Optional[str] = None  # warm-start (e.g. English bst.t7)

    @classmethod
    def from_training_config(cls, cfg: TrainingEngineConfig) -> "PitchConfig":
        extra = dict(cfg.extra)
        extra.pop("quality", None)
        extra.pop("validation_split", None)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: extra[k] for k in list(extra) if k in known}
        kwargs.setdefault("sample_rate", cfg.sample_rate)
        return cls(**kwargs)


class PitchSegmentDataset(torch.utils.data.Dataset):
    """Fixed-length (seq_len) mel segments + pyworld F0 + silence labels,
    per upstream MelDataset (features cached via AuxMelDataset)."""

    def __init__(self, data_list: List[str], root_path: str, sr: int,
                 seq_len: int):
        from phoonnx_train.styletts2.aligner_dataset import AuxMelDataset
        self.inner = AuxMelDataset(data_list, root_path=root_path, sr=sr,
                                   with_f0=True)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int):
        mel, _text, f0 = self.inner[idx]
        n = mel.size(1)
        if n > self.seq_len:
            start = np.random.randint(0, n - self.seq_len)
            mel = mel[:, start:start + self.seq_len]
            f0 = f0[start:start + self.seq_len]
        elif n < self.seq_len:
            mel = F.pad(mel, (0, self.seq_len - n))
            f0 = F.pad(f0, (0, self.seq_len - n))
        is_silence = (f0 == 0).float()
        f0 = torch.nan_to_num(f0, nan=0.0)
        return mel, f0, is_silence


def _jdc_forward_with_detector(model, x):
    """Upstream yl4579/PitchExtractor ``JDCNet.forward`` (classifier +
    voicing detector). The vendored StyleTTS2 JDCNet keeps all the layers but
    its forward skips the detector head, so training runs this instead —
    state_dict stays exactly ``load_F0_models``-compatible.

    x: (b, 1, seq_len, n_mels) -> f0 (b, seq_len, 1), voicing (b, seq_len).
    """
    seq_len = x.shape[-2]
    convblock_out = model.conv_block(x)
    resblock1_out = model.res_block1(convblock_out)
    resblock2_out = model.res_block2(resblock1_out)
    resblock3_out = model.res_block3(resblock2_out)
    poolblock_out = model.pool_block(resblock3_out)

    classifier_out = poolblock_out.permute(0, 2, 1, 3).contiguous().view(
        (-1, seq_len, 512))
    classifier_out, _ = model.bilstm_classifier(classifier_out)
    classifier_out = classifier_out.contiguous().view((-1, 512))
    classifier_out = model.classifier(classifier_out)
    classifier_out = classifier_out.view((-1, seq_len, model.num_class))

    mp1_out = model.maxpool1(convblock_out)
    mp2_out = model.maxpool2(resblock1_out)
    mp3_out = model.maxpool3(resblock2_out)
    concat_out = torch.cat((mp1_out, mp2_out, mp3_out, poolblock_out), dim=1)
    detector_out = model.detector_conv(concat_out)
    detector_out = detector_out.permute(0, 2, 1, 3).contiguous().view(
        (-1, seq_len, 512))
    detector_out, _ = model.bilstm_detector(detector_out)
    detector_out = detector_out.contiguous().view((-1, 512))
    detector_out = model.detector(detector_out)
    detector_out = detector_out.view((-1, seq_len, 2)).sum(axis=-1)

    return classifier_out, detector_out


class PitchModule(pl.LightningModule):

    def __init__(self, config: PitchConfig,
                 train_list: Optional[List[str]] = None,
                 val_list: Optional[List[str]] = None):
        super().__init__()
        from phoonnx_train.styletts2.Utils.JDC.model import JDCNet

        self.config = config
        self.train_list = train_list or []
        self.val_list = val_list or []
        self.model = JDCNet(num_class=1, seq_len=config.seq_len)
        if config.pretrained_path:
            params = torch.load(config.pretrained_path, map_location="cpu")
            params = params.get("net", params.get("model", params))
            model_state = self.model.state_dict()
            filtered = {k: v for k, v in params.items()
                        if k in model_state and v.shape == model_state[k].shape}
            self.model.load_state_dict(filtered, strict=False)
            LOG.info("warm-started pitch extractor from %s", config.pretrained_path)
        if config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        self.save_hyperparameters({"config": config.__dict__})

    def _losses(self, batch):
        mels, f0s, sils = batch
        # (B, n_mels, T) -> (B, 1, T, n_mels) as upstream x.transpose(-1, -2)
        x = mels.unsqueeze(1).transpose(-1, -2)
        model = getattr(self.model, "_orig_mod", self.model)
        f0_pred, sil_pred = _jdc_forward_with_detector(model, x)
        loss_f0 = self.config.lambda_f0 * F.smooth_l1_loss(f0_pred.squeeze(-1), f0s)
        loss_sil = F.binary_cross_entropy_with_logits(sil_pred, sils)
        return loss_f0, loss_sil

    def training_step(self, batch, batch_idx):
        loss_f0, loss_sil = self._losses(batch)
        loss = loss_f0 + loss_sil
        self.log_dict({"train/loss": loss, "train/f0": loss_f0,
                       "train/sil": loss_sil}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss_f0, loss_sil = self._losses(batch)
        self.log_dict({"val_loss": loss_f0 + loss_sil, "val/f0": loss_f0,
                       "val/sil": loss_sil}, prog_bar=True)

    def configure_optimizers(self):
        fused = torch.cuda.is_available()
        return torch.optim.AdamW(self.model.parameters(), lr=self.config.lr,
                                 fused=fused)

    def _dataloader(self, data_list, validation: bool):
        ds = PitchSegmentDataset(data_list, root_path=self.config.root_path,
                                 sr=self.config.sample_rate,
                                 seq_len=self.config.seq_len)
        kwargs: Dict[str, Any] = dict(
            batch_size=self.config.batch_size, shuffle=not validation,
            drop_last=not validation, num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.config.num_workers > 0)
        if self.config.num_workers > 0:
            kwargs["prefetch_factor"] = 4
        return torch.utils.data.DataLoader(ds, **kwargs)

    def train_dataloader(self):
        return self._dataloader(self.train_list, validation=False)

    def val_dataloader(self):
        if not self.val_list:
            return None
        return self._dataloader(self.val_list, validation=True)

    # ------------------------------------------------------------------
    def save_f0_checkpoint(self, path: Path) -> Path:
        """Write a ``.t7`` consumable by ``load_F0_models`` (``f0_path``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model = getattr(self.model, "_orig_mod", self.model)
        torch.save({"net": model.state_dict()}, path)
        return path

    def on_train_epoch_end(self):
        if self.config.save_dir:
            self.save_f0_checkpoint(
                Path(self.config.save_dir) / f"epoch_{self.current_epoch:05d}.t7")


class PitchTrainingEngine(BaseTrainingEngine):
    """Trains the JDC pitch extractor consumed by ``--engine styletts2``."""

    def create_model(self, config: TrainingEngineConfig,
                     dataset_paths: List[Path], **kwargs: Any) -> pl.LightningModule:
        pcfg = PitchConfig.from_training_config(config)
        train_list, val_list, root = _read_lists(dataset_paths)
        if not train_list:
            raise FileNotFoundError(
                "styletts2-pitch needs a train_list.txt (filename|text|speaker "
                f"per line) in the dataset dir(s): {[str(p) for p in dataset_paths]}")
        if not pcfg.root_path:
            pcfg.root_path = root
        return PitchModule(pcfg, train_list=train_list, val_list=val_list, **kwargs)

    def export_onnx(self, checkpoint_path: Path, config_path: Path,
                    output_dir: Path, **kwargs: Any) -> Path:
        raise NotImplementedError(
            "The pitch extractor is a training-time auxiliary model; it is "
            "consumed as a torch checkpoint via f0_path.")

    def quality_presets(self) -> Dict[str, Dict[str, Any]]:
        return {"medium": {}}  # JDCNet has a fixed upstream architecture
