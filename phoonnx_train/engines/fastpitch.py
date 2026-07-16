"""
FastPitch / SpeedySpeech training engine adapter.

Both architectures are configurations of the vendored ``ForwardTTS`` model
(``phoonnx_train/fastpitch/model.py`` — a self-contained, pure-torch port of
coqui-ai/TTS's ``TTS/tts/models/forward_tts.py``, MPL-2.0, see
``phoonnx_train/fastpitch/__init__.py`` for the license note):

- **FastPitch**: FFT-transformer encoder/decoder, pitch predictor on.
- **SpeedySpeech**: residual conv-BN encoder/decoder, no pitch predictor.

Durations are learned unsupervised (no external aligner needed) via the
vendored ``AlignmentNetwork`` + monotonic alignment search, same recipe as
upstream FastPitch/coqui.

Reuses the shared VITS preprocessing pipeline: ``audio_norm_path`` /
``audio_spec_path`` (linear spectrogram) are produced by
``phoonnx_train.preprocess``; the mel target is derived from the linear
spectrogram at train time via ``phoonnx_train.vits.mel_processing`` (no
separate mel cache needed). Pitch (F0) is extracted via
``extra_preprocess`` (pyworld), following the same contract used by the
OptiSpeech training engine.

ONNX export follows the contract consumed by ``phoonnx.engines.fastpitch``
(``FastPitchAdapter``, which reuses ``MixerTTSAdapter``'s feed/parse logic):
inputs ``token_ids`` [B,T] (+ optional ``speaker`` [B] for multi-speaker) ->
output ``mel_spec`` [B, 80, T_mel]. This mirrors
``scripts/conversion/coqui_fastpitch_export/export_fp.py``, which exports
pretrained Coqui FastPitch/SpeedySpeech checkpoints with the same I/O.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import LongTensor
from torch.utils.data import DataLoader, Dataset, random_split

from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig
from phoonnx_train.fastpitch.losses import ForwardTTSLoss
from phoonnx_train.fastpitch.model import ForwardTTS, ForwardTTSArgs
from phoonnx_train.vits.dataset import PhoonnxDataset, Utterance
from phoonnx_train.vits.mel_processing import spec_to_mel_torch

_LOG = logging.getLogger(__name__)

# ONNX opset used for export (matches coqui_fastpitch_export/export_fp.py)
OPSET_VERSION = 14

# Quality tier -> ForwardTTS hyper-param overrides.
# "variant" selects encoder/decoder family + pitch predictor (set via
# ``config.extra['variant']``, default "fastpitch").
_QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "x-low": {
        "hidden_channels": 128,
        "hidden_channels_ffn": 512,
        "encoder_num_layers": 4,
        "decoder_num_layers": 4,
        "num_heads": 1,
    },
    "medium": {
        "hidden_channels": 384,
        "hidden_channels_ffn": 1024,
        "encoder_num_layers": 6,
        "decoder_num_layers": 6,
        "num_heads": 1,
    },
    "high": {
        "hidden_channels": 512,
        "hidden_channels_ffn": 1536,
        "encoder_num_layers": 6,
        "decoder_num_layers": 6,
        "num_heads": 2,
    },
}

# SpeedySpeech-specific overrides layered on top of the tier preset when
# ``variant == "speedyspeech"``.
_SPEEDYSPEECH_OVERRIDES: Dict[str, Any] = {
    "encoder_type": "residual_conv_bn",
    "decoder_type": "residual_conv_bn",
    "use_pitch": False,
}


class UtteranceTensors:
    """Per-utterance training tensors: ids, mel (via spec), optional f0."""

    __slots__ = ("phoneme_ids", "spectrogram", "speaker_id", "pitch")

    def __init__(self, phoneme_ids: LongTensor, spectrogram: torch.Tensor,
                 speaker_id: Optional[LongTensor], pitch: Optional[torch.Tensor]):
        self.phoneme_ids = phoneme_ids
        self.spectrogram = spectrogram
        self.speaker_id = speaker_id
        self.pitch = pitch


class Batch:
    __slots__ = (
        "phoneme_ids", "phoneme_lengths", "mels", "mel_lengths",
        "pitch", "speaker_ids",
    )

    def __init__(self, phoneme_ids, phoneme_lengths, mels, mel_lengths, pitch, speaker_ids):
        self.phoneme_ids = phoneme_ids
        self.phoneme_lengths = phoneme_lengths
        self.mels = mels
        self.mel_lengths = mel_lengths
        self.pitch = pitch
        self.speaker_ids = speaker_ids


class ForwardTTSDataset(Dataset):
    """
    Wraps :class:`phoonnx_train.vits.dataset.PhoonnxDataset` utterances,
    converting the shared linear-spectrogram cache to mel at load time and
    optionally loading a pitch (F0) cache produced by
    :meth:`ForwardTTSTrainingEngine.extra_preprocess`.
    """

    def __init__(self, dataset_paths: List[Path], mel_channels: int,
                 filter_length: int, sample_rate: int,
                 mel_fmin: float, mel_fmax: Optional[float],
                 max_phoneme_ids: Optional[int] = None):
        self._inner = PhoonnxDataset(dataset_paths, max_phoneme_ids=max_phoneme_ids)
        self.mel_channels = mel_channels
        self.filter_length = filter_length
        self.sample_rate = sample_rate
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        # corpus F0 statistics over voiced frames — the pitch target is
        # z-scored so its scale matches the other loss terms and the
        # pitch_mul/pitch_add inference controls operate on a normalized
        # quantity (raw Hz would dwarf every other loss)
        self.pitch_mean, self.pitch_std = self._pitch_stats(dataset_paths)

    def _f0_candidate(self, utt: "Utterance") -> Path:
        return Path(str(utt.audio_spec_path)).with_suffix("").with_suffix(".f0.npy")

    def _pitch_stats(self, dataset_paths: List[Path]):
        import json

        import numpy as np

        stats_path = None
        for p in dataset_paths:
            p = Path(p)
            if p.is_dir():
                stats_path = p / "pitch_stats.json"
                break
        if stats_path and stats_path.is_file():
            stats = json.loads(stats_path.read_text())
            return float(stats["mean"]), float(stats["std"])

        voiced = []
        for utt in self._inner.utterances:
            cand = self._f0_candidate(utt)
            if cand.exists():
                f0 = np.load(cand)
                voiced.append(f0[f0 > 0])
        if not voiced:
            return 0.0, 1.0  # no pitch caches — identity normalization
        allv = np.concatenate(voiced)
        mean = float(allv.mean()) if allv.size else 0.0
        std = float(allv.std()) if allv.size else 1.0
        std = std or 1.0
        if stats_path:
            try:
                stats_path.write_text(json.dumps({"mean": mean, "std": std}))
            except OSError:
                pass
        return mean, std

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> UtteranceTensors:
        utt: Utterance = self._inner.utterances[idx]
        spec = torch.load(utt.audio_spec_path)  # [n_fft//2+1, T]
        mel = spec_to_mel_torch(
            spec.unsqueeze(0), self.filter_length, self.mel_channels,
            self.sample_rate, self.mel_fmin, self.mel_fmax,
        ).squeeze(0)  # [mel_channels, T]

        pitch = None
        # optional sidecar cache written by extra_preprocess: "<stem>.f0.npy"
        # next to audio_norm_path's parent cache dir.
        f0_candidate = self._f0_candidate(utt)
        if f0_candidate.exists():
            import numpy as np

            f0 = np.load(f0_candidate).astype("float32")
            # z-score voiced frames with the corpus stats; unvoiced stays 0
            voiced = f0 > 0
            f0[voiced] = (f0[voiced] - self.pitch_mean) / self.pitch_std
            pitch = torch.from_numpy(f0).unsqueeze(0)  # [1, T_f0]

        return UtteranceTensors(
            phoneme_ids=LongTensor(utt.phoneme_ids),
            spectrogram=mel,
            speaker_id=LongTensor([utt.speaker_id]) if utt.speaker_id is not None else None,
            pitch=pitch,
        )


class ForwardTTSCollate:
    def __init__(self, is_multispeaker: bool, use_pitch: bool):
        self.is_multispeaker = is_multispeaker
        self.use_pitch = use_pitch

    def __call__(self, utterances: List[UtteranceTensors]) -> Batch:
        n = len(utterances)
        max_ph = max(u.phoneme_ids.size(0) for u in utterances)
        max_mel = max(u.spectrogram.size(1) for u in utterances)
        mel_channels = utterances[0].spectrogram.size(0)

        phoneme_ids = torch.zeros(n, max_ph, dtype=torch.long)
        phoneme_lengths = torch.zeros(n, dtype=torch.long)
        mels = torch.zeros(n, mel_channels, max_mel)
        mel_lengths = torch.zeros(n, dtype=torch.long)
        pitch = torch.zeros(n, 1, max_mel) if self.use_pitch else None
        speaker_ids = torch.zeros(n, dtype=torch.long) if self.is_multispeaker else None

        for i, utt in enumerate(utterances):
            pl_ = utt.phoneme_ids.size(0)
            ml = utt.spectrogram.size(1)
            phoneme_ids[i, :pl_] = utt.phoneme_ids
            phoneme_lengths[i] = pl_
            mels[i, :, :ml] = utt.spectrogram
            mel_lengths[i] = ml
            if self.use_pitch and utt.pitch is not None:
                pl_len = min(utt.pitch.size(-1), max_mel)
                pitch[i, :, :pl_len] = utt.pitch[:, :pl_len]
            if speaker_ids is not None and utt.speaker_id is not None:
                speaker_ids[i] = utt.speaker_id

        return Batch(phoneme_ids, phoneme_lengths, mels, mel_lengths, pitch, speaker_ids)


class ForwardTTSModule(pl.LightningModule):
    """LightningModule wrapping the vendored :class:`ForwardTTS` model."""

    def __init__(
        self,
        num_symbols: int,
        num_speakers: int = 1,
        sample_rate: int = 22050,
        filter_length: int = 1024,
        mel_channels: int = 80,
        mel_fmin: float = 0.0,
        mel_fmax: Optional[float] = None,
        variant: str = "fastpitch",
        hidden_channels: int = 384,
        hidden_channels_ffn: int = 1024,
        encoder_num_layers: int = 6,
        decoder_num_layers: int = 6,
        num_heads: int = 1,
        use_pitch: Optional[bool] = None,
        use_energy: bool = False,
        dataset: Optional[List[Path]] = None,
        learning_rate: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.998),
        eps: float = 1e-9,
        weight_decay: float = 1e-6,
        batch_size: int = 8,
        num_workers: int = 1,
        validation_split: float = 0.1,
        num_test_examples: int = 5,
        max_phoneme_ids: Optional[int] = None,
        binary_loss_start_epoch: int = 0,
        binary_loss_warmup_epochs: int = 10,
        **kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters()

        variant_overrides: Dict[str, Any] = {}
        if variant == "speedyspeech":
            variant_overrides.update(_SPEEDYSPEECH_OVERRIDES)
        if use_pitch is not None:
            variant_overrides["use_pitch"] = use_pitch

        args = ForwardTTSArgs(
            num_chars=num_symbols,
            out_channels=mel_channels,
            hidden_channels=hidden_channels,
            hidden_channels_ffn=hidden_channels_ffn,
            encoder_num_layers=encoder_num_layers,
            decoder_num_layers=decoder_num_layers,
            num_heads=num_heads,
            use_energy=use_energy,
            num_speakers=num_speakers,
            **variant_overrides,
        )
        self.model_args = args
        self.model = ForwardTTS(args)
        self.criterion = ForwardTTSLoss()

        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        self._load_datasets(validation_split, num_test_examples, max_phoneme_ids)

    # ------------------------------------------------------------------
    def _load_datasets(self, validation_split: float, num_test_examples: int,
                       max_phoneme_ids: Optional[int]) -> None:
        if not self.hparams.dataset:
            _LOG.debug("No dataset to load")
            return
        full_dataset = ForwardTTSDataset(
            self.hparams.dataset,
            mel_channels=self.hparams.mel_channels,
            filter_length=self.hparams.filter_length,
            sample_rate=self.hparams.sample_rate,
            mel_fmin=self.hparams.mel_fmin,
            mel_fmax=self.hparams.mel_fmax,
            max_phoneme_ids=max_phoneme_ids,
        )
        valid_size = max(0, int(len(full_dataset) * validation_split))
        test_size = min(num_test_examples, max(0, len(full_dataset) - valid_size))
        train_size = len(full_dataset) - valid_size - test_size
        self._train_dataset, self._test_dataset, self._val_dataset = random_split(
            full_dataset, [train_size, test_size, valid_size]
        )

    def _collate(self) -> ForwardTTSCollate:
        return ForwardTTSCollate(
            is_multispeaker=self.hparams.num_speakers > 1,
            use_pitch=bool(self.model_args.use_pitch),
        )

    def train_dataloader(self):
        return DataLoader(self._train_dataset, collate_fn=self._collate(),
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self._val_dataset, collate_fn=self._collate(),
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self._test_dataset, collate_fn=self._collate(),
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.batch_size)

    # ------------------------------------------------------------------
    def forward(self, x, pace: float = 1.0, pitch_mul: float = 1.0,
               pitch_add: float = 0.0, speaker=None):
        return self.model.inference(x, speaker=speaker, pace=pace,
                                    pitch_mul=pitch_mul, pitch_add=pitch_add)

    def _step(self, batch: Batch, binary_loss_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        outputs = self.model(
            x=batch.phoneme_ids,
            x_lengths=batch.phoneme_lengths,
            y=batch.mels,
            y_lengths=batch.mel_lengths,
            pitch=batch.pitch,
            speaker=batch.speaker_ids,
        )
        losses = self.criterion(
            decoder_output=outputs["model_outputs"],
            decoder_target=batch.mels.transpose(1, 2),
            decoder_output_lens=batch.mel_lengths,
            dur_output=outputs["durations_log"],
            dur_target=outputs["durations"],
            input_lens=batch.phoneme_lengths,
            pitch_output=outputs["pitch_avg_pred"],
            pitch_target=outputs["pitch_avg"],
            aligner_logprob=outputs["alignment_logprob"],
            alignment_hard=outputs["alignment_hard"],
            alignment_soft=outputs["alignment_soft"],
            binary_loss_weight=binary_loss_weight,
        )
        return losses

    def training_step(self, batch: Batch, batch_idx: int):
        # ramp the binary alignment loss in over binary_loss_warmup_epochs
        # starting at binary_loss_start_epoch — full-strength binarization
        # against a still-random soft alignment destabilizes the aligner
        warmup = max(1, self.hparams.binary_loss_warmup_epochs)
        binary_loss_weight = min(1.0, max(
            0.0, (self.current_epoch - self.hparams.binary_loss_start_epoch + 1) / warmup))
        losses = self._step(batch, binary_loss_weight=binary_loss_weight)
        self.log_dict({f"train_{k}": v for k, v in losses.items()}, prog_bar=False)
        return losses["loss"]

    def validation_step(self, batch: Batch, batch_idx: int):
        losses = self._step(batch)
        self.log_dict({f"val_{k}": v for k, v in losses.items()})
        return losses["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            eps=self.hparams.eps,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999875)
        return [optimizer], [scheduler]


class ForwardTTSTrainingEngine(BaseTrainingEngine):
    """Training engine adapter for FastPitch / SpeedySpeech (ForwardTTS).

    Registered twice — as ``"fastpitch"`` and (via
    :class:`SpeedySpeechTrainingEngine`) as ``"speedyspeech"`` — the only
    difference being the default ``variant`` used when the caller doesn't
    set ``config.extra["variant"]`` explicitly.
    """

    default_variant = "fastpitch"

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    def create_model(
        self,
        config: TrainingEngineConfig,
        dataset_paths: List[Path],
        **kwargs: Any,
    ) -> pl.LightningModule:
        extra = dict(config.extra)
        extra.setdefault("variant", self.default_variant)
        return ForwardTTSModule(
            num_symbols=config.num_symbols,
            num_speakers=config.num_speakers,
            sample_rate=config.sample_rate,
            dataset=[str(p) for p in dataset_paths],
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
        """Export a ForwardTTS checkpoint to ONNX.

        Contract matches ``coqui_fastpitch_export/export_fp.py`` /
        ``phoonnx.engines.fastpitch.FastPitchAdapter``: ``token_ids`` [B,T]
        (+ optional ``speaker`` [B]) -> ``mel_spec`` [B, 80, T_mel].
        """
        import json

        with open(config_path, "r", encoding="utf-8") as f:
            model_config: Dict[str, Any] = json.load(f)

        module: ForwardTTSModule = ForwardTTSModule.load_from_checkpoint(
            checkpoint_path, dataset=None, map_location="cpu",
        )
        model = module.model
        model.eval()

        num_symbols = model.args.num_chars
        num_speakers = model.args.num_speakers
        multispeaker = num_speakers > 1

        class _InferWrapper(torch.nn.Module):
            """pace / pitch_mul / pitch_add are graph inputs so the
            FastPitchAdapter's speed and pitch controls actually reach the
            model (traced as constants they would be silent no-ops)."""

            def __init__(self, m: ForwardTTS, multispeaker: bool):
                super().__init__()
                self.m = m
                self.multispeaker = multispeaker

            def forward(self, token_ids: torch.Tensor,
                       pace: torch.Tensor,
                       pitch_mul: torch.Tensor,
                       pitch_add: torch.Tensor,
                       speaker: Optional[torch.Tensor] = None) -> torch.Tensor:
                sid = speaker if self.multispeaker else None
                return self.m.inference(token_ids, speaker=sid, pace=pace,
                                        pitch_mul=pitch_mul, pitch_add=pitch_add)

        wrapper = _InferWrapper(model, multispeaker)
        wrapper.eval()

        dummy_ids = torch.randint(low=1, high=max(2, num_symbols - 1), size=(1, 30), dtype=torch.long)
        dummy_controls = (torch.ones(1), torch.ones(1), torch.zeros(1))
        control_names = ["pace", "pitch_mul", "pitch_add"]
        if multispeaker:
            dummy_speaker = torch.zeros(1, dtype=torch.long)
            dummy_args: Tuple[Any, ...] = (dummy_ids, *dummy_controls, dummy_speaker)
            input_names = ["token_ids", *control_names, "speaker"]
            dynamic_axes = {
                "token_ids": {0: "batch_size", 1: "phonemes"},
                "speaker": {0: "batch_size"},
                "mel_spec": {0: "batch_size", 2: "frame"},
            }
        else:
            dummy_args = (dummy_ids, *dummy_controls)
            input_names = ["token_ids", *control_names]
            dynamic_axes = {
                "token_ids": {0: "batch_size", 1: "phonemes"},
                "mel_spec": {0: "batch_size", 2: "frame"},
            }

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{checkpoint_path.stem}.onnx"

        export_kwargs: Dict[str, Any] = dict(
            input_names=input_names,
            output_names=["mel_spec"],
            dynamic_axes=dynamic_axes,
            opset_version=OPSET_VERSION,
        )
        # torch>=2.5 defaults torch.onnx.export to the dynamo-based exporter,
        # which additionally requires the optional 'onnxscript' package.
        # Force the legacy TorchScript-tracing exporter (matches
        # coqui_fastpitch_export/export_fp.py) when the kwarg is available.
        if "dynamo" in torch.onnx.export.__code__.co_varnames:
            export_kwargs["dynamo"] = False

        with torch.no_grad():
            torch.onnx.export(wrapper, dummy_args, str(output_path), **export_kwargs)

        try:
            import onnx as _onnx

            onnx_model = _onnx.load(str(output_path))
            del onnx_model.metadata_props[:]
            for key, value in {
                "model_type": "fastpitch" if module.hparams.variant == "fastpitch" else "speedyspeech",
                "engine": module.hparams.variant,
                "n_speakers": num_speakers,
                "n_vocab": num_symbols,
                "sample_rate": model_config.get("audio", {}).get("sample_rate", module.hparams.sample_rate),
                "alphabet": model_config.get("alphabet", ""),
                "phoneme_type": model_config.get("phoneme_type", ""),
                "phonemizer_model": model_config.get("phonemizer_model", ""),
                "phoneme_id_map": json.dumps(model_config.get("phoneme_id_map", {})),
                "has_espeak": model_config.get("phoneme_type", "") == "espeak",
            }.items():
                meta = onnx_model.metadata_props.add()
                meta.key = key
                meta.value = str(value)
            _onnx.save(onnx_model, str(output_path))
        except ImportError:
            _LOG.warning("onnx package not installed — skipping metadata")

        _LOG.info("Exported ForwardTTS (%s) ONNX model to %s", module.hparams.variant, output_path)
        return output_path

    def quality_presets(self) -> Dict[str, Dict[str, Any]]:
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
        """Extract F0 (pitch) via pyworld; cached alongside the mel cache.

        Same contract as the OptiSpeech training engine's
        ``extra_preprocess`` (pyworld DIO + StoneMask). SpeedySpeech does
        not use pitch, but the field is harmless to compute/cache and lets
        the same preprocessed dataset be reused for either variant.
        """
        import numpy as np

        try:
            import librosa
            import pyworld as pw
        except ImportError:
            _LOG.warning(
                "pyworld/librosa not installed — skipping F0 extraction "
                "(FastPitch pitch predictor will train without a target; "
                "install phoonnx[train,train-fastpitch] for real pitch)."
            )
            return {}

        wav, sr = librosa.load(str(utterance_audio_path), sr=sample_rate, mono=True)
        wav_double = wav.astype(np.float64)
        f0, timeaxis = pw.dio(wav_double, sr)
        f0 = pw.stonemask(wav_double, f0, timeaxis, sr).astype(np.float32)

        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = utterance_audio_path.stem
        f0_path = cache_dir / f"{stem}.f0.npy"
        np.save(f0_path, f0)

        return {"f0_path": str(f0_path)}

    def load_checkpoint(
        self,
        model: pl.LightningModule,
        checkpoint_path: Path,
        **kwargs: Any,
    ) -> pl.LightningModule:
        """Tolerant checkpoint load (skips shape-mismatched tensors, e.g.
        when fine-tuning onto a different phoneme inventory)."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("state_dict", ckpt)
        model_state = model.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered[k] = v
            elif k in model_state:
                _LOG.warning("Shape mismatch for %s: ckpt=%s model=%s — skipping",
                            k, v.shape, model_state[k].shape)
        model.load_state_dict(filtered, strict=False)
        _LOG.info("Loaded %d/%d parameters from checkpoint", len(filtered), len(model_state))
        return model


class SpeedySpeechTrainingEngine(ForwardTTSTrainingEngine):
    """Same ``ForwardTTS`` engine, defaulting to the SpeedySpeech variant
    (residual conv-BN encoder/decoder, no pitch predictor) when
    ``config.extra["variant"]`` isn't set explicitly."""

    default_variant = "speedyspeech"
