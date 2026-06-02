import json
import logging
import os
from pathlib import Path
from typing import Optional

import click
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from phoonnx_train.engines import get_engine, list_engines
from phoonnx_train.engines.base import TrainingEngineConfig
from phoonnx_train.vits.lightning import VitsModel

_LOGGER = logging.getLogger(__package__)


def _validate_engine(ctx, param, value):
    available = list_engines()
    if value.lower() not in [e.lower() for e in available]:
        raise click.BadParameter(
            f"Unknown engine {value!r}. Choose from: {', '.join(available)}"
        )
    return value


def load_state_dict(model, saved_state_dict):
    state_dict = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in saved_state_dict:
            new_state_dict[k] = saved_state_dict[k]
        else:
            # Map old legacy keys to new disentangled keys when loading a
            # non-disentangled checkpoint into a disentangled model.
            mapped = False
            if 'emb_g' in k:
                # Legacy emb_g.weight -> timbre_enc.speaker_emb.weight
                new_k = k.replace('model_g.emb_g', 'model_g.timbre_enc.speaker_emb')
                if new_k in saved_state_dict:
                    new_state_dict[k] = saved_state_dict[new_k]
                    mapped = True
            if not mapped:
                _LOGGER.debug("%s is not in the checkpoint", k)
                new_state_dict[k] = v

    model.load_state_dict(new_state_dict)


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.option('--dataset-dir', required=True,
              type=click.Path(exists=True, file_okay=False),
              help='Path to pre-processed dataset directory')
@click.option('--engine', default='vits',
              type=str,
              callback=_validate_engine,
              help='TTS architecture to train (default: vits)')
@click.option('--checkpoint-epochs', default=1, type=int,
              help='Save checkpoint every N epochs (default: 1)')
@click.option('--quality', default='medium',
              type=str,
              help='Quality/size of model (default: medium)')
@click.option('--resume-from-checkpoint', default=None,
              help='Load an existing checkpoint and resume training')
@click.option('--resume-from-single-speaker-checkpoint', default=None,
              help='Convert a single-speaker checkpoint to multi-speaker')
@click.option('--seed', type=int, default=1234, help='Random seed')
# Common Trainer options
@click.option('--max-epochs', type=int, default=1000, help='Stop training once this number of epochs is reached (default: 1000)')
@click.option('--devices', default=1, help='Number of devices or list of device IDs to train on (default: 1)')
@click.option('--accelerator', default='auto', help='Hardware accelerator to use (cpu, gpu, tpu, mps, etc.)  (default: "auto")')
@click.option('--default-root-dir', type=click.Path(file_okay=False), default=None, help='Default root directory for logs and checkpoints (default: None)')
@click.option('--precision', default=32, help='Precision used in training (e.g. 16, 32, bf16) (default: 32)')
# Model-specific arguments
@click.option('--learning-rate', type=float, default=2e-4, help='Learning rate for optimizer (default: 2e-4)')
@click.option('--batch-size', type=int, default=16, help='Training batch size (default: 16)')
@click.option('--num-workers', type=click.IntRange(min=1), default=os.cpu_count() or 1, help='Number of data loader workers (default: CPU count)')
@click.option('--validation-split', type=float, default=0.05, help='Proportion of data used for validation (default: 0.05)')
@click.option('--discard-encoder', is_flag=True, default=False, help='Discard the encoder weights from base checkpoint (default: False)')
# Disentangled encoder options
@click.option('--disentangled', is_flag=True, help='Enable disentangled timbre/articulation/prosody encoders')
@click.option('--ref-enc-hidden-channels', type=int, default=256, help='Reference encoder hidden channels (default: 256)')
@click.option('--ref-enc-n-layers', type=int, default=3, help='Reference encoder conv layers (default: 3)')
@click.option('--ref-enc-stride', type=int, default=2, help='Reference encoder conv stride (default: 2)')
@click.option('--timbre-dim', type=int, default=0, help='Timbre embedding dimension (0 = gin_channels)')
@click.option('--artic-dim', type=int, default=0, help='Articulation embedding dimension (0 = gin_channels)')
@click.option('--prosody-dim', type=int, default=0, help='Prosody embedding dimension (0 = gin_channels)')
@click.option('--n-emotion-labels', type=int, default=0, help='Number of emotion labels for categorical prosody control (0 = disabled)')
@click.option('--lambda-mi', type=float, default=0.1, help='Weight for mutual information disentanglement loss (default: 0.1)')
@click.option('--lambda-cycle', type=float, default=1.0, help='Weight for cycle consistency loss (default: 1.0)')
@click.option('--lambda-kl-dis', type=float, default=0.01, help='Weight for KL regularization on disentangled latents (default: 0.01)')
def main(
    dataset_dir: str,
    engine: str,
    checkpoint_epochs: int,
    quality: str,
    resume_from_checkpoint: Optional[str],
    resume_from_single_speaker_checkpoint: Optional[str],
    seed: int,
    max_epochs: int,
    devices,
    accelerator: str,
    default_root_dir: Optional[str],
    precision,
    learning_rate,
    batch_size: int,
    num_workers,
    validation_split: float,
    discard_encoder: bool,
    disentangled,
    ref_enc_hidden_channels,
    ref_enc_n_layers,
    ref_enc_stride,
    timbre_dim,
    artic_dim,
    prosody_dim,
    n_emotion_labels,
    lambda_mi,
    lambda_cycle,
    lambda_kl_dis,
):
    logging.basicConfig(level=logging.DEBUG)
    torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # Load dataset config
    # ------------------------------------------------------------------
    dataset_path = Path(dataset_dir)
    config_path = dataset_path / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        dataset_config = json.load(f)

    num_symbols = dataset_config.get("num_symbols", 256)
    num_speakers = dataset_config.get("num_speakers", 1)
    sample_rate = dataset_config.get("audio", {}).get("sample_rate", 22050)

    # ------------------------------------------------------------------
    # Resolve engine + quality preset
    # ------------------------------------------------------------------
    training_engine = get_engine(engine)
    presets = training_engine.quality_presets()
    if quality not in presets:
        _LOGGER.warning(
            "Quality %r not found in engine presets %s — falling back to 'medium'",
            quality, list(presets),
        )
        quality = "medium"
    quality_kwargs = presets.get(quality, {})

    _LOGGER.info(
        "Training engine=%s  quality=%s  symbols=%d  speakers=%d  sr=%d",
        engine, quality, num_symbols, num_speakers, sample_rate,
    )

    # ------------------------------------------------------------------
    # Build model via the engine
    # ------------------------------------------------------------------
    engine_config = TrainingEngineConfig(
        num_symbols=num_symbols,
        num_speakers=num_speakers,
        sample_rate=sample_rate,
        extra={
            **quality_kwargs,
            "batch_size": batch_size,
            "validation_split": validation_split,
            "num_workers": num_workers,
            "learning_rate": learning_rate,
            "disentangled": disentangled,
            "ref_enc_hidden_channels": ref_enc_hidden_channels,
            "ref_enc_n_layers": ref_enc_n_layers,
            "ref_enc_stride": ref_enc_stride,
            "timbre_dim": timbre_dim,
            "artic_dim": artic_dim,
            "prosody_dim": prosody_dim,
            "n_emotion_labels": n_emotion_labels,
            "lambda_mi": lambda_mi,
            "lambda_cycle": lambda_cycle,
            "lambda_kl_dis": lambda_kl_dis,
        },
    )
    model = training_engine.create_model(
        config=engine_config,
        dataset_paths=[dataset_path],
    )
    _LOGGER.info("Model created: %s", type(model).__name__)

    # ------------------------------------------------------------------
    # Disentangled warnings
    # ------------------------------------------------------------------
    if disentangled and num_speakers < 2:
        _LOGGER.warning(
            "Disentangled mode is enabled but the dataset only has 1 speaker. "
            "Timbre/articulation/prosody disentanglement will be weak. "
            "Consider using a multi-speaker dataset for best results."
        )

    # ------------------------------------------------------------------
    # Resume from checkpoint (engine-specific logic)
    # ------------------------------------------------------------------
    if resume_from_checkpoint:
        if disentangled:
            ckpt = VitsModel.load_from_checkpoint(resume_from_checkpoint, dataset=None)
            _LOGGER.debug(
                "Checkpoint params: num_symbols=%d num_speakers=%d sample_rate=%d",
                ckpt.model_g.n_vocab, ckpt.model_g.n_speakers, ckpt.hparams.sample_rate,
            )
            if ckpt.model_g.n_vocab != num_symbols:
                _LOGGER.warning(
                    "Checkpoint num_symbols=%d does not match config num_symbols=%d",
                    ckpt.model_g.n_vocab, num_symbols,
                )
            if ckpt.model_g.n_speakers != num_speakers:
                _LOGGER.warning(
                    "Checkpoint num_speakers=%d does not match config num_speakers=%d",
                    ckpt.model_g.n_speakers, num_speakers,
                )
            if ckpt.hparams.sample_rate != sample_rate:
                _LOGGER.warning(
                    "Checkpoint sample_rate=%d does not match config sample_rate=%d",
                    ckpt.hparams.sample_rate, sample_rate,
                )

            saved_state_dict = ckpt.state_dict()
            # Filter the state dictionary by removing the encoder weights
            enc_key = 'model_g.enc_p.emb.weight'
            if enc_key in saved_state_dict:
                saved_shape = saved_state_dict[enc_key].shape
                current_shape = model.state_dict()[enc_key].shape
                if saved_shape[0] != current_shape[0]:
                    _LOGGER.warning(
                        "Size mismatch detected for '%s': saved shape %s vs current shape %s.",
                        enc_key, saved_shape, current_shape,
                    )
                    discard_encoder = True

                if discard_encoder:
                    _LOGGER.warning(
                        "Skipping encoder weights from the checkpoint. (will be randomly initialized)"
                    )
                    saved_state_dict.pop(enc_key)

            load_state_dict(model, saved_state_dict)
            _LOGGER.info("Successfully loaded model weights from %s", resume_from_checkpoint)
        else:
            training_engine.load_checkpoint(
                model,
                Path(resume_from_checkpoint),
                discard_encoder=discard_encoder,
            )
            _LOGGER.info("Loaded checkpoint: %s", resume_from_checkpoint)

    if resume_from_single_speaker_checkpoint:
        if disentangled:
            assert num_speakers > 1, "--resume-from-single-speaker-checkpoint is only for multi-speaker models."
            _LOGGER.info('Resuming from single-speaker checkpoint: %s', resume_from_single_speaker_checkpoint)

            model_single = VitsModel.load_from_checkpoint(resume_from_single_speaker_checkpoint, dataset=None)
            g_dict = model_single.model_g.state_dict()

            for key in list(g_dict.keys()):
                if key.startswith('dec.cond') or key.startswith('dp.cond') or ('enc.cond_layer' in key):
                    g_dict.pop(key, None)

            load_state_dict(model.model_g, g_dict)
            load_state_dict(model.model_d, model_single.model_d.state_dict())
            _LOGGER.info('Successfully converted single-speaker checkpoint to multi-speaker')
        else:
            training_engine.load_checkpoint(
                model,
                Path(resume_from_single_speaker_checkpoint),
                resume_from_single_speaker_checkpoint=True,
            )
            _LOGGER.info(
                "Loaded single-speaker checkpoint: %s",
                resume_from_single_speaker_checkpoint,
            )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=checkpoint_epochs,
        save_top_k=-1,
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        devices=devices,
        accelerator=accelerator,
        default_root_dir=default_root_dir,
        precision=precision,
        callbacks=[checkpoint_callback],
    )
    _LOGGER.info("Training started!")
    trainer.fit(model)


if __name__ == '__main__':
    main()
