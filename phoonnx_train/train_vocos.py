"""Train a Vocos mel→waveform vocoder on raw audio.

Audio-only training: point it at one or more directories of audio files
(or a phoonnx preprocess output directory — its normalized wav cache is
picked up automatically) and it learns to invert the exact mel
spectrograms phoonnx acoustic models are trained on.

Example::

    python -m phoonnx_train.train_vocos \\
        --audio-dir /data/my_voice/wavs \\
        --audio-dir /data/other_voice/wavs \\
        --sample-rate 22050 \\
        --warm-start charactr/vocos-mel-24khz \\
        --max-epochs 100
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

import click
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from phoonnx_train.vocos.data import MelConfig, crop_samples, find_audio_files
from phoonnx_train.vocos.lightning import VocosTrainingModule

_LOGGER = logging.getLogger(__package__)


@click.command()
@click.option("--audio-dir", "audio_dirs", multiple=True, required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Directory of audio files (repeatable). A phoonnx "
                   "preprocess output dir works too — its cached wavs are used.")
@click.option("--sample-rate", default=22050, type=int,
              help="Target sample rate; audio is resampled (default: 22050)")
@click.option("--batch-size", default=16, type=int)
@click.option("--crop-seconds", default=1.0, type=float,
              help="Length of random training crops in seconds (default: 1.0)")
@click.option("--warm-start", default=None, type=str,
              help="Pretrained Vocos weights: local .ckpt/.bin or HuggingFace repo id")
@click.option("--resume-from-checkpoint", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="Resume a previous training run (restores optimizers/epoch)")
@click.option("--checkpoint-epochs", default=1, type=int,
              help="Save a checkpoint every N epochs (default: 1)")
@click.option("--max-epochs", type=int, default=1000)
@click.option("--devices", default="1", type=str)
@click.option("--accelerator", default="auto")
@click.option("--default-root-dir", type=click.Path(file_okay=False), default=None)
@click.option("--precision", default="32", type=str)
@click.option("--num-workers", type=int, default=4)
@click.option("--seed", type=int, default=1234)
def main(
    audio_dirs: Tuple[str, ...],
    sample_rate: int,
    batch_size: int,
    crop_seconds: float,
    warm_start: Optional[str],
    resume_from_checkpoint: Optional[str],
    checkpoint_epochs: int,
    max_epochs: int,
    devices,
    accelerator: str,
    default_root_dir: Optional[str],
    precision,
    num_workers: int,
    seed: int,
):
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(seed)

    mel = MelConfig(sample_rate=sample_rate)
    files = find_audio_files(audio_dirs)
    _LOGGER.info("Found %d audio files in %s", len(files), list(audio_dirs))

    model = VocosTrainingModule(
        mel=mel.as_dict(),
        batch_size=batch_size,
        num_workers=num_workers,
        crop_samples=crop_samples(crop_seconds, mel),
        audio_files=files,
    )

    if warm_start:
        model.warm_start(warm_start)

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
    trainer.fit(model, ckpt_path=resume_from_checkpoint)
    _LOGGER.info("Training finished. Checkpoints under: %s",
                 Path(trainer.default_root_dir))


if __name__ == "__main__":
    main()
