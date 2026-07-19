import json
import logging
from pathlib import Path
from typing import Optional

import click
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from phoonnx_train.engines import get_engine, list_engines
from phoonnx_train.engines.base import TrainingEngineConfig


def _build_extra(quality_kwargs: dict, engine_params: dict, **cli_values) -> dict:
    """Merge the engine extra bag. Precedence: explicit CLI flag >
    config.json engine_params > CLI default > quality preset."""
    extra = {**quality_kwargs, **engine_params}
    ctx = click.get_current_context(silent=True)
    for name, value in cli_values.items():
        source = ctx.get_parameter_source(name) if ctx else None
        explicit = source is not None and source.name == "COMMANDLINE"
        if explicit or name not in extra:
            extra[name] = value
    return extra

_LOGGER = logging.getLogger(__package__)


def _validate_engine(ctx, param, value):
    available = list_engines()
    if value.lower() not in [e.lower() for e in available]:
        raise click.BadParameter(
            f"Unknown engine {value!r}. Choose from: {', '.join(available)}"
        )
    return value.lower()


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
@click.option('--max-epochs', type=int, default=1000)
@click.option('--devices', default='1', type=str)
@click.option('--accelerator', default='auto')
@click.option('--default-root-dir', type=click.Path(file_okay=False), default=None)
@click.option('--precision', default='32', type=str)
@click.option('--batch-size', type=int, default=16)
@click.option('--validation-split', type=float, default=0.1)
@click.option('--num-workers', type=int, default=4)
@click.option('--discard-encoder', is_flag=True, default=False)
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
    batch_size: int,
    validation_split: float,
    num_workers: int,
    discard_encoder: bool,
):
    logging.basicConfig(level=logging.DEBUG)
    torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # Load dataset config
    # ------------------------------------------------------------------
    dataset_path = Path(dataset_dir)
    config_path = dataset_path / "config.json"
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            dataset_config = json.load(f)
    else:
        # the styletts2* engines use the upstream list layout
        # (train_list.txt/val_list.txt + wavs/), which has no config.json
        dataset_config = {}
        _LOGGER.warning("No %s — using engine defaults", config_path)

    styletts2_engine = engine.startswith("styletts2")
    num_symbols = dataset_config.get("num_symbols",
                                     178 if styletts2_engine else 256)
    num_speakers = dataset_config.get("num_speakers", 1)
    sample_rate = dataset_config.get("audio", {}).get(
        "sample_rate", 24000 if styletts2_engine else 22050)

    # ------------------------------------------------------------------
    # Resolve engine + quality preset
    # ------------------------------------------------------------------
    training_engine = get_engine(engine)
    presets = training_engine.quality_presets()
    if quality not in presets:
        fallback = "medium" if "medium" in presets else next(iter(presets))
        _LOGGER.warning(
            "Quality %r not found in engine presets %s — falling back to %r",
            quality, list(presets), fallback,
        )
        quality = fallback
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
        extra=_build_extra(
            quality_kwargs,
            # engine-specific knobs (asr_path, plbert_dir, stage, backbone,
            # download_aux, ...) ride in an "engine_params" dict in config.json
            dataset_config.get("engine_params", {}),
            batch_size=batch_size,
            validation_split=validation_split,
            num_workers=num_workers,
        ),
    )
    model = training_engine.create_model(
        config=engine_config,
        dataset_paths=[dataset_path],
    )
    _LOGGER.info("Model created: %s", type(model).__name__)

    # ------------------------------------------------------------------
    # Resume from checkpoint (engine-specific logic)
    # ------------------------------------------------------------------
    if resume_from_checkpoint:
        training_engine.load_checkpoint(
            model,
            Path(resume_from_checkpoint),
            discard_encoder=discard_encoder,
        )
        _LOGGER.info("Loaded checkpoint: %s", resume_from_checkpoint)

    if resume_from_single_speaker_checkpoint:
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
        **training_engine.trainer_kwargs(),
    )
    _LOGGER.info("Training started!")
    trainer.fit(model)


if __name__ == '__main__':
    main()
