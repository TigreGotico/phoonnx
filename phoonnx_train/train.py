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
from phoonnx_train.vits.input_features import INPUT_LAYERS, load_input_features

# Matrix multiplications benefit from TF32/reduced precision on Ampere+ GPUs;
# "high" trades a small amount of accuracy for a meaningful speedup and is
# safe for training (not inference-critical precision).
torch.set_float32_matmul_precision('high')


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


def _resumes_training_state(path: str) -> bool:
    """Whether Lightning can resume a training state from ``path``.

    Only a local file is read: it resumes when Lightning wrote it. A URL or a
    remote path goes to Lightning as given, which reads it.
    """
    from phoonnx_train.torch_compat import trusting_torch_load

    if not os.path.isfile(path):
        return True
    with trusting_torch_load():
        try:
            checkpoint = torch.load(path, map_location="cpu", mmap=True)
        except RuntimeError:  # mmap needs the zip format, the torch.save default since 1.6
            checkpoint = torch.load(path, map_location="cpu")
    return "pytorch-lightning_version" in checkpoint


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
@click.option('--log-audio-samples/--no-log-audio-samples', default=False,
              help='Log synthesized validation audio samples to the logger '
                   'each epoch (requires a TensorBoard logger)')
@click.option('--discard-encoder', is_flag=True, default=False)
@click.option('--input-layer', type=click.Choice(INPUT_LAYERS), default='ids',
              help='VITS text-encoder input: an embedding per id (default), or a projection '
                   'of --input-features plus a learned vector per id (residual) or per base '
                   'symbol and mark (composition)')
@click.option('--input-features', type=click.Path(exists=True, dir_okay=False), default=None,
              help='features.json table for --input-layer residual or composition; it must be '
                   'made from the dataset config\'s phoneme_id_map')
# In-training evaluation / early stopping (all opt-in; default = off)
@click.option('--eval-sentences', type=click.Path(exists=True, dir_okay=False), default=None,
              help='Held-out sentences file (one per line); enables in-training scoring')
@click.option('--eval-every', type=int, default=0,
              help='Score the newest checkpoint every N epochs (0 = off, default)')
@click.option('--early-stop-patience', type=int, default=0,
              help='Stop after this many scored epochs without improvement (0 = off)')
@click.option('--eval-speaker-ref-dir', type=click.Path(exists=True, file_okay=False), default=None,
              help='Reference-speaker wav dir enabling the speaker-similarity gate')
@click.option('--min-spk-sim', type=float, default=None,
              help='Similarity gate floor for best-checkpoint selection')
@click.option('--eval-speaker-id', type=int, default=None,
              help='Speaker id used for in-training evaluation synthesis on '
                   'multi-speaker models (default: None = engine default voice)')
@click.option('--eval-seed', type=int, default=1234,
              help='Base seed for per-utterance evaluation synthesis (default: 1234)')
@click.option('--compile', 'use_compile', is_flag=True, default=False,
              help='Compile the model with torch.compile for faster training (default: False)')
@click.option('--compile-mode', default='default',
              type=click.Choice(['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs']),
              help='torch.compile mode (default: default)')
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
    log_audio_samples: bool,
    discard_encoder: bool,
    input_layer: str,
    input_features: Optional[str],
    eval_sentences: Optional[str],
    eval_every: int,
    early_stop_patience: int,
    eval_speaker_ref_dir: Optional[str],
    min_spk_sim: Optional[float],
    eval_speaker_id: Optional[int],
    eval_seed: int,
    use_compile: bool,
    compile_mode: str,
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
    extra = _build_extra(
        quality_kwargs,
        # engine-specific knobs (asr_path, plbert_dir, stage, backbone,
        # download_aux, ...) ride in an "engine_params" dict in config.json
        dataset_config.get("engine_params", {}),
        batch_size=batch_size,
        validation_split=validation_split,
        num_workers=num_workers,
        log_audio_samples=log_audio_samples,
    )
    if input_layer != "ids" or input_features:
        if engine != "vits":
            raise click.UsageError("--input-layer and --input-features apply to the vits engine only")
        if input_layer == "ids" or not input_features:
            raise click.UsageError("--input-features needs --input-layer residual or composition, "
                                   "and those layers need --input-features")
        if "phoneme_id_map" not in dataset_config:
            raise click.UsageError(f"--input-features needs a phoneme_id_map in {config_path}")
        extra["input_layer"] = input_layer
        extra["input_features"] = load_input_features(
            Path(input_features), dataset_config["phoneme_id_map"], num_symbols)
        _LOGGER.info("Text-encoder input: %s over %s (%d features)", input_layer, input_features,
                     len(extra["input_features"]["feature_names"]))
    engine_config = TrainingEngineConfig(
        num_symbols=num_symbols,
        num_speakers=num_speakers,
        sample_rate=sample_rate,
        extra=extra,
    )
    model = training_engine.create_model(
        config=engine_config,
        dataset_paths=[dataset_path],
    )
    _LOGGER.info("Model created: %s", type(model).__name__)

    # ------------------------------------------------------------------
    # Resume from checkpoint (engine-specific logic)
    # ------------------------------------------------------------------
    # A plain --resume-from-checkpoint (same architecture, not discarding the
    # encoder) is a *true resume*: optimizer state, LR scheduler, epoch and
    # global_step must all continue from the checkpoint, otherwise Adam's
    # moment estimates are lost, the epoch counter restarts and max_epochs is
    # miscounted. Lightning only restores those when the checkpoint is handed to
    # Trainer.fit(ckpt_path=...); a manual state_dict load copies weights only.
    # The --discard-encoder and single-speaker paths intentionally change the
    # architecture, so they stay weight-only warm starts (fit_ckpt_path=None).
    # So does a local file that Lightning did not write (e.g. an upstream
    # release): it holds weights and no training state to resume.
    fit_ckpt_path: Optional[str] = None
    if resume_from_checkpoint and not discard_encoder \
            and _resumes_training_state(resume_from_checkpoint):
        fit_ckpt_path = resume_from_checkpoint
        _LOGGER.info("Resuming full training state from checkpoint: %s",
                     resume_from_checkpoint)
    elif resume_from_checkpoint:
        if not discard_encoder:
            _LOGGER.warning("%s is not a Lightning checkpoint: loading its weights only, so "
                            "the optimizer state and the step count start again",
                            resume_from_checkpoint)
        training_engine.load_checkpoint(
            model,
            Path(resume_from_checkpoint),
            discard_encoder=discard_encoder,
        )
        _LOGGER.info("Warm-started weights from checkpoint: %s", resume_from_checkpoint)

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
    callbacks = [checkpoint_callback]

    # ------------------------------------------------------------------
    # Evaluation / early stopping
    # ------------------------------------------------------------------
    # StopFileCallback is ALWAYS added: it watches the run directory for a
    # stop.flag written by an external sidecar scoreboard (the sidecar->trainer
    # bridge). With no flag present it is a no-op, so behaviour is byte-identical
    # to before unless a flag actually appears.
    from phoonnx_train.evaluation.callbacks import (EvalScoreboardCallback,
                                                    StopFileCallback)

    run_dir = Path(default_root_dir) if default_root_dir else Path.cwd()
    run_dir.mkdir(parents=True, exist_ok=True)
    callbacks.append(StopFileCallback(run_dir / "stop.flag"))

    # In-training scoreboard is opt-in: only wired up when both an eval
    # sentences file and a positive --eval-every are given.
    if eval_sentences and eval_every > 0:
        from phoonnx_train.engines.base import BaseTrainingEngine
        from phoonnx_train.evaluation import (CheckpointScorer, MetricsTracker,
                                              SelectionPolicy)
        from phoonnx_train.eval_utils import read_sentences

        # In-training scoring loads each checkpoint through
        # engine.eval_synthesize; engines that never implemented it would fail
        # silently at the first scoring firing (deep inside a swallowed
        # callback exception) after wasting a full training run. Fail loudly at
        # startup instead, naming the engines that DO support it.
        if type(training_engine).eval_synthesize is BaseTrainingEngine.eval_synthesize:
            supported = sorted(
                name for name in list_engines()
                if type(get_engine(name)).eval_synthesize
                is not BaseTrainingEngine.eval_synthesize
            )
            raise click.UsageError(
                f"engine {engine!r} does not support in-training evaluation "
                f"(--eval-sentences / --eval-every); it has no eval_synthesize "
                f"implementation. Engines that do: {', '.join(supported) or 'none'}."
            )

        sentences = read_sentences(eval_sentences)
        eval_dir = run_dir / "eval"
        scorer = CheckpointScorer(
            training_engine, dataset_config, sentences,
            speaker_ref_dir=Path(eval_speaker_ref_dir) if eval_speaker_ref_dir else None,
            speaker_id=eval_speaker_id,
            seed=eval_seed,
        )
        selection = SelectionPolicy(metric="utmos_mean", min_spk_sim=min_spk_sim)
        tracker = MetricsTracker(eval_dir)
        callbacks.append(EvalScoreboardCallback(
            scorer, tracker, selection, eval_dir,
            every_n_epochs=eval_every,
            patience=early_stop_patience or None,
        ))
        _LOGGER.info("In-training evaluation enabled: every %d epoch(s), "
                     "patience=%s, scoreboard=%s", eval_every,
                     early_stop_patience or None, eval_dir)

    trainer = Trainer(
        max_epochs=max_epochs,
        devices=devices,
        accelerator=accelerator,
        default_root_dir=default_root_dir,
        precision=precision,
        callbacks=callbacks,
        **training_engine.trainer_kwargs(),
    )
    if use_compile:
        # torch.compile can raise at call time on unsupported python/torch
        # combinations (e.g. dynamo has no CPython 3.12 backend on torch<2.3).
        # A compile failure must never abort a training run that would have
        # been perfectly fine uncompiled: warn and continue.
        try:
            from phoonnx_train.torch_compat import disable_compile_on_transforms
            if hasattr(model, "model_g") and hasattr(model, "model_d"):
                _LOGGER.info("Compiling model_g/model_d with torch.compile (mode=%s)", compile_mode)
                model.model_g = torch.compile(model.model_g, mode=compile_mode)
                model.model_d = torch.compile(model.model_d, mode=compile_mode)
                # Exclude the VITS spline transforms from graph capture — their
                # data-dependent control flow cannot be traced by dynamo. Only
                # once compile succeeded; export paths never see the wrapper.
                disable_compile_on_transforms()
            elif engine == "rapido":
                # the Rápido loop calls the text, condition and velocity stages and
                # never model.forward, so a compiled wrapper would only rename every
                # checkpoint key to model._orig_mod.* and break the export
                _LOGGER.warning("compile not supported for engine %r — running uncompiled", engine)
            elif hasattr(model, "model") and isinstance(getattr(model, "model"), torch.nn.Module):
                _LOGGER.info("Compiling model with torch.compile (mode=%s)", compile_mode)
                model.model = torch.compile(model.model, mode=compile_mode)
            else:
                _LOGGER.warning("compile not supported for engine %r yet — running uncompiled", engine)
        except Exception as err:
            _LOGGER.warning(
                "torch.compile unavailable: %s — continuing uncompiled", err
            )

    _LOGGER.info("Training started!")
    # torch>=2.6 defaults torch.load(weights_only=True), which rejects our own
    # pickled Lightning checkpoints on ckpt_path resume.
    from phoonnx_train.torch_compat import trusting_torch_load
    with trusting_torch_load():
        trainer.fit(model, ckpt_path=fit_ckpt_path)


if __name__ == '__main__':
    main()
