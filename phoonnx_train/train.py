import json
import logging
import pathlib
from pathlib import Path
import os
import torch
import click
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm.auto import tqdm

from phoonnx_train.vits.lightning import VitsModel


# Set the precision for matrix multiplications
torch.set_float32_matmul_precision('high')

# Allow unpickling those classes
torch.serialization.add_safe_globals([pathlib.PosixPath])
torch.serialization.safe_globals([pathlib.PosixPath])


_LOGGER = logging.getLogger(__package__)

_TORCH_COMPILE_LOGGERS = [
    "torch._dynamo",
    "torch._inductor",
    "torch._functorch",
    "torch._logging",
]


class _CompileStatusCallback(Callback):
    """Displays an elapsed-time indicator while the first training step compiles."""

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx == 0 and trainer.current_epoch == 0:
            self._bar = tqdm(desc="Compiling", bar_format="{desc}: {elapsed}", leave=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        bar = getattr(self, "_bar", None)
        if bar is not None:
            bar.set_description("Compiled")
            bar.close()
            self._bar = None


def load_state_dict(model, saved_state_dict):
    state_dict = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in saved_state_dict:
            new_state_dict[k] = saved_state_dict[k]
        else:
            _LOGGER.debug("%s is not in the checkpoint", k)
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.option('--dataset-dir', required=True, type=click.Path(exists=True, file_okay=False), help='Path to pre-processed dataset directory')
@click.option('--checkpoint-epochs', default=1, type=int, help='Save checkpoint every N epochs (default: 1)')
@click.option('--quality', default='medium', type=click.Choice(['x-low', 'medium', 'high']), help='Quality/size of model (default: medium)')
@click.option('--resume-from-checkpoint', default=None, help='Load an existing checkpoint and resume training')
@click.option('--resume-from-single-speaker-checkpoint', help='For multi-speaker models only. Converts a single-speaker checkpoint to multi-speaker and resumes training')
@click.option('--seed', type=int, default=1234, help='Random seed (default: 1234)')
# Common Trainer options
@click.option('--max-epochs', type=int, default=1000, help='Stop training once this number of epochs is reached (default: 1000)')
@click.option('--devices', default=1, help='Number of devices or list of device IDs to train on (default: 1)')
@click.option('--accelerator', default='auto', help='Hardware accelerator to use (cpu, gpu, tpu, mps, etc.)  (default: "auto")')
@click.option('--default-root-dir', type=click.Path(file_okay=False), default=None, help='Default root directory for logs and checkpoints (default: None)')
@click.option('--precision', type=str, default="32", help='Precision used in training (e.g. 16, 32, bf16) (default: "32")')
# Model-specific arguments
@click.option('--learning-rate', type=float, default=2e-4, help='Learning rate for optimizer (default: 2e-4)')
@click.option('--batch-size', type=int, default=16, help='Training batch size (default: 16)')
@click.option('--num-workers', type=click.IntRange(min=1), default=os.cpu_count() or 1, help='Number of data loader workers (default: CPU count)')
@click.option('--validation-split', type=float, default=0.05, help='Proportion of data used for validation (default: 0.05)')
@click.option('--discard-encoder', type=bool, default=False, help='Discard the encoder weights from base checkpoint (default: False)')
@click.option('--compile', 'use_compile', is_flag=True, default=False, help='Compile the model with torch.compile for faster training (default: False)')
@click.option('--compile-mode', default='default', type=click.Choice(['default', 'reduce-overhead', 'max-autotune', 'reduce-overhead-no-cudagraphs', 'max-autotune-no-cudagraphs']), help='torch.compile mode (default: default)')
@click.option('--notebook', is_flag=True, default=False, help='Notebook-friendly output: suppress all logs below ERROR, show compilation/training progress bars only (default: False)')
def main(
    dataset_dir,
    checkpoint_epochs,
    quality,
    resume_from_checkpoint,
    resume_from_single_speaker_checkpoint,
    seed,
    max_epochs,
    devices,
    accelerator,
    default_root_dir,
    precision,
    learning_rate,
    batch_size,
    num_workers,
    validation_split,
    discard_encoder,
    use_compile,
    compile_mode,
    notebook,
):
    log_level = logging.ERROR if notebook else logging.DEBUG
    logging.basicConfig(level=log_level, force=True)
    if notebook:
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    if use_compile:
        # In notebook mode, suppress to CRITICAL: autotuning trial failures
        # (E0324 ... Runtime error during autotuning: Ignoring this choice) are
        # expected fallbacks, not real errors, but are logged at ERROR level by
        # torch._logging's TorchLoggingHandler — setLevel() alone doesn't reach
        # that handler, so we also call torch._logging.set_logs() which is the
        # canonical PyTorch API for controlling it.
        _compile_log_level = logging.CRITICAL if notebook else logging.ERROR
        import torch._logging as _tl
        _set_logs = getattr(_tl, "set_logs", None)
        if _set_logs is not None:
            _set_logs(dynamo=_compile_log_level, aot=_compile_log_level, inductor=_compile_log_level)
        for name in _TORCH_COMPILE_LOGGERS:
            logging.getLogger(name).setLevel(_compile_log_level)

    dataset_dir = Path(dataset_dir)
    if default_root_dir is None:
        default_root_dir = dataset_dir

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)

    config_path = dataset_dir / 'config.json'
    dataset_path = dataset_dir / 'dataset.jsonl'

    _LOGGER.info(f"config_path: '{config_path}'")
    _LOGGER.info(f"dataset_path: '{dataset_path}'")

    with open(config_path, 'r', encoding='utf-8') as config_file:
        config = json.load(config_file)

    callbacks = []
    if checkpoint_epochs is not None:
        callbacks.append(ModelCheckpoint(every_n_epochs=checkpoint_epochs))
        _LOGGER.info('Checkpoints will be saved every %s epoch(s)', checkpoint_epochs)
    if notebook and use_compile:
        callbacks.append(_CompileStatusCallback())

    trainer = Trainer(
        max_epochs=max_epochs,
        devices=devices,
        accelerator=accelerator,
        default_root_dir=default_root_dir,
        precision=precision,
        callbacks=callbacks,
    )

    dict_args = dict(
        seed=seed,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_workers=num_workers,
        validation_split=validation_split,
    )

    if quality == 'x-low':
        dict_args.update({
            'hidden_channels': 96,
            'inter_channels': 96,
            'filter_channels': 384,
        })
    elif quality == 'high':
        dict_args.update({
            'resblock': '1',
            'resblock_kernel_sizes': (3, 7, 11),
            'resblock_dilation_sizes': ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            'upsample_rates': (8, 8, 2, 2),
            'upsample_initial_channel': 512,
            'upsample_kernel_sizes': (16, 16, 4, 4),
        })

    num_symbols = int(config['num_symbols'])
    num_speakers = int(config['num_speakers'])
    sample_rate = int(config['audio']['sample_rate'])
    _LOGGER.debug(f"Config params: num_symbols={num_symbols} num_speakers={num_speakers} sample_rate={sample_rate}")

    if resume_from_checkpoint:
        # TODO (?) - add a flag to use params from config vs from checkpoint in case of mismatch
        ckpt = VitsModel.load_from_checkpoint(resume_from_checkpoint, dataset=None)
        _LOGGER.debug(f"Checkpoint params: num_symbols={ckpt.model_g.n_vocab} num_speakers={ckpt.model_g.n_speakers} sample_rate={ckpt.hparams.sample_rate}")
        if ckpt.model_g.n_vocab != num_symbols:
            _LOGGER.warning(f"Checkpoint num_symbols={ckpt.model_g.n_vocab} does not match config num_symbols={num_symbols}")
            #-------------
            # commented out this code because this is not supposed to happen if you used the preprocess.py script
            # uncomment if you want to use the encoder from checkpoint + update num_symbols in the .json file manually
            #-------------
            #if ckpt.model_g.n_vocab > num_symbols and not discard_encoder:
            #    num_symbols = ckpt.model_g.n_vocab
            #    _LOGGER.info(f"Training with num_symbols={num_symbols}")
            ###############
        if ckpt.model_g.n_speakers != num_speakers:
            _LOGGER.warning(f"Checkpoint num_speakers={ckpt.model_g.n_speakers} does not match config num_speakers={num_speakers}")
            #num_speakers = ckpt.model_g.n_speakers
        if ckpt.hparams.sample_rate != sample_rate:
            _LOGGER.warning(f"Checkpoint sample_rate={ckpt.hparams.sample_rate} does not match config sample_rate={sample_rate}")
            #sample_rate = ckpt.hparams.sample_rate

    model = VitsModel(
        num_symbols=num_symbols,
        num_speakers=num_speakers,
        sample_rate=sample_rate,
        dataset=[dataset_path],
        **dict_args,
    )
    _LOGGER.info(f"VitsModel params: num_symbols={num_symbols} num_speakers={num_speakers} sample_rate={sample_rate}")

    if resume_from_checkpoint:
        saved_state_dict = ckpt.state_dict()

        # Filter the state dictionary by removing the encoder weights
        enc_key = 'model_g.enc_p.emb.weight'
        if enc_key in saved_state_dict:
            saved_shape = saved_state_dict[enc_key].shape
            current_shape = model.state_dict()[enc_key].shape
            if saved_shape[0] != current_shape[0]:
                _LOGGER.warning(
                    "Size mismatch detected for '%s': saved shape %s vs current shape %s. ",
                    enc_key, saved_shape, current_shape
                )
                discard_encoder = True

            if discard_encoder:
                _LOGGER.warning(
                    "Skipping encoder weights from the checkpoint. (will be randomly initialized)"
                )
                saved_state_dict.pop(enc_key)

        load_state_dict(model, saved_state_dict)
        _LOGGER.info("Successfully loaded model weights.")

    if resume_from_single_speaker_checkpoint:
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

    if use_compile:
        _LOGGER.info('Compiling model with torch.compile (mode=%s)', compile_mode)
        model.model_g = torch.compile(model.model_g, mode=compile_mode)
        model.model_d = torch.compile(model.model_d, mode=compile_mode)

    _LOGGER.info('training started!!')
    trainer.fit(model)


if __name__ == '__main__':
    main()
