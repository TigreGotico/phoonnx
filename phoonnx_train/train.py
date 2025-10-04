import json
import logging
from pathlib import Path
import os
import torch
import click
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from phoonnx_train.vits.lightning import VitsModel

_LOGGER = logging.getLogger(__package__)


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
@click.option('--precision', default=32, help='Precision used in training (e.g. 16, 32, bf16) (default: 32)')
# Model-specific arguments
@click.option('--learning-rate', type=float, default=2e-4, help='Learning rate for optimizer (default: 2e-4)')
@click.option('--batch-size', type=int, default=16, help='Training batch size (default: 16)')
@click.option('--num-workers', type=click.IntRange(min=1), default=1, help='Number of data loader workers (default: 1)')
@click.option('--validation-split', type=float, default=0.05, help='Proportion of data used for validation (default: 0.05)')
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
):
    logging.basicConfig(level=logging.DEBUG)

    dataset_dir = Path(dataset_dir)
    if default_root_dir is None:
        default_root_dir = dataset_dir

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)

    config_path = dataset_dir / 'config.json'
    dataset_path = dataset_dir / 'dataset.jsonl'

    print(f"INFO - config_path: '{config_path}'")
    print(f"INFO - dataset_path: '{dataset_path}'")

    with open(config_path, 'r', encoding='utf-8') as config_file:
        config = json.load(config_file)
        num_symbols = int(config['num_symbols'])
        num_speakers = int(config['num_speakers'])
        sample_rate = int(config['audio']['sample_rate'])

    trainer = Trainer(
        max_epochs=max_epochs,
        devices=devices,
        accelerator=accelerator,
        default_root_dir=default_root_dir,
        precision=precision,
        resume_from_checkpoint=resume_from_checkpoint
    )

    if checkpoint_epochs is not None:
        trainer.callbacks = [ModelCheckpoint(every_n_epochs=checkpoint_epochs)]
        _LOGGER.info('Checkpoints will be saved every %s epoch(s)', checkpoint_epochs)

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

    print(f"VitsModel params: num_symbols={num_symbols} num_speakers={num_speakers} sample_rate={sample_rate}")
    model = VitsModel(
        num_symbols=num_symbols,
        num_speakers=num_speakers,
        sample_rate=sample_rate,
        dataset=[dataset_path],
        **dict_args,
    )

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

    print('training started!!')
    trainer.fit(model)


if __name__ == '__main__':
    main()
