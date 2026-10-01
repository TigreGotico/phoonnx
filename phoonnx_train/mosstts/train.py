"""Training / finetuning CLI for MOSS-TTS-Nano.

Two mutually exclusive ways to start from existing weights:

``--warm-start-from PATH``
    *finetune*: copy weights from an upstream checkpoint directory, a ``.safetensors``
    file, or a previous phoonnx ``.ckpt``. The optimizer and LR schedule start fresh.
``--resume-from PATH.ckpt``
    *resume*: Lightning restores model **and** optimizer/scheduler state and the global
    step, so an interrupted run continues exactly where it stopped.

Example::

    python -m phoonnx_train.mosstts.train \\
        --train-jsonl data/train.codes.jsonl \\
        --tokenizer-model models/MOSS-TTS-Nano/tokenizer.model \\
        --config models/MOSS-TTS-Nano/config.json \\
        --warm-start-from models/MOSS-TTS-Nano \\
        --output-dir runs/moss-pt --max-steps 2000 --batch-size 1
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from phoonnx_train.mosstts.config import MossTTSNanoConfig
from phoonnx_train.mosstts.dataset import (
    MossTTSNanoCollator,
    MossTTSNanoDataset,
    SentencePieceTextTokenizer,
)
from phoonnx_train.mosstts.lightning import SCHEDULER_CHOICES, MossTTSNanoModule

_LOGGER = logging.getLogger("mosstts.train")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finetune MOSS-TTS-Nano.")
    parser.add_argument("--train-jsonl", nargs="+", required=True,
                        help="JSONL(s) produced by prepare_data.py")
    parser.add_argument("--tokenizer-model", required=True, help="SentencePiece tokenizer.model")
    parser.add_argument("--config", default=None,
                        help="config JSON (upstream config.json works); defaults to the "
                             "released MOSS-TTS-Nano geometry")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warm-start-from", default=None, help="finetune: copy weights only")
    parser.add_argument("--resume-from", default=None, help="resume: restore optimizer + scheduler too")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="linear", choices=SCHEDULER_CHOICES)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--channelwise-loss-weight", default="1,32",
                        help="either n_vq+1 weights or 'text,total_audio'")
    parser.add_argument("--prompt-style", default="inference", choices=("inference", "finetuning"))
    parser.add_argument("--precision", default=None,
                        help="Lightning precision string, e.g. bf16-mixed (default: 32 on CPU)")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--attn-implementation", default=None, choices=(None, "eager", "sdpa"))
    parser.add_argument("--save-every-n-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every-n-steps", type=int, default=1)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pl.seed_everything(args.seed, workers=True)

    if args.warm_start_from and args.resume_from:
        raise SystemExit("--warm-start-from and --resume-from are mutually exclusive")

    config = (
        MossTTSNanoConfig.from_json_file(args.config) if args.config else MossTTSNanoConfig()
    )
    if args.attn_implementation:
        config.attn_implementation = args.attn_implementation
        config.local_transformer_attn_implementation = args.attn_implementation

    tokenizer = SentencePieceTextTokenizer(args.tokenizer_model)
    dataset = MossTTSNanoDataset.from_jsonl(
        args.train_jsonl,
        tokenizer=tokenizer,
        config=config,
        max_length=args.max_length,
        prompt_style=args.prompt_style,
    )
    _LOGGER.info("loaded %d training records", len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=MossTTSNanoCollator(config),
        pin_memory=torch.cuda.is_available(),
    )

    module = MossTTSNanoModule(
        config=config,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_train_steps=args.max_steps if args.max_steps > 0 else None,
        channelwise_loss_weight=args.channelwise_loss_weight,
        warm_start_from=args.warm_start_from,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save_json(output_dir / "config.json")
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="moss-tts-nano-{step:07d}",
        save_last=True,
        save_top_k=-1,
        every_n_train_steps=args.save_every_n_steps or None,
    )

    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision or ("bf16-mixed" if torch.cuda.is_available() else "32-true"),
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=args.max_grad_norm if args.max_grad_norm > 0 else None,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[checkpoint_callback],
        logger=CSVLogger(save_dir=str(output_dir), name="logs"),
        enable_progress_bar=True,
    )
    trainer.fit(module, loader, ckpt_path=args.resume_from)
    _LOGGER.info("done; checkpoints in %s", output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
