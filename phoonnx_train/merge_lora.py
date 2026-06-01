#!/usr/bin/env python3
import json
import logging
import os
from pathlib import Path
from typing import Optional

import click
import torch

from phoonnx_train.vits.lightning import VitsModel
from phoonnx_train.vits.lora_config import LoRAConfig, SCOPE_PRESETS
from phoonnx_train.vits.apply_lora import apply_lora, merge_lora, get_lora_state_dict, load_lora_adapter

logging.basicConfig(level=logging.DEBUG)
_LOGGER = logging.getLogger("phoonnx_train.merge_lora")


@click.command(help="Merge a LoRA adapter into a base VITS checkpoint and export to ONNX.")
@click.argument(
    "base_checkpoint",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--lora-adapter",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to LoRA adapter file (.pt or .ckpt containing lora weights).",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the model configuration JSON file.",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path(os.getcwd()),
    help="Output directory for the merged ONNX model and adapter file.",
)
@click.option(
    "--lora-scope",
    type=click.Choice(list(SCOPE_PRESETS.keys())),
    default="full-acoustic",
    help="LoRA scope preset used during training (must match the training config).",
)
@click.option(
    "--lora-rank",
    type=int,
    default=None,
    help="Override LoRA rank (overrides scope preset).",
)
@click.option(
    "--lora-alpha",
    type=float,
    default=None,
    help="Override LoRA alpha (overrides scope preset).",
)
@click.option(
    "--lora-target-modules",
    type=str,
    default=None,
    help="Comma-separated list of target modules (overrides scope preset). E.g., 'dec,enc_q,flow,dp'",
)
@click.option(
    "--export-onnx/--no-export-onnx",
    default=True,
    help="Whether to also export to ONNX after merging (default: --export-onnx).",
)
@click.option(
    "--save-adapter/--no-save-adapter",
    default=True,
    help="Whether to also save the standalone LoRA adapter file (default: --save-adapter).",
)
def main(
    base_checkpoint: Path,
    lora_adapter: Path,
    config: Optional[Path],
    output_dir: Path,
    lora_scope: str,
    lora_rank: Optional[int],
    lora_alpha: Optional[float],
    lora_target_modules: Optional[str],
    export_onnx: bool,
    save_adapter: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_config = LoRAConfig.from_preset(lora_scope)
    if lora_rank is not None:
        lora_config = LoRAConfig(
            rank=lora_rank,
            alpha=lora_alpha or lora_config.alpha,
            dropout=lora_config.dropout,
            target_modules=lora_config.target_modules,
        )
    if lora_alpha is not None:
        lora_config = LoRAConfig(
            rank=lora_config.rank,
            alpha=lora_alpha,
            dropout=lora_config.dropout,
            target_modules=lora_config.target_modules,
        )
    if lora_target_modules is not None:
        modules = tuple(m.strip() for m in lora_target_modules.split(","))
        lora_config = LoRAConfig(
            rank=lora_config.rank,
            alpha=lora_config.alpha,
            dropout=lora_config.dropout,
            target_modules=modules,
        )

    _LOGGER.info("LoRA config: rank=%d, alpha=%.1f, targets=%s",
                 lora_config.rank, lora_config.alpha, lora_config.target_modules)

    _LOGGER.info("Loading base model from %s", base_checkpoint)
    model = VitsModel.load_from_checkpoint(str(base_checkpoint), dataset=None)

    _LOGGER.info("Applying LoRA with config: %s", lora_config)
    apply_lora(model.model_g, lora_config)

    _LOGGER.info("Loading LoRA adapter from %s", lora_adapter)
    adapter_state = torch.load(str(lora_adapter), map_location="cpu", weights_only=True)

    if "state_dict" in adapter_state:
        lora_weights = {
            k.replace("model_g.", "", 1): v
            for k, v in adapter_state["state_dict"].items()
            if "lora_A" in k or "lora_B" in k
        }
        if not lora_weights:
            lora_weights = {
                k: v for k, v in adapter_state["state_dict"].items()
                if "lora_A" in k or "lora_B" in k
            }
    else:
        lora_weights = adapter_state

    load_lora_adapter(model.model_g, lora_weights)

    _LOGGER.info("Merging LoRA weights into base model")
    merge_lora(model.model_g)

    merged_ckpt_path = output_dir / "merged_model.ckpt"
    _LOGGER.info("Saving merged checkpoint to %s", merged_ckpt_path)
    torch.save({"state_dict": model.model_g.state_dict()}, str(merged_ckpt_path))

    if save_adapter:
        standalone_adapter = get_lora_state_dict(model.model_g)
        adapter_path = output_dir / "lora_adapter.pt"
        _LOGGER.info("Saving standalone LoRA adapter to %s", adapter_path)
        torch.save(standalone_adapter, str(adapter_path))

        if config is not None:
            with open(config, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            lora_meta = {
                "lora_rank": lora_config.rank,
                "lora_alpha": lora_config.alpha,
                "lora_target_modules": list(lora_config.target_modules),
                "lora_scope": lora_scope,
            }
            config_data["lora"] = lora_meta
            config_out = output_dir / "config.json"
            with open(config_out, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            _LOGGER.info("Saved config with LoRA metadata to %s", config_out)

    if export_onnx and config is not None:
        from phoonnx_train.export_onnx import cli as export_cli
        from click.testing import CliRunner

        onnx_output = output_dir / "merged_model.onnx"
        _LOGGER.info("Exporting merged model to ONNX: %s", onnx_output)

        runner = CliRunner()
        result = runner.invoke(export_cli, [
            str(merged_ckpt_path),
            "-c", str(config),
            "-o", str(output_dir),
        ])
        if result.exit_code != 0:
            _LOGGER.error("ONNX export failed: %s", result.output)
            raise click.ClickException(f"ONNX export failed:\n{result.output}")
        _LOGGER.info("ONNX export complete")

    _LOGGER.info("Merge complete. Output directory: %s", output_dir)


if __name__ == "__main__":
    main()