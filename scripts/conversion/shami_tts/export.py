#!/usr/bin/env python3
"""Export Tushe/shami-tts (HamsVITS) to a phoonnx-compatible ONNX + config.json.

This script wraps hams-tts's own ONNX exporter (the checkpoint is a HamsVITS
state_dict, so we reuse the upstream model code for the export only) and writes
a phoonnx ``config.json`` that pairs the exported model with the vendored Shami
front-end in ``phoonnx.thirdparty.shami``.

Usage:
    python scripts/conversion/shami_tts/export.py \
        --checkpoint-dir ~/.cache/huggingface/hub/models--Tushe--shami-tts/snapshots/<sha> \
        --output-dir ./shami_phoonnx

The resulting directory can be uploaded to Hugging Face and then referenced from
``phoonnx/voice_index/shami.json``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


def build_config() -> Dict[str, Any]:
    """Build a native phoonnx config.json for the ShamiVITS model."""
    from phoonnx.thirdparty.shami import SYMBOL_TO_ID

    return {
        "phoonnx_version": "1.0",
        "engine": "shami",
        "phoneme_type": "shami",
        "alphabet": "ipa",
        "lang_code": "ar",
        "audio": {"sample_rate": 24000},
        "num_symbols": len(SYMBOL_TO_ID),
        "num_speakers": 1,
        "num_langs": 4,
        "speaker_id_map": {},
        "lang_id_map": {"PAD": 0, "AR": 1, "EN": 2, "NEUTRAL": 3},
        "phonemizer_model": None,
        "add_diacritics": False,
        "inference": {
            "noise_scale": 0.667,
            "length_scale": 1.0,
            "noise_w": 0.8,
        },
        "phoneme_id_map": dict(SYMBOL_TO_ID),
        "pad": "<pad>",
        "blank": "<pad>",
        "bos": "<bos>",
        "eos": "<eos>",
        "add_blank_char": False,
        "add_blank_word": False,
        "use_eos_bos": False,
        "blank_at_start": False,
        "blank_at_end": False,
        "word_sep_token": " ",
        "blank_between": "tokens",
        "engine_params": {},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Tushe/shami-tts to phoonnx ONNX")
    ap.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing hams_vits.pt and hams_vits_config.json",
    )
    ap.add_argument(
        "--output-dir",
        default=os.getcwd(),
        help="Directory to write model.onnx and config.json",
    )
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    try:
        from hams_tts.models.optimize.export_onnx import export
    except ImportError as e:
        raise SystemExit(
            "hams-tts is required for export. Install it with:\n"
            "  pip install git+https://github.com/Al-aminI/hams-levantine-tts.git\n"
            f"Original error: {e}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / "model.onnx"
    export(args.checkpoint_dir, str(onnx_path), opset=args.opset, verify=True)

    config = build_config()
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"Wrote {onnx_path} and {config_path}")


if __name__ == "__main__":
    main()
