"""Standalone Inflect-v2 (Micro/Nano) -> ONNX exporter (no extra package deps
beyond the vendored files in this directory + torch/onnx).

Inflect-v2 (https://github.com/owenawsong/Inflect, Apache-2.0) is a plain VITS
(``jaywalnut310/vits``, MIT — see ``THIRD_PARTY_NOTICES.md`` on the upstream HF
repos) checkpoint: ``SynthesizerTrn`` with a non-stochastic ``DurationPredictor``
(``use_sdp=False``). Its own official ONNX release
(``owensong/Inflect-Micro-v2-ONNX`` / ``owensong/Inflect-Nano-v2-ONNX``) ships
``SynthesizerTrn.infer`` split into two graphs (``duration.onnx`` /
``decode.onnx``) with the flow's latent noise sampled *outside* the graph, so a
caller can seed it explicitly (useful for a JS/WASM browser runtime).

phoonnx's ``VitsAdapter`` already speaks the single-graph piper/coqui-VITS
contract (``input``/``input_lengths``/``scales`` -> waveform, noise sampled
*inside* the graph via ``torch.randn_like`` -> ONNX ``RandomNormalLike``), so
no new engine/adapter is needed: this script exports ``SynthesizerTrn.infer``
directly (the same method the split export traces, just untouched) into that
shape, matching the pattern of ``scripts/conversion/coqui_vits_export``.

Usage::

    # 1. Fetch the upstream PyTorch checkpoint (config.json + model.pth) --
    #    NOT the *-ONNX repo, which ships onnx/ instead of model.pth:
    huggingface-cli download owensong/Inflect-Micro-v2 \\
        --local-dir /tmp/inflect-micro-v2

    # 2. Export
    python export_inflect.py \\
        --model-dir /tmp/inflect-micro-v2 \\
        --out inflect-micro-en.onnx \\
        --model-name Inflect-Micro-v2

This writes ``<out>`` and a matching ``<out>.json`` phoonnx voice config
(piper-shaped ``phoneme_id_map``, ``phoneme_type: espeak``, ``lang_code: en-us``),
ready for ``TTSVoice.load(<out>)``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class InflectVitsExport(nn.Module):
    """Wraps ``SynthesizerTrn.infer`` behind the standard VITS ONNX contract:
    ``(input, input_lengths, scales) -> waveform``, matching phoonnx's
    ``VitsAdapter`` (``scales`` packs ``[noise_scale, length_scale, noise_scale_w]``;
    ``noise_scale_w`` is accepted but unused since Inflect's duration predictor
    is deterministic, i.e. ``use_sdp=False``)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input: torch.Tensor, input_lengths: torch.Tensor,
                scales: torch.Tensor) -> torch.Tensor:
        noise_scale, length_scale, noise_scale_w = scales[0], scales[1], scales[2]
        audio, *_ = self.model.infer(
            input, input_lengths,
            noise_scale=noise_scale, length_scale=length_scale,
            noise_scale_w=noise_scale_w,
        )
        return audio


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(root: Path) -> nn.Module:
    import utils
    from models import SynthesizerTrn
    from text.symbols import symbols

    hps = utils.get_hparams_from_file(str(root / "config.json"))
    model = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).cpu().eval()
    logger = logging.getLogger()
    previous_level = logger.level
    try:
        logger.setLevel(logging.WARNING)
        utils.load_checkpoint(str(root / "model.pth"), model, None)
    finally:
        logger.setLevel(previous_level)
    for module in model.modules():
        if hasattr(module, "remove_weight_norm"):
            try:
                module.remove_weight_norm()
            except Exception:
                pass
    return model


def phoonnx_config(symbols: list, lang_code: str = "en-us") -> dict:
    """Build a piper-shaped phoonnx voice config.json for the exported voice.

    Mirrors the character-level tokenization Inflect's own frontend uses
    (``phonemes_to_tokens`` in ``onnx/inference_onnx.py``): every character of
    the eSpeak IPA output — including spaces and punctuation, which are part
    of the model's own symbol table — is looked up directly, with a blank
    token interleaved between every character (``add_blank_char=True``,
    ``blank_at_start=blank_at_end=True``, no BOS/EOS). phoonnx's
    ``TTSTokenizer.intersperse_blank_char`` reproduces that exactly.
    """
    phoneme_id_map = {symbol: index for index, symbol in enumerate(symbols)}
    return {
        "phoonnx_version": "1.0",
        "phoneme_type": "espeak",
        "alphabet": "ipa",
        "lang_code": lang_code,
        "num_symbols": len(symbols),
        "num_speakers": 1,
        "num_langs": 1,
        "pad": "_",
        "blank": "_",
        "use_eos_bos": False,
        "add_blank_word": False,
        "add_blank_char": True,
        "blank_at_start": True,
        "blank_at_end": True,
        "phoneme_id_map": phoneme_id_map,
        "audio": {"sample_rate": 24000},
        "inference": {
            "noise_scale": 0.667,
            "length_scale": 1.0,
            "noise_w": 0.8,
            "add_diacritics": False,
        },
    }


def export(model_dir: Path, out_path: Path, *, model_name: str, lang_code: str) -> None:
    torch.manual_seed(7)
    from text.symbols import symbols

    model = load_model(model_dir)
    wrapper = InflectVitsExport(model).eval()

    tokens = torch.tensor(
        [[0, 18, 0, 61, 0, 55, 0, 48, 0, 44, 0, 46, 0]], dtype=torch.long,
    )
    lengths = torch.tensor([tokens.shape[1]], dtype=torch.long)
    scales = torch.tensor([0.667, 1.0, 0.8], dtype=torch.float32)

    with torch.no_grad():
        audio = wrapper(tokens, lengths, scales)
    print(f"forward OK -> audio {tuple(audio.shape)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (tokens, lengths, scales),
        str(out_path),
        input_names=["input", "input_lengths", "scales"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 1: "text_len"},
            "input_lengths": {0: "batch"},
            "output": {0: "batch", 2: "wav_len"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print("exported ->", out_path)

    cfg = phoonnx_config(symbols, lang_code=lang_code)
    cfg["_export"] = {
        "model_name": model_name,
        "source_model_sha256": sha256(model_dir / "model.pth"),
        "exporter": "phoonnx scripts/conversion/inflect/export_inflect.py",
        "license": "Apache-2.0 (Inflect-v2); MIT (VITS runtime portions)",
    }
    cfg_path = out_path.with_suffix(out_path.suffix + ".json")
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote config ->", cfg_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Inflect-v2 to a single-graph phoonnx/piper-style ONNX voice.")
    parser.add_argument("--model-dir", type=Path, required=True,
                         help="Local checkout of owensong/Inflect-Micro-v2 or "
                              "owensong/Inflect-Nano-v2 (needs config.json + model.pth)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--lang-code", default="en-us")
    args = parser.parse_args()
    export(args.model_dir.resolve(), args.out.resolve(),
           model_name=args.model_name, lang_code=args.lang_code)


if __name__ == "__main__":
    main()
