"""ZipVoice training engine (``--engine zipvoice``).

Port of the `k2-fsa/ZipVoice <https://github.com/k2-fsa/ZipVoice>`_
(Apache-2.0) flow-matching TTS recipe (Zhu et al., "ZipVoice: Fast and
high-quality zero-shot text-to-speech with flow matching",
arXiv:2506.13053) onto the phoonnx training framework. The model,
optimizer and feature code are vendored in ``phoonnx_train/zipvoice/``
(``model.py`` = ZipVoice with the TTSZipformer text encoder +
flow-matching decoder, ``zipformer.py``/``scaling.py``/``solver.py``,
``optim.py`` = ScaledAdam, ``lr_scheduler.py`` = Eden, ``feature.py`` =
Vocos log-mel fbank) and driven by the Lightning loop in
``phoonnx_train/zipvoice/lightning.py``.

Dataset: the shared ``preprocess.py`` output (``dataset.jsonl`` +
``config.json``, phoneme_ids + cached normalized audio); audio is
resampled to 24 kHz and turned into the 100-bin Vocos log-mel features
ZipVoice expects (cached next to the audio cache).

Export (``export_onnx``) produces the two-graph contract
``phoonnx.engines.zipvoice.ZipVoiceAdapter`` consumes —
``text_encoder.onnx`` (tokens + prompt_tokens + prompt_features_len +
speed → text_condition) and the classifier-free-guidance-folded
``fm_decoder.onnx`` (t, x, text_condition, speech_condition,
guidance_scale → v), as upstream ``onnx_export.py``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig

if TYPE_CHECKING:  # heavy import — only needed for type annotations
    import pytorch_lightning as pl

LOG = logging.getLogger(__name__)

_ONNX_OPSET = 13  # upstream onnx_export.py

# model-size presets: "base" = upstream ZipVoice defaults; "low" is a tiny
# CI/smoke tier (architecture only, not a quality recommendation)
_QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "base": {},
    "low": {
        "fm_decoder_dim": 64,
        "fm_decoder_feedforward_dim": 96,
        "fm_decoder_num_heads": 2,
        "fm_decoder_num_layers": [1, 1, 1],
        "fm_decoder_downsampling_factor": [1, 2, 1],
        "fm_decoder_cnn_module_kernel": [7, 7, 7],
        "text_encoder_dim": 32,
        "text_encoder_feedforward_dim": 48,
        "text_encoder_num_layers": 1,
        "query_head_dim": 8,
        "value_head_dim": 8,
        "pos_head_dim": 4,
        "pos_dim": 16,
        "time_embed_dim": 32,
        "text_embed_dim": 32,
    },
}

# kwargs consumed by the ZipVoice model constructor (everything else in the
# extra bag belongs to the Lightning training loop)
_MODEL_KEYS = {
    "fm_decoder_downsampling_factor", "fm_decoder_num_layers",
    "fm_decoder_cnn_module_kernel", "fm_decoder_feedforward_dim",
    "fm_decoder_num_heads", "fm_decoder_dim", "text_encoder_num_layers",
    "text_encoder_feedforward_dim", "text_encoder_cnn_module_kernel",
    "text_encoder_num_heads", "text_encoder_dim", "time_embed_dim",
    "text_embed_dim", "query_head_dim", "value_head_dim", "pos_head_dim",
    "pos_dim", "feat_dim", "vocab_size", "pad_id",
}


def split_model_params(
    extra: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split the CLI ``extra`` bag into (model_params, trainer kwargs).

    Applies the quality preset first, then explicit ``model_params``, then
    any bare model keys — later sources win. Neither input dict nor the
    preset tables are mutated.
    """
    extra = dict(extra)
    quality = extra.pop("quality", "base")
    preset = _QUALITY_PRESETS.get(quality)
    if preset is None:
        LOG.warning("unknown zipvoice quality %r — using 'base'", quality)
        preset = _QUALITY_PRESETS["base"]
    model_params = {**preset, **extra.pop("model_params", {})}
    for key in list(extra):
        if key in _MODEL_KEYS:
            model_params[key] = extra.pop(key)
    return model_params, extra


def _add_meta(path: Path, meta: Dict[str, str]) -> None:
    import onnx

    model = onnx.load(str(path))
    while len(model.metadata_props):
        model.metadata_props.pop()
    for key, value in meta.items():
        prop = model.metadata_props.add()
        prop.key, prop.value = key, str(value)
    onnx.save(model, str(path))


class ZipVoiceTrainingEngine(BaseTrainingEngine):
    """Flow-matching (ZipVoice) training engine."""

    def create_model(self, config: TrainingEngineConfig,
                     dataset_paths: List[Path],
                     **kwargs: Any) -> "pl.LightningModule":
        from phoonnx_train.zipvoice.lightning import ZipVoiceModule

        model_params, extra = split_model_params(config.extra)
        return ZipVoiceModule(
            num_symbols=config.num_symbols,
            sample_rate=24000,  # ZipVoice features are defined at 24 kHz
            source_sample_rate=config.sample_rate,
            dataset=[str(p) for p in dataset_paths],
            model_params=model_params,
            **extra, **kwargs)

    def export_onnx(self, checkpoint_path: Path, config_path: Path,
                    output_dir: Path, **kwargs: Any) -> Path:
        import torch

        from phoonnx_train.zipvoice.lightning import (
            OnnxFlowMatchingModel,
            OnnxTextModel,
            ZipVoiceModule,
        )
        from phoonnx_train.zipvoice.scaling_converter import (
            convert_scaled_to_non_scaled,
        )

        checkpoint_path = Path(checkpoint_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        module = ZipVoiceModule.load_from_checkpoint(
            str(checkpoint_path), map_location="cpu", dataset=None)
        model = module.model.eval()
        # swap training-only scaled modules (SwooshL/R, balancers, ...) for
        # export-friendly equivalents, as upstream onnx_export.py
        convert_scaled_to_non_scaled(model, inplace=True, is_onnx=True)

        from phoonnx_train.torch_compat import onnx_export_kwargs
        export_kwargs: Dict[str, Any] = onnx_export_kwargs()

        # text encoder graph
        text_model = OnnxTextModel(model).eval()
        tokens = torch.tensor([[2, 3, 4, 5]], dtype=torch.int64)
        prompt_tokens = torch.tensor([[0, 1]], dtype=torch.int64)
        prompt_features_len = torch.tensor(10, dtype=torch.int64)
        speed = torch.tensor(1.0, dtype=torch.float32)
        text_path = output_dir / "text_encoder.onnx"
        # check_trace=False: the graph is shape-dynamic (token-duration
        # expand); the exported ONNX is validated by running it instead
        traced = torch.jit.trace(
            text_model, (tokens, prompt_tokens, prompt_features_len, speed),
            check_trace=False)
        torch.onnx.export(
            traced, (tokens, prompt_tokens, prompt_features_len, speed),
            str(text_path), verbose=False, opset_version=_ONNX_OPSET,
            input_names=["tokens", "prompt_tokens", "prompt_features_len",
                         "speed"],
            output_names=["text_condition"],
            dynamic_axes={"tokens": {0: "N", 1: "T"},
                          "prompt_tokens": {0: "N", 1: "T"},
                          "text_condition": {0: "N", 1: "T"}},
            **export_kwargs)
        _add_meta(text_path, {"version": "1",
                              "comment": "ZipVoice text encoder",
                              "use_espeak": "1"})

        # flow-matching decoder graph (CFG folded in, as upstream)
        fm_model = OnnxFlowMatchingModel(model).eval()
        feat_dim = model.feat_dim
        seq_len = 200
        t = torch.tensor(0.5, dtype=torch.float32)
        x = torch.randn(1, seq_len, feat_dim)
        cond = torch.randn(1, seq_len, feat_dim)
        speech = torch.randn(1, seq_len, feat_dim)
        guidance = torch.tensor(1.0, dtype=torch.float32)
        fm_path = output_dir / "fm_decoder.onnx"
        traced = torch.jit.trace(fm_model, (t, x, cond, speech, guidance),
                                 check_trace=False)
        torch.onnx.export(
            traced, (t, x, cond, speech, guidance), str(fm_path),
            verbose=False, opset_version=_ONNX_OPSET,
            input_names=["t", "x", "text_condition", "speech_condition",
                         "guidance_scale"],
            output_names=["v"],
            dynamic_axes={"x": {0: "N", 1: "T"},
                          "text_condition": {0: "N", 1: "T"},
                          "speech_condition": {0: "N", 1: "T"},
                          "v": {0: "N", 1: "T"}},
            **export_kwargs)
        _add_meta(fm_path, {
            "version": "1", "comment": "ZipVoice flow-matching decoder",
            "feat_dim": str(feat_dim), "sample_rate": "24000",
            "n_fft": "1024", "hop_length": "256", "window_length": "1024",
            "num_mels": "100"})

        # phoneme map for the adapter, if the dataset config is available
        if Path(config_path).is_file():
            with open(config_path, "r", encoding="utf-8") as fh:
                dataset_config = json.load(fh)
            (output_dir / "config.json").write_text(json.dumps({
                "engine": "zipvoice",
                "sample_rate": 24000,
                "num_mels": 100,
                "phoneme_id_map": dataset_config.get("phoneme_id_map", {}),
            }, ensure_ascii=False))
        return fm_path

    def quality_presets(self) -> Dict[str, Dict[str, Any]]:
        return _QUALITY_PRESETS

    def load_checkpoint(self, model: "pl.LightningModule",
                        checkpoint_path: Path,
                        **kwargs: Any) -> "pl.LightningModule":
        """Load Lightning or upstream ``{"model": ...}`` checkpoints."""
        import torch

        ckpt = torch.load(checkpoint_path, map_location="cpu",
                          weights_only=True)
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        stripped = {}
        for k, v in state.items():
            for prefix in ("model.", "module."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
            stripped[k] = v
        model_state = model.model.state_dict()
        filtered = {k: v for k, v in stripped.items()
                    if k in model_state and v.shape == model_state[k].shape}
        model.model.load_state_dict(filtered, strict=False)
        LOG.info("loaded %d/%d tensors from %s", len(filtered),
                 len(model_state), checkpoint_path)
        return model
