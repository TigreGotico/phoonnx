"""
VITS training engine adapter.

Wraps the existing ``phoonnx_train.vits`` module behind the
``BaseTrainingEngine`` interface so the shared CLI tools can drive it.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # heavy imports — only needed for type annotations
    import pytorch_lightning as pl
    import torch

from phoonnx.compat import to_piper_config, to_sherpa_tokens
from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig

_LOG = logging.getLogger(__name__)

# ONNX opset used for export
OPSET_VERSION = 15

# Quality tier → VITS hyper-param overrides
_QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "x-low": {
        "inter_channels": 96,
        "hidden_channels": 96,
        "filter_channels": 384,
        "n_heads": 2,
        "n_layers": 4,
    },
    "medium": {
        "inter_channels": 192,
        "hidden_channels": 192,
        "filter_channels": 768,
        "n_heads": 2,
        "n_layers": 6,
    },
    "high": {
        "inter_channels": 256,
        "hidden_channels": 256,
        "filter_channels": 1024,
        "n_heads": 4,
        "n_layers": 8,
    },
}


class VitsTrainingEngine(BaseTrainingEngine):

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    def create_model(
        self,
        config: TrainingEngineConfig,
        dataset_paths: List[Path],
        **kwargs: Any,
    ) -> "pl.LightningModule":
        """Build a VitsModel LightningModule from *config*."""
        from phoonnx_train.vits.lightning import VitsModel

        # The VITS dataset loader expects jsonl *files*; the CLI passes the
        # pre-processed dataset *directory* — map dirs to their dataset.jsonl.
        dataset_files = [
            Path(p) / "dataset.jsonl" if Path(p).is_dir() else Path(p)
            for p in dataset_paths
        ]

        return VitsModel(
            num_symbols=config.num_symbols,
            num_speakers=config.num_speakers,
            sample_rate=config.sample_rate,
            dataset=[str(p) for p in dataset_files],
            **config.extra,
            **kwargs,
        )

    def export_onnx(
        self,
        checkpoint_path: Path,
        config_path: Path,
        output_dir: Path,
        **kwargs: Any,
    ) -> Path:
        """Export a VITS checkpoint to ONNX with optional side-car files."""
        import torch

        from phoonnx_train.vits.lightning import VitsModel

        # Load config
        with open(config_path, "r", encoding="utf-8") as f:
            model_config: Dict[str, Any] = json.load(f)

        sample_rate = model_config.get("audio", {}).get("sample_rate", 22050)
        phoneme_id_map = model_config.get("phoneme_id_map", {})
        alphabet = model_config.get("alphabet", "")
        phoneme_type = model_config.get("phoneme_type", "")
        phonemizer_model = model_config.get("phonemizer_model", "")

        # Load model
        model: VitsModel = VitsModel.load_from_checkpoint(
            checkpoint_path, dataset=None,
        )
        model_g = model.model_g
        num_symbols = model_g.n_vocab
        num_speakers = model_g.n_speakers

        model_g.eval()
        with torch.no_grad():
            model_g.dec.remove_weight_norm()
        # a feature input layer exports as the id table it computes
        model_g.enc_p.bake_input()

        # Forward wrapper
        def infer_forward(
            text: "torch.Tensor",
            text_lengths: "torch.Tensor",
            scales: "torch.Tensor",
            sid: "Optional[torch.Tensor]" = None,
        ) -> "torch.Tensor":
            noise_scale = scales[0]
            length_scale = scales[1]
            noise_scale_w = scales[2]
            audio = model_g.infer(
                text, text_lengths,
                noise_scale=noise_scale,
                length_scale=length_scale,
                noise_scale_w=noise_scale_w,
                sid=sid,
            )[0].unsqueeze(1)
            return audio

        model_g.forward = infer_forward

        # Build dummy inputs on the model's own device. A checkpoint trained on a
        # GPU restores there, and inputs built on CPU regardless would meet its
        # weights mid-trace and abort the export on exactly the machines that
        # trained the model.
        device = next(model_g.parameters()).device
        sequences = torch.randint(low=0, high=num_symbols, size=(1, 50),
                                  dtype=torch.long, device=device)
        sequence_lengths = torch.LongTensor([sequences.size(1)]).to(device)
        scales = torch.FloatTensor([0.667, 1.0, 0.8]).to(device)
        sid = torch.LongTensor([0]).to(device) if num_speakers > 1 else None

        input_names = ["input", "input_lengths", "scales"]
        output_names = ["output"]
        dynamic_axes = {
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 1: "channels", 2: "time"},
        }
        dummy_input = (sequences, sequence_lengths, scales)
        if sid is not None:
            input_names.append("sid")
            dynamic_axes["sid"] = {0: "batch_size"}
            dummy_input = (*dummy_input, sid)

        output_path = output_dir / f"{checkpoint_path.name}.onnx"
        from phoonnx_train.torch_compat import onnx_export_kwargs
        export_kwargs = onnx_export_kwargs()
        torch.onnx.export(
            model=model_g,
            args=dummy_input,
            f=str(output_path),
            verbose=False,
            opset_version=OPSET_VERSION,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            **export_kwargs,
        )

        # Metadata
        try:
            import onnx as _onnx

            onnx_model = _onnx.load(str(output_path))
            del onnx_model.metadata_props[:]
            for key, value in {
                "model_type": "vits",
                "n_speakers": num_speakers,
                "n_vocab": num_symbols,
                "sample_rate": sample_rate,
                "alphabet": alphabet,
                "phoneme_type": phoneme_type,
                "phonemizer_model": phonemizer_model,
                "phoneme_id_map": json.dumps(phoneme_id_map),
                "has_espeak": phoneme_type == "espeak",
            }.items():
                meta = onnx_model.metadata_props.add()
                meta.key = key
                meta.value = str(value)
            _onnx.save(onnx_model, str(output_path))
        except ImportError:
            _LOG.warning("onnx package not installed — skipping metadata")

        write_sidecars(model_config, output_dir, checkpoint_path.stem,
                       generate_tokens=kwargs.get("generate_tokens", False),
                       piper=kwargs.get("piper", False))

        _LOG.info("Exported VITS ONNX model to %s", output_path)
        return output_path

    def quality_presets(self) -> Dict[str, Dict[str, Any]]:
        """Return VITS quality tier → hyper-parameter overrides."""
        return _QUALITY_PRESETS

    # ------------------------------------------------------------------
    # Held-out evaluation synthesis
    # ------------------------------------------------------------------

    def eval_synthesize(
        self,
        checkpoint_path: Path,
        config: Dict[str, Any],
        *,
        vocoder_path: Optional[Path] = None,
        device: str = "cpu",
    ) -> Any:
        """Load a VITS checkpoint and return a CPU synthesis callable.

        VITS is an end-to-end model, so ``vocoder_path`` is ignored.
        """
        import numpy as np
        import torch

        from phoonnx_train.torch_compat import trusting_torch_load
        from phoonnx_train.vits.lightning import VitsModel

        with trusting_torch_load():
            model = VitsModel.load_from_checkpoint(
                str(checkpoint_path), dataset=None, map_location=device
            )
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            model.model_g.dec.remove_weight_norm()

        inf = (config or {}).get("inference", {})
        default_scales = [
            float(inf.get("noise_scale", 0.667)),
            float(inf.get("length_scale", 1.0)),
            float(inf.get("noise_w", 0.8)),
        ]

        def synth(ids: List[int], scales: List[float], sid: Optional[int] = None):
            # scales=None/[] means the voice config's inference defaults;
            # when set it is [noise_scale, length_scale, noise_w].
            scales = scales or default_scales
            text = torch.LongTensor(ids).unsqueeze(0).to(device)
            text_lengths = torch.LongTensor([len(ids)]).to(device)
            scales_t = torch.FloatTensor(scales).to(device)
            sid_t = (
                torch.LongTensor([sid]).to(device) if sid is not None else None
            )
            with torch.no_grad():
                audio = model(text, text_lengths, scales_t, sid=sid_t)
            return (
                audio.detach().cpu().float().squeeze().numpy().astype(np.float32)
            )

        return synth

    # ------------------------------------------------------------------
    # Optional
    # ------------------------------------------------------------------

    def load_checkpoint(
        self,
        model: "pl.LightningModule",
        checkpoint_path: Path,
        discard_encoder: bool = False,
        resume_from_single_speaker_checkpoint: bool = False,
    ) -> "pl.LightningModule":
        """VITS-specific checkpoint loading with encoder size mismatch handling."""
        from phoonnx_train.torch_compat import trusting_torch_load
        from phoonnx_train.vits.lightning import VitsModel

        # ------------------------------------------------------------------
        # Single-speaker → multi-speaker conversion
        # ------------------------------------------------------------------
        if resume_from_single_speaker_checkpoint:
            if model.model_g.n_speakers <= 1:
                raise ValueError(
                    "--resume-from-single-speaker-checkpoint is only for "
                    "multi-speaker models."
                )

            with trusting_torch_load():
                ckpt_single = VitsModel.load_from_checkpoint(
                    checkpoint_path, dataset=None
                )
            if model.model_g.enc_p.input_layer is None and ckpt_single.model_g.enc_p.input_layer is not None:
                ckpt_single.model_g.enc_p.bake_input()
            g_dict = ckpt_single.model_g.state_dict()
            self._drop_input_layer(g_dict, model.model_g.state_dict(), ckpt_single.hparams, model.hparams,
                                   discard_encoder, prefix="enc_p.input_layer.")

            # Strip speaker-conditional layers so they stay randomly init'd
            for key in list(g_dict.keys()):
                if (
                    key.startswith("dec.cond")
                    or key.startswith("dp.cond")
                    or ("enc.cond_layer" in key)
                ):
                    g_dict.pop(key, None)

            self._tolerant_load(model.model_g, g_dict)
            self._tolerant_load(model.model_d, ckpt_single.model_d.state_dict())
            _LOG.info(
                "Converted single-speaker checkpoint to multi-speaker: %s",
                checkpoint_path,
            )
            return model

        # ------------------------------------------------------------------
        # Normal checkpoint loading
        # ------------------------------------------------------------------
        with trusting_torch_load():
            ckpt = VitsModel.load_from_checkpoint(checkpoint_path, dataset=None)
        if model.model_g.enc_p.input_layer is None and ckpt.model_g.enc_p.input_layer is not None:
            _LOG.info("Baking the checkpoint's feature input layer into an id embedding")
            ckpt.model_g.enc_p.bake_input()
        saved_state_dict = ckpt.state_dict()
        model_state = model.state_dict()
        self._drop_input_layer(saved_state_dict, model_state, ckpt.hparams, model.hparams, discard_encoder,
                               prefix="model_g.enc_p.input_layer.")

        enc_key = "model_g.enc_p.emb.weight"
        # a feature input layer has no id embedding: the saved one is not loaded
        if enc_key in saved_state_dict and enc_key in model_state:
            if saved_state_dict[enc_key].shape != model_state[enc_key].shape:
                _LOG.warning(
                    "Encoder embedding size mismatch (%s vs %s) — "
                    "discarding encoder weights",
                    saved_state_dict[enc_key].shape,
                    model_state[enc_key].shape,
                )
                saved_state_dict.pop(enc_key)
            elif discard_encoder:
                _LOG.warning(
                    "Discarding encoder weights as requested (--discard-encoder)."
                )
                saved_state_dict.pop(enc_key)

        new_state = {}
        for k, v in model_state.items():
            new_state[k] = saved_state_dict.get(k, v)
        model.load_state_dict(new_state)
        _LOG.info("Loaded checkpoint: %s", checkpoint_path)
        return model

    @staticmethod
    def _drop_input_layer(saved_state_dict: Dict[str, Any], model_state: Dict[str, Any],
                          saved_hparams: Any, hparams: Any, discard_encoder: bool, prefix: str) -> None:
        """Leave out the saved feature input layer weights that do not fit this model.

        The projection fits when the feature names are the same. The per-id, per-base
        and per-mark tables fit only when the symbols are the same, because their rows
        follow the symbols. ``--discard-encoder`` leaves out the whole layer.
        """
        saved_features = saved_hparams.get("input_features") or {}
        features = hparams.get("input_features") or {}
        same_names = saved_features.get("feature_names") == features.get("feature_names")
        same_symbols = saved_features.get("symbols") == features.get("symbols")
        for key in [k for k in saved_state_dict if k.startswith(prefix)]:
            fits = (not discard_encoder and key in model_state
                    and saved_state_dict[key].shape == model_state[key].shape
                    and (same_names if key.startswith(prefix + "proj.") else same_symbols))
            if not fits:
                _LOG.warning("Not loading %s: it does not fit this model's feature table", key)
                saved_state_dict.pop(key)

    @staticmethod
    def _tolerant_load(model: "torch.nn.Module", state_dict: Dict[str, Any]) -> None:
        """Load state dict, keeping current values for missing keys."""
        model_state = model.state_dict()
        new_state = {}
        for k, v in model_state.items():
            new_state[k] = state_dict.get(k, v)
        model.load_state_dict(new_state)


# ------------------------------------------------------------------
# Side-car exporters
# ------------------------------------------------------------------

def _write_tokens_txt(phoneme_id_map: Dict[str, Any], path: Path) -> None:
    """Write the sherpa-onnx ``tokens.txt`` of :func:`phoonnx.compat.to_sherpa_tokens`."""
    path.write_text(to_sherpa_tokens(phoneme_id_map), encoding="utf-8")


def _write_piper_json(model_config: Dict[str, Any], path: Path) -> None:
    """Write the Piper-shaped config of :func:`phoonnx.compat.to_piper_config`."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_piper_config(model_config), f, indent=2, ensure_ascii=False)


def write_sidecars(model_config: Dict[str, Any], output_dir: Path, stem: str,
                   generate_tokens: bool = False, piper: bool = False) -> None:
    """Write the configs that sit beside an exported ONNX model.

    ``<stem>.json`` is always written: the training config in phoonnx's own
    format, so the export folder loads on its own. ``--piper`` adds
    ``<stem>.piper.json`` (:func:`phoonnx.compat.to_piper_config`) and
    ``--generate-tokens`` adds sherpa-onnx's ``<stem>.tokens.txt``
    (:func:`phoonnx.compat.to_sherpa_tokens`). ``phoonnx-voices export`` turns the
    folder into a Piper or sherpa-onnx voice, after it checks that the target gives
    the ids of the voice.
    """
    native_path = output_dir / f"{stem}.json"
    with open(native_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)
    _LOG.info("Wrote phoonnx config: %s", native_path)

    if generate_tokens:
        tokens_path = output_dir / f"{stem}.tokens.txt"
        _write_tokens_txt(model_config.get("phoneme_id_map", {}), tokens_path)
        _LOG.info("Generated tokens.txt: %s", tokens_path)

    if piper:
        piper_path = output_dir / f"{stem}.piper.json"
        _write_piper_json(model_config, piper_path)
        _LOG.info("Generated piper JSON: %s", piper_path)
