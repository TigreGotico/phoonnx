"""
OptiSpeech training engine adapter.

Bridges the OptiSpeech model architecture into the phoonnx_train
engine system.  This allows training OptiSpeech models via the
shared CLI::

    python -m phoonnx_train.preprocess --engine optispeech ...
    python -m phoonnx_train.train      --engine optispeech ...
    python -m phoonnx_train.export_onnx model.ckpt --engine optispeech ...

Key differences from VITS:
- Requires F0 (pitch) and energy extraction during preprocessing (pyworld)
- Uses manual optimization (GAN training with generator pretraining)
- Uses Conformer/Transformer encoder + alignment module
- Exports 3 ONNX outputs: wav, wav_lengths, durations
- Embeds full inference config in ONNX metadata
"""
import logging
from pathlib import Path
from typing import Any, Dict, List

import json
import onnx
import librosa
import numpy as np
import pytorch_lightning as pl
import torch

from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig
from phoonnx_train.optispeech.model.optispeech import OptiSpeech

_LOG = logging.getLogger(__name__)

OPSET_VERSION = 16

# Quality tier → OptiSpeech hyper-param overrides
# dim controls the hidden dimension across encoder/decoder/predictors
_QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "x-low": {
        "dim": 128,
    },
    "medium": {
        "dim": 256,
    },
    "high": {
        "dim": 384,
    },
}


class OptiSpeechTrainingEngine(BaseTrainingEngine):
    """
    Training engine for OptiSpeech (FastSpeech2 + GAN).

    This engine depends on the ``optispeech`` package being installed.
    It imports from ``optispeech.model``, ``optispeech.dataset``, and
    ``optispeech.onnx`` at method call time to avoid hard import errors
    when optispeech is not installed.
    """

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    def create_model(
            self,
            config: TrainingEngineConfig,
            dataset_paths: List[Path],
            **kwargs: Any,
    ) -> pl.LightningModule:
        """
        Build the OptiSpeech LightningModule.

        Expects ``config.extra`` to contain the Hydra-style sub-configs
        that OptiSpeech uses (generator, vocoder, discriminator, etc.)
        or a path to a YAML/JSON config that will be parsed.

        For the initial integration, this creates the model from the
        serialized config stored in the dataset directory.
        """
        # OptiSpeech uses Hydra DictConfig internally.  For phoonnx_train
        # integration, we pass a plain dict through config.extra and let
        # OptiSpeech reconstruct its config objects.
        model_kwargs = dict(config.extra)
        model_kwargs.setdefault("dim", 256)

        _LOG.info(
            "Creating OptiSpeech model (dim=%d, speakers=%d, sr=%d)",
            model_kwargs.get("dim", 256),
            config.num_speakers,
            config.sample_rate,
        )

        # The actual model construction requires the full Hydra config
        # objects (generator, vocoder, discriminator, etc.) which are
        # architecture-specific.  We store the config path and let
        # OptiSpeech handle instantiation.
        config_path = model_kwargs.pop("optispeech_config", None)
        if config_path:
            return self._create_from_hydra_config(config_path, config)

        raise ValueError(
            "OptiSpeech training requires an optispeech_config path in "
            "config.extra pointing to a Hydra YAML configuration. "
            "Example: --optispeech-config path/to/config.yaml"
        )

    def _create_from_hydra_config(
            self,
            config_path: str,
            engine_config: TrainingEngineConfig,
    ) -> pl.LightningModule:
        """Build OptiSpeech model from a Hydra config file."""
        # TODO - get rid of yda
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate

        config_dir = str(Path(config_path).parent)
        config_name = Path(config_path).stem

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name=config_name)

        return instantiate(cfg.model)

    def export_onnx(
            self,
            checkpoint_path: Path,
            config_path: Path,
            output_dir: Path,
            **kwargs: Any,
    ) -> Path:
        """
        Export an OptiSpeech checkpoint to ONNX with embedded metadata.

        Uses the same export procedure as optispeech/onnx/export.py but
        invoked through the engine interface.
        """
        _LOG.info("Loading OptiSpeech checkpoint: %s", checkpoint_path)

        model = OptiSpeech.load_from_checkpoint(
            str(checkpoint_path), map_location="cpu"
        )
        model.eval()

        # Access model internals
        generator = model.generator
        is_multi_speaker = model.num_speakers > 1
        is_multi_language = model.text_processor.is_multi_language

        # Remove alignment module (not needed at inference)
        if hasattr(generator, "alignment_module"):
            del generator.alignment_module

        # Build ONNX forward
        def _infer_forward(x, x_lengths, scales, sids=None, lids=None):
            d_factor = scales[0]
            p_factor = scales[1]
            e_factor = scales[2]
            outputs = generator.synthesise(
                x, x_lengths,
                sids=sids, lids=lids,
                d_factor=d_factor, p_factor=p_factor, e_factor=e_factor,
            )
            return outputs["wav"], outputs["wav_lengths"], outputs["durations"]

        generator.forward = _infer_forward

        # Dummy inputs
        num_symbols = len(model.text_processor.tokenizer.input_symbols)
        dummy_len = 50
        x = torch.randint(0, num_symbols, (1, dummy_len), dtype=torch.long)
        x_lengths = torch.LongTensor([dummy_len])
        scales = torch.FloatTensor([1.0, 1.0, 1.0])

        input_names = ["x", "x_lengths", "scales"]
        output_names = ["wav", "wav_lengths", "durations"]
        dynamic_axes = {
            "x": {0: "batch_size", 1: "time"},
            "x_lengths": {0: "batch_size"},
            "wav": {0: "batch_size", 2: "frames"},
            "wav_lengths": {0: "batch_size"},
            "durations": {0: "batch_size", 1: "time"},
        }
        dummy_input = [x, x_lengths, scales]

        if is_multi_speaker:
            dummy_input.append(torch.LongTensor([0]))
            input_names.append("sids")
            dynamic_axes["sids"] = {0: "batch_size"}

        if is_multi_language:
            dummy_input.append(torch.LongTensor([0]))
            input_names.append("lids")
            dynamic_axes["lids"] = {0: "batch_size"}

        output_path = output_dir / f"{checkpoint_path.stem}.onnx"
        output_dir.mkdir(parents=True, exist_ok=True)

        _LOG.info("Exporting ONNX to %s (opset %d)", output_path, OPSET_VERSION)

        generator._jit_is_scripting = True
        torch.onnx.export(
            generator,
            f=str(output_path),
            args=tuple(dummy_input),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
        )

        # Embed inference metadata
        self._add_inference_metadata(output_path, model)

        _LOG.info("Exported OptiSpeech ONNX model to %s", output_path)
        return output_path

    def _add_inference_metadata(self, onnx_path: Path, model) -> None:
        """Embed inference config in the ONNX model metadata."""

        onnx_model = onnx.load(str(onnx_path))

        text_processor = model.text_processor
        tokenizer = text_processor.tokenizer
        languages = list(text_processor.languages)

        infer_dict = {
            "name": getattr(model.hparams, "data_args", {}).get("name", "optispeech"),
            "sample_rate": model.sample_rate,
            "inference_args": dict(model.inference_args)
            if hasattr(model.inference_args, "__iter__")
            else {"d_factor": 1.0, "p_factor": 1.0, "e_factor": 1.0},
            "input_symbols": tokenizer.input_symbols,
            "special_symbols": tokenizer.special_symbols,
            "speakers": [],
            "languages": languages,
            "text_processor": text_processor.asdict(),
        }

        meta = onnx_model.metadata_props.add()
        meta.key = "inference"
        meta.value = json.dumps(infer_dict)

        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, str(onnx_path))

    def quality_presets(self) -> Dict[str, Dict[str, Any]]:
        return _QUALITY_PRESETS

    # ------------------------------------------------------------------
    # Optional
    # ------------------------------------------------------------------

    def extra_preprocess(
            self,
            utterance_audio_path: Path,
            cache_dir: Path,
            sample_rate: int,
            **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Extract F0 (pitch) and energy features using pyworld.

        OptiSpeech requires these additional features beyond the
        mel spectrograms that the shared preprocessing already produces.

        Returns dict with keys: f0_path, energy_path
        """
        # Load audio
        wav, sr = librosa.load(str(utterance_audio_path), sr=sample_rate, mono=True)

        # F0 extraction via pyworld (DIO + StoneMask)
        import pyworld as pw
        wav_double = wav.astype(np.float64)
        f0, timeaxis = pw.dio(wav_double, sr)
        f0 = pw.stonemask(wav_double, f0, timeaxis, sr)
        f0 = f0.astype(np.float32)

        # Energy (RMS per frame)
        hop_length = kwargs.get("hop_length", 256)
        n_fft = kwargs.get("n_fft", 1024)
        energy = librosa.feature.rms(
            y=wav, frame_length=n_fft, hop_length=hop_length
        ).squeeze()

        # Cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = utterance_audio_path.stem
        f0_path = cache_dir / f"{stem}.f0.npy"
        energy_path = cache_dir / f"{stem}.energy.npy"
        np.save(f0_path, f0)
        np.save(energy_path, energy)

        return {
            "f0_path": str(f0_path),
            "energy_path": str(energy_path),
        }

    def load_checkpoint(
            self,
            model: pl.LightningModule,
            checkpoint_path: Path,
    ) -> pl.LightningModule:
        """Load OptiSpeech checkpoint."""
        import torch

        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model_state = model.state_dict()

        # Tolerant load: skip mismatched shapes
        filtered = {}
        for k, v in state_dict.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered[k] = v
            elif k in model_state:
                _LOG.warning(
                    "Shape mismatch for %s: ckpt=%s model=%s — skipping",
                    k, v.shape, model_state[k].shape,
                )

        model.load_state_dict(filtered, strict=False)
        _LOG.info("Loaded %d/%d parameters from checkpoint", len(filtered), len(model_state))
        return model
