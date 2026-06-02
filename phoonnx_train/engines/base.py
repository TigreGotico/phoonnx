"""
Base class for training engines.

Each TTS architecture has different:
  - PyTorch LightningModule structure
  - Dataset / collation requirements
  - Feature extraction needs (mel only vs mel+f0+energy)
  - Preprocessing pipeline
  - ONNX export procedure

A TrainingEngine encapsulates these differences so that the shared
CLI tools (preprocess, train, export_onnx) can work with any
architecture.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import pytorch_lightning as pl


@dataclass
class TrainingEngineConfig:
    """
    Shared training configuration that every engine receives.
    Engine-specific params go in ``extra``.
    """

    num_symbols: int = 256
    num_speakers: int = 1
    sample_rate: int = 22050

    # Engine-specific key→value bag
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseTrainingEngine(ABC):
    """
    Contract for a training engine.

    Subclasses **must** implement:

    - ``create_model``      — build the LightningModule
    - ``export_onnx``       — checkpoint → ONNX
    - ``quality_presets``    — map quality name → model hyper-params

    Subclasses **may** override:

    - ``extra_preprocess``  — additional feature extraction beyond
                              the shared phonemization + audio norm
    - ``extra_cli_options`` — click options specific to this engine
    - ``load_checkpoint``   — custom checkpoint loading logic
    """

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    @abstractmethod
    def create_model(
        self,
        config: TrainingEngineConfig,
        dataset_paths: List[Path],
        **kwargs: Any,
    ) -> pl.LightningModule:
        """
        Build and return the LightningModule for this architecture.
        """

    @abstractmethod
    def export_onnx(
        self,
        checkpoint_path: Path,
        config_path: Path,
        output_dir: Path,
        **kwargs: Any,
    ) -> Path:
        """
        Export a trained checkpoint to ONNX.

        Returns the path to the exported ``.onnx`` file.
        """

    @abstractmethod
    def quality_presets(self) -> Dict[str, Dict[str, Any]]:
        """
        Map quality tier names to model hyper-parameters.

        Example::

            {
                "x-low":  {"hidden_channels": 96, "inter_channels": 96, ...},
                "medium": {"hidden_channels": 192, ...},
                "high":   {"hidden_channels": 256, ...},
            }
        """

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
        Run engine-specific feature extraction *beyond* the shared
        phonemization + audio normalization.

        For example, OptiSpeech needs F0 (pitch) and energy extraction
        via pyworld, while VITS does not.

        Returns a dict of extra fields to merge into the utterance
        record in dataset.jsonl.  Default returns empty.
        """
        return {}

    def extra_cli_options(self) -> List[Any]:
        """
        Return additional ``click.option`` decorators that should be
        added to the training CLI for this engine.  Default is empty.
        """
        return []

    def load_checkpoint(
        self,
        model: pl.LightningModule,
        checkpoint_path: Path,
        **kwargs: Any,
    ) -> pl.LightningModule:
        """
        Load weights from an existing checkpoint into *model*.

        Default implementation does a standard state_dict load with
        missing-key tolerance.

        Engines may accept additional keyword arguments (e.g.
        ``discard_encoder``, ``resume_from_single_speaker_checkpoint``)
        to control load behaviour.
        """
        import torch

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("state_dict", ckpt)
        model_state = model.state_dict()
        filtered = {
            k: v for k, v in state_dict.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        model.load_state_dict(filtered, strict=False)
        return model
