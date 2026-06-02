"""
Disentangled VITS training engine adapter.

Wraps the ``phoonnx_train.vits_disentangled`` module behind the
``BaseTrainingEngine`` interface.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch

from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig

_LOG = logging.getLogger(__name__)

OPSET_VERSION = 15

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


class DisentangledVitsTrainingEngine(BaseTrainingEngine):
    def create_model(
        self,
        config: TrainingEngineConfig,
        dataset_paths: List[Path],
        **kwargs: Any,
    ) -> pl.LightningModule:
        """Build a DisentangledVitsModel LightningModule from *config*."""
        from phoonnx_train.vits_disentangled.lightning import DisentangledVitsModel

        return DisentangledVitsModel(
            num_symbols=config.num_symbols,
            num_speakers=config.num_speakers,
            sample_rate=config.sample_rate,
            dataset=[str(p) for p in dataset_paths],
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
        """Export a Disentangled VITS checkpoint to ONNX."""
        from phoonnx_train.vits_disentangled.lightning import DisentangledVitsModel

        with open(config_path, "r", encoding="utf-8") as f:
            model_config: Dict[str, Any] = json.load(f)

        sample_rate = model_config.get("audio", {}).get("sample_rate", 22050)
        phoneme_id_map = model_config.get("phoneme_id_map", {})
        alphabet = model_config.get("alphabet", "")
        phoneme_type = model_config.get("phoneme_type", "")
        phonemizer_model = model_config.get("phonemizer_model", "")

        model: DisentangledVitsModel = DisentangledVitsModel.load_from_checkpoint(
            checkpoint_path, dataset=None,
        )
        model_g = model.model_g
        num_symbols = model_g.n_vocab
        num_speakers = model_g.n_speakers

        model_g.eval()
        with torch.no_grad():
            model_g.dec.remove_weight_norm()

        # Forward wrapper that accepts the disentangled reference inputs
        def infer_forward(
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            scales: torch.Tensor,
            sid: Optional[torch.Tensor] = None,
            timbre_ref_mel: Optional[torch.Tensor] = None,
            artic_ref_mel: Optional[torch.Tensor] = None,
            prosody_ref_mel: Optional[torch.Tensor] = None,
            emotion_id: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            noise_scale = scales[0]
            length_scale = scales[1]
            noise_scale_w = scales[2]
            audio = model_g.infer(
                text, text_lengths,
                noise_scale=noise_scale,
                length_scale=length_scale,
                noise_scale_w=noise_scale_w,
                sid=sid,
                timbre_ref_mel=timbre_ref_mel,
                artic_ref_mel=artic_ref_mel,
                prosody_ref_mel=prosody_ref_mel,
                emotion_id=emotion_id,
            )[0].unsqueeze(1)
            return audio

        model_g.forward = infer_forward

        # Build dummy inputs
        sequences = torch.randint(low=0, high=num_symbols, size=(1, 50), dtype=torch.long)
        sequence_lengths = torch.LongTensor([sequences.size(1)])
        scales = torch.FloatTensor([0.667, 1.0, 0.8])
        sid = torch.LongTensor([0]) if num_speakers > 1 else None

        # Dummy reference mels and emotion id for tracing
        dummy_ref_mel = torch.zeros(1, 80, 50)
        dummy_emotion_id = torch.LongTensor([0])

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

        input_names.extend([
            "timbre_ref_mel", "artic_ref_mel", "prosody_ref_mel", "emotion_id",
        ])
        dynamic_axes.update({
            "timbre_ref_mel": {0: "batch_size", 2: "time"},
            "artic_ref_mel": {0: "batch_size", 2: "time"},
            "prosody_ref_mel": {0: "batch_size", 2: "time"},
        })
        dummy_input = (*dummy_input, dummy_ref_mel, dummy_ref_mel, dummy_ref_mel, dummy_emotion_id)

        output_path = output_dir / f"{checkpoint_path.name}.onnx"
        torch.onnx.export(
            model=model_g,
            args=dummy_input,
            f=str(output_path),
            verbose=False,
            opset_version=OPSET_VERSION,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

        # Metadata
        try:
            import onnx as _onnx

            onnx_model = _onnx.load(str(output_path))
            del onnx_model.metadata_props[:]
            for key, value in {
                "model_type": "disentangled-vits",
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

        generate_tokens = kwargs.get("generate_tokens", False)
        piper = kwargs.get("piper", False)

        if generate_tokens:
            tokens_path = output_dir / "tokens.txt"
            _write_tokens_txt(phoneme_id_map, tokens_path)
            _LOG.info("Generated tokens.txt: %s", tokens_path)

        if piper:
            piper_path = output_dir / f"{checkpoint_path.stem}.json"
            _write_piper_json(model_config, piper_path)
            _LOG.info("Generated piper JSON: %s", piper_path)

        _LOG.info("Exported Disentangled VITS ONNX model to %s", output_path)
        return output_path

    def quality_presets(self) -> Dict[str, Dict[str, Any]]:
        return _QUALITY_PRESETS

    def load_checkpoint(
        self,
        model: pl.LightningModule,
        checkpoint_path: Path,
        discard_encoder: bool = False,
        resume_from_single_speaker_checkpoint: bool = False,
    ) -> pl.LightningModule:
        """Disentangled VITS-specific checkpoint loading."""
        from phoonnx_train.vits_disentangled.lightning import DisentangledVitsModel

        if resume_from_single_speaker_checkpoint:
            if model.model_g.n_speakers <= 1:
                raise ValueError(
                    "--resume-from-single-speaker-checkpoint is only for "
                    "multi-speaker models."
                )

            ckpt_single = DisentangledVitsModel.load_from_checkpoint(
                checkpoint_path, dataset=None
            )
            g_dict = ckpt_single.model_g.state_dict()

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

        ckpt = DisentangledVitsModel.load_from_checkpoint(checkpoint_path, dataset=None)
        saved_state_dict = ckpt.state_dict()
        model_state = model.state_dict()

        enc_key = "model_g.enc_p.emb.weight"
        if enc_key in saved_state_dict:
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
    def _tolerant_load(model: torch.nn.Module, state_dict: Dict[str, Any]) -> None:
        model_state = model.state_dict()
        new_state = {}
        for k, v in model_state.items():
            new_state[k] = state_dict.get(k, v)
        model.load_state_dict(new_state)


# ------------------------------------------------------------------
# Side-car exporters
# ------------------------------------------------------------------

def _write_tokens_txt(phoneme_id_map: Dict[str, Any], path: Path) -> None:
    if not phoneme_id_map:
        path.write_text("", encoding="utf-8")
        return
    entries = []
    max_id = -1
    for phoneme, idx in phoneme_id_map.items():
        if isinstance(idx, int):
            entries.append((idx, phoneme))
            max_id = max(max_id, idx)
        elif isinstance(idx, list):
            for i in idx:
                if isinstance(i, int):
                    entries.append((i, phoneme))
                    max_id = max(max_id, i)
    tokens = [f"<UNK_{i}>" for i in range(max_id + 1)]
    for idx, phoneme in entries:
        tokens[idx] = phoneme
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8")


def _write_piper_json(model_config: Dict[str, Any], path: Path) -> None:
    piper_cfg = {
        "audio": model_config.get("audio", {}),
        "inference": model_config.get("inference", {}),
        "num_symbols": model_config.get("num_symbols"),
        "num_speakers": model_config.get("num_speakers"),
        "phoneme_id_map": model_config.get("phoneme_id_map", {}),
        "speaker_id_map": model_config.get("speaker_id_map", {}),
        "language": model_config.get("language", {}),
        "espeak": model_config.get("espeak", {}),
        "phoneme_type": model_config.get("phoneme_type", ""),
        "phonemizer_model": model_config.get("phonemizer_model", ""),
        "alphabet": model_config.get("alphabet", ""),
    }
    piper_cfg = {k: v for k, v in piper_cfg.items() if v is not None}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(piper_cfg, f, indent=2, ensure_ascii=False)
