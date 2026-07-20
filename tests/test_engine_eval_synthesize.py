"""Tests for the standalone ``eval_synthesize`` interface added to
``phoonnx_train.engines.base.BaseTrainingEngine`` and implemented by the
Matcha-TTS and OptiSpeech training engines.

Every test builds a tiny real (CPU-sized) model, trains it for a single
step so a checkpoint exists, then drives ``eval_synthesize`` end to end —
no mocking of the model internals.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from phoonnx_train.engines import get_engine
from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig


def _tiny_trainer(max_steps=1):
    return pl.Trainer(
        max_steps=max_steps,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


# ----------------------------------------------------------------------
# Matcha
# ----------------------------------------------------------------------

_M_NFEATS = 20
_M_TX = 6
_M_TY = 16


class _MatchaDummyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return idx


def _matcha_collate(batch):
    b = len(batch)
    return {
        "x": torch.randint(1, 30, (b, _M_TX)),
        "x_lengths": torch.tensor([_M_TX] * b),
        "y": torch.randn(b, _M_NFEATS, _M_TY),
        "y_lengths": torch.tensor([_M_TY] * b),
    }


def _build_matcha_checkpoint(tmp_path) -> Path:
    eng = get_engine("matcha")
    cfg = TrainingEngineConfig(
        num_symbols=30,
        num_speakers=1,
        sample_rate=22050,
        extra={
            "quality": "x-low",
            "n_feats": _M_NFEATS,
            "mel_mean": 0.0,
            "mel_std": 1.0,
        },
    )
    model = eng.create_model(cfg, [])
    dl = DataLoader(_MatchaDummyDataset(), batch_size=2, collate_fn=_matcha_collate)
    trainer = _tiny_trainer(max_steps=1)
    trainer.fit(model, dl)
    ckpt = tmp_path / "matcha.ckpt"
    trainer.save_checkpoint(str(ckpt))
    return ckpt


def _export_tiny_raw_vocoder(tmp_path, n_feats: int, hop_length: int) -> Path:
    """A minimal torch module exported to ONNX with the 'raw waveform'
    vocoder contract: single input mel [1, n_feats, T] -> single output
    wav [1, T*hop_length]. Not trained — only exercises the eval_synthesize
    ONNX-vocoder code path end to end."""

    class TinyVocoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.ConvTranspose1d(
                n_feats, 1, kernel_size=hop_length * 2, stride=hop_length,
                padding=hop_length // 2,
            )

        def forward(self, mel):
            wav = self.up(mel).squeeze(1)
            return torch.tanh(wav)

    module = TinyVocoder()
    module.eval()
    dummy = torch.randn(1, n_feats, 10)
    onnx_path = tmp_path / "tiny_vocoder.onnx"
    from phoonnx_train.torch_compat import onnx_export_kwargs

    torch.onnx.export(
        module,
        (dummy,),
        str(onnx_path),
        input_names=["mels"],
        output_names=["wav"],
        dynamic_axes={"mels": {0: "batch", 2: "time"}, "wav": {0: "batch", 1: "time"}},
        opset_version=15,
        **onnx_export_kwargs(),
    )
    return onnx_path


class MatchaEvalSynthesizeTests(unittest.TestCase):
    def test_griffinlim_fallback_produces_finite_audio(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            ckpt = _build_matcha_checkpoint(tmp_path)
            eng = get_engine("matcha")
            synth = eng.eval_synthesize(ckpt, config={"n_timesteps": 2})
            torch.manual_seed(1234)
            wav = synth([1, 2, 3, 4, 5], [0.667, 1.0], None)
            self.assertIsInstance(wav, np.ndarray)
            self.assertEqual(wav.dtype, np.float32)
            self.assertEqual(wav.ndim, 1)
            self.assertGreater(wav.size, 0)
            self.assertTrue(np.isfinite(wav).all())

    def test_onnx_vocoder_path_runs(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            ckpt = _build_matcha_checkpoint(tmp_path)
            vocoder_path = _export_tiny_raw_vocoder(
                tmp_path, n_feats=_M_NFEATS, hop_length=256
            )
            eng = get_engine("matcha")
            synth = eng.eval_synthesize(
                ckpt,
                config={"n_timesteps": 2, "vocoder": {"vocoder_type": "raw"}},
                vocoder_path=str(vocoder_path),
            )
            torch.manual_seed(1234)
            wav = synth([1, 2, 3, 4, 5], [0.667, 1.0], None)
            self.assertIsInstance(wav, np.ndarray)
            self.assertEqual(wav.ndim, 1)
            self.assertGreater(wav.size, 0)
            self.assertTrue(np.isfinite(wav).all())

    def test_determinism_same_seed_same_output(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            ckpt = _build_matcha_checkpoint(tmp_path)
            eng = get_engine("matcha")
            synth = eng.eval_synthesize(ckpt, config={"n_timesteps": 2})

            torch.manual_seed(4242)
            wav_a = synth([1, 2, 3, 4, 5], [0.667, 1.0], None)
            torch.manual_seed(4242)
            wav_b = synth([1, 2, 3, 4, 5], [0.667, 1.0], None)
            np.testing.assert_array_equal(wav_a, wav_b)


# ----------------------------------------------------------------------
# OptiSpeech
# ----------------------------------------------------------------------

_O_NFEATS = 40
_O_HOP = 256
_O_TX = 8
_O_TY = 24


def _optispeech_tiny_extra(**over):
    extra = {
        "quality": "x-low",
        "dim": 32,
        "n_feats": _O_NFEATS,
        "hop_length": _O_HOP,
        "n_fft": 256,
        "win_length": 256,
        "encoder_intermediate_dim": 48,
        "decoder_intermediate_dim": 48,
        "encoder_num_layers": 2,
        "decoder_num_layers": 2,
        "vocoder_dim": 32,
        "vocoder_intermediate_dim": 48,
        "vocoder_num_layers": 2,
        "duration_intermediate_dim": 32,
        "pitch_intermediate_dim": 32,
        "energy_intermediate_dim": 32,
        "pitch_num_layers": 2,
        "segment_size": 8,
        "batch_size": 2,
        "warmup_steps": 1,
    }
    extra.update(over)
    return extra


class _OptiDummyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return idx


def _optispeech_collate(batch):
    b = len(batch)
    return {
        "x": torch.randint(1, 30, (b, _O_TX)),
        "x_lengths": torch.tensor([_O_TX] * b),
        "mel": torch.randn(b, _O_NFEATS, _O_TY),
        "mel_lengths": torch.tensor([_O_TY] * b),
        "pitches": torch.randn(b, _O_TY),
        "energies": torch.randn(b, _O_TY),
        "sids": None,
        "lids": None,
        "wav": np.random.randn(b, _O_TY * _O_HOP).astype(np.float32),
    }


def _build_optispeech_checkpoint(tmp_path) -> Path:
    eng = get_engine("optispeech")
    cfg = TrainingEngineConfig(
        num_symbols=30,
        num_speakers=1,
        sample_rate=22050,
        extra=_optispeech_tiny_extra(),
    )
    model = eng.create_model(cfg, [])
    dl = DataLoader(_OptiDummyDataset(), batch_size=2, collate_fn=_optispeech_collate)
    trainer = _tiny_trainer(max_steps=1)
    trainer.fit(model, dl)
    ckpt = tmp_path / "optispeech.ckpt"
    trainer.save_checkpoint(str(ckpt))
    return ckpt


class OptiSpeechEvalSynthesizeTests(unittest.TestCase):
    def test_self_vocoding_produces_finite_audio(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            ckpt = _build_optispeech_checkpoint(tmp_path)
            eng = get_engine("optispeech")
            synth = eng.eval_synthesize(ckpt, config={})
            wav = synth([1, 2, 3, 4, 5, 6], [1.0, 1.0, 1.0], None)
            self.assertIsInstance(wav, np.ndarray)
            self.assertEqual(wav.dtype, np.float32)
            self.assertEqual(wav.ndim, 1)
            self.assertGreater(wav.size, 0)
            self.assertTrue(np.isfinite(wav).all())

    def test_vocoder_path_ignored_with_debug_log(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            ckpt = _build_optispeech_checkpoint(tmp_path)
            eng = get_engine("optispeech")
            # Passing a bogus vocoder_path must not raise or be used —
            # OptiSpeech self-vocodes.
            synth = eng.eval_synthesize(
                ckpt, config={}, vocoder_path="/nonexistent/vocoder.onnx"
            )
            wav = synth([1, 2, 3, 4, 5, 6], [1.0, 1.0, 1.0], None)
            self.assertGreater(wav.size, 0)
            self.assertTrue(np.isfinite(wav).all())

    def test_determinism_same_seed_same_output(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            ckpt = _build_optispeech_checkpoint(tmp_path)
            eng = get_engine("optispeech")
            synth = eng.eval_synthesize(ckpt, config={})

            torch.manual_seed(99)
            wav_a = synth([1, 2, 3, 4, 5, 6], [1.0, 1.0, 1.0], None)
            torch.manual_seed(99)
            wav_b = synth([1, 2, 3, 4, 5, 6], [1.0, 1.0, 1.0], None)
            np.testing.assert_array_equal(wav_a, wav_b)


# ----------------------------------------------------------------------
# Default NotImplementedError path
# ----------------------------------------------------------------------

class _NoEvalSynthesisEngine(BaseTrainingEngine):
    """Minimal concrete engine that does not override eval_synthesize."""

    def create_model(self, config, dataset_paths, **kwargs):
        raise NotImplementedError

    def export_onnx(self, checkpoint_path, config_path, output_dir, **kwargs):
        raise NotImplementedError

    def quality_presets(self):
        return {}


class DefaultEvalSynthesizeTests(unittest.TestCase):
    def test_unsupported_engine_raises_clear_error(self):
        eng = _NoEvalSynthesisEngine()
        with self.assertRaises(NotImplementedError) as ctx:
            eng.eval_synthesize(Path("unused.ckpt"), config={})
        self.assertIn("eval_synthesize", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
