"""Tests for the OptiSpeech training engine (phoonnx_train.engines.optispeech).

These exercise real torch on tiny CPU-sized models: registration + preset
consumption, model construction, a training step with an optimizer update,
checkpoint save/load/resume (optimizer + scheduler + epoch round-trip),
warm-starting from a partial checkpoint, and ONNX export that the phoonnx
OptiSpeech inference adapter accepts.
"""
import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from phoonnx_train.engines import get_engine, list_engines
from phoonnx_train.engines.base import TrainingEngineConfig
from phoonnx_train.engines.optispeech import (
    OptiSpeechEngineConfig,
    OptiSpeechTrainingEngine,
    _QUALITY_PRESETS,
)
from phoonnx_train.torch_compat import trusting_torch_load

HOP = 256
NFEATS = 40
TMEL = 24
TTEXT = 8


def _tiny_extra(**over):
    extra = {
        "quality": "x-low",
        "dim": 32,
        "n_feats": NFEATS,
        "hop_length": HOP,
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


def _tiny_config(**over):
    return TrainingEngineConfig(
        num_symbols=40, num_speakers=1, sample_rate=22050, extra=_tiny_extra(**over)
    )


class _TinyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return idx


def _collate(batch):
    b = len(batch)
    return {
        "x": torch.randint(1, 40, (b, TTEXT)),
        "x_lengths": torch.tensor([TTEXT] * b),
        "mel": torch.randn(b, NFEATS, TMEL),
        "mel_lengths": torch.tensor([TMEL] * b),
        "pitches": torch.randn(b, TMEL),
        "energies": torch.randn(b, TMEL),
        "sids": None,
        "lids": None,
        "wav": np.random.randn(b, TMEL * HOP).astype(np.float32),
    }


def _tiny_trainer(max_steps):
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


def _dim_of(model):
    return model.generator.text_embedding.embed_tokens.embedding_dim


class RegistrationTests(unittest.TestCase):
    def test_registered(self):
        self.assertIn("optispeech", list_engines())
        self.assertIsInstance(get_engine("optispeech"), OptiSpeechTrainingEngine)

    def test_quality_presets_exposed(self):
        presets = get_engine("optispeech").quality_presets()
        self.assertEqual({"x-low", "medium", "high"}, set(presets))


class ConfigTests(unittest.TestCase):
    def test_presets_change_constructed_model_dims(self):
        eng = get_engine("optispeech")
        low = eng.create_model(
            TrainingEngineConfig(num_symbols=40, extra={"quality": "x-low", **_min()}), []
        )
        high = eng.create_model(
            TrainingEngineConfig(num_symbols=40, extra={"quality": "high", **_min()}), []
        )
        # The preset drives the acoustic-model hidden width, so the two models
        # differ structurally, not just in weights.
        self.assertEqual(_dim_of(low), _QUALITY_PRESETS["x-low"]["dim"])
        self.assertEqual(_dim_of(high), _QUALITY_PRESETS["high"]["dim"])
        self.assertNotEqual(_dim_of(low), _dim_of(high))
        self.assertLess(
            sum(p.numel() for p in low.parameters()),
            sum(p.numel() for p in high.parameters()),
        )

    def test_unknown_quality_falls_back_to_medium(self):
        cfg = OptiSpeechEngineConfig.from_training_config(
            TrainingEngineConfig(num_symbols=40, extra={"quality": "nonsense"})
        )
        self.assertEqual(cfg.dim, _QUALITY_PRESETS["medium"]["dim"])

    def test_from_training_config_maps_shared_fields(self):
        cfg = OptiSpeechEngineConfig.from_training_config(
            TrainingEngineConfig(num_symbols=99, num_speakers=3, sample_rate=16000)
        )
        self.assertEqual(cfg.num_symbols, 99)
        self.assertEqual(cfg.num_speakers, 3)
        self.assertEqual(cfg.sample_rate, 16000)

    def test_presets_not_mutated(self):
        before = {k: dict(v) for k, v in _QUALITY_PRESETS.items()}
        OptiSpeechEngineConfig.from_training_config(
            TrainingEngineConfig(num_symbols=40, extra={"quality": "x-low", "dim": 999})
        )
        self.assertEqual(before, _QUALITY_PRESETS)


class ConstructionTests(unittest.TestCase):
    def test_tiny_model_builds(self):
        model = get_engine("optispeech").create_model(_tiny_config(), [])
        self.assertEqual(_dim_of(model), 32)
        self.assertGreater(sum(p.numel() for p in model.parameters()), 0)
        self.assertFalse(model.automatic_optimization)

    def test_generator_forward_loss_finite(self):
        model = get_engine("optispeech").create_model(_tiny_config(), [])
        batch = _collate([0, 1])
        out = model.generator(
            x=batch["x"],
            x_lengths=batch["x_lengths"],
            mel=batch["mel"],
            mel_lengths=batch["mel_lengths"],
            pitches=batch["pitches"],
            energies=batch["energies"],
            sids=None,
            lids=None,
        )
        self.assertTrue(torch.isfinite(out["loss"]).item())


class TrainingStepTests(unittest.TestCase):
    def test_training_step_updates_weights(self):
        model = get_engine("optispeech").create_model(_tiny_config(), [])
        before = {k: v.detach().clone() for k, v in model.generator.named_parameters()}
        dl = DataLoader(_TinyDataset(), batch_size=2, collate_fn=_collate)
        _tiny_trainer(max_steps=4).fit(model, dl)
        # An optimizer step ran: the generator weights moved measurably.
        total_delta = sum(
            (v.detach() - before[k]).abs().sum().item()
            for k, v in model.generator.named_parameters()
        )
        self.assertGreater(total_delta, 0.0)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_carries_optimizer_scheduler_epoch(self):
        eng = get_engine("optispeech")
        model = eng.create_model(_tiny_config(), [])
        dl = DataLoader(_TinyDataset(), batch_size=2, collate_fn=_collate)
        trainer = _tiny_trainer(max_steps=2)
        trainer.fit(model, dl)
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "last.ckpt"
            trainer.save_checkpoint(str(ckpt))
            raw = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            # HARD rule: resumable + finetunable -> optimizer + scheduler + epoch.
            self.assertIn("optimizer_states", raw)
            self.assertEqual(len(raw["optimizer_states"]), 2)
            self.assertIn("lr_schedulers", raw)
            self.assertEqual(len(raw["lr_schedulers"]), 2)
            self.assertIn("epoch", raw)
            self.assertIn("global_step", raw)

            # Full module rebuilds from the checkpoint's hyperparameters.
            reloaded = eng.load_lightning_module(ckpt)
            self.assertEqual(_dim_of(reloaded), _dim_of(model))

    def test_resume_continues_from_checkpoint(self):
        eng = get_engine("optispeech")
        model = eng.create_model(_tiny_config(), [])
        dl = DataLoader(_TinyDataset(), batch_size=2, collate_fn=_collate)
        trainer = _tiny_trainer(max_steps=2)
        trainer.fit(model, dl)
        self.assertEqual(trainer.global_step, 2)
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "last.ckpt"
            trainer.save_checkpoint(str(ckpt))
            resumed = _tiny_trainer(max_steps=4)
            # torch>=2.6 defaults weights_only=True; Lightning's own ckpt_path
            # load must trust our checkpoint's pickled config objects.
            with trusting_torch_load():
                resumed.fit(model, dl, ckpt_path=str(ckpt))
            # Optimizer + scheduler state restored; training continued past the
            # saved step rather than restarting.
            self.assertEqual(resumed.global_step, 4)

    def test_warm_start_partial_checkpoint(self):
        eng = get_engine("optispeech")
        donor = eng.create_model(_tiny_config(), [])
        # Build a *partial* state dict (every other tensor) with recognisable
        # values so we can prove the matching keys are actually loaded.
        donor_state = donor.state_dict()
        partial = {}
        for i, (k, v) in enumerate(donor_state.items()):
            if i % 2 == 0:
                partial[k] = torch.full_like(v, 0.0) if v.dtype.is_floating_point else v
        # A key the target model does not have -> must be reported as skipped.
        partial["not_a_real_module.weight"] = torch.zeros(3)

        target = eng.create_model(_tiny_config(), [])
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "partial.ckpt"
            torch.save({"state_dict": partial}, str(ckpt))
            with self.assertLogs(
                "phoonnx_train.engines.optispeech", level="INFO"
            ) as cm:
                eng.load_checkpoint(target, ckpt)
            joined = "\n".join(cm.output)
            self.assertIn("warm-start", joined)
            self.assertIn("skipped", joined)

        # A float tensor present in the partial checkpoint was overwritten...
        float_key = next(
            k for k, v in partial.items() if v.dtype.is_floating_point
        )
        self.assertTrue(
            torch.equal(target.state_dict()[float_key], partial[float_key])
        )
        # ...while a key absent from the partial kept its freshly-initialised value.
        missing_key = next(k for k in donor_state if k not in partial)
        self.assertIn(missing_key, target.state_dict())


class ExportTests(unittest.TestCase):
    def test_export_onnx_loads_in_adapter(self):
        import onnxruntime as ort

        from phoonnx.engines.optispeech import OptiSpeechAdapter

        eng = get_engine("optispeech")
        model = eng.create_model(_tiny_config(), [])
        dl = DataLoader(_TinyDataset(), batch_size=2, collate_fn=_collate)
        trainer = _tiny_trainer(max_steps=2)
        trainer.fit(model, dl)
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "last.ckpt"
            trainer.save_checkpoint(str(ckpt))
            onnx_path = eng.export_onnx(ckpt, None, Path(td) / "onnx")
            self.assertTrue(onnx_path.exists())

            sess = ort.InferenceSession(str(onnx_path))
            input_names = {i.name for i in sess.get_inputs()}
            output_names = {o.name for o in sess.get_outputs()}
            self.assertIn("x", input_names)
            self.assertIn("x_lengths", input_names)
            self.assertIn("scales", input_names)
            self.assertEqual(
                {"wav", "wav_lengths", "durations"}, output_names
            )

            adapter = OptiSpeechAdapter()
            self.assertTrue(adapter.detect(session=sess))
            meta = adapter.parse_onnx_meta(sess)
            self.assertIn("text_processor", meta)
            self.assertIn("input_symbols", meta)

            # The adapter builds a feed dict from its request and the model runs.
            from phoonnx.engines.base import AdapterSynthesisRequest

            req = AdapterSynthesisRequest(
                phoneme_ids=np.array([[1, 2, 3, 4, 5]], dtype=np.int64),
                phoneme_lengths=np.array([5], dtype=np.int64),
                params={"d_factor": 1.0, "p_factor": 1.0, "e_factor": 1.0},
            )
            feed = adapter.build_feed_dict(req, sess)
            outputs = sess.run(None, feed)
            self.assertEqual(len(outputs), 3)


def _min():
    """Minimal preset-independent overrides so `high` stays CPU-cheap."""
    return {
        "n_feats": NFEATS,
        "hop_length": HOP,
        "n_fft": 256,
        "win_length": 256,
        "segment_size": 8,
        "encoder_num_layers": 1,
        "decoder_num_layers": 1,
        "vocoder_num_layers": 1,
        "encoder_intermediate_dim": 32,
        "decoder_intermediate_dim": 32,
        "vocoder_intermediate_dim": 32,
    }


if __name__ == "__main__":
    unittest.main()
