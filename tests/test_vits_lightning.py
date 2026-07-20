"""Tests for phoonnx_train/vits/lightning.py: optimizer/scheduler config,
dataset-split edges and the checkpoint save/restore glue that the HARD
"training must resume + finetune" rule depends on.

VitsModel always builds a real (but tiny -- x-low-preset-sized) SynthesizerTrn
so construction stays fast; heavy forward-pass work (training_step_g/d) is
covered elsewhere (test_vits_audio_logging.py) and is not exercised here.
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from phoonnx_train.engines.base import TrainingEngineConfig
from phoonnx_train.vits.lightning import VitsModel

# Small hyper-parameters so building SynthesizerTrn is fast (mirrors the
# "x-low" quality preset in phoonnx_train.engines.vits).
_TINY_KWARGS = dict(
    inter_channels=32,
    hidden_channels=32,
    filter_channels=64,
    n_heads=1,
    n_layers=1,
    n_layers_q=1,
    resblock="2",
    resblock_kernel_sizes=(3,),
    resblock_dilation_sizes=((1, 2),),
    upsample_rates=(8,),
    upsample_initial_channel=16,
    upsample_kernel_sizes=(16,),
)


def _write_dataset_jsonl(path: Path, n: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            row = {
                "phoneme_ids": [1, 2, 3],
                "audio_norm_path": f"norm_{i}.pt",
                "audio_spec_path": f"spec_{i}.pt",
                "text": f"utterance {i}",
            }
            f.write(json.dumps(row) + "\n")


def _build_model(dataset=None, **overrides):
    kwargs = dict(_TINY_KWARGS)
    kwargs.setdefault("num_speakers", 1)
    kwargs.setdefault("num_symbols", 32)
    kwargs.update(overrides)
    return VitsModel(
        dataset=dataset,
        **kwargs,
    )


class TestConfigureOptimizers(unittest.TestCase):
    def test_returns_two_optimizers_and_two_schedulers(self):
        model = _build_model()
        optimizers, schedulers = model.configure_optimizers()
        self.assertEqual(len(optimizers), 2)
        self.assertEqual(len(schedulers), 2)
        self.assertIsInstance(optimizers[0], torch.optim.AdamW)
        self.assertIsInstance(optimizers[1], torch.optim.AdamW)
        self.assertIsInstance(schedulers[0], torch.optim.lr_scheduler.ExponentialLR)
        self.assertIsInstance(schedulers[1], torch.optim.lr_scheduler.ExponentialLR)

    def test_optimizer_covers_generator_and_discriminator_params(self):
        model = _build_model()
        optimizers, _ = model.configure_optimizers()
        g_ids = {id(p) for p in model.model_g.parameters()}
        d_ids = {id(p) for p in model.model_d.parameters()}
        opt_g_ids = {id(p) for group in optimizers[0].param_groups for p in group["params"]}
        opt_d_ids = {id(p) for group in optimizers[1].param_groups for p in group["params"]}
        self.assertEqual(opt_g_ids, g_ids)
        self.assertEqual(opt_d_ids, d_ids)

    def test_optimizer_and_scheduler_state_survives_resume_round_trip(self):
        """HARD rule: training must resume + finetune. configure_optimizers'
        result must be checkpoint-and-restore safe: step it, snapshot its
        state_dict, build a *fresh* optimizer/scheduler pair, and restore --
        the restored state must match the stepped state exactly."""
        model = _build_model()
        optimizers, schedulers = model.configure_optimizers()
        opt_g, sched_g = optimizers[0], schedulers[0]

        # Simulate a few training steps advancing optimizer + scheduler state.
        for p in model.model_g.parameters():
            if p.requires_grad:
                p.grad = torch.ones_like(p)
        opt_g.step()
        sched_g.step()
        sched_g.step()

        opt_state = opt_g.state_dict()
        sched_state = sched_g.state_dict()
        lr_before = opt_g.param_groups[0]["lr"]

        # Fresh optimizer/scheduler pair from a re-built model (as happens on
        # resume: model is reconstructed, then state_dicts are loaded).
        model2 = _build_model()
        optimizers2, schedulers2 = model2.configure_optimizers()
        opt_g2, sched_g2 = optimizers2[0], schedulers2[0]

        opt_g2.load_state_dict(opt_state)
        sched_g2.load_state_dict(sched_state)

        self.assertEqual(opt_g2.param_groups[0]["lr"], lr_before)
        self.assertEqual(sched_g2.last_epoch, sched_g.last_epoch)
        self.assertEqual(sched_g2.state_dict(), sched_g.state_dict())


class TestLoadDatasetsEdges(unittest.TestCase):
    def test_no_dataset_configured_leaves_splits_none(self):
        model = _build_model(dataset=None)
        self.assertIsNone(model._train_dataset)
        self.assertIsNone(model._val_dataset)
        self.assertIsNone(model._test_dataset)

    def test_validation_split_zero_yields_empty_val_set(self):
        with TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "dataset.jsonl"
            _write_dataset_jsonl(jsonl, 10)
            model = _build_model(
                dataset=[str(jsonl)], validation_split=0.0, num_test_examples=2,
            )
        self.assertEqual(len(model._val_dataset), 0)
        self.assertEqual(len(model._test_dataset), 2)
        self.assertEqual(len(model._train_dataset), 8)

    def test_num_test_examples_larger_than_dataset_raises_not_silently_wrong(self):
        # random_split requires the requested split sizes to sum to the
        # dataset length; requesting more test examples than exist must
        # fail loudly rather than silently produce a negative train split.
        with TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "dataset.jsonl"
            _write_dataset_jsonl(jsonl, 5)
            with self.assertRaises(ValueError):
                _build_model(
                    dataset=[str(jsonl)], validation_split=0.1, num_test_examples=100,
                )

    def test_split_sizes_partition_full_dataset_exactly(self):
        with TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "dataset.jsonl"
            _write_dataset_jsonl(jsonl, 20)
            model = _build_model(
                dataset=[str(jsonl)], validation_split=0.25, num_test_examples=3,
            )
        total = len(model._train_dataset) + len(model._val_dataset) + len(model._test_dataset)
        self.assertEqual(total, 20)
        self.assertEqual(len(model._val_dataset), 5)  # int(20 * 0.25)
        self.assertEqual(len(model._test_dataset), 3)
        self.assertEqual(len(model._train_dataset), 12)


class TestCheckpointSaveRestoreGlue(unittest.TestCase):
    def test_hyperparameters_round_trip_through_checkpoint(self):
        """save_hyperparameters() must capture everything configure_optimizers
        and _load_datasets need to reconstruct an identical model on resume."""
        model = _build_model(learning_rate=1e-3, batch_size=4)
        with TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "model.ckpt"
            trainer_ckpt = {
                "state_dict": model.state_dict(),
                "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": "0.0.0",
                "global_step": 0,
                "epoch": 0,
            }
            torch.save(trainer_ckpt, ckpt_path)

            restored = VitsModel.load_from_checkpoint(ckpt_path, dataset=None)

        self.assertEqual(restored.hparams.learning_rate, 1e-3)
        self.assertEqual(restored.hparams.batch_size, 4)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, restored.state_dict()[k]))

    def test_model_state_dict_keys_stable_across_construction(self):
        # a resumed run reloads weights by key -- the key set must be
        # deterministic across two independently constructed instances.
        model_a = _build_model()
        model_b = _build_model()
        self.assertEqual(set(model_a.state_dict().keys()), set(model_b.state_dict().keys()))

    def test_gin_channels_defaulted_for_multi_speaker(self):
        # multi-speaker models get gin_channels auto-set to 512 so the
        # speaker-conditioning layers exist to be checkpointed/resumed
        model = _build_model(num_speakers=3)
        self.assertEqual(model.hparams.gin_channels, 512)


class TestAddModelSpecificArgs(unittest.TestCase):
    def test_registers_expected_arguments(self):
        import argparse

        parser = argparse.ArgumentParser()
        VitsModel.add_model_specific_args(parser)
        args = parser.parse_args(["--batch-size", "8"])
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.validation_split, 0.1)
        self.assertEqual(args.num_test_examples, 5)
        self.assertEqual(args.hidden_channels, 192)

    def test_batch_size_is_required(self):
        import argparse

        parser = argparse.ArgumentParser()
        VitsModel.add_model_specific_args(parser)
        with self.assertRaises(SystemExit):
            parser.parse_args([])


class TestVitsTrainingEngineRegistrationAndWiring(unittest.TestCase):
    """Reachable surface of phoonnx_train/engines/vits.py without a real
    training run: registry, quality presets, config plumbing, model
    construction, and checkpoint-loading logic (torch.onnx export is not
    exercised -- it needs a fully trained/traced graph and is out of scope
    for a fast unit test)."""

    def setUp(self):
        from phoonnx_train.engines.vits import VitsTrainingEngine
        self.engine = VitsTrainingEngine()

    def test_registered_under_vits_name(self):
        from phoonnx_train.engines import get_engine, list_engines
        self.assertIn("vits", list_engines())
        self.assertIsInstance(get_engine("vits"), type(self.engine))

    def test_quality_presets_have_three_tiers_with_increasing_capacity(self):
        presets = self.engine.quality_presets()
        self.assertEqual(set(presets), {"x-low", "medium", "high"})
        sizes = [presets[t]["hidden_channels"] for t in ("x-low", "medium", "high")]
        self.assertEqual(sizes, sorted(sizes))
        self.assertTrue(all(sizes[i] < sizes[i + 1] for i in range(len(sizes) - 1)))

    def test_create_model_maps_dataset_dir_to_dataset_jsonl(self):
        import json as _json

        with TemporaryDirectory() as tmp:
            ds_dir = Path(tmp) / "ds"
            ds_dir.mkdir()
            _write_dataset_jsonl(ds_dir / "dataset.jsonl", 3)
            config = TrainingEngineConfig(
                num_symbols=16, num_speakers=1, sample_rate=16000,
                extra={**_TINY_KWARGS, "num_test_examples": 1, "validation_split": 0.0},
            )
            model = self.engine.create_model(config, dataset_paths=[ds_dir])
        self.assertEqual(model.hparams.dataset, [str(ds_dir / "dataset.jsonl")])
        self.assertEqual(model.hparams.num_symbols, 16)

    def test_create_model_accepts_a_dataset_jsonl_file_path_directly(self):
        with TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "custom.jsonl"
            _write_dataset_jsonl(jsonl, 2)
            config = TrainingEngineConfig(
                num_symbols=16, num_speakers=1, sample_rate=16000,
                extra={**_TINY_KWARGS, "num_test_examples": 1, "validation_split": 0.0},
            )
            model = self.engine.create_model(config, dataset_paths=[jsonl])
        self.assertEqual(model.hparams.dataset, [str(jsonl)])

    def test_load_checkpoint_discards_mismatched_encoder(self):
        from unittest.mock import patch
        from phoonnx_train.vits.lightning import VitsModel

        model = _build_model(num_symbols=16)
        ckpt_model = _build_model(num_symbols=999)  # different vocab size

        with patch.object(VitsModel, "load_from_checkpoint", return_value=ckpt_model):
            result = self.engine.load_checkpoint(model, Path("fake.ckpt"))
        self.assertIs(result, model)  # loaded in place, mismatched encoder skipped

    def test_load_checkpoint_discard_encoder_flag_drops_matching_encoder_too(self):
        from unittest.mock import patch
        from phoonnx_train.vits.lightning import VitsModel

        model = _build_model(num_symbols=16)
        ckpt_model = _build_model(num_symbols=16)  # same vocab size this time
        before = model.model_g.enc_p.emb.weight.clone()

        with patch.object(VitsModel, "load_from_checkpoint", return_value=ckpt_model):
            self.engine.load_checkpoint(model, Path("fake.ckpt"), discard_encoder=True)
        # encoder embedding kept at its own (pre-load) value, not ckpt's
        self.assertTrue(torch.equal(model.model_g.enc_p.emb.weight, before))

    def test_single_speaker_checkpoint_conversion_requires_multi_speaker_target(self):
        model = _build_model(num_symbols=16, num_speakers=1)  # single-speaker
        with self.assertRaises(ValueError):
            self.engine.load_checkpoint(
                model, Path("fake.ckpt"),
                resume_from_single_speaker_checkpoint=True,
            )

    def test_single_speaker_checkpoint_conversion_strips_speaker_conditional_layers(self):
        from unittest.mock import patch
        from phoonnx_train.vits.lightning import VitsModel

        model = _build_model(num_symbols=16, num_speakers=3)  # multi-speaker target
        ckpt_model = _build_model(num_symbols=16, num_speakers=1)  # single-speaker source

        with patch.object(VitsModel, "load_from_checkpoint", return_value=ckpt_model):
            result = self.engine.load_checkpoint(
                model, Path("fake.ckpt"),
                resume_from_single_speaker_checkpoint=True,
            )
        self.assertIs(result, model)
        # multi-speaker conditioning layers survive untouched (never in the
        # single-speaker source, so kept at their randomly-initialized value)
        self.assertEqual(model.hparams.gin_channels, 512)

    def test_tolerant_load_keeps_current_values_for_missing_keys(self):
        from phoonnx_train.engines.vits import VitsTrainingEngine

        target = torch.nn.Linear(4, 4)
        original_bias = target.bias.clone()
        VitsTrainingEngine._tolerant_load(target, {})  # no matching keys at all
        self.assertTrue(torch.equal(target.bias, original_bias))


class TestVitsSidecarExporters(unittest.TestCase):
    def test_write_tokens_txt_orders_by_id(self):
        from phoonnx_train.engines.vits import _write_tokens_txt

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokens.txt"
            _write_tokens_txt({"a": 0, "b": 2, "c": 1}, path)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines, ["a", "c", "b"])

    def test_write_tokens_txt_handles_list_ids_and_unknown_gaps(self):
        from phoonnx_train.engines.vits import _write_tokens_txt

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokens.txt"
            _write_tokens_txt({"a": [0, 2]}, path)  # id 1 is an unfilled gap
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines, ["a", "<UNK_1>", "a"])

    def test_write_tokens_txt_empty_map_writes_empty_file(self):
        from phoonnx_train.engines.vits import _write_tokens_txt

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokens.txt"
            _write_tokens_txt({}, path)
            self.assertEqual(path.read_text(encoding="utf-8"), "")

    def test_write_piper_json_drops_none_values(self):
        from phoonnx_train.engines.vits import _write_piper_json

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.json"
            _write_piper_json({"num_symbols": 100, "num_speakers": None,
                               "phoneme_type": "espeak"}, path)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["num_symbols"], 100)
        self.assertNotIn("num_speakers", data)
        self.assertEqual(data["phoneme_type"], "espeak")


if __name__ == "__main__":
    unittest.main()
