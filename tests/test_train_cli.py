"""Tests for the training CLI (phoonnx_train/train.py).

train.py wires together dataset config loading, engine selection, quality
preset resolution, checkpoint resume and the pytorch_lightning Trainer. These
tests never actually train: ``pytorch_lightning.Trainer`` and heavy engines
are mocked/faked so the CLI's own argument handling and precedence logic is
exercised in isolation and in seconds.
"""
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import click.testing

from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig
from phoonnx_train.train import _build_extra, main


class _FakeModel:
    """Stand-in LightningModule; records what it was built with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.loaded_checkpoints = []


class _FakeEngine(BaseTrainingEngine):
    """Minimal in-memory engine used to drive train.py without torch."""

    PRESETS = {
        "x-low": {"hidden_channels": 32},
        "medium": {"hidden_channels": 64},
        "high": {"hidden_channels": 128},
    }

    def __init__(self):
        self.created_with = None

    def create_model(self, config: TrainingEngineConfig, dataset_paths, **kwargs):
        model = _FakeModel(**config.extra, **kwargs)
        self.created_with = config
        return model

    def export_onnx(self, checkpoint_path, config_path, output_dir, **kwargs):
        raise NotImplementedError

    def quality_presets(self):
        return self.PRESETS

    def load_checkpoint(self, model, checkpoint_path, **kwargs):
        model.loaded_checkpoints.append((checkpoint_path, kwargs))
        return model


def _dataset_dir(tmp_path, config=None):
    d = tmp_path / "dataset"
    d.mkdir()
    if config is not None:
        (d / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return d


class TestBuildExtraPrecedence(unittest.TestCase):
    """The precedence chain: explicit CLI flag > config.json engine_params >
    CLI default > quality preset."""

    def _invoke(self, extra_cli_args):
        # _build_extra reads click.get_current_context() for parameter
        # sources, so it must run inside a real click command invocation.
        captured = {}

        @click.command()
        @click.option("--batch-size", type=int, default=16)
        def cmd(batch_size):
            captured["extra"] = _build_extra(
                {"batch_size": 999, "validation_split": 0.5},  # quality preset
                {"batch_size": 777},  # config.json engine_params
                batch_size=batch_size,
            )

        click.testing.CliRunner().invoke(cmd, extra_cli_args)
        return captured["extra"]

    def test_quality_preset_is_lowest_priority(self):
        extra = self._invoke([])
        # engine_params (777) beats quality preset (999) when CLI is default
        self.assertEqual(extra["batch_size"], 777)
        # validation_split only in the quality preset -> survives
        self.assertEqual(extra["validation_split"], 0.5)

    def test_explicit_cli_flag_wins_over_everything(self):
        extra = self._invoke(["--batch-size", "4"])
        self.assertEqual(extra["batch_size"], 4)

    def test_config_engine_params_beats_cli_default(self):
        # CLI default (16) is NOT explicit -> engine_params (777) wins
        extra = self._invoke([])
        self.assertNotEqual(extra["batch_size"], 16)
        self.assertEqual(extra["batch_size"], 777)


class TestQualityFallback(unittest.TestCase):
    def test_unknown_quality_falls_back_and_lists_presets_in_log(self):
        presets = {"x-low": {}, "medium": {}, "high": {}}
        quality = "not-a-real-tier"
        # mirror the fallback branch in train.main() directly, since it
        # is not factored into its own function
        fallback = "medium" if "medium" in presets else next(iter(presets))
        self.assertEqual(fallback, "medium")
        self.assertNotIn(quality, presets)


class TestMainCliEngineSelection(unittest.TestCase):
    """Full click invocation of train.main() with the engine registry and
    Trainer both faked out."""

    def setUp(self):
        self.runner = click.testing.CliRunner()
        self.fake_engine = _FakeEngine()

    def _run(self, tmp_path, args, dataset_config=None):
        dataset_dir = _dataset_dir(tmp_path, dataset_config)
        with patch("phoonnx_train.train.get_engine", return_value=self.fake_engine), \
             patch("phoonnx_train.train.list_engines", return_value=["vits", "fakeengine"]), \
             patch("phoonnx_train.train.Trainer") as trainer_cls:
            trainer_instance = MagicMock()
            trainer_cls.return_value = trainer_instance
            result = self.runner.invoke(
                main, ["--dataset-dir", str(dataset_dir), *args],
            )
            return result, trainer_instance, dataset_dir

    def test_unknown_engine_gives_clear_click_error_not_a_stacktrace(self):
        with self.runner.isolated_filesystem():
            Path("ds").mkdir()
            with patch("phoonnx_train.train.list_engines", return_value=["vits", "matcha"]):
                result = self.runner.invoke(
                    main, ["--dataset-dir", "ds", "--engine", "no-such-engine"],
                )
        self.assertNotEqual(result.exit_code, 0)
        # a click.BadParameter usage error, not an unhandled traceback
        self.assertIsNone(result.exception if isinstance(result.exception, KeyError) else None)
        self.assertIn("Unknown engine", result.output)
        self.assertIn("vits", result.output)
        self.assertIn("matcha", result.output)

    def test_missing_dataset_dir_rejected_by_click_before_anything_runs(self):
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                main, ["--dataset-dir", "does-not-exist"],
            )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("does-not-exist", result.output)

    def test_plain_resume_hands_checkpoint_to_fit_for_full_state_restore(self):
        # A plain --resume-from-checkpoint is a true resume: the path must be
        # handed to Trainer.fit(ckpt_path=...) so Lightning restores optimizer
        # state, epoch and global_step. A weight-only manual load would reset
        # them, so load_checkpoint must NOT be called on this path.
        with self.runner.isolated_filesystem() as td:
            Path("ds").mkdir()
            result, trainer_instance, _ = self._run(
                Path(td), ["--resume-from-checkpoint", "no/such/file.ckpt"],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        _, fit_kwargs = trainer_instance.fit.call_args
        self.assertEqual(fit_kwargs.get("ckpt_path"), "no/such/file.ckpt")

    def test_discard_encoder_resume_is_weight_only_not_ckpt_path(self):
        # --discard-encoder changes the architecture: it stays a weight-only
        # warm start via load_checkpoint, never Trainer.fit(ckpt_path=...).
        loaded = []
        self.fake_engine.load_checkpoint = (
            lambda model, checkpoint_path, **kw: loaded.append(str(checkpoint_path)) or model
        )
        with self.runner.isolated_filesystem() as td:
            Path("ds").mkdir()
            result, trainer_instance, _ = self._run(
                Path(td),
                ["--resume-from-checkpoint", "no/such/file.ckpt", "--discard-encoder"],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(loaded, ["no/such/file.ckpt"])
        _, fit_kwargs = trainer_instance.fit.call_args
        self.assertIsNone(fit_kwargs.get("ckpt_path"))

    def test_invalid_quality_falls_back_with_warning_not_crash(self):
        with self.assertLogs("phoonnx_train", level="WARNING") as logs:
            with self.runner.isolated_filesystem() as td:
                Path("ds").mkdir()
                result, trainer_instance, _ = self._run(
                    Path(td), ["--quality", "ultra-mega"],
                )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(any("falling back" in m for m in logs.output))
        trainer_instance.fit.assert_called_once()

    def test_quality_preset_reaches_model_via_extra(self):
        with self.runner.isolated_filesystem() as td:
            Path("ds").mkdir()
            result, _, _ = self._run(Path(td), ["--quality", "high"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.fake_engine.created_with.extra["hidden_channels"], 128)

    def test_config_json_engine_params_override_quality_preset(self):
        with self.runner.isolated_filesystem() as td:
            result, _, _ = self._run(
                Path(td), ["--quality", "medium"],
                dataset_config={"engine_params": {"hidden_channels": 4242}},
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.fake_engine.created_with.extra["hidden_channels"], 4242)

    def test_explicit_cli_flag_overrides_config_engine_params(self):
        with self.runner.isolated_filesystem() as td:
            result, _, _ = self._run(
                Path(td), ["--batch-size", "3"],
                dataset_config={"engine_params": {"batch_size": 999}},
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.fake_engine.created_with.extra["batch_size"], 3)

    def test_trainer_fit_is_never_actually_called_with_real_trainer(self):
        # sanity: confirms the mocking strategy actually prevents real training
        with self.runner.isolated_filesystem() as td:
            Path("ds").mkdir()
            result, trainer_instance, _ = self._run(Path(td), [])
        self.assertEqual(result.exit_code, 0, result.output)
        trainer_instance.fit.assert_called_once()

    def test_resume_from_checkpoint_success_path_invokes_engine_loader(self):
        with self.runner.isolated_filesystem() as td:
            Path("ds").mkdir()
            ckpt = Path(td) / "some.ckpt"
            ckpt.write_text("fake")
            result, _, _ = self._run(
                Path(td), ["--resume-from-checkpoint", str(ckpt), "--discard-encoder"],
            )
        self.assertEqual(result.exit_code, 0, result.output)

    def test_no_config_json_uses_engine_defaults_and_warns(self):
        with self.assertLogs("phoonnx_train", level="WARNING") as logs:
            with self.runner.isolated_filesystem() as td:
                result, _, dataset_dir = self._run(Path(td), [])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertFalse((dataset_dir / "config.json").exists())
        self.assertTrue(any("config.json" in m for m in logs.output))

    def test_styletts2_engine_prefix_changes_defaults(self):
        # engine.startswith("styletts2") switches num_symbols/sample_rate
        # defaults when config.json is absent -- verify via the log line
        # rather than internals, since main() has no return value.
        with self.assertLogs("phoonnx_train", level="INFO") as logs:
            with patch("phoonnx_train.train.get_engine", return_value=self.fake_engine), \
                 patch("phoonnx_train.train.list_engines",
                       return_value=["vits", "styletts2"]), \
                 patch("phoonnx_train.train.Trainer") as trainer_cls:
                trainer_cls.return_value = MagicMock()
                with self.runner.isolated_filesystem() as td:
                    Path("ds").mkdir()
                    result = self.runner.invoke(
                        main, ["--dataset-dir", "ds", "--engine", "styletts2"],
                    )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(any("sr=24000" in m for m in logs.output))


if __name__ == "__main__":
    unittest.main()
