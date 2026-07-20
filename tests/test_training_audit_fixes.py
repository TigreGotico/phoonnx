"""Regression tests for the training-stack audit fixes.

Each test class targets one audit finding:

  * ``TestCompileResumeMatrix`` — VitsModel checkpoint key reconciliation across
    the full compiled/uncompiled save x compiled/uncompiled load matrix
    (on_save_checkpoint always cleans; on_load_checkpoint adapts to the CURRENT
    model, including the reverse clean-checkpoint -> compiled-model case).
  * ``TestGuardedCompile`` — a failing torch.compile warns and continues
    uncompiled instead of aborting the run.
  * ``TestEvalWithoutSynthesizeGuard`` — enabling in-training eval on an engine
    with no eval_synthesize fails loudly at startup.
  * ``TestOptiSpeechEvalTokenization`` — the OptiSpeech eval hook tokenizes
    differently from the generic phoonnx pipeline (proving the bug), and the
    scorer prefers the hook.
  * ``TestSizeStableFast`` — a fully-written file is judged stable quickly.
  * ``TestScorePendingCheckpoints`` — every unscored on-disk checkpoint is
    scored on a gated firing, not only the newest.
  * ``TestMelBasisCacheKeys`` — mel_basis caches key on the full parameter set.
"""
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from click.testing import CliRunner

from phoonnx_train import train as train_module
from phoonnx_train.vits.lightning import VitsModel


# ---------------------------------------------------------------------------
# #1 compile x resume matrix
# ---------------------------------------------------------------------------
class _CompiledWrapper(torch.nn.Module):
    """Stand-in for torch._dynamo OptimizedModule: exposes ``_orig_mod`` and
    prefixes its state-dict keys with ``_orig_mod.`` exactly like the real one.
    Used because torch.compile cannot dynamo-compile on this py/torch combo."""

    def __init__(self, orig: torch.nn.Module):
        super().__init__()
        self._orig_mod = orig


def _bare_model():
    """A VitsModel instance without running __init__ (no dataset/torch build);
    we only exercise the checkpoint hooks, which read model_g/model_d."""
    m = VitsModel.__new__(VitsModel)
    torch.nn.Module.__init__(m)
    m.model_g = torch.nn.Linear(2, 2)
    m.model_d = torch.nn.Linear(2, 2)
    return m


def _compile(model):
    model.model_g = _CompiledWrapper(model.model_g)
    model.model_d = _CompiledWrapper(model.model_d)
    return model


class TestCompileResumeMatrix(unittest.TestCase):
    CLEAN = {
        "model_g.enc.weight": torch.tensor([1.0]),
        "model_d.disc.bias": torch.tensor([2.0]),
        "unrelated": torch.tensor([3.0]),
    }
    COMPILED = {
        "model_g._orig_mod.enc.weight": torch.tensor([1.0]),
        "model_d._orig_mod.disc.bias": torch.tensor([2.0]),
        "unrelated": torch.tensor([3.0]),
    }

    def _save(self, model, state):
        ckpt = {"state_dict": dict(state)}
        model.on_save_checkpoint(ckpt)
        return set(ckpt["state_dict"])

    def _load(self, model, state):
        ckpt = {"state_dict": dict(state)}
        model.on_load_checkpoint(ckpt)
        return set(ckpt["state_dict"])

    # on_save always produces clean keys, regardless of compile state
    def test_save_uncompiled_is_clean(self):
        self.assertEqual(self._save(_bare_model(), self.CLEAN), set(self.CLEAN))

    def test_save_compiled_strips_to_clean(self):
        self.assertEqual(
            self._save(_compile(_bare_model()), self.COMPILED), set(self.CLEAN)
        )

    # on_load adapts to the CURRENT model — the four cells
    def test_load_clean_into_uncompiled(self):
        self.assertEqual(self._load(_bare_model(), self.CLEAN), set(self.CLEAN))

    def test_load_compiled_into_uncompiled(self):
        # compiled checkpoint, uncompiled model -> strip (old behaviour)
        self.assertEqual(self._load(_bare_model(), self.COMPILED), set(self.CLEAN))

    def test_load_compiled_into_compiled(self):
        self.assertEqual(
            self._load(_compile(_bare_model()), self.COMPILED), set(self.COMPILED)
        )

    def test_load_clean_into_compiled(self):
        # THE reverse-direction bug: clean checkpoint must gain the _orig_mod
        # prefix so a --compile resume's strict restore does not crash.
        self.assertEqual(
            self._load(_compile(_bare_model()), self.CLEAN), set(self.COMPILED)
        )

    def test_loaded_state_actually_restores_into_compiled_model(self):
        # End-to-end: a clean checkpoint loads via load_state_dict into a
        # compiled model without missing/unexpected keys.
        model = _compile(_bare_model())
        real = {
            f"model_g._orig_mod.{k}": v
            for k, v in torch.nn.Linear(2, 2).state_dict().items()
        }
        real.update(
            {
                f"model_d._orig_mod.{k}": v
                for k, v in torch.nn.Linear(2, 2).state_dict().items()
            }
        )
        clean = {k.replace("._orig_mod.", "."): v for k, v in real.items()}
        ckpt = {"state_dict": clean}
        model.on_load_checkpoint(ckpt)
        combined = torch.nn.Module()
        combined.model_g = model.model_g
        combined.model_d = model.model_d
        missing, unexpected = combined.load_state_dict(
            ckpt["state_dict"], strict=False
        )
        self.assertEqual(list(missing), [])
        self.assertEqual(list(unexpected), [])

    def test_missing_state_dict_is_noop(self):
        ckpt = {}
        _bare_model().on_load_checkpoint(ckpt)
        _bare_model().on_save_checkpoint(ckpt)
        self.assertEqual(ckpt, {})


# ---------------------------------------------------------------------------
# #2 guarded torch.compile
# ---------------------------------------------------------------------------
class _FakeEngine:
    def __init__(self, with_model_g_d=True):
        self.with_model_g_d = with_model_g_d

    def quality_presets(self):
        return {"medium": {}}

    def trainer_kwargs(self):
        return {}

    def create_model(self, config, dataset_paths, **kwargs):
        m = SimpleNamespace()
        if self.with_model_g_d:
            m.model_g = torch.nn.Linear(2, 2)
            m.model_d = torch.nn.Linear(2, 2)
        return m

    def load_checkpoint(self, model, path, **kwargs):
        return model


class _FakeTrainer:
    instances = []

    def __init__(self, *a, **k):
        self.fit_calls = []
        _FakeTrainer.instances.append(self)

    def fit(self, model, ckpt_path=None):
        self.fit_calls.append((model, ckpt_path))


def _invoke_main(tmp_dir, extra_args, engine):
    _FakeTrainer.instances = []
    with mock.patch.object(train_module, "get_engine", return_value=engine), \
         mock.patch.object(train_module, "Trainer", _FakeTrainer):
        return CliRunner().invoke(
            train_module.main,
            ["--dataset-dir", tmp_dir, "--engine", "vits",
             "--max-epochs", "1", *extra_args],
        )


class TestGuardedCompile(unittest.TestCase):
    def test_compile_failure_warns_and_continues(self):
        def boom(m, mode):
            raise RuntimeError("dynamo has no CPython 3.12 backend")

        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.object(train_module.torch, "compile", side_effect=boom), \
                 self.assertLogs(train_module._LOGGER, level="WARNING") as logs:
                result = _invoke_main(tmp_dir, ["--compile"], _FakeEngine())
            self.assertEqual(result.exit_code, 0, result.output)
            # Training still ran uncompiled instead of aborting.
            self.assertEqual(len(_FakeTrainer.instances), 1)
            self.assertEqual(len(_FakeTrainer.instances[0].fit_calls), 1)
            self.assertTrue(
                any("torch.compile unavailable" in m for m in logs.output)
            )


# ---------------------------------------------------------------------------
# #3 eval-every on an engine without eval_synthesize
# ---------------------------------------------------------------------------
class TestEvalWithoutSynthesizeGuard(unittest.TestCase):
    def test_usage_error_when_engine_lacks_eval_synthesize(self):
        from phoonnx_train.engines.base import BaseTrainingEngine

        # _FakeEngine does not implement eval_synthesize; patch the base check
        # to see it as unsupported (it is not a BaseTrainingEngine subclass).
        engine = _FakeEngine()
        with tempfile.TemporaryDirectory() as tmp_dir:
            sents = Path(tmp_dir) / "s.txt"
            sents.write_text("hello\n", encoding="utf-8")
            # Make the engine look like it inherits the base (unimplemented)
            # eval_synthesize by giving type(engine) the base method.
            with mock.patch.object(
                type(engine), "eval_synthesize",
                BaseTrainingEngine.eval_synthesize, create=True
            ):
                result = _invoke_main(
                    tmp_dir,
                    ["--eval-sentences", str(sents), "--eval-every", "1"],
                    engine,
                )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("does not support in-training evaluation", result.output)


# ---------------------------------------------------------------------------
# #4 OptiSpeech eval tokenization
# ---------------------------------------------------------------------------
class TestOptiSpeechEvalTokenization(unittest.TestCase):
    CONFIG = {
        "num_symbols": 178,
        "num_speakers": 1,
        "audio": {"sample_rate": 22050},
        "lang_code": "en-us",
        "phoneme_type": "espeak",
        "alphabet": "ipa",
        "phoneme_id_map": {"a": 1, "b": 2},
        "engine_params": {"languages": ["en-us"]},
    }

    def test_engine_hook_differs_from_phoonnx_pipeline(self):
        try:
            import piper_phonemize  # noqa: F401
        except ImportError:
            self.skipTest("piper-phonemize required for the OptiSpeech IPA tokenizer")
        from phoonnx_train.engines.optispeech import OptiSpeechTrainingEngine

        engine = OptiSpeechTrainingEngine()
        hook_ids = engine.eval_text_to_ids("hello world", self.CONFIG)
        self.assertIsInstance(hook_ids, list)
        self.assertTrue(all(isinstance(i, int) for i in hook_ids))

        # The generic phoonnx pipeline produces a DIFFERENT id sequence for the
        # same sentence (different phoneme set / blank / BOS-EOS handling) —
        # feeding those to an OptiSpeech-trained model scores garbage.
        from phoonnx_train.evaluation import scorer as scorer_mod
        try:
            ph, tok, lang, _sr, _sc = scorer_mod.build_encoder(self.CONFIG)
            _, generic_ids = scorer_mod.text_to_ids("hello world", ph, tok, lang)
        except Exception:
            self.skipTest("generic phoonnx pipeline unavailable in this env")
        self.assertNotEqual(hook_ids, generic_ids)

    def test_scorer_prefers_engine_hook(self):
        from phoonnx_train.evaluation import scorer as scorer_mod
        from phoonnx_train.engines.base import BaseTrainingEngine

        sentinel = [7, 7, 7]

        class HookEngine(BaseTrainingEngine):
            def create_model(self, *a, **k):
                raise NotImplementedError

            def export_onnx(self, *a, **k):
                raise NotImplementedError

            def quality_presets(self):
                return {}

            def eval_text_to_ids(self, text, config):
                return list(sentinel)

        s = scorer_mod.CheckpointScorer.__new__(scorer_mod.CheckpointScorer)
        s.engine = HookEngine()
        s.config = self.CONFIG
        base = BaseTrainingEngine.eval_text_to_ids
        s._uses_engine_tokenizer = (
            getattr(type(s.engine), "eval_text_to_ids", base) is not base
        )
        self.assertTrue(s._uses_engine_tokenizer)
        self.assertEqual(s._text_to_ids("anything"), sentinel)

    def test_scorer_falls_back_without_hook(self):
        from phoonnx_train.evaluation import scorer as scorer_mod
        from phoonnx_train.engines.base import BaseTrainingEngine

        class PlainEngine(BaseTrainingEngine):
            def create_model(self, *a, **k):
                raise NotImplementedError

            def export_onnx(self, *a, **k):
                raise NotImplementedError

            def quality_presets(self):
                return {}

        s = scorer_mod.CheckpointScorer.__new__(scorer_mod.CheckpointScorer)
        s.engine = PlainEngine()
        base = BaseTrainingEngine.eval_text_to_ids
        s._uses_engine_tokenizer = (
            getattr(type(s.engine), "eval_text_to_ids", base) is not base
        )
        s.ph = s.tokenizer = None
        s.lang = ""
        self.assertFalse(s._uses_engine_tokenizer)
        with mock.patch.object(scorer_mod, "text_to_ids",
                               return_value=(["p"], [9, 9])):
            self.assertEqual(s._text_to_ids("x"), [9, 9])


# ---------------------------------------------------------------------------
# #5 size_stable fast path
# ---------------------------------------------------------------------------
class TestSizeStableFast(unittest.TestCase):
    def test_static_file_stable_under_two_seconds(self):
        from phoonnx_train import eval_utils

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ckpt"
            p.write_bytes(b"complete-checkpoint-bytes")
            start = time.monotonic()
            # Real sleep, default wait — must not stall multiple seconds.
            stable = eval_utils.size_stable(p)
            elapsed = time.monotonic() - start
        self.assertTrue(stable)
        self.assertLess(elapsed, 2.0)


# ---------------------------------------------------------------------------
# #6 score every pending checkpoint on a gated firing
# ---------------------------------------------------------------------------
class TestScorePendingCheckpoints(unittest.TestCase):
    def _make_row(self, epoch, mean):
        from phoonnx_train.evaluation.scorer import EvalRow

        return EvalRow(
            epoch=epoch, step=epoch, checkpoint=f"epoch={epoch}.ckpt",
            n_sentences=1, aggregates={"utmos_mean": mean, "utmos_std": 0.0,
                                       "utmos_min": mean, "utmos_max": mean},
            perutt=[("s", {"utmos": mean}, None)],
        )

    def test_all_unscored_epochs_scored_in_one_firing(self):
        from phoonnx_train.evaluation.callbacks import EvalScoreboardCallback
        from phoonnx_train.evaluation.selection import SelectionPolicy
        from phoonnx_train.evaluation.tracker import MetricsTracker

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ckpt_dir = d / "ck"
            ckpt_dir.mkdir()
            # --checkpoint-epochs 2 style: epochs 1, 3, 5 on disk.
            for e in (1, 3, 5):
                (ckpt_dir / f"epoch={e}-step={e}.ckpt").write_bytes(b"w")

            scored = []

            scorer = SimpleNamespace(metrics=["utmos"])
            scorer.score = lambda ckpt, epoch, work_dir=None: (
                scored.append(epoch) or self._make_row(epoch, 3.0 + epoch)
            )

            out = d / "eval"
            tracker = MetricsTracker(out)
            selection = SelectionPolicy(metric="utmos_mean")
            cb = EvalScoreboardCallback(
                scorer, tracker, selection, out,
                every_n_epochs=2, checkpoint_dir=ckpt_dir,
            )
            trainer = SimpleNamespace(current_epoch=5, should_stop=False,
                                      checkpoint_callback=None)
            with mock.patch(
                "phoonnx_train.evaluation.callbacks.size_stable",
                return_value=True,
            ):
                cb.on_train_epoch_end(trainer, None)

            self.assertEqual(sorted(scored), [1, 3, 5])
            self.assertEqual(tracker.done_epochs(), {1, 3, 5})


# ---------------------------------------------------------------------------
# #7 mel_basis cache keyed on the full parameter set
# ---------------------------------------------------------------------------
class TestMelBasisCacheKeys(unittest.TestCase):
    def test_vits_same_fmax_different_n_mels(self):
        from phoonnx_train.vits import mel_processing as mp

        mp.mel_basis.clear()
        y = torch.zeros(1, 4096)
        a = mp.mel_spectrogram_torch(y, 1024, 80, 22050, 256, 1024, 0.0, 8000.0)
        b = mp.mel_spectrogram_torch(y, 1024, 40, 22050, 256, 1024, 0.0, 8000.0)
        # Same fmax (8000) but different n_mels -> distinct filterbanks and
        # distinct mel-channel dims, not aliased to one cached bank.
        self.assertEqual(a.shape[1], 80)
        self.assertEqual(b.shape[1], 40)
        self.assertEqual(len(mp.mel_basis), 2)

    def test_matcha_same_fmax_different_n_mels(self):
        from phoonnx_train.matcha import audio as ma

        ma.mel_basis.clear()
        ma.hann_window.clear()
        y = torch.zeros(1, 4096)
        a = ma.mel_spectrogram(y, 1024, 80, 22050, 256, 1024, 0.0, 8000.0)
        b = ma.mel_spectrogram(y, 1024, 40, 22050, 256, 1024, 0.0, 8000.0)
        self.assertEqual(a.shape[1], 80)
        self.assertEqual(b.shape[1], 40)
        self.assertEqual(len(ma.mel_basis), 2)

    def test_eps_constants_named(self):
        from phoonnx_train.vits import mel_processing as mp
        from phoonnx_train.matcha import audio as ma

        self.assertEqual(mp.VITS_STFT_EPS, 1e-6)
        self.assertEqual(ma.MEL_STFT_EPS, 1e-9)


if __name__ == "__main__":
    unittest.main()
