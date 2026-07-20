"""Tests for cold-start latency fixes: ORT-optimized-graph caching, voice
warmup, and the silent-provider-fallback warning.
"""
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from phoonnx import providers
from phoonnx.providers import (
    CACHE_DIR_ENV_VAR,
    CPU_PROVIDER,
    make_session,
)


def _build_tiny_onnx_model(path: Path) -> None:
    """Write a minimal but real single-node ONNX model to *path*."""
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "tiny", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


class TestOrtOptimizedGraphCache(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.model_path = Path(self.tmpdir.name) / "model.onnx"
        _build_tiny_onnx_model(self.model_path)
        self.cache_dir = Path(self.tmpdir.name) / "cache"

        env_patcher = patch.dict(os.environ, {}, clear=False)
        env_patcher.start()
        self.addCleanup(env_patcher.stop)
        os.environ.pop(CACHE_DIR_ENV_VAR, None)

    def test_no_cache_dir_leaves_behaviour_unchanged(self):
        session = make_session(self.model_path, providers=[CPU_PROVIDER])
        self.assertIn(CPU_PROVIDER, session.get_providers())
        self.assertFalse(self.cache_dir.exists())

    def test_first_load_writes_the_optimized_cache_file(self):
        make_session(self.model_path, providers=[CPU_PROVIDER], cache_dir=self.cache_dir)
        cached_files = list(self.cache_dir.glob("*.ort_optimized.onnx"))
        self.assertEqual(len(cached_files), 1)

    def test_second_load_reuses_the_cached_file_without_rewriting_it(self):
        make_session(self.model_path, providers=[CPU_PROVIDER], cache_dir=self.cache_dir)
        cached_files = list(self.cache_dir.glob("*.ort_optimized.onnx"))
        self.assertEqual(len(cached_files), 1)
        cache_path = cached_files[0]
        mtime_after_first_load = cache_path.stat().st_mtime_ns

        with patch.object(providers.onnxruntime, "InferenceSession",
                           wraps=providers.onnxruntime.InferenceSession) as spy:
            session = make_session(self.model_path, providers=[CPU_PROVIDER], cache_dir=self.cache_dir)

        # Second load must construct the session directly from the cached
        # optimized file (with optimization disabled), not from the raw model.
        (loaded_path, ), kwargs = spy.call_args
        self.assertEqual(str(loaded_path), str(cache_path))
        self.assertEqual(
            kwargs["sess_options"].graph_optimization_level,
            providers.onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL,
        )
        self.assertEqual(cache_path.stat().st_mtime_ns, mtime_after_first_load)
        self.assertIn(CPU_PROVIDER, session.get_providers())

    def test_env_var_is_honoured_when_no_cache_dir_argument_is_given(self):
        with patch.dict(os.environ, {CACHE_DIR_ENV_VAR: str(self.cache_dir)}):
            make_session(self.model_path, providers=[CPU_PROVIDER])
        self.assertTrue(list(self.cache_dir.glob("*.ort_optimized.onnx")))

    def test_explicit_cache_dir_argument_wins_over_env_var(self):
        other_dir = Path(self.tmpdir.name) / "other_cache"
        with patch.dict(os.environ, {CACHE_DIR_ENV_VAR: str(self.cache_dir)}):
            make_session(self.model_path, providers=[CPU_PROVIDER], cache_dir=other_dir)
        self.assertTrue(list(other_dir.glob("*.ort_optimized.onnx")))
        self.assertFalse(self.cache_dir.exists())

    def test_corrupt_cache_file_falls_back_and_is_removed(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = providers._cache_key_path(self.model_path, [CPU_PROVIDER], self.cache_dir)
        cache_path.write_bytes(b"not a valid onnx model")
        self.assertTrue(cache_path.is_file())

        with patch.object(providers.LOG, "warning") as warn:
            session = make_session(self.model_path, providers=[CPU_PROVIDER], cache_dir=self.cache_dir)

        self.assertTrue(warn.called)
        warned = " ".join(str(c) for c in warn.call_args_list)
        self.assertIn("cached optimized model", warned)
        self.assertIn(CPU_PROVIDER, session.get_providers())
        # the corrupt bytes were discarded — what's on disk now (if anything)
        # is a freshly rebuilt, valid optimized model, not the garbage above.
        if cache_path.is_file():
            self.assertNotEqual(cache_path.read_bytes(), b"not a valid onnx model")

    def test_cache_key_changes_when_provider_list_changes(self):
        make_session(self.model_path, providers=[CPU_PROVIDER], cache_dir=self.cache_dir)
        key_a = set(p.name for p in self.cache_dir.glob("*.ort_optimized.onnx"))

        make_session(self.model_path, providers=[("CPUExecutionProvider", {"foo": "bar"})],
                     cache_dir=self.cache_dir)
        key_b = set(p.name for p in self.cache_dir.glob("*.ort_optimized.onnx"))

        # Different provider spec -> different cache key -> an extra file, not
        # a reused/corrupted one.
        self.assertEqual(len(key_a), 1)
        self.assertEqual(len(key_b), 2)
        self.assertTrue(key_a.issubset(key_b))

    def test_cache_key_changes_when_model_file_changes(self):
        make_session(self.model_path, providers=[CPU_PROVIDER], cache_dir=self.cache_dir)
        first_keys = set(self.cache_dir.glob("*.ort_optimized.onnx"))

        # touch the model so mtime/size differ
        import time
        time.sleep(0.01)
        _build_tiny_onnx_model(self.model_path)

        make_session(self.model_path, providers=[CPU_PROVIDER], cache_dir=self.cache_dir)
        second_keys = set(self.cache_dir.glob("*.ort_optimized.onnx"))
        self.assertEqual(len(first_keys | second_keys), 2)


class TestProviderFallbackWarning(unittest.TestCase):
    def test_warns_when_requested_provider_is_missing_from_active_session(self):
        fake_session = MagicMock()
        fake_session.get_providers.return_value = [CPU_PROVIDER]
        with patch.object(providers.LOG, "warning") as warn:
            providers._warn_on_provider_fallback(["CUDAExecutionProvider", CPU_PROVIDER], fake_session)
        self.assertTrue(warn.called)
        joined = " ".join(str(c) for c in warn.call_args_list)
        self.assertIn("CUDAExecutionProvider", joined)
        self.assertIn(CPU_PROVIDER, joined)

    def test_no_warning_when_requested_provider_is_active(self):
        fake_session = MagicMock()
        fake_session.get_providers.return_value = ["CUDAExecutionProvider", CPU_PROVIDER]
        with patch.object(providers.LOG, "warning") as warn:
            providers._warn_on_provider_fallback(["CUDAExecutionProvider", CPU_PROVIDER], fake_session)
        warn.assert_not_called()

    def test_no_crash_when_get_providers_raises(self):
        fake_session = MagicMock()
        fake_session.get_providers.side_effect = RuntimeError("boom")
        # Must not raise.
        providers._warn_on_provider_fallback(["CUDAExecutionProvider"], fake_session)

    def test_make_session_surfaces_the_warning_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.onnx"
            _build_tiny_onnx_model(model_path)
            with patch.object(providers, "resolve_providers",
                               return_value=["CUDAExecutionProvider", CPU_PROVIDER]):
                with patch.object(providers.LOG, "warning") as warn:
                    make_session(model_path, providers=["CUDAExecutionProvider"])
            self.assertTrue(warn.called)
            warned = " ".join(str(c) for c in warn.call_args_list)
            self.assertIn("CUDAExecutionProvider", warned)


class TestTTSVoiceWarmup(unittest.TestCase):
    def _make_voice(self):
        from phoonnx.voice import TTSVoice

        voice = TTSVoice.__new__(TTSVoice)
        voice.session = MagicMock()
        voice.config = MagicMock()
        voice.phonetic_spellings = None
        voice.phonemizer = MagicMock()
        adapter = MagicMock()
        adapter.default_params.return_value = {"noise_scale": 0.667}
        adapter.synthesize.return_value = MagicMock(audio=np.zeros(4, dtype=np.float32))
        voice.adapter = adapter
        return voice

    def test_warmup_runs_one_inference_with_a_valid_feed_dict(self):
        voice = self._make_voice()
        voice.warmup()

        self.assertEqual(voice.adapter.synthesize.call_count, 1)
        (request, session), _ = voice.adapter.synthesize.call_args
        self.assertIs(session, voice.session)
        self.assertEqual(request.phoneme_ids.shape, (1, 1))
        self.assertEqual(int(request.phoneme_lengths[0]), 1)
        self.assertEqual(request.params["noise_scale"], 0.667)

    def test_warmup_no_ops_and_logs_debug_for_an_unsupported_adapter(self):
        voice = self._make_voice()
        voice.adapter.synthesize.side_effect = KeyError("unknown input name")

        from phoonnx.voice import LOG as voice_log
        with patch.object(voice_log, "debug") as debug:
            voice.warmup()  # must not raise

        self.assertTrue(debug.called)
        logged = " ".join(str(c) for c in debug.call_args_list)
        self.assertIn("warmup skipped", logged)

    def test_load_with_warmup_true_triggers_one_inference(self):
        from phoonnx.voice import TTSVoice

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.onnx"
            _build_tiny_onnx_model(model_path)

            fake_adapter = MagicMock()
            fake_adapter.default_params.return_value = {}
            fake_adapter.synthesize.return_value = MagicMock(audio=np.zeros(4, dtype=np.float32))

            with patch("phoonnx.voice.detect_engine", return_value=fake_adapter):
                voice = TTSVoice.load(model_path=model_path, config_path=str(model_path) + ".missing",
                                       providers=[CPU_PROVIDER], warmup=True)

        self.assertEqual(fake_adapter.synthesize.call_count, 1)
        self.assertEqual(voice.adapter, fake_adapter)

    def test_load_with_warmup_false_does_not_run_inference(self):
        from phoonnx.voice import TTSVoice

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.onnx"
            _build_tiny_onnx_model(model_path)

            fake_adapter = MagicMock()
            fake_adapter.default_params.return_value = {}

            with patch("phoonnx.voice.detect_engine", return_value=fake_adapter):
                TTSVoice.load(model_path=model_path, config_path=str(model_path) + ".missing",
                               providers=[CPU_PROVIDER], warmup=False)

        fake_adapter.synthesize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
