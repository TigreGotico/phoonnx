"""The intra-op thread count reaches every session a voice opens."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phoonnx import providers
from phoonnx.providers import (CPU_PROVIDER, THREADS_ENV_VAR, make_session, onnx_threads,
                               resolve_threads)
from phoonnx.voice import TTSVoice
from tests.test_shared_sessions import _build_tiny_onnx_model


def _threads(session) -> int:
    return session.get_session_options().intra_op_num_threads


class _TinyModel(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.model_path = Path(tmp.name) / "tiny.onnx"
        _build_tiny_onnx_model(self.model_path)
        env = patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(THREADS_ENV_VAR, None)
        providers._SHARED_SESSIONS.clear()


class TestMakeSession(_TinyModel):
    def test_unset_leaves_the_onnxruntime_default(self):
        self.assertEqual(_threads(make_session(self.model_path, providers=[CPU_PROVIDER])), 0)

    def test_the_argument_sets_the_count(self):
        session = make_session(self.model_path, providers=[CPU_PROVIDER], threads=2)
        self.assertEqual(_threads(session), 2)

    def test_the_environment_sets_the_count(self):
        os.environ[THREADS_ENV_VAR] = "3"
        self.assertEqual(_threads(make_session(self.model_path, providers=[CPU_PROVIDER])), 3)

    def test_the_scope_wins_over_the_environment(self):
        os.environ[THREADS_ENV_VAR] = "3"
        with onnx_threads(2):
            self.assertEqual(_threads(make_session(self.model_path, providers=[CPU_PROVIDER])), 2)

    def test_the_argument_wins_over_the_scope(self):
        with onnx_threads(2):
            session = make_session(self.model_path, providers=[CPU_PROVIDER], threads=1)
        self.assertEqual(_threads(session), 1)

    def test_a_none_scope_keeps_the_outer_one(self):
        with onnx_threads(2), onnx_threads(None):
            self.assertEqual(_threads(make_session(self.model_path, providers=[CPU_PROVIDER])), 2)

    def test_different_counts_do_not_share_a_session(self):
        one = make_session(self.model_path, providers=[CPU_PROVIDER], threads=1)
        two = make_session(self.model_path, providers=[CPU_PROVIDER], threads=2)
        self.assertIsNot(one, two)
        self.assertIs(two, make_session(self.model_path, providers=[CPU_PROVIDER], threads=2))

    def test_the_optimized_graph_cache_keeps_the_count(self):
        with tempfile.TemporaryDirectory() as cache:
            for _ in range(2):  # build the cache file, then load from it
                providers._SHARED_SESSIONS.clear()
                session = make_session(self.model_path, providers=[CPU_PROVIDER],
                                       cache_dir=cache, threads=2)
                self.assertEqual(_threads(session), 2)

    def test_a_bad_value_names_its_source(self):
        os.environ[THREADS_ENV_VAR] = "many"
        with self.assertRaises(ValueError) as caught:
            resolve_threads()
        self.assertIn(THREADS_ENV_VAR, str(caught.exception))
        for bad in (0, -1, "x"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                resolve_threads(bad)


class TestTheVoice(_TinyModel):
    def test_load_sets_the_count_of_the_voice_session(self):
        voice = TTSVoice.load(model_path=self.model_path, providers=[CPU_PROVIDER], threads=2)
        self.assertEqual(_threads(voice.session), 2)
        self.assertEqual(voice.onnx_threads, 2)

    def test_a_session_opened_during_synthesis_gets_the_voice_count(self):
        voice = TTSVoice.load(model_path=self.model_path, providers=[CPU_PROVIDER], threads=2)
        opened = []

        def infer(*args, **kwargs):
            opened.append(make_session(self.model_path, providers=[CPU_PROVIDER]))
            raise StopIteration

        with patch.object(voice, "_infer", side_effect=infer):
            from phoonnx.voice import Utterance
            with self.assertRaises(StopIteration):
                voice.render(Utterance(text="", symbols=None, ids=(1, 2)))
        self.assertEqual(_threads(opened[0]), 2)


if __name__ == "__main__":
    unittest.main()
