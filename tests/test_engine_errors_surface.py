"""A bug inside an engine reaches the caller once, not as a silent retry."""
import inspect
import unittest
from unittest.mock import MagicMock

import numpy as np

from phoonnx.engines import _REGISTRY, _register_builtins
from phoonnx.engines.base import AdapterSynthesisRequest, BaseOnnxAdapter


class _BrokenParser(BaseOnnxAdapter):
    def __init__(self):
        self.calls = 0

    @staticmethod
    def default_params():
        return {}

    def build_feed_dict(self, request, session):
        return {}

    def parse_outputs(self, outputs, request, output_names=None):
        self.calls += 1
        raise TypeError("a bug inside the engine")


class _BrokenFeed(_BrokenParser):
    def build_feed_dict(self, request, session):
        self.calls += 1
        raise TypeError("a bug while building the feed")


class TestEngineErrors(unittest.TestCase):
    def test_a_type_error_in_parse_outputs_reaches_the_caller_once(self):
        adapter = _BrokenParser()
        session = MagicMock()
        session.run.return_value = [np.zeros(4, dtype=np.float32)]
        request = AdapterSynthesisRequest(phoneme_ids=np.zeros((1, 3), dtype=np.int64),
                                          phoneme_lengths=np.array([3], dtype=np.int64))
        with self.assertRaisesRegex(TypeError, "a bug inside the engine"):
            adapter.synthesize(request, session)
        self.assertEqual(adapter.calls, 1)

    def test_a_type_error_in_build_feed_dict_reaches_the_caller_once(self):
        adapter = _BrokenFeed()
        request = AdapterSynthesisRequest(phoneme_ids=np.zeros((1, 3), dtype=np.int64),
                                          phoneme_lengths=np.array([3], dtype=np.int64))
        with self.assertRaisesRegex(TypeError, "a bug while building the feed"):
            adapter.synthesize(request, MagicMock())
        self.assertEqual(adapter.calls, 1)

    def test_every_engine_takes_output_names(self):
        _register_builtins()
        for name, adapter in _REGISTRY.items():
            with self.subTest(engine=name):
                self.assertIn("output_names", inspect.signature(adapter.parse_outputs).parameters)


if __name__ == "__main__":
    unittest.main()
