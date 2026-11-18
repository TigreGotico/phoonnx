"""The Engine enum names the adapter registry, and the removed hooks warn."""
import unittest

from phoonnx.config import ENGINE_ALIASES, Engine
from phoonnx.engines import _REGISTRY, _register_builtins
from phoonnx.engines.base import BaseOnnxAdapter
from phoonnx.engines.vits import VitsAdapter

# Config-format names a published config may carry. Detection maps each to an
# adapter by probing the model, so none of them is a registry entry.
FORMAT_NAMES = {"phoonnx", "piper", "mimic3", "coqui", "transformers"}


class TestEngineEnumMatchesRegistry(unittest.TestCase):
    def setUp(self):
        _register_builtins()

    def test_every_registered_engine_is_an_enum_member(self):
        self.assertEqual(set(_REGISTRY) - {e.value for e in Engine}, set())

    def test_every_enum_member_is_registered_or_a_config_format(self):
        self.assertEqual({e.value for e in Engine} - set(_REGISTRY), FORMAT_NAMES)

    def test_kokoro_is_the_one_alias_and_not_a_registry_name(self):
        self.assertEqual(ENGINE_ALIASES, {"kokoro": "styletts2"})
        self.assertIs(Engine("kokoro"), Engine.STYLETTS2)
        self.assertNotIn("kokoro", _REGISTRY)


class TestRemovedHooksWarn(unittest.TestCase):
    def test_parse_config_warns_and_returns_nothing(self):
        with self.assertWarnsRegex(DeprecationWarning, r"parse_config.*removed in phoonnx \d+\.0\.0"):
            self.assertEqual(VitsAdapter().parse_config({"inference": {"noise_scale": 0.4}}), {})

    def test_param_labels_warns_and_names_the_default_params(self):
        adapter = VitsAdapter()
        with self.assertWarnsRegex(DeprecationWarning, r"param_labels.*removed in phoonnx \d+\.0\.0"):
            labels = adapter.param_labels()
        self.assertEqual(labels, {k: k for k in adapter.default_params()})

    def test_no_adapter_overrides_a_removed_hook(self):
        for name, cls in _REGISTRY.items():
            for hook in ("parse_config", "param_labels"):
                with self.subTest(engine=name, hook=hook):
                    self.assertIs(getattr(cls, hook), getattr(BaseOnnxAdapter, hook))

    def test_setup_methods_attribute_is_gone(self):
        self.assertFalse(hasattr(BaseOnnxAdapter, "SETUP_METHODS"))


if __name__ == "__main__":
    unittest.main()
