"""Bounds and pinning for the loaded-voice cache.

The cache used to grow without limit, so a server answering requests for many
voices grew until something killed it. These pin the two settings that replace
that: a ceiling, and a set of voices that ceiling may not evict.
"""
import unittest
from unittest.mock import MagicMock, patch


def _plugin(testcase, **config):
    """Build the plugin with its model manager and loading stubbed out.

    The patches stay active for the whole test: get_model is exercised after
    construction, and it calls the same collaborators.
    """
    from phoonnx.opm import PhoonnxTTSPlugin
    patchers = [
        patch("phoonnx.opm.TTSModelManager"),
        patch.object(PhoonnxTTSPlugin, "get_default_voice",
                     return_value=MagicMock()),
        patch.object(PhoonnxTTSPlugin, "get_voice_info",
                     side_effect=lambda v: MagicMock(
                         load=MagicMock(return_value=f"model:{v}"))),
        patch.object(PhoonnxTTSPlugin, "_providers", return_value=None),
    ]
    for pat in patchers:
        pat.start()
        testcase.addCleanup(pat.stop)
    return PhoonnxTTSPlugin(config=dict(config))


class TestCacheCeiling(unittest.TestCase):

    def test_unset_limit_keeps_everything(self):
        # The previous behaviour, kept as the default so no deployment
        # silently starts evicting.
        plugin = _plugin(self)
        for i in range(5):
            plugin.get_model(f"v{i}")
        self.assertEqual(len(plugin.voices), 5)

    def test_limit_evicts_least_recently_used(self):
        plugin = _plugin(self, max_loaded_voices=2)
        plugin.get_model("a")
        plugin.get_model("b")
        plugin.get_model("a")      # touch: "b" is now the oldest
        plugin.get_model("c")
        self.assertEqual(len(plugin.voices), 2)
        self.assertIn("a", plugin.voices)
        self.assertIn("c", plugin.voices)
        self.assertNotIn("b", plugin.voices)

    def test_invalid_limit_is_ignored(self):
        for bad in ("nonsense", -1, 0):
            with self.subTest(value=bad):
                plugin = _plugin(self, max_loaded_voices=bad)
                self.assertIsNone(plugin.max_loaded_voices)


class TestPinning(unittest.TestCase):

    def test_pinned_voice_is_loaded_at_startup(self):
        plugin = _plugin(self, pinned_voices=["keeper"])
        self.assertIn("keeper", plugin.voices)

    def test_pinned_voice_survives_eviction_pressure(self):
        plugin = _plugin(self, pinned_voices=["keeper"], max_loaded_voices=2)
        for i in range(6):
            plugin.get_model(f"passing{i}")
        self.assertIn("keeper", plugin.voices,
                      "a pinned voice must never be evicted")

    def test_limit_is_raised_to_fit_the_pins(self):
        # A limit smaller than the pin set is a contradiction; the pins win,
        # because they were named explicitly.
        plugin = _plugin(self, pinned_voices=["a", "b", "c"], max_loaded_voices=1)
        self.assertEqual(plugin.max_loaded_voices, 3)
        for v in ("a", "b", "c"):
            self.assertIn(v, plugin.voices)

    def test_a_pinned_voice_that_cannot_load_does_not_stop_startup(self):
        from phoonnx.opm import PhoonnxTTSPlugin
        with patch("phoonnx.opm.TTSModelManager"), \
                patch.object(PhoonnxTTSPlugin, "get_default_voice",
                             return_value=MagicMock()), \
                patch.object(PhoonnxTTSPlugin, "get_voice_info",
                             side_effect=RuntimeError("no such voice")), \
                patch.object(PhoonnxTTSPlugin, "_providers", return_value=None):
            plugin = PhoonnxTTSPlugin(config={"pinned_voices": ["missing"]})
        self.assertEqual(len(plugin.voices), 0)


if __name__ == "__main__":
    unittest.main()
