"""A speaker and the speech-variation scales chosen per request.

A multi-speaker voice served over HTTP was pinned to whatever speaker the
server's config named, and the response cache keyed audio by voice and sentence,
so every request for a voice returned the same recording whatever speaker it
asked for. The request now selects the speaker, the speaking rate and the two
sampling noises, and each combination is its own cache identity.
"""
import inspect
import unittest
from unittest.mock import MagicMock, patch


def _plugin(testcase, **config):
    from phoonnx.opm import PhoonnxTTSPlugin

    patchers = [
        patch("phoonnx.opm.TTSModelManager"),
        patch.object(PhoonnxTTSPlugin, "get_default_voice",
                     return_value=MagicMock()),
        patch.object(PhoonnxTTSPlugin, "get_voice_info",
                     side_effect=lambda v: MagicMock(
                         load=MagicMock(return_value=MagicMock()),
                         config=MagicMock(speaker_id_map={"p225": 0, "p226": 1},
                                          noise_scale=0.667, length_scale=1.0,
                                          noise_w_scale=0.8))),
        patch.object(PhoonnxTTSPlugin, "_providers", return_value=None),
    ]
    for pat in patchers:
        pat.start()
        testcase.addCleanup(pat.stop)
    return PhoonnxTTSPlugin(config=dict(config))


def _sent_config(testcase, plugin, **call_kwargs):
    with patch("phoonnx.opm.SynthesisConfig") as cfg, \
            patch("phoonnx.opm.wave.open"):
        plugin.get_tts("hey mycroft", "/tmp/out.wav", voice="v", **call_kwargs)
    return cfg.call_args.kwargs


class TestTheManagerForwardsThem(unittest.TestCase):

    def test_every_option_is_named_in_the_signature(self):
        from phoonnx.opm import PhoonnxTTSPlugin
        params = inspect.signature(PhoonnxTTSPlugin.get_tts).parameters
        for name in ("speaker_id", "speaker", "length_scale", "noise_scale",
                     "noise_w_scale"):
            self.assertIn(name, params,
                          f"{name} must be declared or the plugin manager "
                          f"drops it before get_tts is called")


class TestTheRequestReachesTheEngine(unittest.TestCase):

    def test_a_speaker_index_sent_as_a_query_string(self):
        got = _sent_config(self, _plugin(self), speaker_id="57")
        self.assertEqual(got["speaker_id"], 57)

    def test_a_speaker_named_through_the_voice_map(self):
        got = _sent_config(self, _plugin(self), speaker="p226")
        self.assertEqual(got["speaker_id"], 1)

    def test_the_request_beats_the_configured_speaker(self):
        got = _sent_config(self, _plugin(self, speaker_id=3), speaker_id="57")
        self.assertEqual(got["speaker_id"], 57)

    def test_the_configured_speaker_applies_when_none_is_sent(self):
        got = _sent_config(self, _plugin(self, speaker_id=3))
        self.assertEqual(got["speaker_id"], 3)

    def test_scales_sent_as_query_strings_arrive_as_numbers(self):
        got = _sent_config(self, _plugin(self, length_scale=1.0),
                           length_scale="1.3", noise_scale="0.4",
                           noise_w_scale="0.9")
        self.assertEqual((got["length_scale"], got["noise_scale"],
                          got["noise_w_scale"]), (1.3, 0.4, 0.9))

    def test_unsent_scales_keep_the_voice_defaults(self):
        got = _sent_config(self, _plugin(self))
        self.assertEqual((got["length_scale"], got["noise_scale"],
                          got["noise_w_scale"]), (1.0, 0.667, 0.8))


class TestEachVariantHasItsOwnCacheEntry(unittest.TestCase):
    """The cache keys audio by sentence inside a voice identity, so a variant
    that does not change the identity is served the first variant's audio."""

    def _id(self, plugin, **kwargs):
        return plugin._get_ctxt(dict(voice="piper/en_GB-vctk-medium",
                                     **kwargs)).tts_id

    def test_two_speakers_do_not_share_audio(self):
        plugin = _plugin(self)
        self.assertNotEqual(self._id(plugin, speaker_id="3"),
                            self._id(plugin, speaker_id="50"))

    def test_two_speaking_rates_do_not_share_audio(self):
        plugin = _plugin(self)
        self.assertNotEqual(self._id(plugin, length_scale="0.9"),
                            self._id(plugin, length_scale="1.2"))

    def test_the_same_variant_still_shares_its_cache(self):
        plugin = _plugin(self)
        self.assertEqual(self._id(plugin, speaker_id="3", noise_scale="0.5"),
                         self._id(plugin, speaker_id="3", noise_scale="0.5"))

    def test_a_plain_request_keeps_the_plain_identity(self):
        plugin = _plugin(self)
        self.assertNotIn("#", self._id(plugin))

    def test_the_real_voice_still_reaches_get_tts(self):
        plugin = _plugin(self)
        ctxt = plugin._get_ctxt({"voice": "piper/en_GB-vctk-medium",
                                 "speaker_id": "3"})
        self.assertEqual(ctxt.synth_kwargs["voice"], "piper/en_GB-vctk-medium")


if __name__ == "__main__":
    unittest.main()
