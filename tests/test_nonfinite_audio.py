"""A model that returns NaN or inf samples makes synthesis raise.

Cast to int16, a NaN waveform is written as silence of the right length, so the
call looked like a success and a TTS server answered 200 with a silent WAV.
"""
import os
import tempfile
import unittest

import numpy as np

from phoonnx.config import SynthesisConfig, VoiceConfig
from phoonnx.engines.vits import VitsAdapter
from phoonnx.errors import NonFiniteAudioError, PhoonnxError
from phoonnx.voice import TTSVoice
from tests.test_failed_synthesis_cache import _plugin
from tests.test_prepare_render import CONFIG, RAW, TEXT, _engine_audio, _session

STREAM_PARTS = SynthesisConfig(normalize_audio=False)
"""With no whole-sentence step, ``stream`` yields the engine's parts directly."""


class _PoisonAdapter(VitsAdapter):
    """The real graph's audio, with *poison* written into the samples *where* selects."""

    def __init__(self, poison=None, where=slice(None)):
        super().__init__()
        self.poison, self.where = poison, where

    def synthesize(self, request, session):
        result = super().synthesize(request, session)
        if self.poison is not None:
            result.audio[..., self.where] = self.poison
        return result


def _voice(poison=None, where=slice(None)):
    return TTSVoice(config=VoiceConfig.from_dict(dict(CONFIG)), session=_session(),
                    adapter=_PoisonAdapter(poison, where))


BAD = [("nan", np.nan, slice(None)), ("one nan sample", np.nan, 3),
       ("inf", np.inf, 3), ("-inf", -np.inf, slice(None))]


class TestNonFiniteAudioRaises(unittest.TestCase):
    def test_synthesize(self):
        for name, poison, where in BAD:
            for config in (SynthesisConfig(), RAW):
                with self.subTest(name, normalize=config.normalize_audio):
                    with self.assertRaises(NonFiniteAudioError) as caught:
                        list(_voice(poison, where).synthesize(TEXT, config))
                    self.assertIn("non-finite", str(caught.exception))

    def test_stream(self):
        for name, poison, where in BAD:
            for config in (SynthesisConfig(), STREAM_PARTS):
                with self.subTest(name, normalize=config.normalize_audio):
                    with self.assertRaises(NonFiniteAudioError):
                        list(_voice(poison, where).stream(TEXT, config))

    def test_render(self):
        voice = _voice(np.nan)
        utterance = next(iter(voice.prepare(TEXT)))
        with self.assertRaises(NonFiniteAudioError):
            voice.render(utterance)

    def test_the_error_names_the_voice_and_is_a_runtime_error(self):
        voice = _voice(np.nan)
        voice.model_path = "/models/kokoro-82m/af_heart.onnx"
        with self.assertRaises(RuntimeError) as caught:
            list(voice.synthesize(TEXT))
        self.assertIsInstance(caught.exception, PhoonnxError)
        self.assertIn("'af_heart'", str(caught.exception))


class TestFiniteAudioStillWorks(unittest.TestCase):
    def test_synthesize_and_stream_give_the_graph_audio(self):
        voice = _voice()
        expected = [np.clip(_engine_audio(voice.session, u.ids), -1.0, 1.0)
                    for u in voice.prepare(TEXT, RAW)]
        for chunks in (list(voice.synthesize(TEXT, RAW)), list(voice.stream(TEXT, STREAM_PARTS))):
            self.assertEqual(len(chunks), len(expected))
            for chunk, audio in zip(chunks, expected):
                np.testing.assert_allclose(chunk.audio_float_array, audio, rtol=1e-6)

    def test_a_huge_finite_sample_is_clipped_not_refused(self):
        chunks = list(_voice(1e30, 3).synthesize(TEXT, RAW))
        self.assertEqual([c.audio_int16_array[3] for c in chunks], [32767, 32767])


class TestPluginGetTts(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.target = os.path.join(self.dir, "audio.wav")

    def test_nan_audio_raises_and_leaves_no_wav(self):
        for name, poison, where in BAD:
            with self.subTest(name):
                plugin = _plugin(self, synth_side_effect=_voice(poison, where).synthesize_wav)
                with self.assertRaises(NonFiniteAudioError):
                    plugin.get_tts(TEXT, self.target)
                self.assertEqual(os.listdir(self.dir), [])

    def test_finite_audio_writes_the_wav(self):
        plugin = _plugin(self, synth_side_effect=_voice().synthesize_wav)
        self.assertEqual(plugin.get_tts(TEXT, self.target), (self.target, None))
        self.assertGreater(os.path.getsize(self.target), 44)


if __name__ == "__main__":
    unittest.main()
