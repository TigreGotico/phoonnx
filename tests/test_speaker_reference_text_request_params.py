"""The cloning reference transcription in the request of ``TTSVoice.render``.

The transcription goes on the per-call ``AdapterSynthesisRequest.params``, not
on adapter state, because the threaded ovos-tts-server shares one adapter
between concurrent requests. OmniVoice and Spark-TTS read it from
``request.params["speaker_reference_text"]`` and tokenize it themselves, so it
must be normalised as the target text is. The voice here reads letters, and the
adapter only records its requests, so no model runs. The expected normalised
forms are written by hand.
"""
import unittest

import numpy as np

from phoonnx.config import SynthesisConfig, VoiceConfig
from phoonnx.engines.base import AdapterSynthesisResult
from phoonnx.engines.vits import VitsAdapter
from phoonnx.voice import TTSVoice, Utterance
from tests.test_prepare_render import CONFIG


class _Recorder(VitsAdapter):
    def __init__(self):
        super().__init__()
        self.requests = []

    def synthesize(self, request, session):
        self.requests.append(request)
        return AdapterSynthesisResult(audio=np.zeros(4, dtype=np.float32))


def _voice():
    return TTSVoice(config=VoiceConfig.from_dict(dict(CONFIG)), session=object(), adapter=_Recorder())


def _reference_text(voice, syn_config):
    voice.render(Utterance("", None, (1, 2, 3)), syn_config)
    return voice.adapter.requests[-1].params.get("speaker_reference_text")


class TestSpeakerReferenceTextReachesRequestParams(unittest.TestCase):
    def test_speaker_reference_text_lands_in_request_params(self):
        voice = _voice()
        syn_config = SynthesisConfig(speaker_reference_text="A reference clip transcript")
        self.assertEqual(_reference_text(voice, syn_config), "A reference clip transcript")

    def test_absent_when_no_reference_text_was_given(self):
        voice = _voice()
        voice.render(Utterance("", None, (1, 2, 3)), SynthesisConfig())
        self.assertNotIn("speaker_reference_text", voice.adapter.requests[0].params)

    def test_each_chunk_of_a_multi_chunk_call_sees_its_own_fresh_params(self):
        """render is called once per chunk; the params are built again from syn_config each time."""
        voice = _voice()
        syn_config = SynthesisConfig(speaker_reference_text="Shared reference text")

        voice.render(Utterance("", None, (1, 2)), syn_config)
        voice.render(Utterance("", None, (3, 4)), syn_config)

        first, second = voice.adapter.requests
        for request in (first, second):
            self.assertEqual(request.params.get("speaker_reference_text"), "Shared reference text")
        self.assertIsNot(first.params, second.params)


class TestSpeakerReferenceTextIsNormalised(unittest.TestCase):
    def test_digits_read_as_the_target_text_reads_them(self):
        voice = _voice()
        text = "I have 3 cats."
        reference = _reference_text(voice, SynthesisConfig(speaker_reference_text=text))
        self.assertEqual(reference, "I have three cats.")
        (target,) = voice.prepare(text)
        self.assertEqual(reference, target.text)

    def test_the_reference_language_reads_the_reference(self):
        voice = _voice()
        syn_config = SynthesisConfig(speaker_reference_text="Tenho 3 gatos.", speaker_reference_lang="pt-PT")
        self.assertEqual(_reference_text(voice, syn_config), "Tenho três gatos.")


if __name__ == "__main__":
    unittest.main()
