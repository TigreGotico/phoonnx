"""Regression test for phoonnx/voice.py:TTSVoice.phoneme_ids_to_audio.

PR #404 moved a cloning reference's transcription off adapter instance state
(shared across concurrent requests under the threaded ovos-tts-server) and onto
the per-call ``AdapterSynthesisRequest.params`` dict instead, so OmniVoice/
Spark-TTS's ``synthesize`` reads ``request.params["speaker_reference_text"]``
rather than something stashed by a possibly-different, interleaved request's
``encode_text``. This test locks down the other half of that fix: the line in
``phoneme_ids_to_audio`` that actually puts it there.

Deleting that one line degrades silently -- ``request.params`` would just be
missing the key, which the engines already treat as "no reference transcription
was given" (their normal, warning-free no-cloning path) -- so nothing else in
the non-training suite catches its absence. This test drives
``phoneme_ids_to_audio`` directly against a mocked adapter and asserts on the
``AdapterSynthesisRequest`` the adapter actually received.
"""
import types
import unittest
from unittest.mock import MagicMock

import numpy as np

from phoonnx.config import Alphabet, SynthesisConfig
from phoonnx.engines.base import AdapterSynthesisResult
from phoonnx.voice import TTSVoice


def _make_voice():
    """A TTSVoice with a mocked adapter and no real ONNX/phonemizer work.

    ``config.alphabet`` doesn't matter here -- ``phoneme_ids_to_audio`` builds
    ``request.params`` before any alphabet dispatch, straight from
    ``syn_config``, so a plain phoneme-model config exercises the same code.
    """
    voice = TTSVoice.__new__(TTSVoice)
    voice.config = types.SimpleNamespace(
        alphabet=Alphabet.IPA,
        lang_code="en",
        sample_rate=22050,
        noise_scale=None,
        length_scale=None,
        noise_w_scale=None,
        engine_params={},
        hop_length=256,
    )
    voice.adapter = MagicMock()
    voice.adapter.default_params.return_value = {}
    voice.adapter.synthesize.return_value = AdapterSynthesisResult(
        audio=np.zeros(4, dtype=np.float32))
    voice._alignment_session = None
    voice.session = MagicMock()
    # the reference transcription also feeds _prompt_token_ids (in-context
    # engines' phoneme prompt) via the voice's own phonemizer -- stub both.
    voice.phonemize = MagicMock(return_value=[["r", "e", "f"]])
    voice.phonemes_to_ids = MagicMock(return_value=[1, 2, 3])
    return voice


class TestSpeakerReferenceTextReachesRequestParams(unittest.TestCase):
    def test_speaker_reference_text_lands_in_request_params(self):
        voice = _make_voice()
        syn_config = SynthesisConfig(speaker_reference_text="A reference clip transcript")

        voice.phoneme_ids_to_audio([1, 2, 3], syn_config=syn_config)

        request = voice.adapter.synthesize.call_args[0][0]
        self.assertEqual(request.params.get("speaker_reference_text"),
                         "A reference clip transcript")

    def test_absent_when_no_reference_text_was_given(self):
        voice = _make_voice()
        voice.phoneme_ids_to_audio([1, 2, 3], syn_config=SynthesisConfig())

        request = voice.adapter.synthesize.call_args[0][0]
        self.assertNotIn("speaker_reference_text", request.params)

    def test_each_chunk_of_a_multi_chunk_call_sees_its_own_fresh_params(self):
        """phoneme_ids_to_audio is called once per sentence/chunk by
        TTSVoice.synthesize; params (including speaker_reference_text) are
        rebuilt from syn_config on every call, not shared/mutated in place."""
        voice = _make_voice()
        syn_config = SynthesisConfig(speaker_reference_text="Shared reference text")

        voice.phoneme_ids_to_audio([1, 2], syn_config=syn_config)
        voice.phoneme_ids_to_audio([3, 4], syn_config=syn_config)

        self.assertEqual(voice.adapter.synthesize.call_count, 2)
        first_request = voice.adapter.synthesize.call_args_list[0][0][0]
        second_request = voice.adapter.synthesize.call_args_list[1][0][0]
        for request in (first_request, second_request):
            self.assertEqual(request.params.get("speaker_reference_text"),
                             "Shared reference text")
        # each chunk got its own params dict -- mutating one must never affect the other
        self.assertIsNot(first_request.params, second_request.params)


if __name__ == "__main__":
    unittest.main()
