"""A synthesis that fails must not leave anything behind.

The plugin is handed the path the cache expects the finished audio at, and
the cache decides whether a sentence is already synthesized by whether that
file exists. Creating it before the audio exists means a failure leaves a
valid, empty WAV — 44 bytes of header — in the cache. Every later request for
that sentence is then answered with silence and HTTP 200, and because the file
is there, nothing ever tries again. It is the worst shape a bug can take: the
caller is told it worked.
"""
import os
import tempfile
import unittest
import wave
from unittest.mock import MagicMock, patch


def _plugin(testcase, synth_side_effect=None):
    from phoonnx.opm import PhoonnxTTSPlugin

    model = MagicMock()
    if synth_side_effect is not None:
        model.synthesize_wav.side_effect = synth_side_effect
    else:
        def write_some_audio(sentence, wav_out, params):
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2)
            wav_out.setframerate(22050)
            wav_out.writeframes(b"\x01\x02" * 2000)
        model.synthesize_wav.side_effect = write_some_audio

    patchers = [
        patch("phoonnx.opm.TTSModelManager"),
        patch.object(PhoonnxTTSPlugin, "get_default_voice", return_value=MagicMock()),
        patch.object(PhoonnxTTSPlugin, "get_voice_info",
                     side_effect=lambda v: MagicMock(
                         load=MagicMock(return_value=model))),
        patch.object(PhoonnxTTSPlugin, "_providers", return_value=None),
        patch.object(PhoonnxTTSPlugin, "_resolve_speaker", return_value=None),
    ]
    for pat in patchers:
        pat.start()
        testcase.addCleanup(pat.stop)
    return PhoonnxTTSPlugin(config={})


class TestFailedSynthesisLeavesNothing(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.target = os.path.join(self.dir, "audio.wav")

    def test_a_failure_does_not_leave_a_file_the_cache_would_serve(self):
        plugin = _plugin(self, synth_side_effect=RuntimeError("engine exploded"))
        with self.assertRaises(RuntimeError):
            plugin.get_tts("hello", self.target)
        self.assertFalse(os.path.exists(self.target),
                         "an empty WAV here is served as success forever")

    def test_the_engine_error_is_what_the_caller_sees(self):
        """Closing an unwritten WAV raises, and would hide the real cause.

        `wave` refuses to close a file it was never given parameters for, so
        the engine's own exception was replaced by "# channels not specified"
        and the reason for the failure never reached the log.
        """
        plugin = _plugin(self, synth_side_effect=RuntimeError("engine exploded"))
        with self.assertRaises(RuntimeError) as caught:
            plugin.get_tts("hello", self.target)
        self.assertIn("engine exploded", str(caught.exception))

    def test_a_failure_leaves_no_partial_file_either(self):
        plugin = _plugin(self, synth_side_effect=RuntimeError("boom"))
        with self.assertRaises(RuntimeError):
            plugin.get_tts("hello", self.target)
        self.assertEqual(os.listdir(self.dir), [],
                         "no leftovers, including the temporary file")

    def test_an_interrupt_is_cleaned_up_too(self):
        # KeyboardInterrupt/SystemExit do not derive from Exception, and a
        # server being shut down mid-synthesis is exactly when this happens.
        plugin = _plugin(self, synth_side_effect=KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            plugin.get_tts("hello", self.target)
        self.assertEqual(os.listdir(self.dir), [])

    def test_a_successful_synthesis_still_writes_the_audio(self):
        plugin = _plugin(self)
        out, _ = plugin.get_tts("hello", self.target)
        self.assertEqual(out, self.target)
        self.assertTrue(os.path.exists(self.target))
        with wave.open(self.target) as w:
            self.assertGreater(w.getnframes(), 0, "must be real audio")
        self.assertEqual(os.listdir(self.dir), ["audio.wav"],
                         "the temporary file must not survive success")

    def test_a_failure_does_not_destroy_previously_cached_audio(self):
        # The retry of a sentence that already synthesized once must not lose
        # the good audio if the retry fails.
        plugin = _plugin(self)
        plugin.get_tts("hello", self.target)
        good = os.path.getsize(self.target)

        plugin2 = _plugin(self, synth_side_effect=RuntimeError("later failure"))
        with self.assertRaises(RuntimeError):
            plugin2.get_tts("hello", self.target)
        self.assertTrue(os.path.exists(self.target))
        self.assertEqual(os.path.getsize(self.target), good,
                         "the good audio must survive a failed retry")


class TestATruncatedWriteIsNotPublished(unittest.TestCase):
    """A disk that fills up must not look like a finished synthesis.

    The last write of a WAV happens when the file is closed, so a full disk
    surfaces there and nowhere else. Swallowing that error would move a
    half-written file into the cache and return it as success — silence
    served with a 200, permanently, which is the failure this whole change
    exists to prevent.
    """

    def test_an_error_while_closing_is_not_swallowed(self):
        import os
        import tempfile
        from phoonnx.opm import PhoonnxTTSPlugin

        directory = tempfile.mkdtemp()
        target = os.path.join(directory, "audio.wav")

        class ShortWriter:
            """Accepts the audio, then fails on the final flush, as ENOSPC does."""

            def __init__(self, *a, **kw):
                pass

            def setnchannels(self, *a): pass
            def setsampwidth(self, *a): pass
            def setframerate(self, *a): pass
            def writeframes(self, *a): pass

            def close(self):
                raise OSError(28, "No space left on device")

        def write_audio(sentence, wav_out, params):
            wav_out.setnchannels(1)
            wav_out.writeframes(b"\x01\x02" * 100)

        model = MagicMock()
        model.synthesize_wav.side_effect = write_audio
        patchers = [
            patch("phoonnx.opm.TTSModelManager"),
            patch.object(PhoonnxTTSPlugin, "get_default_voice", return_value=MagicMock()),
            patch.object(PhoonnxTTSPlugin, "get_voice_info",
                     side_effect=lambda v: MagicMock(
                         load=MagicMock(return_value=model))),
            patch.object(PhoonnxTTSPlugin, "_providers", return_value=None),
            patch.object(PhoonnxTTSPlugin, "_resolve_speaker", return_value=None),
            patch("phoonnx.opm.wave.open", ShortWriter),
        ]
        for pat in patchers:
            pat.start()
            self.addCleanup(pat.stop)

        plugin = PhoonnxTTSPlugin(config={})
        with self.assertRaises(OSError):
            plugin.get_tts("hello", target)
        self.assertFalse(os.path.exists(target),
                         "a truncated file must never be published as finished")
        self.assertEqual(os.listdir(directory), [], "and nothing is left behind")


class TestConcurrentRequestsForTheSameSentence(unittest.TestCase):

    def test_neither_caller_fails_because_they_shared_a_temp_name(self):
        """A shared "<target>.part" made the slower caller fail with ENOENT.

        Two requests for the same sentence and voice land on the same
        destination, so a temporary name derived from it collided: whichever
        finished second found its file already renamed away.
        """
        import os
        import tempfile
        import threading

        directory = tempfile.mkdtemp()
        target = os.path.join(directory, "audio.wav")
        plugin = _plugin(self)
        errors = []

        def go():
            try:
                plugin.get_tts("the same sentence", target)
            except BaseException as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=go, daemon=True) for _ in range(2)]
        [t.start() for t in threads]
        [t.join(timeout=15) for t in threads]

        self.assertEqual(errors, [], "neither caller may fail")
        self.assertTrue(os.path.exists(target))
        self.assertEqual(os.listdir(directory), ["audio.wav"],
                         "no temporary file may survive")


if __name__ == "__main__":
    unittest.main()
