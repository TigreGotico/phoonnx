"""Cloning parameters have to survive the trip from the request to the engine.

The plugin manager forwards only the arguments it can see in ``get_tts``'s
signature — it filters everything else out before the plugin is called. A
cloning engine therefore never saw ``speaker_reference`` and failed with
"needs a reference clip", even though the caller had sent one, because the
plugin did not declare the parameter.
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
                         load=MagicMock(return_value=MagicMock()))),
        patch.object(PhoonnxTTSPlugin, "_providers", return_value=None),
    ]
    for pat in patchers:
        pat.start()
        testcase.addCleanup(pat.stop)
    return PhoonnxTTSPlugin(config=dict(config))


class TestTheManagerCanSeeTheParameters(unittest.TestCase):
    """The filter is by parameter NAME, so **kwargs would not have worked."""

    def test_cloning_parameters_are_named_in_the_signature(self):
        from phoonnx.opm import PhoonnxTTSPlugin
        params = inspect.signature(PhoonnxTTSPlugin.get_tts).parameters
        for name in ("speaker_reference", "speaker_reference_text",
                     "speaker_reference_lang", "ref_wav", "ref_text",
                     "ref_lang"):
            self.assertIn(name, params,
                          f"{name} must be declared or the plugin manager "
                          f"drops it before get_tts is called")

    def test_the_manager_forwards_what_the_signature_declares(self):
        # The real filter, reproduced: this is what OVOSPluginManager does.
        from phoonnx.opm import PhoonnxTTSPlugin
        sent = {"voice": "chatterbox/base/en", "lang": "en-US",
                "speaker_reference": "/tmp/ref.wav", "bogus_param": "x"}
        allowed = {k: v for k, v in sent.items()
                   if k in inspect.signature(PhoonnxTTSPlugin.get_tts).parameters}
        self.assertIn("speaker_reference", allowed)
        self.assertNotIn("bogus_param", allowed,
                         "unknown parameters must still be filtered out")


class TestPerRequestOverridesConfig(unittest.TestCase):

    def _captured_config(self, plugin, **call_kwargs):
        with patch("phoonnx.opm.SynthesisConfig") as cfg, \
                patch("phoonnx.opm.wave.open"):
            plugin.get_tts("hello", "/tmp/out.wav", **call_kwargs)
        return cfg.call_args.kwargs

    def test_the_request_value_reaches_the_engine(self):
        plugin = _plugin(self)
        got = self._captured_config(plugin, speaker_reference="/tmp/mine.wav")
        self.assertEqual(got["speaker_reference"], "/tmp/mine.wav")

    def test_the_request_beats_the_configured_default(self):
        # One server, a different cloned voice per request.
        plugin = _plugin(self, speaker_reference="/etc/configured.wav")
        got = self._captured_config(plugin, speaker_reference="/tmp/mine.wav")
        self.assertEqual(got["speaker_reference"], "/tmp/mine.wav")

    def test_the_configured_default_still_applies_when_nothing_is_sent(self):
        plugin = _plugin(self, speaker_reference="/etc/configured.wav")
        got = self._captured_config(plugin)
        self.assertEqual(got["speaker_reference"], "/etc/configured.wav")

    def test_short_aliases_work(self):
        plugin = _plugin(self)
        got = self._captured_config(plugin, ref_wav="/tmp/a.wav",
                                    ref_text="olá", ref_lang="pt")
        self.assertEqual(got["speaker_reference"], "/tmp/a.wav")
        self.assertEqual(got["speaker_reference_text"], "olá")
        self.assertEqual(got["speaker_reference_lang"], "pt")

    def test_the_long_name_wins_over_its_alias(self):
        plugin = _plugin(self)
        got = self._captured_config(plugin, speaker_reference="/tmp/long.wav",
                                    ref_wav="/tmp/short.wav")
        self.assertEqual(got["speaker_reference"], "/tmp/long.wav")


class TestClonedAudioIsNotSharedBetweenReferences(unittest.TestCase):
    """Two people cloned onto the same sentence must not collide.

    The cache identifies a voice by ``plugin_id/voice/lang`` and keys audio by
    the sentence, so without the reference in that identity the second caller
    is served the first caller's cloned audio — the wrong voice, and a leak of
    someone else's speech, delivered as a perfectly good WAV.
    """

    def _ctxt(self, plugin, **kwargs):
        return plugin._get_ctxt(kwargs)

    def test_different_references_get_different_cache_identities(self):
        plugin = _plugin(self)
        a = self._ctxt(plugin, voice="chatterbox/base/en",
                       speaker_reference="/tmp/alice.wav")
        b = self._ctxt(plugin, voice="chatterbox/base/en",
                       speaker_reference="/tmp/bob.wav")
        self.assertNotEqual(a.tts_id, b.tts_id,
                            "alice and bob must not share a cache entry")

    def test_the_same_reference_still_shares_its_cache(self):
        plugin = _plugin(self)
        a = self._ctxt(plugin, voice="v", speaker_reference="/tmp/alice.wav")
        b = self._ctxt(plugin, voice="v", speaker_reference="/tmp/alice.wav")
        self.assertEqual(a.tts_id, b.tts_id,
                         "caching must still work for identical requests")

    def test_a_reference_transcription_also_separates_the_cache(self):
        # In-context engines condition on the transcription too, so the same
        # clip with a different transcription is different audio.
        plugin = _plugin(self)
        a = self._ctxt(plugin, voice="v", speaker_reference="/tmp/a.wav",
                       speaker_reference_text="ola")
        b = self._ctxt(plugin, voice="v", speaker_reference="/tmp/a.wav",
                       speaker_reference_text="hello")
        self.assertNotEqual(a.tts_id, b.tts_id)

    def test_requests_without_a_reference_are_unaffected(self):
        plugin = _plugin(self)
        plain = self._ctxt(plugin, voice="piper/en_US-amy-low")
        self.assertNotIn("#", plain.tts_id,
                         "an ordinary voice keeps its plain cache identity")

    def test_the_real_voice_still_reaches_get_tts(self):
        # Only the cache identity is decorated; the model to load must not be.
        plugin = _plugin(self)
        ctxt = self._ctxt(plugin, voice="chatterbox/base/en",
                          speaker_reference="/tmp/alice.wav")
        self.assertEqual(ctxt.synth_kwargs["voice"], "chatterbox/base/en")


class TestReferenceFromAUrl(unittest.TestCase):
    """A remote caller cannot put a file on the server's disk."""

    def test_a_url_is_downloaded_and_read(self):
        import numpy as np
        from phoonnx.voice import _load_reference_audio

        wav = self._wav_bytes()
        with patch("requests.get") as get:
            get.return_value.__enter__.return_value = self._response(wav)
            audio, sr = _load_reference_audio("https://example.org/ref.wav")
        self.assertEqual(sr, 16000)
        self.assertGreater(len(audio), 0)
        self.assertEqual(np.asarray(audio).ndim, 1)

    def test_an_oversized_download_is_refused(self):
        from phoonnx.voice import _load_reference_audio
        with patch("requests.get") as get:
            get.return_value.__enter__.return_value = self._response(
                b"x" * (33 * 1024 * 1024))
            with self.assertRaises(ValueError) as caught:
                _load_reference_audio("https://example.org/huge.wav")
        self.assertIn("larger than", str(caught.exception))

    def test_a_plain_path_is_still_read_directly(self):
        import tempfile, os
        from phoonnx.voice import _load_reference_audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(self._wav_bytes())
            path = f.name
        self.addCleanup(lambda: os.unlink(path))
        with patch("requests.get") as get:
            audio, sr = _load_reference_audio(path)
        get.assert_not_called()
        self.assertEqual(sr, 16000)

    @staticmethod
    def _wav_bytes():
        import io, wave, struct
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(struct.pack("<1600h", *([1000] * 1600)))
        return buf.getvalue()

    @staticmethod
    def _response(payload: bytes):
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.iter_content.return_value = [payload]
        # Not a redirect: the fetcher follows those itself so it can check
        # each hop, and a MagicMock is truthy.
        r.is_redirect = False
        r.is_permanent_redirect = False
        return r

    def setUp(self):
        # The host check does real DNS; the addresses it allows are covered
        # by TestTheFetchRefusesTheInsideOfTheNetwork.
        patcher = patch("phoonnx.voice._check_reference_url")
        patcher.start()
        self.addCleanup(patcher.stop)


class TestTheFetchRefusesTheInsideOfTheNetwork(unittest.TestCase):
    """The URL comes from the caller, and the server sits on a private network.

    Without a check, a public TTS endpoint will fetch from its own loopback,
    its container neighbours, or a cloud metadata service on the caller's
    behalf and report back whether it worked.
    """

    def test_loopback_is_refused(self):
        from phoonnx.voice import _check_reference_url
        with self.assertRaises(ValueError) as caught:
            _check_reference_url("http://127.0.0.1:9666/ref.wav")
        self.assertIn("inside this network", str(caught.exception))

    def test_a_private_range_is_refused(self):
        from phoonnx.voice import _check_reference_url
        for host in ("10.0.0.5", "192.168.1.10", "172.16.0.3",
                     "169.254.169.254"):
            with self.subTest(host=host):
                with self.assertRaises(ValueError):
                    _check_reference_url(f"http://{host}/ref.wav")

    def test_a_non_http_scheme_is_refused(self):
        from phoonnx.voice import _check_reference_url
        for url in ("file:///etc/passwd", "ftp://example.org/a.wav",
                    "gopher://example.org/a.wav"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    _check_reference_url(url)

    def test_a_public_address_is_allowed(self):
        from phoonnx.voice import _check_reference_url
        with patch("phoonnx.voice.socket.getaddrinfo",
                   return_value=[(2, 1, 6, "", ("93.184.216.34", 80))]):
            _check_reference_url("https://example.org/ref.wav")

    def test_a_redirect_into_the_network_is_refused(self):
        """A public URL that redirects inward is the easy way around a check
        that only looks at the address the caller typed."""
        from phoonnx import voice

        redirect = MagicMock()
        redirect.is_redirect = True
        redirect.is_permanent_redirect = False
        redirect.url = "https://example.org/ref.wav"
        redirect.headers = {"Location": "http://169.254.169.254/latest/meta-data"}

        def resolve(host, *a, **kw):
            # Only the outward-looking name resolves publicly; the redirect
            # target is a literal address and resolves to itself.
            if host == "example.org":
                return [(2, 1, 6, "", ("93.184.216.34", 80))]
            return [(2, 1, 6, "", (host, 80))]

        with patch("requests.get") as get:
            get.return_value.__enter__.return_value = redirect
            with patch("phoonnx.voice.socket.getaddrinfo", side_effect=resolve):
                with self.assertRaises(ValueError) as caught:
                    voice._fetch_reference_audio("https://example.org/ref.wav")
        self.assertIn("inside this network", str(caught.exception))


class TestTheFetchGivesUp(unittest.TestCase):

    def test_a_trickle_cannot_hold_the_worker_forever(self):
        """requests' timeout is per read, so a sender dripping one byte at a
        time resets it endlessly and keeps a synthesis worker occupied."""
        import time
        from phoonnx import voice

        def dribble(chunk_size=None):
            while True:
                time.sleep(0.05)
                yield b"x"

        response = MagicMock()
        response.raise_for_status.return_value = None
        response.is_redirect = False
        response.is_permanent_redirect = False
        response.iter_content.side_effect = dribble

        with patch("requests.get") as get, \
                patch("phoonnx.voice._check_reference_url"):
            get.return_value.__enter__.return_value = response
            started = time.monotonic()
            with self.assertRaises(ValueError) as caught:
                voice._fetch_reference_audio("https://example.org/slow.wav",
                                             deadline=0.5)
            elapsed = time.monotonic() - started
        self.assertIn("longer than", str(caught.exception))
        self.assertLess(elapsed, 5, "the deadline must actually end it")


class TestClonedCachesAreBounded(unittest.TestCase):

    def test_a_flood_of_references_does_not_grow_without_limit(self):
        """Each distinct reference makes a cache identity, and the reference
        comes from the request, so the count is whatever a caller decides."""
        from ovos_plugin_manager.templates.tts import TTSContext
        plugin = _plugin(self)
        for i in range(plugin.MAX_CLONED_CACHES * 3):
            plugin._get_ctxt({"voice": "v", "speaker_reference": f"/tmp/{i}.wav"})
        self.assertLessEqual(len(plugin._cloned_caches),
                             plugin.MAX_CLONED_CACHES)

    def test_the_newest_identities_are_the_ones_kept(self):
        plugin = _plugin(self)
        ids = []
        for i in range(plugin.MAX_CLONED_CACHES + 5):
            ctxt = plugin._get_ctxt({"voice": "v",
                                     "speaker_reference": f"/tmp/{i}.wav"})
            ids.append(ctxt.tts_id)
        self.assertIn(ids[-1], plugin._cloned_caches)
        self.assertNotIn(ids[0], plugin._cloned_caches)


if __name__ == "__main__":
    unittest.main()
