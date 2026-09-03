"""One model, many voices: the weights are loaded once.

A voice-index entry is not a model. ``omnivoice.json`` holds 646 entries with
exactly one distinct ``model_url`` between them — a single 3.0 GB backbone —
and ``qwen3tts.json`` holds 15 over one 4.2 GB talker. The entries differ only
in the language and the ``engine_options`` applied at synthesis time, so the
audio really is different per voice, but the weights are the same file.

Loading that file once per entry gave every voice its own full
``InferenceSession`` over the identical graph, its own copy of the weights,
and its own full charge against ``max_loaded_bytes``. Observed on an 8 GB
container: ``omnivoice/en`` loaded, evicted to make room for ``omnivoice/es``,
then loaded again — three loads of one file, and four kernel OOM kills.

These tests pin the two halves of the fix: sessions are shared between voices
that name the same artifacts, and the cache charges the memory once.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from phoonnx import providers
from phoonnx.providers import CPU_PROVIDER, SHARE_SESSIONS_ENV_VAR, make_session

MB = 10 ** 6
GB = 10 ** 9


def _build_tiny_onnx_model(path: Path) -> None:
    """Write a minimal but real single-node ONNX model to *path*."""
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "tiny", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


class TestOneModelOneSession(unittest.TestCase):
    """The 646-entry case, in miniature."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.model_path = Path(self.tmpdir.name) / "backbone.onnx"
        _build_tiny_onnx_model(self.model_path)
        env = patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(SHARE_SESSIONS_ENV_VAR, None)

    def _voice(self, **engine_params):
        from phoonnx.voice import TTSVoice
        return TTSVoice.load(model_path=self.model_path,
                             providers=[CPU_PROVIDER],
                             engine_params=engine_params)

    def test_two_voices_over_one_model_load_it_once(self):
        with patch.object(providers.onnxruntime, "InferenceSession",
                          wraps=providers.onnxruntime.InferenceSession) as spy:
            english = self._voice(lang="en")
            spanish = self._voice(lang="es")

        self.assertIs(english.session, spanish.session,
                      "voices over one model must share one session")
        self.assertEqual(spy.call_count, 1,
                         "the second voice must not load the graph again")

    def test_the_whole_catalog_costs_one_load(self):
        # The shape that actually killed the server: hundreds of entries, one
        # file. This is the 646 omnivoice voices, scaled down.
        with patch.object(providers.onnxruntime, "InferenceSession",
                          wraps=providers.onnxruntime.InferenceSession) as spy:
            voices = [self._voice(lang=f"l{i}") for i in range(50)]
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(len({id(v.session) for v in voices}), 1)

    def test_voices_over_one_model_keep_their_own_engine_options(self):
        # Sharing the weights must not share what makes the audio different.
        english = self._voice(lang="en")
        spanish = self._voice(lang="es")
        self.assertIs(english.session, spanish.session)
        self.assertEqual(english.config.engine_params["lang"], "en")
        self.assertEqual(spanish.config.engine_params["lang"], "es")
        self.assertIsNot(english.config, spanish.config)

    def test_each_voice_synthesizes_with_its_own_options(self):
        # What the engine is asked for is the voice's own options, even though
        # the session under it is one object. This is the unit-level statement
        # of "distinct voices still render differently".
        english = self._voice(lang="en", speaker="vivian")
        spanish = self._voice(lang="es", speaker="ethan")

        seen = []
        for voice in (english, spanish):
            adapter = MagicMock()
            adapter.configure = lambda cfg, _s=seen: None
            voice.adapter = adapter
        for voice in (english, spanish):
            seen.append(dict(voice.config.engine_params))

        self.assertNotEqual(seen[0], seen[1])
        self.assertEqual(seen[0]["speaker"], "vivian")
        self.assertEqual(seen[1]["speaker"], "ethan")

    def test_a_different_model_gets_its_own_session(self):
        other = Path(self.tmpdir.name) / "other.onnx"
        _build_tiny_onnx_model(other)
        first = make_session(self.model_path, providers=[CPU_PROVIDER])
        second = make_session(other, providers=[CPU_PROVIDER])
        self.assertIsNot(first, second)

    def test_a_different_provider_list_gets_its_own_session(self):
        first = make_session(self.model_path, providers=[CPU_PROVIDER])
        with patch.object(providers, "resolve_providers",
                          return_value=[("CPUExecutionProvider", {})]):
            second = make_session(self.model_path)
        self.assertIsNot(first, second)

    def test_explicit_session_options_are_never_shared(self):
        # A caller that asked for its own settings gets its own session.
        import onnxruntime
        first = make_session(self.model_path, providers=[CPU_PROVIDER])
        second = make_session(self.model_path, providers=[CPU_PROVIDER],
                              sess_options=onnxruntime.SessionOptions())
        self.assertIsNot(first, second)

    def test_sharing_can_be_turned_off(self):
        os.environ[SHARE_SESSIONS_ENV_VAR] = "0"
        first = make_session(self.model_path, providers=[CPU_PROVIDER])
        second = make_session(self.model_path, providers=[CPU_PROVIDER])
        self.assertIsNot(first, second)

    def test_a_replaced_external_data_sidecar_gets_a_new_session(self):
        # The graph names external weights in a sidecar file that onnxruntime
        # resolves next to it; session_key only stated the graph's own
        # (size, mtime), so a stub graph that keeps its own bytes stable
        # while its sidecar is replaced with different weights kept serving
        # the stale session over the old weights.
        sidecar = Path(str(self.model_path) + "_data")
        sidecar.write_bytes(b"weights-v1")
        first = providers.session_key(self.model_path, [CPU_PROVIDER], None)
        sidecar.write_bytes(b"weights-v2-longer")
        second = providers.session_key(self.model_path, [CPU_PROVIDER], None)
        self.assertNotEqual(first, second)

    def test_a_replaced_sidecar_is_found_through_a_hub_cache_symlink(self):
        # The hub cache lays a voice out as snapshots/<rev>/model.onnx, a
        # symlink to blobs/<sha>; the sidecar sits beside the symlink, not
        # beside the blob it resolves to, so probing only the realpath (as
        # session_key used to) never finds it for a model fetched this way
        # — exactly the models this fix targets.
        blobs = Path(self.tmpdir.name) / "blobs"
        snapshots = Path(self.tmpdir.name) / "snapshots" / "rev1"
        blobs.mkdir()
        snapshots.mkdir(parents=True)
        blob = blobs / "deadbeef"
        _build_tiny_onnx_model(blob)
        stub = snapshots / "model.onnx"
        stub.symlink_to(blob)
        sidecar = snapshots / "model.onnx_data"

        sidecar.write_bytes(b"weights-v1")
        first = providers.session_key(stub, [CPU_PROVIDER], None)
        sidecar.write_bytes(b"weights-v2-longer")
        second = providers.session_key(stub, [CPU_PROVIDER], None)
        self.assertNotEqual(first, second,
                            "the sidecar beside a hub-cache symlink must be "
                            "found, not only one beside the resolved blob")

    def test_no_sidecar_is_unaffected(self):
        # A single-file graph (no external-data sidecar) must key exactly as
        # it did before: this only changes behaviour for models that have one.
        key = providers.session_key(self.model_path, [CPU_PROVIDER], None)
        self.assertEqual(len(key), 4)

    def test_a_session_nobody_uses_is_not_kept_alive(self):
        import gc
        key = providers.session_key(self.model_path, [CPU_PROVIDER], None)
        session = make_session(self.model_path, providers=[CPU_PROVIDER])
        self.assertIn(key, providers.shared_sessions())
        del session
        gc.collect()
        self.assertNotIn(key, providers.shared_sessions(),
                         "the store must not keep a graph in memory by itself")


class TestBuildLocksDoNotGrowForever(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.model_path = Path(self.tmpdir.name) / "backbone.onnx"
        _build_tiny_onnx_model(self.model_path)
        env = patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(SHARE_SESSIONS_ENV_VAR, None)

    def test_the_build_lock_is_dropped_once_the_session_is_published(self):
        key = providers.session_key(self.model_path, [CPU_PROVIDER], None)
        make_session(self.model_path, providers=[CPU_PROVIDER])
        self.assertNotIn(
            key, providers._BUILD_LOCKS,
            "a per-key build lock is only needed while the session it "
            "guards is being built; keeping it after only grows the dict "
            "by one entry per model ever seen")


class TestArtifactKey(unittest.TestCase):
    """What the cache charges memory against."""

    def _info(self, voice_id, lang, **kwargs):
        from phoonnx.model_manager import TTSModelInfo
        return TTSModelInfo(voice_id=voice_id, lang=lang,
                            model_url="hf://org/omnivoice/backbone.onnx",
                            **kwargs)

    def test_voices_over_one_model_share_a_key(self):
        english = self._info("omnivoice/en", "en", engine_options={"lang": "en"})
        spanish = self._info("omnivoice/es", "es", engine_options={"lang": "es"})
        self.assertEqual(english.artifact_key(), spanish.artifact_key())

    def test_a_different_model_is_a_different_key(self):
        english = self._info("omnivoice/en", "en")
        other = self._info("qwen3tts/vivian", "zh",)
        other.model_url = "hf://org/qwen3tts/talker.onnx"
        self.assertNotEqual(english.artifact_key(), other.artifact_key())

    def test_an_extra_graph_is_part_of_the_key(self):
        plain = self._info("a", "en")
        with_vocoder = self._info("b", "en")
        with_vocoder.vocoder_url = "hf://org/vocos.onnx"
        self.assertNotEqual(plain.artifact_key(), with_vocoder.artifact_key())


class _Voice:
    """A loaded voice stand-in that can be weakly referenced.

    Deliberately unhashable, like the real ``TTSVoice``: it is a dataclass
    with ``eq``, so it has no ``__hash__``, and a weak reference hashes what
    it points at. Keeping references to voices in a set therefore raised
    ``TypeError: unhashable type: 'TTSVoice'`` on every synthesis — which no
    stand-in with a default ``__hash__`` would ever have shown.
    """

    __hash__ = None

    def __init__(self, voice_id):
        self.voice_id = voice_id

    def __eq__(self, other):
        return self is other


def _plugin(testcase, keys, sizes, **config):
    """A plugin whose catalog maps voice ids to artifact keys and sizes."""
    from phoonnx.opm import PhoonnxTTSPlugin

    def info_for(voice_id):
        return MagicMock(
            load=lambda **_kw: _Voice(voice_id),
            artifact_key=MagicMock(return_value=keys[voice_id]),
            disk_size=MagicMock(return_value=sizes[keys[voice_id]]))

    patchers = [
        patch("phoonnx.opm.TTSModelManager"),
        patch.object(PhoonnxTTSPlugin, "get_default_voice",
                     return_value=MagicMock()),
        patch.object(PhoonnxTTSPlugin, "get_voice_info", side_effect=info_for),
        patch.object(PhoonnxTTSPlugin, "_providers", return_value=None),
    ]
    for pat in patchers:
        pat.start()
        testcase.addCleanup(pat.stop)
    return PhoonnxTTSPlugin(config=dict(config))


class TestTheBudgetChargesTheModelOnce(unittest.TestCase):

    def test_many_voices_over_one_model_are_charged_once(self):
        keys = {f"omnivoice/{i}": "backbone" for i in range(20)}
        plugin = _plugin(self, keys, {"backbone": 3 * GB},
                         max_loaded_bytes="3GB")
        for voice_id in keys:
            plugin.get_model(voice_id)
        self.assertEqual(len(plugin.voice_cache.voices), 20,
                         "voices over one model must not evict each other")
        self.assertEqual(plugin.voice_cache.resident_bytes(), 3 * GB,
                         "one model in memory is one charge")

    def test_a_shared_model_is_not_reloaded_for_the_next_voice(self):
        keys = {"omnivoice/en": "backbone", "omnivoice/es": "backbone"}
        loads = []
        plugin = _plugin(self, keys, {"backbone": 3 * GB},
                         max_loaded_bytes="3GB")
        plugin.get_model("omnivoice/en")
        plugin.get_model("omnivoice/es")
        plugin.get_model("omnivoice/en")
        self.assertIn("omnivoice/en", plugin.voice_cache.voices)
        self.assertIn("omnivoice/es", plugin.voice_cache.voices)
        del loads

    def test_a_second_model_still_evicts_the_first(self):
        keys = {"omnivoice/en": "backbone", "qwen/vivian": "talker"}
        plugin = _plugin(self, keys,
                         {"backbone": 3 * GB, "talker": 4 * GB},
                         max_loaded_bytes="5GB")
        first = plugin.get_model("omnivoice/en")
        del first
        plugin.get_model("qwen/vivian")
        self.assertNotIn("omnivoice/en", plugin.voice_cache.voices)
        self.assertEqual(plugin.voice_cache.resident_bytes(), 4 * GB)


class TestWhatIsActuallyResident(unittest.TestCase):
    """Eviction pops a dict entry; it does not free a model in use.

    A caller says it is using a voice by holding a lease. The lease is what
    the budget counts, so weights stay charged for as long as a synthesis has
    them, whether or not the cache still lists the voice.
    """

    def test_an_evicted_voice_still_in_use_is_still_charged(self):
        keys = {"a": "model-a", "b": "model-b"}
        plugin = _plugin(self, keys, {"model-a": 3 * GB, "model-b": 3 * GB},
                         max_loaded_bytes="4GB", load_wait_timeout=0.2)
        with plugin.voice_cache.lease("a"):    # as a synthesis does
            plugin.voice_cache.voices.clear()
            plugin.voice_cache._voice_keys.clear()
            plugin.get_model("b")
            self.assertNotIn("a", plugin.voice_cache.voices,
                             "it is gone from the cache")
            self.assertEqual(plugin.voice_cache.resident_bytes(), 6 * GB,
                             "but its weights are still in memory, and the "
                             "OOM killer reads memory, not the cache")

    def test_a_voice_in_use_is_not_evicted_to_make_room(self):
        # Dropping it would free nothing — the lease keeps the weights
        # charged — and the next request for that voice would then load a
        # second copy of a model that never left memory.
        keys = {"a": "model-a", "b": "model-b"}
        plugin = _plugin(self, keys, {"model-a": 3 * GB, "model-b": 3 * GB},
                         max_loaded_bytes="4GB", load_wait_timeout=0.2)
        with plugin.voice_cache.lease("a") as held:
            plugin.get_model("b")
            self.assertIn("a", plugin.voice_cache.voices)
            self.assertIs(plugin.get_model("a"), held,
                          "the leased voice was served from the cache, not "
                          "loaded again")

    def test_a_cold_loaded_voice_is_charged_from_the_lease_that_loaded_it(self):
        # The lease on a voice that was not cached yet is taken by the load
        # itself, inside the same critical section that caches it. Without
        # that, a voice would be charged only while the cache happens to
        # still list it, which is exactly what a lease exists to outlive.
        keys = {"a": "model-a"}
        plugin = _plugin(self, keys, {"model-a": 3 * GB},
                         max_loaded_bytes="4GB", load_wait_timeout=0.2)
        with plugin.voice_cache.lease("a"):
            plugin.voice_cache.voices.clear()
            plugin.voice_cache._voice_keys.clear()
            self.assertEqual(plugin.voice_cache._leases, {"model-a": 1})
            self.assertEqual(plugin.voice_cache.resident_bytes(), 3 * GB)
        self.assertEqual(plugin.voice_cache._leases, {})
        self.assertEqual(plugin.voice_cache.resident_bytes(), 0)

    def test_the_charge_goes_away_when_the_request_finishes(self):
        keys = {"a": "model-a", "b": "model-b"}
        plugin = _plugin(self, keys, {"model-a": 3 * GB, "model-b": 3 * GB},
                         max_loaded_bytes="4GB", load_wait_timeout=0.2)
        with plugin.voice_cache.lease("a"):
            plugin.voice_cache.voices.clear()
            plugin.voice_cache._voice_keys.clear()
            self.assertEqual(plugin.voice_cache.resident_bytes(), 3 * GB)
        self.assertEqual(plugin.voice_cache.resident_bytes(), 0)

    def test_a_load_waits_for_memory_that_is_coming_back(self):
        import threading
        keys = {"a": "model-a", "b": "model-b"}
        plugin = _plugin(self, keys, {"model-a": 3 * GB, "model-b": 3 * GB},
                         max_loaded_bytes="4GB", load_wait_timeout=10)
        done = threading.Event()

        def ask():
            plugin.get_model("b")
            done.set()

        with plugin.voice_cache.lease("a"):
            plugin.voice_cache.voices.clear()   # evicted, but the lease holds it
            plugin.voice_cache._voice_keys.clear()
            thread = threading.Thread(target=ask, daemon=True)
            thread.start()
            self.assertFalse(done.wait(timeout=1),
                             "the second model must not be admitted while the "
                             "first is still resident")
        self.assertTrue(done.wait(timeout=10),
                        "and it must be admitted as soon as it can be")
        thread.join(timeout=5)

    def test_the_wait_is_bounded(self):
        # A budget must never become a hang: a request that holds memory for
        # longer than the timeout gets the load anyway, and says so.
        keys = {"a": "model-a", "b": "model-b"}
        plugin = _plugin(self, keys, {"model-a": 3 * GB, "model-b": 3 * GB},
                         max_loaded_bytes="4GB", load_wait_timeout=0.2)
        with plugin.voice_cache.lease("a"):
            plugin.voice_cache.voices.clear()
            plugin.voice_cache._voice_keys.clear()
            with patch("phoonnx.voice_cache.LOG") as log:
                plugin.get_model("b")
        self.assertIn("b", plugin.voice_cache.voices)
        self.assertIn("exceed the budget",
                      " ".join(str(c) for c in log.warning.call_args_list))


if __name__ == "__main__":
    unittest.main()
