"""A memory budget for the loaded-voice cache, in bytes rather than voices.

A count cannot bound memory on a mixed catalog. Measured on a server with an
8 GiB limit and ``max_loaded_voices: 3``: four concurrent omnivoice requests
pinned the container at 8 GiB / 8 GiB and left 6.479 GiB resident, and a
five-worker sweep killed the process.

The budget bounds *peak* memory, not only the steady state, because peak is
what the OOM killer reads. A cold load reserves its bytes before it allocates
them, so concurrent loads of different voices are admitted only while they fit.
The guarantee is:

    peak resident bytes <= max(budget, pinned bytes + the largest single voice)

A single voice larger than the whole budget still loads, because refusing to
serve it is worse; it is the only way to exceed the budget, and it does so
alone. These tests pin that bound, and that the count bound still works exactly
as before.
"""
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

MB = 10 ** 6
GB = 10 ** 9


def _plugin(testcase, sizes=None, load_hook=None, **config):
    """Build the plugin with its model manager and loading stubbed out.

    ``sizes`` maps voice id -> on-disk bytes; anything not listed is 10 MB, the
    order of a small piper voice.
    """
    from phoonnx.opm import PhoonnxTTSPlugin

    sizes = sizes or {}

    def info_for(voice_id):
        def load(**_kwargs):
            if load_hook is not None:
                load_hook(voice_id)
            return f"model:{voice_id}"

        return MagicMock(load=load,
                         disk_size=MagicMock(
                             return_value=sizes.get(voice_id, 10 * MB)))

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


class TestSizeParsing(unittest.TestCase):

    def test_human_friendly_sizes(self):
        from phoonnx import voice_cache as vc
        cases = {
            "6GB": 6 * 10 ** 9,
            "512MB": 512 * 10 ** 6,
            "1.5GB": 1_500_000_000,
            "2GiB": 2 * 2 ** 30,
            "512 MiB": 512 * 2 ** 20,
            "1024": 1024,
            1024: 1024,
            "900KB": 900_000,
            "4TB": 4 * 10 ** 12,
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(vc.parse_max_bytes(value), expected)

    def test_infinite_and_nan_budgets_are_rejected_not_crashes(self):
        # YAML ".inf" and json.loads("Infinity"/"NaN") both land here as
        # floats, and int(inf) raises out of __init__, killing plugin startup.
        from phoonnx import voice_cache as vc
        for value in (float("inf"), float("-inf"), float("nan"),
                      "1e400", 1e400):
            with self.subTest(value=value):
                self.assertIsNone(vc.parse_max_bytes(value))

    def test_an_infinite_budget_does_not_break_startup(self):
        plugin = _plugin(self, max_loaded_bytes=float("inf"))
        self.assertIsNone(plugin.voice_cache.max_loaded_bytes)
        plugin.get_model("v")
        self.assertIn("v", plugin.voice_cache.voices)

    def test_unset_means_no_budget(self):
        from phoonnx import voice_cache as vc
        for value in (None, "", 0):
            with self.subTest(value=value):
                self.assertIsNone(vc.parse_max_bytes(value))

    def test_invalid_sizes_are_rejected_not_read_as_zero(self):
        # A zero budget would evict every voice on every request, so a typo
        # must leave the cache unbounded rather than unusable.
        from phoonnx import voice_cache as vc
        for value in ("nonsense", "6 gigabytes", "GB", "-1", -1, "1.2.3",
                      "6GBB", "0.4", True, [1], "12PB"):
            with self.subTest(value=value):
                self.assertIsNone(vc.parse_max_bytes(value))

    def test_an_invalid_budget_leaves_the_cache_unbounded(self):
        plugin = _plugin(self, max_loaded_bytes="nonsense")
        self.assertIsNone(plugin.voice_cache.max_loaded_bytes)
        for i in range(4):
            plugin.get_model(f"v{i}")
        self.assertEqual(len(plugin.voice_cache.voices), 4)


class TestByteBudget(unittest.TestCase):

    def test_a_voice_over_the_remaining_budget_evicts_the_lru(self):
        plugin = _plugin(self, sizes={"a": 200 * MB, "b": 200 * MB,
                                      "big": 700 * MB},
                         max_loaded_bytes="1GB")
        plugin.get_model("a")
        plugin.get_model("b")
        plugin.get_model("a")       # touch: "b" is now the oldest
        plugin.get_model("big")
        self.assertIn("big", plugin.voice_cache.voices)
        self.assertNotIn("b", plugin.voice_cache.voices)
        self.assertIn("a", plugin.voice_cache.voices)

    def test_many_small_voices_but_only_one_big_one(self):
        # The case a count gets wrong: with max_loaded_voices=3 these small
        # voices would evict for no reason, and three big ones would fit and
        # kill the server.
        plugin = _plugin(self, sizes={"omni": 2200 * MB},
                         max_loaded_bytes="3GB")
        for i in range(20):
            plugin.get_model(f"small{i}")   # 10 MB each -> 200 MB resident
        self.assertEqual(len(plugin.voice_cache.voices), 20)

        plugin.get_model("omni")
        self.assertIn("omni", plugin.voice_cache.voices)
        self.assertLessEqual(plugin.voice_cache.resident_bytes(), 3 * GB)

        # Another small voice still fits alongside it; the budget is never
        # exceeded by voices that could have been evicted.
        plugin.get_model("small20")
        self.assertIn("omni", plugin.voice_cache.voices)
        self.assertLessEqual(plugin.voice_cache.resident_bytes(), 3 * GB)

    def test_two_big_voices_cannot_be_resident_together(self):
        plugin = _plugin(self, sizes={"omniA": 2200 * MB, "omniB": 2200 * MB},
                         max_loaded_bytes="3GB")
        plugin.get_model("omniA")
        plugin.get_model("omniB")
        self.assertIn("omniB", plugin.voice_cache.voices)
        self.assertNotIn("omniA", plugin.voice_cache.voices)

    def test_a_voice_bigger_than_the_whole_budget_is_refused_not_loaded(self):
        # Loading it is not a degraded path, it is a guaranteed OOM kill; a
        # 15.4 GB voice against a 5 GB budget in an 8 GiB cgroup produced
        # exactly that, and the container restart looped straight back into
        # the same load. The known size is refused before it is ever loaded.
        from phoonnx.voice_cache import VoiceExceedsMemoryBudget
        plugin = _plugin(self, sizes={"huge": 8 * GB}, max_loaded_bytes="1GB")
        plugin.get_model("small")
        with self.assertRaises(VoiceExceedsMemoryBudget) as ctx:
            plugin.get_model("huge")
        self.assertIn("huge", str(ctx.exception))
        self.assertIn(str(8 * GB), str(ctx.exception))
        self.assertIn(str(10 ** 9), str(ctx.exception))
        self.assertNotIn("huge", plugin.voice_cache.voices)
        self.assertIn("small", plugin.voice_cache.voices,
                      "a refused load must not disturb what is already resident")
        self.assertEqual(plugin.voice_cache._voice_keys.keys() & plugin.voice_cache.voices.keys(),
                         plugin.voice_cache.voices.keys())

    def test_a_refused_voice_leaves_no_accounting_behind(self):
        from phoonnx.voice_cache import VoiceExceedsMemoryBudget
        plugin = _plugin(self, sizes={"huge": 8 * GB}, max_loaded_bytes="1GB")
        with self.assertRaises(VoiceExceedsMemoryBudget):
            plugin.get_model("huge")
        self.assertNotIn("huge", plugin.voice_cache.voices)
        self.assertNotIn("huge", plugin.voice_cache._voice_keys)
        self.assertNotIn("huge", plugin.voice_cache._key_bytes)
        self.assertEqual(plugin.voice_cache._reserved_bytes, 0)
        # A voice that actually fits still loads normally afterwards.
        model = plugin.get_model("small")
        self.assertEqual(model, "model:small")
        self.assertIn("small", plugin.voice_cache.voices)

    def test_a_voice_that_fits_alone_but_faces_pressure_still_evicts_not_refuses(self):
        # Regression guard: a voice that individually fits the budget, but not
        # alongside what is already resident, still follows the existing
        # evict-and-load path (or the bounded-wait-then-overflow path once
        # nothing is evictable, exercised in TestCyclicVoiceResidency) rather
        # than the outright refusal above, which is reserved for a voice that
        # cannot ever fit on its own.
        plugin = _plugin(self, sizes={"a": 700 * MB, "b": 700 * MB},
                         max_loaded_bytes="1GB", load_wait_timeout=0.2)
        plugin.get_model("a")
        model = plugin.get_model("b")
        self.assertEqual(model, "model:b")
        self.assertIn("b", plugin.voice_cache.voices)
        self.assertNotIn("a", plugin.voice_cache.voices,
                         "b displaces a rather than being refused")

    def test_no_budget_means_no_measuring(self):
        # Deployments that set neither bound must be untouched, and must not
        # pay for a measurement they never asked for.
        info = MagicMock(disk_size=MagicMock(return_value=1))
        plugin = _plugin(self)
        with patch.object(plugin, "get_voice_info", return_value=info):
            plugin.get_model("v")
        info.disk_size.assert_not_called()

    def test_an_unmeasurable_voice_is_still_served(self):
        plugin = _plugin(self, max_loaded_bytes="1GB")
        broken = MagicMock(load=lambda **_k: "model:broken",
                           disk_size=MagicMock(side_effect=OSError("no")))
        with patch.object(plugin, "get_voice_info", return_value=broken):
            self.assertEqual(plugin.get_model("broken"), "model:broken")
        self.assertIn("broken", plugin.voice_cache.voices)


class TestBothBoundsTogether(unittest.TestCase):

    def test_the_count_still_evicts_when_the_budget_is_roomy(self):
        plugin = _plugin(self, max_loaded_voices=2, max_loaded_bytes="100GB")
        for name in ("a", "b", "c"):
            plugin.get_model(name)
        self.assertEqual(set(plugin.voice_cache.voices), {"b", "c"})

    def test_the_budget_still_evicts_when_the_count_is_roomy(self):
        sizes = {f"small{i}": 20 * MB for i in range(10)}
        sizes["big"] = 900 * MB
        plugin = _plugin(self, max_loaded_voices=50,
                         sizes=sizes, max_loaded_bytes="1GB")
        for i in range(10):
            plugin.get_model(f"small{i}")
        plugin.get_model("big")
        self.assertIn("big", plugin.voice_cache.voices)
        self.assertLess(len(plugin.voice_cache.voices), 11)
        self.assertLessEqual(plugin.voice_cache.resident_bytes(), GB)


class TestPinningUnderBudgetPressure(unittest.TestCase):

    def test_a_pinned_voice_is_never_evicted_by_the_budget(self):
        plugin = _plugin(self, pinned_voices=["keeper"],
                         sizes={"keeper": 900 * MB, "big": 900 * MB},
                         max_loaded_bytes="1GB")
        self.assertIn("keeper", plugin.voice_cache.voices)
        for i in range(5):
            plugin.get_model(f"big{i}" if i else "big")
        self.assertIn("keeper", plugin.voice_cache.voices,
                      "a pinned voice must never be evicted")

    def test_pins_raise_the_effective_ceiling(self):
        # Three pins of 900 MB against a 1 GB budget: the pins win, exactly as
        # they win against a too-small max_loaded_voices.
        plugin = _plugin(self, pinned_voices=["p1", "p2", "p3"],
                         sizes={"p1": 900 * MB, "p2": 900 * MB,
                                "p3": 900 * MB},
                         max_loaded_bytes="1GB")
        for pin in ("p1", "p2", "p3"):
            self.assertIn(pin, plugin.voice_cache.voices)
        self.assertGreater(plugin.voice_cache.resident_bytes(), GB)

    def test_pins_over_the_budget_are_reported_at_startup(self):
        with patch("phoonnx.voice_cache.LOG") as log:
            _plugin(self, pinned_voices=["p1", "p2"],
                    sizes={"p1": 900 * MB, "p2": 900 * MB},
                    max_loaded_bytes="1GB")
        logged = " ".join(str(c) for c in log.error.call_args_list)
        self.assertIn("max_loaded_bytes", logged)
        self.assertIn(str(1800 * MB), logged)

    def test_a_pin_bigger_than_the_whole_budget_is_refused_not_exempt(self):
        # Pinning does not exempt a voice from the guaranteed-OOM refusal.
        # get_model's own startup loop catches the failure so the service
        # still starts; the voice simply never becomes resident.
        plugin = _plugin(self, pinned_voices=["huge"],
                         sizes={"huge": 8 * GB}, max_loaded_bytes="1GB")
        self.assertNotIn("huge", plugin.voice_cache.voices)

    def test_an_all_pinned_cache_serves_a_new_voice_anyway(self):
        plugin = _plugin(self, pinned_voices=["p1"],
                         sizes={"p1": 900 * MB, "extra": 900 * MB},
                         max_loaded_bytes="1GB")
        self.assertEqual(plugin.get_model("extra"), "model:extra")
        self.assertIn("p1", plugin.voice_cache.voices)
        self.assertIn("extra", plugin.voice_cache.voices)


class TestConcurrentEviction(unittest.TestCase):

    def test_eviction_under_concurrent_loads_keeps_the_cache_consistent(self):
        # Eviction runs in the same critical section that installs a voice and
        # releases its loading gate. If it corrupted either, callers would hang
        # or the size bookkeeping would drift away from the cache contents.
        # Three of these 30 MB voices fit in the 100 MB budget at once, so
        # three loads may overlap; a fourth waits for room. The barrier is
        # sized to what admission control admits together.
        barrier = threading.Barrier(3, timeout=10)
        started = []
        lock = threading.Lock()

        def hook(voice_id):
            with lock:
                started.append(voice_id)
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass

        plugin = _plugin(self, load_hook=hook, max_loaded_bytes="100MB",
                         sizes={f"v{i}": 30 * MB for i in range(8)})

        results = {}

        def ask(i):
            results[i] = plugin.get_model(f"v{i}")

        threads = [threading.Thread(target=ask, args=(i,), daemon=True)
                   for i in range(8)]
        [t.start() for t in threads]
        [t.join(timeout=20) for t in threads]

        self.assertFalse([t for t in threads if t.is_alive()],
                         "a stranded loading gate hangs callers forever")
        self.assertEqual(len(results), 8)
        self.assertEqual(plugin.voice_cache._loading, {},
                         "every loading gate must be released")
        self.assertEqual(set(plugin.voice_cache._voice_keys), set(plugin.voice_cache.voices),
                         "size bookkeeping must match the cache exactly")
        self.assertLessEqual(plugin.voice_cache.resident_bytes(), 100 * MB)

    def test_repeated_concurrent_requests_for_an_evicted_voice(self):
        plugin = _plugin(self, max_loaded_bytes="50MB",
                         sizes={"a": 30 * MB, "b": 30 * MB})

        def hammer():
            for _ in range(20):
                plugin.get_model("a")
                plugin.get_model("b")

        threads = [threading.Thread(target=hammer, daemon=True)
                   for _ in range(4)]
        [t.start() for t in threads]
        [t.join(timeout=30) for t in threads]
        self.assertFalse([t for t in threads if t.is_alive()])
        self.assertEqual(plugin.voice_cache._loading, {})
        self.assertEqual(set(plugin.voice_cache._voice_keys), set(plugin.voice_cache.voices))


class TestPeakMemoryUnderConcurrentColdLoads(unittest.TestCase):
    """The bound that matters: what is resident at the worst instant.

    Bounding only the steady state is no protection at all. The live incident
    this budget exists for was four concurrent omnivoice loads: each one ended
    up evicting the previous, so the cache looked well behaved afterwards while
    the process had already been killed at the peak.
    """

    def _run(self, voices, sizes, budget, threads=None, expect_errors=False):
        """Load ``voices`` concurrently and return the peak bytes observed.

        Peak is sampled as (bytes being loaded right now) + (bytes resident),
        counting a voice once: a voice that has become resident is dropped from
        the in-flight side of the sum. ``expect_errors`` is for voices that
        must all be refused (``VoiceExceedsMemoryBudget``): every result is
        an error rather than a failure.
        """
        lock = threading.Lock()
        live = {}
        peak = [0]
        plugin = None

        def sample():
            if plugin is None:   # a pinned voice loading from __init__
                return
            with lock:
                inflight = sum(size for v, size in live.items()
                               if v not in plugin.voice_cache.voices)
                total = inflight + plugin.voice_cache.resident_bytes()
                peak[0] = max(peak[0], total)

        def hook(voice_id):
            with lock:
                live[voice_id] = sizes.get(voice_id, 10 * MB)
            sample()
            # Long enough that any admitted-together loads genuinely overlap.
            time.sleep(0.2)
            sample()

        plugin = _plugin(self, sizes=sizes, load_hook=hook,
                         max_loaded_bytes=budget,
                         **(threads or {}))

        errors = []

        def ask(voice_id):
            try:
                plugin.get_model(voice_id)
            except BaseException as exc:  # pragma: no cover - a failure is the report
                errors.append(exc)
            finally:
                sample()
                with lock:
                    live.pop(voice_id, None)

        workers = [threading.Thread(target=ask, args=(v,), daemon=True)
                   for v in voices]
        [t.start() for t in workers]
        [t.join(timeout=60) for t in workers]
        self.assertFalse([t for t in workers if t.is_alive()],
                         "admission control must never deadlock")
        if expect_errors:
            from phoonnx.voice_cache import VoiceExceedsMemoryBudget
            self.assertEqual(len(errors), len(voices))
            self.assertTrue(
                all(isinstance(e, VoiceExceedsMemoryBudget) for e in errors),
                f"unexpected errors: {errors}")
        else:
            self.assertFalse(errors, f"loads failed: {errors}")
        self.assertEqual(plugin.voice_cache._loading, {})
        self.assertEqual(set(plugin.voice_cache._voice_keys), set(plugin.voice_cache.voices))
        return plugin, peak[0]

    def test_four_concurrent_omnivoice_loads_stay_within_the_budget(self):
        # The reviewer's repro: without reservation this peaks at 4 x 2.2 GB.
        sizes = {f"omni{i}": 2200 * MB for i in range(4)}
        plugin, peak = self._run(list(sizes), sizes, "3GB")
        self.assertLessEqual(peak, 3 * GB,
                             f"peak {peak} bytes exceeded the 3 GB budget")
        # No two of these fit together, so the peak is one voice, not four.
        self.assertLessEqual(peak, 2200 * MB)
        self.assertLessEqual(plugin.voice_cache.resident_bytes(), 3 * GB)

    def test_voices_that_fit_together_still_load_together(self):
        # Admission control must not serialise loads that the budget allows;
        # that would trade an OOM for a queue.
        barrier = threading.Barrier(3, timeout=15)
        plugin = _plugin(self, load_hook=lambda _v: barrier.wait(),
                         sizes={f"v{i}": 30 * MB for i in range(3)},
                         max_loaded_bytes="100MB")
        threads = [threading.Thread(target=plugin.get_model, args=(f"v{i}",),
                                    daemon=True) for i in range(3)]
        [t.start() for t in threads]
        [t.join(timeout=20) for t in threads]
        self.assertFalse([t for t in threads if t.is_alive()])
        self.assertEqual(len(plugin.voice_cache.voices), 3)

    def test_a_voice_larger_than_the_budget_is_refused_under_concurrency(self):
        # None of these can ever fit, by definition, so none may be let
        # through: waiting forever for room that will never appear used to
        # be avoided by loading anyway, which is a guaranteed OOM kill
        # instead of a wait. Every concurrent request must be refused, and
        # admission control must still terminate rather than deadlock.
        sizes = {f"huge{i}": 4 * GB for i in range(3)}
        plugin, peak = self._run(list(sizes), sizes, "1GB",
                                 expect_errors=True)
        self.assertEqual(len(plugin.voice_cache.voices), 0)
        self.assertEqual(peak, 0)

    def test_pinned_voices_raise_the_peak_ceiling_but_only_by_themselves(self):
        # "keeper" fits alone under a 3 GB budget: pinning raises the ceiling
        # by exactly its own size, not by one voice per concurrent request.
        # (A pin bigger than the whole budget is refused just like any other
        # voice — see TestPinningUnderBudgetPressure below — pinning does not
        # exempt a voice from the guaranteed-OOM refusal.)
        sizes = {"keeper": 2 * GB, "a": 900 * MB, "b": 900 * MB,
                 "c": 900 * MB}
        plugin, peak = self._run(["a", "b", "c"], sizes, "3GB",
                                 threads={"pinned_voices": ["keeper"]})
        self.assertIn("keeper", plugin.voice_cache.voices)
        self.assertLessEqual(peak, 2 * GB + 900 * MB,
                             "pins raise the ceiling by their own size, "
                             "not by one voice per concurrent request")


class _CyclicVoice:
    """A loaded voice stand-in shaped like the real ``TTSVoice``.

    ``TTSVoice`` holds a reference back to itself through ``adapter.voice``,
    so releasing the caller's last reference does not make the refcount drop
    to zero — CPython only reclaims it on a cyclic collection. A leaf stand-in
    would hide any residency rule that quietly depends on the collector,
    because a plain ``del`` already frees a leaf object.
    """

    def __init__(self, voice_id):
        self.voice_id = voice_id
        self.adapter = _Adapter(self)

    def __eq__(self, other):
        return self is other

    __hash__ = None


class _Adapter:
    def __init__(self, voice):
        self.voice = voice


def _cyclic_plugin(testcase, keys, sizes, **config):
    from phoonnx.opm import PhoonnxTTSPlugin

    def info_for(voice_id):
        return MagicMock(
            load=lambda **_kw: _CyclicVoice(voice_id),
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


class TestCyclicVoiceResidency(unittest.TestCase):
    """A voice in a reference cycle must not delay the next load.

    Residency is the lease, not the garbage collector: weights are charged
    from the moment a caller takes a lease until the moment it ends, so a
    voice nobody is using is free immediately even though the cycle it sits
    in means the object itself will not be reclaimed for a while yet.
    """

    def test_a_released_cyclic_voice_frees_its_charge_without_waiting(self):
        import gc
        # Automatic collection off, so nothing can break the cycle by
        # coincidence. A budget that depended on the collector at all would
        # sit out the whole load_wait_timeout below.
        gc.disable()
        self.addCleanup(gc.enable)

        keys = {"a": "model-a", "b": "model-b"}
        # A timeout of a few seconds, not the 300s default: if this regresses,
        # the test must fail fast rather than eat minutes of CI time sitting
        # out the real production timeout.
        plugin = _cyclic_plugin(self, keys,
                                {"model-a": 3 * GB, "model-b": 3 * GB},
                                max_loaded_bytes="4GB",
                                load_wait_timeout=3)
        with plugin.voice_cache.lease("a"):
            self.assertEqual(plugin.voice_cache.resident_bytes(), 3 * GB,
                             "a leased voice is charged for")
        plugin.voice_cache.voices.clear()          # evicted, cycle still alive
        plugin.voice_cache._voice_keys.clear()

        started = time.monotonic()
        with patch("gc.collect") as collect:
            plugin.get_model("b")
        elapsed = time.monotonic() - started

        self.assertIn("b", plugin.voice_cache.voices)
        collect.assert_not_called()
        self.assertLess(elapsed, 1.0,
                        "an unleased voice's budget charge is gone at once, "
                        "not after the load_wait_timeout")


class _WeakrefableVoice:
    """A loaded-voice stand-in that supports weak references, unlike a str."""

    def __init__(self, voice_id):
        self.voice_id = voice_id


def _plugin_shared_keys(testcase, keys, sizes, **config):
    """A plugin whose catalog maps voice ids to artifact keys, real load()."""
    from phoonnx.opm import PhoonnxTTSPlugin

    def info_for(voice_id):
        return MagicMock(
            load=lambda **_kw: _WeakrefableVoice(voice_id),
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


class TestSharedArtifactRefusal(unittest.TestCase):
    """Several catalog entries can name the same oversized artifact.

    The orpheus/canopylabs/en family is eight voice ids sharing one 15.4 GB
    model; against an 8 GiB deployment every one of the eight crash-looped in
    turn. Each voice id gets its own refusal (dedup is keyed by voice id, not
    by artifact key, same as every other load), and none of those repeated
    refusals may leave the shared key looking resident, charged, or reserved
    — a rejected load must be as if it never happened, every time.
    """

    KEYS = {f"orpheus/en/{name}": "orpheus-3b-en-onnx"
            for name in ("tara", "leah", "jess", "leo", "dan", "mia",
                         "zac", "zoe")}
    SIZES = {"orpheus-3b-en-onnx": 15_400_000_000}  # ~15.4 GB, one shared model

    def test_every_voice_over_one_shared_oversized_artifact_is_refused(self):
        from phoonnx.voice_cache import VoiceExceedsMemoryBudget
        plugin = _plugin_shared_keys(self, self.KEYS, self.SIZES,
                                     max_loaded_bytes="8GB")
        for voice_id in self.KEYS:
            with self.subTest(voice_id=voice_id):
                with self.assertRaises(VoiceExceedsMemoryBudget):
                    plugin.get_model(voice_id)
        self.assertEqual(plugin.voice_cache.voices, {})
        self.assertEqual(plugin.voice_cache._voice_keys, {})
        self.assertEqual(plugin.voice_cache._key_bytes, {})
        self.assertEqual(plugin.voice_cache._leases, {})
        self.assertEqual(plugin.voice_cache._reserved_bytes, 0)
        self.assertEqual(plugin.voice_cache._loading, {},
                         "every loading gate must be released on refusal, "
                         "not just the first one")

    def test_repeated_refusals_of_the_same_artifact_do_not_admit_it(self):
        # A caller that retries the same voice id after a refusal must be
        # refused again, not admitted because an earlier attempt left the
        # shared key half-tracked.
        from phoonnx.voice_cache import VoiceExceedsMemoryBudget
        plugin = _plugin_shared_keys(self, self.KEYS, self.SIZES,
                                     max_loaded_bytes="8GB")
        for _ in range(3):
            with self.assertRaises(VoiceExceedsMemoryBudget):
                plugin.get_model("orpheus/en/dan")
        self.assertNotIn("orpheus-3b-en-onnx",
                         {plugin.voice_cache._voice_keys.get(v, v) for v in plugin.voice_cache.voices})
        self.assertEqual(plugin.voice_cache._reserved_bytes, 0)

        # A small, unrelated voice still loads normally afterwards; the
        # repeated rejections did not poison the budget.
        other_keys = dict(self.KEYS, small="small-model")
        plugin2 = _plugin_shared_keys(self, other_keys,
                                      dict(self.SIZES, **{"small-model": 10 * MB}),
                                      max_loaded_bytes="8GB")
        for voice_id in self.KEYS:
            with self.assertRaises(VoiceExceedsMemoryBudget):
                plugin2.get_model(voice_id)
        model = plugin2.get_model("small")
        self.assertEqual(model.voice_id, "small")
        self.assertIn("small", plugin2.voice_cache.voices)

    def test_concurrent_requests_across_the_family_all_refuse_cleanly(self):
        # Eight concurrent requests, one per sibling voice id sharing the one
        # oversized artifact: dedup is per voice id, so all eight run their
        # own _reserve concurrently rather than waiting on each other, and
        # none may deadlock or corrupt the shared key's accounting.
        from phoonnx.voice_cache import VoiceExceedsMemoryBudget
        plugin = _plugin_shared_keys(self, self.KEYS, self.SIZES,
                                     max_loaded_bytes="8GB")
        errors = {}

        def ask(voice_id):
            try:
                plugin.get_model(voice_id)
            except BaseException as exc:  # pragma: no cover - reported below
                errors[voice_id] = exc

        threads = [threading.Thread(target=ask, args=(v,), daemon=True)
                   for v in self.KEYS]
        [t.start() for t in threads]
        [t.join(timeout=20) for t in threads]
        self.assertFalse([t for t in threads if t.is_alive()],
                         "a shared oversized artifact must not deadlock "
                         "concurrent siblings")
        self.assertEqual(set(errors), set(self.KEYS))
        self.assertTrue(
            all(isinstance(e, VoiceExceedsMemoryBudget) for e in errors.values()),
            f"unexpected errors: {errors}")
        self.assertEqual(plugin.voice_cache.voices, {})
        self.assertEqual(plugin.voice_cache._reserved_bytes, 0)
        self.assertEqual(plugin.voice_cache._loading, {})


class TestPostLoadEviction(unittest.TestCase):
    """The eviction that runs once a cold load's real size is known.

    ``_track`` used to run before ``_evict_for`` in ``get_model``'s post-load
    section, which registers the incoming voice's own key as resident before
    the eviction call gets to look at it — so ``_evict_for``'s "already
    resident, nothing to do" guard fired on every post-load call, evicting
    nothing, regardless of whether the just-measured size actually fit.
    """

    def test_a_measured_size_that_overflows_the_budget_evicts_others(self):
        # "extra" is admitted *during* "big"'s own (unlocked) load — a voice
        # request another caller made while "big" was loading, exactly the
        # kind of admission a long cold load has time for. Once "big"'s real
        # size is known post-load, the two together no longer fit, and only
        # the post-load eviction step (not the pre-load one, which already
        # ran before "extra" existed) can make room.
        plugin_holder = {}
        sizes = {"extra": 2 * GB, "big": 3 * GB}

        class _V:
            def __init__(self, voice_id):
                self.voice_id = voice_id

        def info_for(voice_id):
            def load(**_kw):
                if voice_id == "big":
                    plugin_holder["p"].get_model("extra")
                return _V(voice_id)
            return MagicMock(load=load,
                             disk_size=MagicMock(return_value=sizes[voice_id]))

        from phoonnx.opm import PhoonnxTTSPlugin
        patchers = [
            patch("phoonnx.opm.TTSModelManager"),
            patch.object(PhoonnxTTSPlugin, "get_default_voice",
                         return_value=MagicMock()),
            patch.object(PhoonnxTTSPlugin, "get_voice_info", side_effect=info_for),
            patch.object(PhoonnxTTSPlugin, "_providers", return_value=None),
        ]
        for pat in patchers:
            pat.start()
            self.addCleanup(pat.stop)

        plugin = PhoonnxTTSPlugin(config={"max_loaded_bytes": "4GB",
                                          "load_wait_timeout": 0.2})
        plugin_holder["p"] = plugin
        plugin.get_model("big")

        self.assertIn("big", plugin.voice_cache.voices)
        self.assertNotIn("extra", plugin.voice_cache.voices,
                         "the post-load eviction must evict a voice that no "
                         "longer fits once the real size is known")
        import gc
        gc.collect()
        self.assertLessEqual(plugin.voice_cache.resident_bytes(), 4 * GB)


if __name__ == "__main__":
    unittest.main()
