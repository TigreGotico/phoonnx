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
        from phoonnx.opm import PhoonnxTTSPlugin as P
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
                self.assertEqual(P._parse_max_bytes(value), expected)

    def test_infinite_and_nan_budgets_are_rejected_not_crashes(self):
        # YAML ".inf" and json.loads("Infinity"/"NaN") both land here as
        # floats, and int(inf) raises out of __init__, killing plugin startup.
        from phoonnx.opm import PhoonnxTTSPlugin as P
        for value in (float("inf"), float("-inf"), float("nan"),
                      "1e400", 1e400):
            with self.subTest(value=value):
                self.assertIsNone(P._parse_max_bytes(value))

    def test_an_infinite_budget_does_not_break_startup(self):
        plugin = _plugin(self, max_loaded_bytes=float("inf"))
        self.assertIsNone(plugin.max_loaded_bytes)
        plugin.get_model("v")
        self.assertIn("v", plugin.voices)

    def test_unset_means_no_budget(self):
        from phoonnx.opm import PhoonnxTTSPlugin as P
        for value in (None, "", 0):
            with self.subTest(value=value):
                self.assertIsNone(P._parse_max_bytes(value))

    def test_invalid_sizes_are_rejected_not_read_as_zero(self):
        # A zero budget would evict every voice on every request, so a typo
        # must leave the cache unbounded rather than unusable.
        from phoonnx.opm import PhoonnxTTSPlugin as P
        for value in ("nonsense", "6 gigabytes", "GB", "-1", -1, "1.2.3",
                      "6GBB", "0.4", True, [1], "12PB"):
            with self.subTest(value=value):
                self.assertIsNone(P._parse_max_bytes(value))

    def test_an_invalid_budget_leaves_the_cache_unbounded(self):
        plugin = _plugin(self, max_loaded_bytes="nonsense")
        self.assertIsNone(plugin.max_loaded_bytes)
        for i in range(4):
            plugin.get_model(f"v{i}")
        self.assertEqual(len(plugin.voices), 4)


class TestByteBudget(unittest.TestCase):

    def test_a_voice_over_the_remaining_budget_evicts_the_lru(self):
        plugin = _plugin(self, sizes={"a": 200 * MB, "b": 200 * MB,
                                      "big": 700 * MB},
                         max_loaded_bytes="1GB")
        plugin.get_model("a")
        plugin.get_model("b")
        plugin.get_model("a")       # touch: "b" is now the oldest
        plugin.get_model("big")
        self.assertIn("big", plugin.voices)
        self.assertNotIn("b", plugin.voices)
        self.assertIn("a", plugin.voices)

    def test_many_small_voices_but_only_one_big_one(self):
        # The case a count gets wrong: with max_loaded_voices=3 these small
        # voices would evict for no reason, and three big ones would fit and
        # kill the server.
        plugin = _plugin(self, sizes={"omni": 2200 * MB},
                         max_loaded_bytes="3GB")
        for i in range(20):
            plugin.get_model(f"small{i}")   # 10 MB each -> 200 MB resident
        self.assertEqual(len(plugin.voices), 20)

        plugin.get_model("omni")
        self.assertIn("omni", plugin.voices)
        self.assertLessEqual(sum(plugin._voice_bytes[v] for v in plugin.voices),
                             3 * GB)

        # Another small voice still fits alongside it; the budget is never
        # exceeded by voices that could have been evicted.
        plugin.get_model("small20")
        self.assertIn("omni", plugin.voices)
        self.assertLessEqual(sum(plugin._voice_bytes[v] for v in plugin.voices),
                             3 * GB)

    def test_two_big_voices_cannot_be_resident_together(self):
        plugin = _plugin(self, sizes={"omniA": 2200 * MB, "omniB": 2200 * MB},
                         max_loaded_bytes="3GB")
        plugin.get_model("omniA")
        plugin.get_model("omniB")
        self.assertIn("omniB", plugin.voices)
        self.assertNotIn("omniA", plugin.voices)

    def test_a_voice_bigger_than_the_whole_budget_still_loads_and_warns(self):
        plugin = _plugin(self, sizes={"huge": 8 * GB}, max_loaded_bytes="1GB")
        plugin.get_model("small")
        with patch("phoonnx.opm.LOG") as log:
            model = plugin.get_model("huge")
        self.assertEqual(model, "model:huge")
        self.assertIn("huge", plugin.voices)
        self.assertNotIn("small", plugin.voices,
                         "everything evictable goes before overshooting")
        warned = " ".join(str(c) for c in log.warning.call_args_list)
        self.assertIn("huge", warned)
        self.assertIn(str(8 * GB), warned)
        self.assertIn(str(10 ** 9), warned)

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
        self.assertIn("broken", plugin.voices)


class TestBothBoundsTogether(unittest.TestCase):

    def test_the_count_still_evicts_when_the_budget_is_roomy(self):
        plugin = _plugin(self, max_loaded_voices=2, max_loaded_bytes="100GB")
        for name in ("a", "b", "c"):
            plugin.get_model(name)
        self.assertEqual(set(plugin.voices), {"b", "c"})

    def test_the_budget_still_evicts_when_the_count_is_roomy(self):
        sizes = {f"small{i}": 20 * MB for i in range(10)}
        sizes["big"] = 900 * MB
        plugin = _plugin(self, max_loaded_voices=50,
                         sizes=sizes, max_loaded_bytes="1GB")
        for i in range(10):
            plugin.get_model(f"small{i}")
        plugin.get_model("big")
        self.assertIn("big", plugin.voices)
        self.assertLess(len(plugin.voices), 11)
        self.assertLessEqual(sum(plugin._voice_bytes[v] for v in plugin.voices),
                             GB)


class TestPinningUnderBudgetPressure(unittest.TestCase):

    def test_a_pinned_voice_is_never_evicted_by_the_budget(self):
        plugin = _plugin(self, pinned_voices=["keeper"],
                         sizes={"keeper": 900 * MB, "big": 900 * MB},
                         max_loaded_bytes="1GB")
        self.assertIn("keeper", plugin.voices)
        for i in range(5):
            plugin.get_model(f"big{i}" if i else "big")
        self.assertIn("keeper", plugin.voices,
                      "a pinned voice must never be evicted")

    def test_pins_raise_the_effective_ceiling(self):
        # Three pins of 900 MB against a 1 GB budget: the pins win, exactly as
        # they win against a too-small max_loaded_voices.
        plugin = _plugin(self, pinned_voices=["p1", "p2", "p3"],
                         sizes={"p1": 900 * MB, "p2": 900 * MB,
                                "p3": 900 * MB},
                         max_loaded_bytes="1GB")
        for pin in ("p1", "p2", "p3"):
            self.assertIn(pin, plugin.voices)
        self.assertGreater(plugin._resident_bytes(), GB)

    def test_pins_over_the_budget_are_reported_at_startup(self):
        with patch("phoonnx.opm.LOG") as log:
            _plugin(self, pinned_voices=["p1", "p2"],
                    sizes={"p1": 900 * MB, "p2": 900 * MB},
                    max_loaded_bytes="1GB")
        logged = " ".join(str(c) for c in log.error.call_args_list)
        self.assertIn("max_loaded_bytes", logged)
        self.assertIn(str(1800 * MB), logged)

    def test_an_all_pinned_cache_serves_a_new_voice_anyway(self):
        plugin = _plugin(self, pinned_voices=["p1"],
                         sizes={"p1": 900 * MB, "extra": 900 * MB},
                         max_loaded_bytes="1GB")
        self.assertEqual(plugin.get_model("extra"), "model:extra")
        self.assertIn("p1", plugin.voices)
        self.assertIn("extra", plugin.voices)


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
        self.assertEqual(plugin._loading, {},
                         "every loading gate must be released")
        self.assertEqual(set(plugin._voice_bytes), set(plugin.voices),
                         "size bookkeeping must match the cache exactly")
        self.assertLessEqual(plugin._resident_bytes(), 100 * MB)

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
        self.assertEqual(plugin._loading, {})
        self.assertEqual(set(plugin._voice_bytes), set(plugin.voices))


class TestPeakMemoryUnderConcurrentColdLoads(unittest.TestCase):
    """The bound that matters: what is resident at the worst instant.

    Bounding only the steady state is no protection at all. The live incident
    this budget exists for was four concurrent omnivoice loads: each one ended
    up evicting the previous, so the cache looked well behaved afterwards while
    the process had already been killed at the peak.
    """

    def _run(self, voices, sizes, budget, threads=None):
        """Load ``voices`` concurrently and return the peak bytes observed.

        Peak is sampled as (bytes being loaded right now) + (bytes resident),
        counting a voice once: a voice that has become resident is dropped from
        the in-flight side of the sum.
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
                               if v not in plugin.voices)
                total = inflight + plugin._resident_bytes()
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
        self.assertFalse(errors, f"loads failed: {errors}")
        self.assertEqual(plugin._loading, {})
        self.assertEqual(set(plugin._voice_bytes), set(plugin.voices))
        return plugin, peak[0]

    def test_four_concurrent_omnivoice_loads_stay_within_the_budget(self):
        # The reviewer's repro: without reservation this peaks at 4 x 2.2 GB.
        sizes = {f"omni{i}": 2200 * MB for i in range(4)}
        plugin, peak = self._run(list(sizes), sizes, "3GB")
        self.assertLessEqual(peak, 3 * GB,
                             f"peak {peak} bytes exceeded the 3 GB budget")
        # No two of these fit together, so the peak is one voice, not four.
        self.assertLessEqual(peak, 2200 * MB)
        self.assertLessEqual(plugin._resident_bytes(), 3 * GB)

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
        self.assertEqual(len(plugin.voices), 3)

    def test_a_voice_larger_than_the_budget_still_loads_under_concurrency(self):
        # It cannot fit by definition, so it must be let through alone rather
        # than wait forever for room that will never appear.
        sizes = {f"huge{i}": 4 * GB for i in range(3)}
        plugin, peak = self._run(list(sizes), sizes, "1GB")
        self.assertEqual(len(plugin.voices), 1)
        self.assertLessEqual(peak, 4 * GB,
                             "one oversized voice may exceed the budget, "
                             "three at once may not")

    def test_pinned_voices_raise_the_peak_ceiling_but_only_by_themselves(self):
        sizes = {"keeper": 2 * GB, "a": 900 * MB, "b": 900 * MB,
                 "c": 900 * MB}
        plugin, peak = self._run(["a", "b", "c"], sizes, "1GB",
                                 threads={"pinned_voices": ["keeper"]})
        self.assertIn("keeper", plugin.voices)
        self.assertLessEqual(peak, 2 * GB + 900 * MB,
                             "pins raise the ceiling by their own size, "
                             "not by one voice per concurrent request")


if __name__ == "__main__":
    unittest.main()
