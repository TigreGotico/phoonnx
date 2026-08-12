"""One load per cold voice, however many callers arrive at once.

The load runs outside the cache lock because it is slow, so without a gate
every simultaneous caller starts its own. That is not a rare race: the proxy in
front of this service times out long before a heavy voice finishes loading, so
callers retry, and every retry is another concurrent request for the same
voice.
"""
import threading
import time
import unittest
from unittest.mock import MagicMock, patch


def _plugin(testcase, load_delay=0.2, **config):
    """Build the plugin with a slow, counting loader."""
    from phoonnx.opm import PhoonnxTTSPlugin

    counts = {"loads": 0, "concurrent": 0, "peak": 0}
    lock = threading.Lock()

    def slow_load(**_kwargs):
        with lock:
            counts["loads"] += 1
            counts["concurrent"] += 1
            counts["peak"] = max(counts["peak"], counts["concurrent"])
        time.sleep(load_delay)
        with lock:
            counts["concurrent"] -= 1
        return f"model-{counts['loads']}"

    patchers = [
        patch("phoonnx.opm.TTSModelManager"),
        patch.object(PhoonnxTTSPlugin, "get_default_voice",
                     return_value=MagicMock()),
        patch.object(PhoonnxTTSPlugin, "get_voice_info",
                     side_effect=lambda v: MagicMock(load=slow_load)),
        patch.object(PhoonnxTTSPlugin, "_providers", return_value=None),
    ]
    for pat in patchers:
        pat.start()
        testcase.addCleanup(pat.stop)
    return PhoonnxTTSPlugin(config=dict(config)), counts


class TestInFlightLoads(unittest.TestCase):

    def test_simultaneous_callers_cause_one_load(self):
        plugin, counts = _plugin(self)
        results = []

        def ask():
            results.append(plugin.get_model("cold"))

        threads = [threading.Thread(target=ask, daemon=True) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]

        self.assertEqual(counts["loads"], 1,
                         "eight callers must share one load, not start eight")
        self.assertEqual(counts["peak"], 1)
        self.assertEqual(len(set(results)), 1,
                         "every caller must get the same model instance")

    def test_a_failed_load_does_not_strand_the_waiters(self):
        from phoonnx.opm import PhoonnxTTSPlugin
        attempts = {"n": 0}

        def failing(**_kwargs):
            attempts["n"] += 1
            time.sleep(0.1)
            raise RuntimeError("load failed")

        patchers = [
            patch("phoonnx.opm.TTSModelManager"),
            patch.object(PhoonnxTTSPlugin, "get_default_voice",
                         return_value=MagicMock()),
            patch.object(PhoonnxTTSPlugin, "get_voice_info",
                         side_effect=lambda v: MagicMock(load=failing)),
            patch.object(PhoonnxTTSPlugin, "_providers", return_value=None),
        ]
        for pat in patchers:
            pat.start()
            self.addCleanup(pat.stop)
        plugin = PhoonnxTTSPlugin(config={})

        errors = []

        def ask():
            try:
                plugin.get_model("broken")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=ask, daemon=True) for _ in range(4)]
        [t.start() for t in threads]
        for t in threads:
            t.join(timeout=10)
            self.assertFalse(t.is_alive(), "a failed load must not hang waiters")
        self.assertEqual(len(errors), 4, "every caller should see the failure")

    def test_different_voices_still_load_in_parallel(self):
        # The gate is per voice; it must not serialise unrelated loads.
        plugin, counts = _plugin(self, load_delay=0.3)
        threads = [threading.Thread(target=plugin.get_model, args=(f"v{i}",))
                   for i in range(4)]
        started = time.time()
        [t.start() for t in threads]
        [t.join() for t in threads]
        elapsed = time.time() - started
        self.assertEqual(counts["loads"], 4)
        # 0.3s of load each: fully parallel is ~0.3s, two-at-a-time is ~0.6s,
        # fully serial is ~1.2s. The threshold has to reject partial
        # serialisation, not just total serialisation.
        self.assertLess(elapsed, 0.55,
                        "four different voices should load concurrently")


class TestGateIsAlwaysReleased(unittest.TestCase):
    """A failure before the load must not strand the callers behind it."""

    def _plugin_with_unknown_voice(self):
        from phoonnx.opm import PhoonnxTTSPlugin
        patchers = [
            patch("phoonnx.opm.TTSModelManager"),
            patch.object(PhoonnxTTSPlugin, "get_default_voice",
                         return_value=MagicMock()),
            # The voice id comes from the request, so an unknown one is
            # reachable from outside and must not wedge the service.
            patch.object(PhoonnxTTSPlugin, "get_voice_info",
                         side_effect=Exception("Unknown voice: nope")),
            patch.object(PhoonnxTTSPlugin, "_providers", return_value=None),
        ]
        for pat in patchers:
            pat.start()
            self.addCleanup(pat.stop)
        return PhoonnxTTSPlugin(config={})

    def test_an_unknown_voice_does_not_hang_the_next_caller(self):
        plugin = self._plugin_with_unknown_voice()
        with self.assertRaises(Exception):
            plugin.get_model("nope")
        # The gate must be gone, or every later caller waits on it forever.
        self.assertEqual(plugin._loading, {})

        done = threading.Event()

        def second():
            try:
                plugin.get_model("nope")
            except Exception:
                pass
            done.set()

        threading.Thread(target=second, daemon=True).start()
        self.assertTrue(done.wait(timeout=10),
                        "a second caller for an unknown voice must not hang")

    def test_concurrent_callers_for_an_unknown_voice_all_return(self):
        plugin = self._plugin_with_unknown_voice()
        errors = []

        def ask():
            try:
                plugin.get_model("nope")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=ask, daemon=True) for _ in range(6)]
        [t.start() for t in threads]
        for t in threads:
            t.join(timeout=10)
            self.assertFalse(t.is_alive(), "no caller may hang")
        self.assertEqual(len(errors), 6)
        self.assertEqual(plugin._loading, {})


class TestFailedLoadIsNotRetriedPerCaller(unittest.TestCase):

    def test_one_attempt_serves_every_waiting_caller(self):
        from phoonnx.opm import PhoonnxTTSPlugin
        attempts = {"n": 0}

        def failing(**_kwargs):
            attempts["n"] += 1
            time.sleep(0.2)
            raise RuntimeError("load failed")

        patchers = [
            patch("phoonnx.opm.TTSModelManager"),
            patch.object(PhoonnxTTSPlugin, "get_default_voice",
                         return_value=MagicMock()),
            patch.object(PhoonnxTTSPlugin, "get_voice_info",
                         side_effect=lambda v: MagicMock(load=failing)),
            patch.object(PhoonnxTTSPlugin, "_providers", return_value=None),
        ]
        for pat in patchers:
            pat.start()
            self.addCleanup(pat.stop)
        plugin = PhoonnxTTSPlugin(config={})

        errors = []

        def ask():
            try:
                plugin.get_model("broken")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=ask, daemon=True) for _ in range(8)]
        started = time.time()
        [t.start() for t in threads]
        for t in threads:
            t.join(timeout=15)
        elapsed = time.time() - started

        self.assertEqual(len(errors), 8, "every caller must see the failure")
        self.assertEqual(attempts["n"], 1,
                         "a failing load must not be retried once per caller")
        self.assertLess(elapsed, 1.0,
                        "waiters must not queue behind serial retries")


if __name__ == "__main__":
    unittest.main()


class TestGateReleasedWithTheCacheInsert(unittest.TestCase):
    """Waiters must never wake into a gap where the voice is not yet cached.

    If the gate is released before the loaded voice is stored, a waiter that
    runs in between finds no cached voice and no gate, elects itself, and
    starts a second cold load of the same voice — the duplicate the gate is
    there to prevent. The window is a few instructions wide, so the test does
    not race for it: it widens the window by making the gate's own event yield
    to the waiters as it is set.
    """

    def test_a_waiter_woken_at_the_release_point_does_not_reload(self):
        from phoonnx import opm as opm_module
        from phoonnx.opm import PhoonnxTTSPlugin

        class YieldingEvent(threading.Event):
            """Hands the CPU to the waiters at the moment the gate opens."""

            def set(self):
                super().set()
                # Long enough for a waiter released into an empty cache to
                # elect itself and start its own load.
                time.sleep(0.3)

        real_loading = opm_module._Loading

        def yielding_loading():
            gate = real_loading()
            gate.done = YieldingEvent()
            return gate

        loads = {"n": 0}
        first_load_started = threading.Event()

        def counting(**_kwargs):
            loads["n"] += 1
            first_load_started.set()
            return f"model-{loads['n']}"

        patchers = [
            patch("phoonnx.opm.TTSModelManager"),
            patch.object(PhoonnxTTSPlugin, "get_default_voice",
                         return_value=MagicMock()),
            patch.object(PhoonnxTTSPlugin, "get_voice_info",
                         side_effect=lambda v: MagicMock(load=counting)),
            patch.object(PhoonnxTTSPlugin, "_providers", return_value=None),
            patch.object(opm_module, "_Loading", yielding_loading),
        ]
        for pat in patchers:
            pat.start()
            self.addCleanup(pat.stop)
        plugin = PhoonnxTTSPlugin(config={})

        results = []
        owner = threading.Thread(target=lambda: results.append(
            plugin.get_model("cold")), daemon=True)
        owner.start()
        # The waiter must be blocked on the gate before the gate is released.
        self.assertTrue(first_load_started.wait(timeout=10))
        waiter = threading.Thread(target=lambda: results.append(
            plugin.get_model("cold")), daemon=True)
        waiter.start()

        owner.join(timeout=15)
        waiter.join(timeout=15)
        self.assertFalse(owner.is_alive() or waiter.is_alive(),
                         "neither caller may hang")
        self.assertEqual(loads["n"], 1,
                         "the voice must be cached before the gate is released")
        self.assertEqual(len(set(map(id, results))), 1,
                         "both callers must get the same instance")
