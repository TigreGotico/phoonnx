"""A voice already in the catalog must never look unknown.

The registry is rebuilt wholesale — by a refresh, or by the first caller that
asks for an id nobody has asked for before — while other threads are looking
voices up in it. Clearing it and repopulating it in place made every entry
disappear for the length of the rebuild, and an unknown voice is not a slow
answer, it is an error: the caller raises, and in the loading cache that
failure is shared with every thread waiting on the same voice.
"""
import threading
import unittest
from unittest.mock import patch

from phoonnx.model_manager import TTSModelManager


class _Entry:
    """Stand-in for TTSModelInfo, with a construction cost.

    The real one parses a config dict per entry across a few thousand entries;
    the sleep stands in for that, so the rebuild is wide enough for a reader
    to land inside it.
    """

    def __init__(self, **fields):
        self.voice_id = fields["voice_id"]
        threading.Event().wait(0.0005)


class TestRegistryRebuildIsAtomic(unittest.TestCase):

    def setUp(self):
        patcher = patch("phoonnx.model_manager.TTSModelInfo", _Entry)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.manager = TTSModelManager.__new__(TTSModelManager)
        self.manager.voices = {}
        self.manager._registry_lock = threading.RLock()
        self.manager.cache = {f"voice/{i}": {"voice_id": f"voice/{i}"}
                              for i in range(400)}

        # The bundled indexes are read off disk and merged into the cache;
        # this test is about what the rebuild does to readers, so it merges
        # nothing and rebuilds from the cache it was given.
        patcher = patch.object(TTSModelManager, "voice_index_files",
                               return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)

    def _hammer(self, rebuilds=8, readers=4, reads=400, lookup=None):
        lookup = lookup or self.manager.get_voice
        misses = []
        stop = threading.Event()

        def rebuild():
            for _ in range(rebuilds):
                self.manager.merge_default_voices()
            stop.set()

        def read():
            while not stop.is_set():
                for _ in range(reads):
                    if lookup("voice/7") is None:
                        misses.append(len(self.manager.voices))
                    if stop.is_set():
                        return

        threads = [threading.Thread(target=rebuild)]
        threads += [threading.Thread(target=read) for _ in range(readers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        return misses

    def test_a_known_voice_never_looks_unknown_during_a_rebuild(self):
        self.manager.merge_default_voices()
        self.assertIsNotNone(self.manager.get_voice("voice/7"))

        misses = self._hammer()
        self.assertEqual(
            misses, [],
            f"a known voice read as unknown {len(misses)} times while the "
            f"registry was rebuilt (registry sizes seen: "
            f"{sorted(set(misses))[:5]})")

    def test_a_reader_that_does_not_take_the_lock_is_safe_too(self):
        """``self.voices`` is read directly elsewhere (listing by language,
        downloading by id), so the rebuild must also be invisible to a reader
        holding no lock at all — which is what publishing the new registry in
        a single assignment buys, over and above the lock."""
        self.manager.merge_default_voices()
        # Fewer, shorter rebuilds than the locked case: an unlocked reader
        # spins as fast as the interpreter allows and starves the rebuilder.
        misses = self._hammer(rebuilds=3, readers=2,
                              lookup=lambda vid: self.manager.voices.get(vid))
        self.assertEqual(
            misses, [],
            f"an unlocked reader saw a known voice as unknown {len(misses)} "
            f"times (registry sizes seen: {sorted(set(misses))[:5]})")

    def test_the_registry_is_complete_after_concurrent_rebuilds(self):
        self._hammer()
        self.assertEqual(len(self.manager.voices), 400)

    def test_an_unparseable_entry_is_skipped_not_fatal(self):
        self.manager.cache["voice/bad"] = {"nope": 1}
        self.manager.merge_default_voices()
        self.assertIsNone(self.manager.get_voice("voice/bad"))
        self.assertIsNotNone(self.manager.get_voice("voice/7"))


if __name__ == "__main__":
    unittest.main()
