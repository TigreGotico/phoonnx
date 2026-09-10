import json
import unittest

from tests.differential import runner


class TestConfigDifferential(unittest.TestCase):
    """Every loader's output, pinned against a committed snapshot.

    Regenerate with ``python -m tests.differential.runner`` when a change is
    *meant* to alter what a config loads into, and justify each moved case in
    the pull request.
    """

    def test_the_corpus_matches_the_golden_snapshot(self):
        golden = json.load(open(runner.GOLDEN, encoding="utf-8"))
        current = json.loads(json.dumps(runner.snapshot(), sort_keys=True, default=str))
        self.assertEqual(sorted(golden), sorted(current), "corpus cases were added or removed")
        for key in sorted(golden):
            with self.subTest(case=key):
                self.assertEqual(golden[key], current[key])


if __name__ == "__main__":
    unittest.main()
