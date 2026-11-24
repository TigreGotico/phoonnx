import unittest

from scriptconv.frontend import Chunk, Stages

from phoonnx.config import SynthesisConfig


class TestTheChunkSetting(unittest.TestCase):
    def test_each_documented_value_is_accepted(self):
        for value in ("sentence", "clause", "paragraph", "none", Chunk.CLAUSE, None):
            with self.subTest(value=value):
                self.assertEqual(SynthesisConfig(stages=Stages(chunk=value)).stages.chunk, value)

    def test_a_bad_value_fails_when_the_config_is_built(self):
        with self.assertRaises(ValueError) as caught:
            SynthesisConfig(stages=Stages(chunk="word"))
        self.assertIn("'word'", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
