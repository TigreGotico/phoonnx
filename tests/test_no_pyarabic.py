import pathlib
import unittest


class TestNoPyarabicAnywhere(unittest.TestCase):
    def test_tree_is_pyarabic_free(self):
        root = pathlib.Path("phoonnx")
        hits = [str(f) for f in root.rglob("*.py")
                if "pyarabic" in f.read_text(errors="ignore")]
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
