"""Every Hub URL in the bundled voice index names a commit.

A URL that names ``main`` loads whatever the repo holds today, so an upload to the Hub changes
every installed release at once. A commit sha keeps a release on the files it shipped with.
"""
import re
import unittest
from pathlib import Path

from phoonnx.model_manager import TTSModelManager

HUB_STRING = re.compile(r'"(https?://huggingface\.co/[^"]*)"')
PINNED = re.compile(r"https://huggingface\.co/[^/]+/[^/]+/(?:resolve|raw)/[0-9a-f]{40}/[^?#]+")


class TestVoiceIndexPinned(unittest.TestCase):

    def test_every_hub_url_names_a_commit(self):
        files = sorted(Path(TTSModelManager.voice_index_path()).glob("*.json"))
        self.assertGreater(len(files), 0)
        urls = [(path.name, url) for path in files for url in HUB_STRING.findall(path.read_text(encoding="utf-8"))]
        self.assertGreater(len(urls), 0)
        unpinned = [f"{name}: {url}" for name, url in urls if not PINNED.fullmatch(url)]
        if unpinned:
            self.fail(f"{len(unpinned)} of {len(urls)} Hub URLs do not name a commit:\n" + "\n".join(unpinned[:20]))


if __name__ == "__main__":
    unittest.main()
