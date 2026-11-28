"""alVoCat is a Vocos vocoder fine-tuned on Catalan and released as CC-BY-NC-4.0.
BSC publishes the same graph bytes as the Apache-2.0 universal vocoder
BSC-LT/vocos-mel-22khz, so a voice in another language cites that release."""
import json
import unittest
from pathlib import Path

from phoonnx.model_manager import TTSModelManager

ALVOCAT = "alvocat"


class TestVoiceIndexVocoderLicence(unittest.TestCase):

    def test_only_catalan_voices_use_alvocat(self):
        entries = [(path.name, voice_id, entry)
                   for path in sorted(Path(TTSModelManager.voice_index_path()).glob("*.json"))
                   for voice_id, entry in json.loads(path.read_text(encoding="utf-8")).items()]
        self.assertGreater(len(entries), 3000)
        users = [(name, voice_id, entry.get("lang", "")) for name, voice_id, entry in entries
                 if ALVOCAT in (entry.get("vocoder_url") or "")]
        self.assertGreater(len(users), 0)
        wrong = [f"{name}: {voice_id} ({lang})" for name, voice_id, lang in users
                 if not lang.lower().startswith("ca")]
        self.assertEqual(wrong, [])

    def test_english_vocos_voices_cite_the_apache_release(self):
        index = Path(TTSModelManager.voice_index_path())
        voices = {**json.loads((index / "mixertts.json").read_text(encoding="utf-8")),
                  **json.loads((index / "fastpitch.json").read_text(encoding="utf-8"))}
        vocos = {voice_id: entry["vocoder_url"] for voice_id, entry in voices.items()
                 if entry.get("vocoder_type") == "vocos"}
        self.assertEqual(len(vocos), 5)
        for voice_id, url in vocos.items():
            self.assertIn("/BSC-LT/vocos-mel-22khz/resolve/", url, voice_id)
            self.assertTrue(url.endswith("/mel_spec_22khz_univ.onnx"), voice_id)


if __name__ == "__main__":
    unittest.main()
