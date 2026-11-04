"""Region tags of bundled voices whose training source names the variety.

A bare tag means the language's motherland variety (``en`` is British
English, ``pt`` is European Portuguese). A voice trained on another variety
must name it, because the tag ranks the voice for a request and selects the
phonemizer variety for espeak voices.
"""
import json
import unittest

from phoonnx.model_manager import TTSModelManager

# voice id -> tag, each taken from the voice's own training source
REGIONAL_VOICES = {
    # jaywalnut310/vits configs/ljs_base.json uses english_cleaners2, which
    # calls phonemize(text, language='en-us', backend='espeak', ...)
    "hf_community/kakao-enterprise/vits-ljs": "en-US",
    # nipponjo/mixer-tts-pytorch utils/phonemizer.py and phonemize_lj.py:
    # EspeakBackend(language='en-us', ...)
    "nipponjo/mixer-tts-ljspeech-80": "en-US",
    "nipponjo/mixer-tts-ljspeech-128": "en-US",
    "nipponjo/mixer-tts-ljspeech-384": "en-US",
    # coqui-ai/TTS release tts_models--en--ljspeech--glow-tts config.json:
    # "phoneme_language": "en-us"
    "coqui-en-ljspeech-glow-tts": "en-US",
    # the upstream repository is wasmdashai/vits-eng-us-ljs
    "hf_community/wasmdashai/vits-eng-us-ljs": "en-US",
    # ai4bharat/indic-parler-tts card: "The model officially supports Indian
    # English accents through its English voices"; Thoma and Mary are the
    # recommended English speakers
    "indic_parler/thoma/en": "en-IN",
    "indic_parler/mary/en": "en-IN",
    # hexgrad/Kokoro-82M VOICES.md: "Brazilian Portuguese ... espeak-ng pt-br"
    "kokoro/pf_dora": "pt-BR",
    "kokoro/pm_alex": "pt-BR",
    "kokoro/pm_santa": "pt-BR",
    # hexgrad/Kokoro-82M VOICES.md: "### British English ... lang_code='b'
    # ... espeak-ng en-gb fallback" for the bf_ and bm_ voices
    **{f"kokoro/{name}": "en-GB" for name in (
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis")},
    # hexgrad/kLegacy v0.19 README: "'b' => British English => en-gb"
    **{f"kokoro-v019/{name}": "en-GB" for name in (
        "bf_emma", "bf_isabella", "bm_george", "bm_lewis")},
    # hexgrad/Kokoro-82M-v1.1-zh card: "Vale, an older British female"
    "kokoro-v11/bf_vale": "en-GB",
    # kyutai-labs/pocket-tts README links each voice to a VCTK clip, and VCTK
    # speaker-info.txt gives the accent: p315, p339, p333, p360 and p361 are
    # American, p303 is Canadian
    "pockettts/en/george": "en-US",
    "pockettts/en/jane": "en-US",
    "pockettts/en/mary": "en-US",
    "pockettts/en/michael": "en-US",
    "pockettts/en/eve": "en-US",
    "pockettts/en/azelma": "en-CA",
}

# Piper voices trained with espeak voice "en", which is espeak-ng's British
# English (gmw/en). Their .onnx.json has "espeak": {"voice": "en"}.
MOTHERLAND_VOICES = {
    "piper_community/agentvibe/en_16Speakers": "en",
    "piper_community/jstlntchh/en_Scaramouche": "en",
}


def _bundled_entries():
    entries = {}
    for path in TTSModelManager.voice_index_files():
        with open(path, encoding="utf-8") as f:
            entries.update(json.load(f))
    return entries


class TestRegionalVoiceTags(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.entries = _bundled_entries()
        cls.langs = {voice_id: entry["lang"]
                     for voice_id, entry in cls.entries.items()}

    def test_voices_trained_on_a_named_variety_carry_its_region(self):
        for voice_id, tag in REGIONAL_VOICES.items():
            with self.subTest(voice=voice_id):
                self.assertEqual(self.langs[voice_id], tag)

    def test_voices_trained_on_espeak_en_keep_the_bare_tag(self):
        for voice_id, tag in MOTHERLAND_VOICES.items():
            with self.subTest(voice=voice_id):
                self.assertEqual(self.langs[voice_id], tag)


if __name__ == "__main__":
    unittest.main()
