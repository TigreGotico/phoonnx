import unittest

from phoonnx.config import PhonemeType, VoiceConfig, get_phonemizer, Alphabet


class TestPhonemeTypeDefault(unittest.TestCase):
    def test_none_phoneme_type_falls_back_to_unicode(self):
        cfg = VoiceConfig(
            tokenizer=None, num_symbols=0, num_speakers=1, num_langs=1,
            sample_rate=16000, lang_code=None, phoneme_type=None,
            alphabet=Alphabet.UNICODE, phonemizer_model=None,
        )
        self.assertEqual(cfg.phoneme_type, PhonemeType.GRAPHEMES)

    def test_explicit_phoneme_type_string_round_trips(self):
        cfg = VoiceConfig(
            tokenizer=None, num_symbols=0, num_speakers=1, num_langs=1,
            sample_rate=16000, lang_code=None, phoneme_type="espeak",
            alphabet=Alphabet.UNICODE, phonemizer_model=None,
        )
        self.assertEqual(cfg.phoneme_type, PhonemeType.ESPEAK)

    def test_transformers_vocab_only_config_defaults_phoneme_type(self):
        # a bare vocab.json/tokenizer_config.json voice carries no "phoneme_type"
        # key in its config dict, mirroring a real transformers-style export
        vocab = {"a": 0, "b": 1, "<blank>": 2}
        cfg = VoiceConfig.from_dict({}, vocab=vocab)
        self.assertEqual(cfg.phoneme_type, PhonemeType.GRAPHEMES)
        # first phonemization must not crash with PhonemeType(None)
        get_phonemizer(cfg.phoneme_type, alphabet=cfg.alphabet)


if __name__ == "__main__":
    unittest.main()
