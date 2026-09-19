import json
import os
import tempfile
import unittest

from phoonnx.config import Alphabet, Engine, PhonemeType, VoiceConfig
from phoonnx.config_loaders import (LOADERS, CanonicalLoader, ChatterboxLoader, CoquiLoader,
                                    LoadedFields, LoadRequest, Mimic3Loader, PhoonnxLoader,
                                    PiperLoader, RawTextLoader, TokensTxtLoader,
                                    TransformersLoader, VoskLoader, resolve_overrides)
from phoonnx.util import normalize_lang


def _tokens_txt():
    """Path to a throwaway mimic3/sherpa style phonemes.txt."""
    path = os.path.join(tempfile.mkdtemp(), "phonemes.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(f"{i} {c}" for i, c in enumerate("abcdef_ ")))
    return path


def _index(loader_cls):
    return LOADERS.index(loader_cls)


class TestRegistryOrder(unittest.TestCase):
    def test_raw_text_loaders_come_before_every_shape_sniffing_loader(self):
        first_shape = min(_index(c) for c in (PhoonnxLoader, PiperLoader, Mimic3Loader,
                                              CoquiLoader))
        raw = [c for c in LOADERS if issubclass(c, RawTextLoader)]
        self.assertTrue(raw)
        self.assertTrue(all(_index(c) < first_shape for c in raw))

    def test_phoonnx_is_probed_before_piper(self):
        self.assertLess(_index(PhoonnxLoader), _index(PiperLoader))

    def test_vosk_is_probed_before_piper(self):
        self.assertLess(_index(VoskLoader), _index(PiperLoader))

    def test_a_vosk_config_is_never_claimed_by_the_piper_loader(self):
        # a real alphacep vosk-tts config.json is piper-shaped (espeak
        # stanza, list-valued phoneme_id_map) but - unlike a real piper
        # config - carries no phoneme_type; PiperLoader already rejects that,
        # so it never reaches PiperLoader regardless of probing order
        cfg = {"espeak": {"voice": "ru"},
               "phoneme_id_map": {"a0": [10], "sch": [20], "_": [0]}}
        self.assertFalse(PiperLoader.detect(LoadRequest(config=dict(cfg))))
        self.assertEqual(VoiceConfig.from_dict(dict(cfg)).engine, Engine.VOSK)

    def test_a_piper_config_with_the_vosk_inventory_is_not_stolen(self):
        # the romanised-Russian inventory alone is not the signature: a real
        # piper config (declares phoneme_type) that happens to also define
        # "a0"/"sch" phonemes must still load as piper, not vosk
        cfg = {"phoneme_type": "espeak", "espeak": {"voice": "ru"},
               "phoneme_id_map": {"a0": [10], "sch": [20], "_": [0]}}
        self.assertFalse(VoskLoader.detect(LoadRequest(config=dict(cfg))))
        self.assertEqual(VoiceConfig.from_dict(dict(cfg)).engine, Engine.PIPER)

    def test_companion_file_loaders_are_probed_last(self):
        shape = max(_index(c) for c in (PhoonnxLoader, PiperLoader, Mimic3Loader, CoquiLoader))
        self.assertGreater(_index(TransformersLoader), shape)
        self.assertGreater(_index(TokensTxtLoader), shape)

    def test_a_native_config_is_never_claimed_by_the_piper_loader(self):
        # a native espeak voice carries a piper-shaped phoneme_id_map, so the
        # registry order is what keeps it out of the piper loader
        cfg = {"phoonnx_version": "1.0", "phoneme_type": "espeak",
               "phoneme_id_map": {"a": [1], "_": [0]}, "lang_code": "pt"}
        self.assertTrue(PiperLoader.detect(LoadRequest(config=dict(cfg))))
        self.assertEqual(VoiceConfig.from_dict(dict(cfg)).engine, Engine.PHOONNX)

    def test_canonical_loader_is_the_final_fallback(self):
        self.assertIs(LOADERS[-1], CanonicalLoader)
        self.assertTrue(CanonicalLoader.detect(LoadRequest(config={})))

    def test_every_engine_is_registered_once(self):
        engines = [c.ENGINE for c in LOADERS if issubclass(c, RawTextLoader)]
        self.assertEqual(len(engines), len(set(engines)))
        self.assertIn(Engine.CHATTERBOX, engines)


class TestResolveOverrides(unittest.TestCase):
    def test_caller_kwargs_win_over_loader_values(self):
        loaded = LoadedFields(lang_code="pt", phoneme_type=PhonemeType.ESPEAK,
                              alphabet=Alphabet.IPA, engine=Engine.COQUI)
        resolve_overrides(loaded, LoadRequest(config={}, lang_code="de",
                                              phoneme_type=PhonemeType.GRUUT,
                                              alphabet=Alphabet.ARPA, engine=Engine.MATCHA))
        self.assertEqual(loaded.lang_code, "de")
        self.assertEqual(loaded.phoneme_type, PhonemeType.GRUUT)
        self.assertEqual(loaded.alphabet, Alphabet.ARPA)
        self.assertEqual(loaded.engine, Engine.MATCHA)

    def test_pinned_fields_survive_the_caller(self):
        loaded = LoadedFields(lang_code="pt", alphabet=Alphabet.UNICODE,
                              pinned=frozenset({"lang_code", "alphabet"}))
        resolve_overrides(loaded, LoadRequest(config={}, lang_code="de",
                                              alphabet=Alphabet.ARPA))
        self.assertEqual(loaded.lang_code, "pt")
        self.assertEqual(loaded.alphabet, Alphabet.UNICODE)

    def test_loader_value_is_kept_when_the_caller_passes_nothing(self):
        loaded = LoadedFields(lang_code="pt", engine=Engine.PIPER)
        resolve_overrides(loaded, LoadRequest(config={}))
        self.assertEqual(loaded.lang_code, "pt")
        self.assertEqual(loaded.engine, Engine.PIPER)

    def test_engine_falls_back_to_phoonnx(self):
        loaded = LoadedFields()
        resolve_overrides(loaded, LoadRequest(config={}))
        self.assertEqual(loaded.engine, Engine.PHOONNX)


class TestPinnedFormats(unittest.TestCase):
    def test_piper_text_models_pin_their_character_alphabet(self):
        cfg = {"piper_version": "1.0", "phoneme_type": "text",
               "phoneme_id_map": {"a": [1], "_": [0]}}
        voice = VoiceConfig.from_dict(cfg, alphabet=Alphabet.ARPA)
        self.assertEqual(voice.alphabet, Alphabet.UNICODE)
        self.assertEqual(voice.phoneme_type, PhonemeType.UNICODE)

    def test_mimic3_takes_its_language_from_the_config_not_the_caller(self):
        # mimic3 gruut vocabularies are language-specific: honouring a caller
        # language here would phonemize into a vocabulary the model never saw
        cfg = {"phonemizer": "gruut", "phonemes": {}, "text_language": "de"}
        voice = VoiceConfig.from_dict(cfg, tokens_txt=_tokens_txt(), lang_code="pt")
        self.assertEqual(voice.lang_code, normalize_lang("de"))

    def test_mimic3_symbols_models_pin_their_grapheme_alphabet(self):
        cfg = {"phonemizer": "symbols", "phonemes": {}, "text_language": "de"}
        voice = VoiceConfig.from_dict(cfg, tokens_txt=_tokens_txt(), alphabet=Alphabet.ARPA)
        self.assertEqual(voice.alphabet, Alphabet.UNICODE)
        self.assertEqual(voice.phoneme_type, PhonemeType.GRAPHEMES)

    def test_a_transformers_tokenizer_config_language_beats_the_caller(self):
        # the vocabulary was built for that language; the caller cannot re-point it
        voice = VoiceConfig.from_dict({}, vocab={"a": 1, "_": 0},
                                      tokenizer_config={"language": "ro"},
                                      lang_code="pt")
        self.assertEqual(voice.lang_code, normalize_lang("ro"))

    def test_a_caller_language_survives_a_tokenizer_config_without_one(self):
        voice = VoiceConfig.from_dict({}, vocab={"a": 1, "_": 0},
                                      tokenizer_config={"add_blank": True},
                                      lang_code="pt")
        self.assertEqual(voice.lang_code, normalize_lang("pt"))

    def test_a_config_naming_a_codec_engine_pins_it(self):
        voice = VoiceConfig.from_dict({"engine": "llasa"}, engine=Engine.PIPER)
        self.assertEqual(voice.engine, Engine.LLASA)

    def test_chatterbox_needs_its_bpe_tokenizer(self):
        with self.assertRaises(ValueError):
            ChatterboxLoader.load(LoadRequest(config={"engine": "chatterbox"}))

    def test_mimic3_needs_its_tokens_file(self):
        with self.assertRaises(ValueError):
            VoiceConfig.from_dict({"phonemizer": "gruut", "phonemes": {}})


# (engine, sample rate its codec runs at, alphabet the adapter expects)
RAW_TEXT_ENGINES = [
    (Engine.CHATTERBOX, 24000, Alphabet.UNICODE),
    (Engine.LLASA, 16000, Alphabet.GRAPHEMES),
    (Engine.NEUTTS, 24000, Alphabet.GRAPHEMES),
    (Engine.ORPHEUS, 24000, Alphabet.GRAPHEMES),
    (Engine.MOSSTTS, 48000, Alphabet.GRAPHEMES),
    (Engine.SUPERTONIC, 44100, Alphabet.GRAPHEMES),
    (Engine.POCKETTTS, 24000, Alphabet.GRAPHEMES),
    (Engine.SPARKTTS, 16000, Alphabet.GRAPHEMES),
    (Engine.INDIC_PARLER, 44100, Alphabet.GRAPHEMES),
    (Engine.OMNIVOICE, 24000, Alphabet.GRAPHEMES),
    (Engine.MAGPIE, 22050, Alphabet.GRAPHEMES),
    (Engine.ARKTTS, 44100, Alphabet.GRAPHEMES),
    (Engine.QWEN3TTS, 24000, Alphabet.GRAPHEMES),
    (Engine.OUTETTS, 24000, Alphabet.GRAPHEMES),
]


class TestRawTextLoaderDefaults(unittest.TestCase):
    def test_the_table_covers_every_registered_raw_text_loader(self):
        registered = {c.ENGINE for c in LOADERS if issubclass(c, RawTextLoader)}
        self.assertEqual(registered, {e for e, _, _ in RAW_TEXT_ENGINES})

    def test_each_engine_declares_its_codec_sample_rate_and_alphabet(self):
        for engine, sample_rate, alphabet in RAW_TEXT_ENGINES:
            with self.subTest(engine=engine):
                cls = next(c for c in LOADERS
                           if issubclass(c, RawTextLoader) and c.ENGINE is engine)
                self.assertEqual(cls.SAMPLE_RATE, sample_rate)
                self.assertEqual(cls.ALPHABET, alphabet)

    def test_a_voice_is_built_with_its_engine_sample_rate(self):
        # Chatterbox is excluded: it cannot load without its tokenizer.json
        for engine, sample_rate, alphabet in RAW_TEXT_ENGINES:
            if engine is Engine.CHATTERBOX:
                continue
            with self.subTest(engine=engine):
                voice = VoiceConfig.from_dict({"engine": engine.value})
                self.assertEqual(voice.engine, engine)
                self.assertEqual(voice.sample_rate, sample_rate)
                self.assertEqual(voice.alphabet, alphabet)
                self.assertEqual(voice.phoneme_type, PhonemeType.UNICODE)

    def test_a_config_sample_rate_is_never_overwritten(self):
        voice = VoiceConfig.from_dict({"engine": "llasa", "audio": {"sample_rate": 48000}})
        self.assertEqual(voice.sample_rate, 48000)


def _vosk_cfg():
    # a real vosk config: piper-shaped, but no phoneme_type of its own
    return {"espeak": {"voice": "ru"},
            "phoneme_id_map": {"a0": [10], "a1": [11], "bj": [12], "sch": [20], "_": [0]}}


class TestVoskLoader(unittest.TestCase):
    def test_detects_by_phoneme_inventory(self):
        self.assertTrue(VoskLoader.detect(LoadRequest(config=_vosk_cfg())))

    def test_rejects_configs_without_the_vosk_phoneme_set(self):
        cfg = {"phoneme_id_map": {"a": [1], "_": [0]}}
        self.assertFalse(VoskLoader.detect(LoadRequest(config=cfg)))

    def test_rejects_configs_that_declare_a_phoneme_type(self):
        # the inventory alone is not the signature - a real piper config that
        # happens to also define "a0"/"sch" phonemes must not be stolen
        cfg = dict(_vosk_cfg(), phoneme_type="espeak")
        self.assertFalse(VoskLoader.detect(LoadRequest(config=cfg)))

    def test_an_explicit_non_vosk_engine_beats_shape_sniffing(self):
        cfg = dict(_vosk_cfg(), engine="piper")
        self.assertFalse(VoskLoader.detect(LoadRequest(config=cfg)))

    def test_load_defaults_lang_to_russian(self):
        voice = VoiceConfig.from_dict(_vosk_cfg())
        self.assertEqual(voice.engine, Engine.VOSK)
        self.assertEqual(voice.phoneme_type, PhonemeType.VOSK)
        self.assertEqual(voice.alphabet, Alphabet.VOSK)
        self.assertEqual(voice.lang_code, normalize_lang("ru"))

    def test_a_caller_lang_code_overrides_the_russian_default(self):
        voice = VoiceConfig.from_dict(_vosk_cfg(), lang_code="ru-RU")
        self.assertEqual(voice.lang_code, normalize_lang("ru-RU"))

    def test_tokenizer_does_not_fold_compound_phonemes(self):
        # the phonemizer already emits complete tokens; folding would merge
        # unrelated neighbours into a spurious compound key
        voice = VoiceConfig.from_dict(_vosk_cfg())
        self.assertFalse(voice.tokenizer.fold_compounds)


if __name__ == "__main__":
    unittest.main()
