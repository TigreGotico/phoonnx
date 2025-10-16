import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence
from phoonnx.util import LOG
from phoonnx.phoneme_ids import (load_phoneme_ids, BlankBetween,
                                 DEFAULT_BLANK_WORD_TOKEN, DEFAULT_BLANK_TOKEN,
                                 DEFAULT_PAD_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN)

DEFAULT_NOISE_SCALE = 0.667
DEFAULT_LENGTH_SCALE = 1.0
DEFAULT_NOISE_W_SCALE = 0.8


class Engine(str, Enum):
    """voices trained with these frameworks are explicitly supported.
    This mainly affects the format of .json file and possibly tokenization"""
    PHOONNX = "phoonnx"
    PIPER = "piper"
    MIMIC3 = "mimic3"
    COQUI = "coqui"


class Alphabet(str, Enum):
    UNICODE = "unicode"
    IPA = "ipa"
    ARPA = "arpa" # en
    SAMPA = "sampa"
    XSAMPA = "x-sampa"
    HANGUL = "hangul" # ko
    KANA = "kana" # ja
    HIRA = "hira" # ja
    HEPBURN = "hepburn" # ja romanization
    KUNREI = "kunrei" # ja romanization
    NIHON = "nihon" # ja romanization
    PINYIN = "pinyin" # zh
    ERAAB = "eraab" # fa
    COTOVIA = "cotovia" # gl
    HANZI = "hanzi" # zh
    BUCKWALTER = "buckwalter" # ar



class PhonemeType(str, Enum):
    RAW = "raw"  # direct phonemes
    UNICODE = "unicode"  # unicode codepoints
    GRAPHEMES = "graphemes" # text characters

    MISAKI = "misaki"
    ESPEAK = "espeak"
    GRUUT = "gruut"
    GORUUT = "goruut"
    EPITRAN = "epitran"
    BYT5 = "byt5"
    CHARSIU = "charsiu"  # technically same as byt5, but needs special handling for whitespace
    TRANSPHONE = "transphone"
    MIRANDESE = "mwl_phonemizer"

    DEEPPHONEMIZER = "deepphonemizer" # en
    OPENPHONEMIZER = "openphonemizer" # en
    G2PEN = "g2pen" # en

    TUGAPHONE = "tugaphone"  # pt
    G2PFA = "g2pfa"
    OPENJTALK = "openjtalk" # ja
    CUTLET = "cutlet" # ja
    PYKAKASI = "pykakasi" # ja
    COTOVIA = "cotovia"  # galician  (no ipa!)
    PHONIKUD = "phonikud"  # hebrew
    MANTOQ = "mantoq"  # arabic
    VIPHONEME = "viphoneme" # vietnamese
    G2PK = "g2pk" # korean
    KOG2PK = "kog2p" # korean
    G2PC = "g2pc" # chinese
    G2PM = "g2pm" # chinese
    PYPINYIN = "pypinyin" # chinese
    XPINYIN = "xpinyin" # chinese
    JIEBA = "jieba" # chinese  (not a real phonemizer!)


@dataclass
class VoiceConfig:
    """TTS model configuration"""

    num_symbols: int
    """Number of phonemes."""

    num_speakers: int
    """Number of speakers."""

    num_langs: int
    """Number of langs."""

    sample_rate: int
    """Sample rate of output audio."""

    lang_code: Optional[str]
    """Name of espeak-ng voice or alphabet."""

    phoneme_id_map: Optional[Mapping[str, Sequence[int]]]
    """Phoneme -> [id,]. Used for phoneme-based models."""

    phoneme_type: PhonemeType
    """espeak, byt5, text, cotovia, or graphemes."""

    alphabet: Optional[Alphabet]

    phonemizer_model: Optional[str]
    """for phonemizers that allow changing base model """

    speaker_id_map: Mapping[str, int] = field(default_factory=dict)
    """Speaker -> id"""

    lang_id_map: Mapping[str, int] = field(default_factory=dict)
    """lang-code -> id"""

    # Info about what framework was used to train the model
    engine: Engine = Engine.PHOONNX

    # Inference settings
    length_scale: float = DEFAULT_LENGTH_SCALE
    noise_scale: float = DEFAULT_NOISE_SCALE
    noise_w_scale: float = DEFAULT_NOISE_W_SCALE
    add_diacritics: bool = None # arabic and hebrew

    # tokenization settings
    blank_at_start: bool = True
    blank_at_end: bool = True
    include_whitespace: Optional[bool] = True
    pad_token: Optional[str] = DEFAULT_PAD_TOKEN
    blank_token: Optional[str] = DEFAULT_PAD_TOKEN
    bos_token: Optional[str] = DEFAULT_BOS_TOKEN
    eos_token: Optional[str] = DEFAULT_EOS_TOKEN
    word_sep_token: Optional[str] = DEFAULT_BLANK_WORD_TOKEN
    blank_between: BlankBetween = BlankBetween.TOKENS_AND_WORDS

    def __post_init__(self):
        """
        Finalize dataclass defaults after initialization.
        
        If `add_diacritics` is None, sets it to False; if `lang_code` is present and starts with "ar", sets `add_diacritics` to True. Ensures `lang_code` is set to "und" when not provided.
        """
        if self.add_diacritics is None:
            self.add_diacritics = False
            if self.lang_code and self.lang_code.startswith("ar"):
                self.add_diacritics = True
        self.lang_code = self.lang_code or "und"

    @staticmethod
    def is_mimic3(config: dict[str, Any]) -> bool:
        # https://huggingface.co/mukowaty/mimic3-voices

        # mimic3 models indicate a phonemizer strategy in their config
        if ("phonemizer" not in config or
                not isinstance(config["phonemizer"], str)):
            return False

        # mimic3 models include a "phonemes" section with token info
        if "phonemes" not in config or not isinstance(config["phonemes"], dict):
            return False

        # validate phonemizer type as expected by mimic3
        phonemizer = config["phonemizer"]
        # class Phonemizer(str, Enum):
        #     SYMBOLS = "symbols"
        #     GRUUT = "gruut"
        #     ESPEAK = "espeak"
        #     EPITRAN = "epitran"
        if phonemizer not in ["symbols", "gruut", "espeak", "epitran"]:
            return False

        return True

    @staticmethod
    def is_piper(config: dict[str, Any]) -> bool:
        if "piper_version" in config:
            return True
        # piper models indicate a phonemizer strategy in their config
        if ("phoneme_type" not in config or
                not isinstance(config["phoneme_type"], str)):
            return False

        # piper models include a "phoneme_id_map" section mapping phonemes to int
        if "phoneme_id_map" not in config or not isinstance(config["phoneme_id_map"], dict):
            return False

        # validate phonemizer type as expected by piper
        phonemizer = config["phoneme_type"]
        if phonemizer not in ["text", "espeak"]:
            return False

        return True

    @staticmethod
    def is_coqui_vits(config: dict[str, Any]) -> bool:
        # coqui vits grapheme models include a "characters" section with token info
        if "characters" not in config or not isinstance(config["characters"], dict):
            return False

        # double check this was trained with coqui
        if config["characters"].get("characters_class", "") not in ["TTS.tts.models.vits.VitsCharacters",
                                                                    "TTS.tts.utils.text.characters.Graphemes"]:
            return False

        return True

    @staticmethod
    def is_phoonnx(config: dict[str, Any]) -> bool:
        return "phoonnx_version" in config

    @staticmethod
    def from_dict(config: dict[str, Any],
                  phonemes_txt: Optional[str] = None,
                  lang_code: Optional[str] = None,
                  phoneme_type_str: Optional[str] = None) -> "VoiceConfig":
        """
        Create a VoiceConfig from a configuration dictionary and an optional external phonemes file.
        
        Constructs a VoiceConfig by deriving engine, alphabet, phoneme mapping, and inference/synthesis settings from the provided config and optional phonemes_txt. Detects model type (Phoonnx, Piper, Mimic3, Coqui) and applies model-specific defaults and mappings; optional lang_code and phoneme_type_str override values in the config.
        
        Parameters:
            config (dict[str, Any]): Parsed model configuration dictionary.
            phonemes_txt (Optional[str]): Path to an external phonemes file (.txt or .json) used to build or override the phoneme id mapping.
            lang_code (Optional[str]): Optional language code to override the config's language.
            phoneme_type_str (Optional[str]): Optional phoneme type name to override the config's phoneme type.
        
        Returns:
            VoiceConfig: A populated VoiceConfig instance with fields set from the config and any provided phonemes file.
        
        Raises:
            ValueError: If the model is detected as Mimic3 but no phonemes_txt is provided.
        """
        blank_type = BlankBetween.TOKENS_AND_WORDS
        lang_code = lang_code or config.get("lang_code")
        phoneme_type_str = phoneme_type_str or config.get("phoneme_type")
        phoneme_id_map = config.get("phoneme_id_map")
        alphabet = config.get("alphabet")
        engine = Engine.PHOONNX
        diacritics = False

        if phonemes_txt:
            if phonemes_txt.endswith(".txt"):
                # either from mimic3 models or as an override at runtime
                with open(phonemes_txt, "r", encoding="utf-8") as ids_file:
                    phoneme_id_map = load_phoneme_ids(ids_file)
            elif phonemes_txt.endswith(".json"):
                with open(phonemes_txt) as ids_file:
                    phoneme_id_map = json.load(ids_file)

        if VoiceConfig.is_phoonnx(config):
            engine = Engine.PHOONNX

            lang_code = lang_code or config.get("lang_code")
            phoneme_type_str = config.get("phoneme_type", PhonemeType.ESPEAK.value)
            alphabet = Alphabet(config.get("alphabet", "ipa"))
            diacritics = config.get("inference", {}).get("add_diacritics", True)

            config["pad"] =  DEFAULT_PAD_TOKEN
            config["blank"] = DEFAULT_BLANK_TOKEN
            config["bos"] = DEFAULT_BOS_TOKEN
            config["eos"] = DEFAULT_EOS_TOKEN
        # check if model was trained for PiperTTS
        elif VoiceConfig.is_piper(config):
            engine = Engine.PIPER

            lang_code = lang_code or (config.get("language", {}).get("code") or
                         config.get("espeak", {}).get("voice"))
            diacritics = lang_code.startswith("ar")
            phoneme_type_str = config.get("phoneme_type", PhonemeType.ESPEAK.value)
            if phoneme_type_str == "text":
                phoneme_type_str = PhonemeType.UNICODE.value
                alphabet = Alphabet.UNICODE
            else:
                alphabet = Alphabet.IPA

            # not configurable in piper
            config["pad"] =  DEFAULT_PAD_TOKEN
            config["blank"] = DEFAULT_BLANK_TOKEN
            config["bos"] = DEFAULT_BOS_TOKEN
            config["eos"] = DEFAULT_EOS_TOKEN

        # check if model was trained for Mimic3
        elif VoiceConfig.is_mimic3(config):
            engine = Engine.MIMIC3

            if not phonemes_txt:
                raise ValueError("mimic3 models require an external phonemes.txt file in addition to the config")
            lang_code = config.get("text_language")
            phoneme_type_str = config.get("phonemizer", PhonemeType.GRUUT.value)
            # read phoneme settings
            phoneme_cfg = config.get("phonemes", {})
            blank_type = BlankBetween(phoneme_cfg.get("blank_between", "tokens_and_words"))
            config.update(phoneme_cfg)

            if phoneme_type_str == "symbols":
                # Mimic3 "symbols" models are grapheme models
                # symbol map comes from phonemes_txt
                phoneme_type_str = PhonemeType.GRAPHEMES.value
                alphabet = Alphabet.UNICODE
            else:
                alphabet = Alphabet.IPA

        # check if model was trained with Coqui
        elif VoiceConfig.is_coqui_vits(config):
            engine = Engine.COQUI
            phoneme_type_str = PhonemeType.GRAPHEMES.value
            alphabet = Alphabet.UNICODE

            # NOTE: lang code usually not provided and often wrong :(
            ds = config.get("datasets", [])
            if ds and not lang_code:
                lang_code = ds[0].get("language")

            characters_config = config.get("characters", {})
            if config.get("add_blank", True):
                blank_type = BlankBetween.TOKENS
                characters_config["blank"] = characters_config.get("blank") or "<BLNK>"
            config.update(characters_config)
            # For Coqui VITS grapheme models, build phoneme_id_map from characters
            characters = characters_config.get("characters")
            punctuations = characters_config.get("punctuations")

            if not config.get("enable_eos_bos_chars", True):
                config["bos"] = config["eos"] = None

            # Construct vocabulary based on the order defined in the original Graphemes class
            # [PAD, EOS, BOS, BLANK, CHARACTERS, PUNCTUATIONS]
            vocab_list = []

            if characters_config.get("pad") is not None:
                vocab_list.append(characters_config["pad"])

            # ?? - haven't see any coqui model
            # adding bos and eos to vocab_list

            #if characters_config.get("eos") is not None:
            #    vocab_list.append(characters_config["eos"])
            #if characters_config.get("bos") is not None:
            #    vocab_list.append(characters_config["bos"])

            if punctuations:
                vocab_list.extend(list(punctuations))
            if characters:
                vocab_list.extend(list(characters))


            if characters_config.get("blank") is not None:
                vocab_list.append(characters_config["blank"])

            # Ensure unique characters and sort if needed (though not strictly necessary for map creation)
            # This part of logic was previously in Graphemes, now implicitly handled by set/list conversion
            phoneme_id_map = {char: idx for idx, char in enumerate(vocab_list)}

        phoneme_type = PhonemeType(phoneme_type_str)
        LOG.debug(f"phonemizer: {phoneme_type}")
        inference = config.get("inference", {})

        include_whitespace = " " in config.get("characters", "") or " " in config.get("phoneme_id_map", {})
        return VoiceConfig(
            num_langs=config.get("num_langs", 1),
            num_symbols=config.get("num_symbols", 256),
            num_speakers=config.get("num_speakers", 1),
            sample_rate=config.get("audio", {}).get("sample_rate", 16000),
            noise_scale=inference.get("noise_scale", DEFAULT_NOISE_SCALE),
            length_scale=inference.get("length_scale", DEFAULT_LENGTH_SCALE),
            noise_w_scale=inference.get("noise_w", DEFAULT_NOISE_W_SCALE),
            add_diacritics=diacritics,
            lang_code=lang_code,
            alphabet=alphabet,
            engine=engine,
            phonemizer_model=config.get("phonemizer_model"),
            phoneme_id_map=phoneme_id_map,
            phoneme_type=phoneme_type,
            speaker_id_map=config.get("speaker_id_map", {}),
            blank_between=blank_type,
            include_whitespace=include_whitespace,
            blank_at_start=config.get("blank_at_start", True),
            blank_at_end=config.get("blank_at_end", True),
            pad_token=config.get("pad"),
            blank_token=config.get("blank"),
            bos_token=config.get("bos"),
            eos_token=config.get("eos"),
            word_sep_token=config.get("word_sep_token") or config.get("blank_word", " ")
        )


@dataclass
class SynthesisConfig:
    """Configuration for synthesis."""

    speaker_id: Optional[int] = None
    """Index of speaker to use (multi-speaker voices only)."""

    lang_id: Optional[int] = None
    """Index of lang to use (multi-lang voices only)."""

    length_scale: Optional[float] = None
    """Phoneme length scale (< 1 is faster, > 1 is slower)."""

    noise_scale: Optional[float] = None
    """Amount of generator noise to add."""

    noise_w_scale: Optional[float] = None
    """Amount of phoneme width noise to add."""

    normalize_audio: bool = True
    """Enable/disable scaling audio samples to fit full range."""

    volume: float = 1.0
    """Multiplier for audio samples (< 1 is quieter, > 1 is louder)."""

    enable_phonetic_spellings: bool = True

    """for arabic and hebrew models"""
    add_diacritics: bool = True


def get_phonemizer(phoneme_type: PhonemeType,
                   alphabet: Alphabet = Alphabet.IPA,
                   model: Optional[str] = None) -> 'Phonemizer':
    from phoonnx.phonemizers import (EpitranPhonemizer, EspeakPhonemizer, OpenPhonemizer, OpenJTaklPhonemizer,
                                     ByT5Phonemizer, CharsiuPhonemizer, DeepPhonemizer, PersianPhonemizer,
                                     G2pCPhonemizer, G2pMPhonemizer, G2PKPhonemizer, G2PEnPhonemizer,
                                     TransphonePhonemizer, MirandesePhonemizer, GoruutPhonemizer, TugaphonePhonemizer,
                                     GruutPhonemizer, GraphemePhonemizer, MantoqPhonemizer, MisakiPhonemizer,
                                     KoG2PPhonemizer, PypinyinPhonemizer, PyKakasiPhonemizer, CotoviaPhonemizer,
                                     CutletPhonemizer, PhonikudPhonemizer, VIPhonemePhonemizer, XpinyinPhonemizer,
                                     UnicodeCodepointPhonemizer, JiebaPhonemizer, RawPhonemes)
    if phoneme_type == PhonemeType.ESPEAK:
        phonemizer = EspeakPhonemizer()
    elif phoneme_type == PhonemeType.BYT5:
        phonemizer = ByT5Phonemizer(model)
    elif phoneme_type == PhonemeType.TUGAPHONE:
        phonemizer = TugaphonePhonemizer()
    elif phoneme_type == PhonemeType.CHARSIU:
        phonemizer = CharsiuPhonemizer(model)
    elif phoneme_type == PhonemeType.GRUUT:
        phonemizer = GruutPhonemizer()
    elif phoneme_type == PhonemeType.GORUUT:
        phonemizer = GoruutPhonemizer()
    elif phoneme_type == PhonemeType.EPITRAN:
        phonemizer = EpitranPhonemizer()
    elif phoneme_type == PhonemeType.MISAKI:
        phonemizer = MisakiPhonemizer()
    elif phoneme_type == PhonemeType.TRANSPHONE:
        phonemizer = TransphonePhonemizer()
    elif phoneme_type == PhonemeType.MIRANDESE:
        phonemizer = MirandesePhonemizer()
    elif phoneme_type == PhonemeType.DEEPPHONEMIZER:
        phonemizer = DeepPhonemizer(model)
    elif phoneme_type == PhonemeType.OPENPHONEMIZER:
        phonemizer = OpenPhonemizer()
    elif phoneme_type == PhonemeType.G2PEN:
        phonemizer = G2PEnPhonemizer(alphabet=alphabet)
    elif phoneme_type == PhonemeType.OPENJTALK:
        phonemizer = OpenJTaklPhonemizer(alphabet=alphabet)
    elif phoneme_type == PhonemeType.PYKAKASI:
        phonemizer = PyKakasiPhonemizer(alphabet=alphabet)
    elif phoneme_type == PhonemeType.CUTLET:
        phonemizer = CutletPhonemizer(alphabet=alphabet)
    elif phoneme_type == PhonemeType.G2PFA:
        phonemizer = PersianPhonemizer(alphabet=alphabet)
    elif phoneme_type == PhonemeType.PHONIKUD:
        phonemizer = PhonikudPhonemizer()
    elif phoneme_type == PhonemeType.MANTOQ:
        phonemizer = MantoqPhonemizer()
    elif phoneme_type == PhonemeType.VIPHONEME:
        phonemizer = VIPhonemePhonemizer()
    elif phoneme_type == PhonemeType.KOG2PK:
        phonemizer = KoG2PPhonemizer(alphabet=alphabet)
    elif phoneme_type == PhonemeType.G2PK:
        phonemizer = G2PKPhonemizer(alphabet=alphabet)
    elif phoneme_type == PhonemeType.PYPINYIN:
        phonemizer = PypinyinPhonemizer(alphabet=alphabet)
    elif phoneme_type == PhonemeType.XPINYIN:
        phonemizer = XpinyinPhonemizer(alphabet=alphabet)
    elif phoneme_type == PhonemeType.JIEBA:
        phonemizer = JiebaPhonemizer()
    elif phoneme_type == PhonemeType.G2PC:
        phonemizer = G2pCPhonemizer(alphabet=alphabet)
    elif phoneme_type == PhonemeType.G2PM:
        phonemizer = G2pMPhonemizer(alphabet=alphabet)
    elif phoneme_type == PhonemeType.COTOVIA:
        phonemizer = CotoviaPhonemizer()
    elif phoneme_type == PhonemeType.UNICODE:
        phonemizer = UnicodeCodepointPhonemizer()
    elif phoneme_type == PhonemeType.GRAPHEMES:
        phonemizer = GraphemePhonemizer()
    elif phoneme_type == PhonemeType.RAW:
        phonemizer = RawPhonemes()
    else:
        raise ValueError("invalid phonemizer")
    return phonemizer



if __name__ == "__main__":
    config_files = [
        "/home/miro/PycharmProjects/phoonnx_tts/sabela_cotovia_vits.json",
        "/home/miro/PycharmProjects/phoonnx_tts/celtia_vits.json",
        "/home/miro/PycharmProjects/phoonnx_tts/mimic3_gruut.json",
        "/home/miro/PycharmProjects/phoonnx_tts/mimic3_espeak.json",
        "/home/miro/PycharmProjects/phoonnx_tts/mimic3_epitran.json",
        "/home/miro/PycharmProjects/phoonnx_tts/mimic3_symbols.json",
        "/home/miro/PycharmProjects/phoonnx_tts/piper_espeak.json",
        "/home/miro/PycharmProjects/phoonnx_tts/vits-coqui-pt-cv/config.json",
        "/home/miro/PycharmProjects/phoonnx_tts/phonikud/model.config.json"
    ]
    phoneme_txts = [
        None,
        None,
        "/home/miro/PycharmProjects/phoonnx_tts/mimic3_ap/phonemes.txt",
        "/home/miro/PycharmProjects/phoonnx_tts/mimic3_ap/phonemes.txt",
        "/home/miro/PycharmProjects/phoonnx_tts/mimic3_ap/phonemes.txt",
        "/home/miro/PycharmProjects/phoonnx_tts/mimic3_ap/phonemes.txt",
        None,
        None,
        None
    ]
    print("Testing model config file parsing\n###############")
    for idx, cfile in enumerate(config_files):
        print(f"\nConfig file: {cfile}")
        with open(cfile) as f:
            config = json.load(f)
        print("Mimic3:", VoiceConfig.is_mimic3(config))
        print("Piper:", VoiceConfig.is_piper(config))
        print("Coqui:", VoiceConfig.is_coqui_vits(config))
        print("Phoonx:", VoiceConfig.is_phoonnx(config))
        cfg = VoiceConfig.from_dict(config, phoneme_txts[idx])
        print(cfg)

