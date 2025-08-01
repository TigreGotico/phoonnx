import json
import string
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence
from phoonnx.phoneme_ids import (load_phoneme_ids, BlankBetween,
                                 DEFAULT_BLANK_WORD_TOKEN, DEFAULT_BLANK_TOKEN, DEFAULT_PAD_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN)


DEFAULT_NOISE_SCALE = 0.667
DEFAULT_LENGTH_SCALE = 1.0
DEFAULT_NOISE_W_SCALE = 0.8

try:
    from ovos_utils.log import LOG
except ImportError:
    import logging
    LOG = logging.getLogger(__name__)


class PhonemeType(str, Enum):
    RAW = "raw"  # direct phonemes
    UNICODE = "unicode"  # unicode codepoints
    GRAPHEMES = "graphemes" # text characters

    MISAKI = "misaki"
    ESPEAK = "espeak"
    GRUUT = "gruut"
    EPITRAN = "epitran"
    BYT5 = "byt5"
    CHARSIU = "charsiu"  # technically same as byt5, but needs special handling for whitespace

    DEEPPHONEMIZER = "deepphonemizer" # en
    OPENPHONEMIZER = "openphonemizer" # en
    G2PEN = "g2pen" # en

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

    speaker_id_map: Mapping[str, int] = field(default_factory=dict)
    """Speaker -> id"""

    lang_id_map: Mapping[str, int] = field(default_factory=dict)
    """lang-code -> id"""

    # Inference settings
    length_scale: float = DEFAULT_LENGTH_SCALE
    noise_scale: float = DEFAULT_NOISE_SCALE
    noise_w_scale: float = DEFAULT_NOISE_W_SCALE

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
        # phoonnx models indicate a phonemizer strategy in their config
        if ("phoneme_type" not in config or
                not isinstance(config["phoneme_type"], str)):
            return False

        if "lang_code" not in config:
            return False

        # validate phonemizer type as expected
        phonemizer = config["phoneme_type"]
        if phonemizer not in list(PhonemeType):
            return False

        return True

    @staticmethod
    def is_cotovia(config: dict[str, Any]) -> bool:
        # no way to determine unless explicitly configured unfortunately
        # afaik only the sabela galician model uses this
        # will fallback to coqui "graphemes" if "cotovia" not specified,
        # this will work but will make mistakes
        if (not VoiceConfig.is_coqui_vits(config)
                or not VoiceConfig.is_phoonnx(config)):
            return False

        return config["phoneme_type"] == PhonemeType.COTOVIA

    @staticmethod
    def from_dict(config: dict[str, Any],
                  phonemes_txt: Optional[str] = None,
                  lang_code: Optional[str] = None,
                  phoneme_type_str: Optional[str] = None) -> "VoiceConfig":
        """Load configuration from a dictionary."""
        blank_type = BlankBetween.TOKENS_AND_WORDS
        lang_code = lang_code or config.get("lang_code")
        phoneme_type_str = phoneme_type_str or config.get("phoneme_type")
        phoneme_id_map = config.get("phoneme_id_map")

        if phonemes_txt:
            if phonemes_txt.endswith(".txt"):
                # either from mimic3 models or as an override at runtime
                with open(phonemes_txt, "r", encoding="utf-8") as ids_file:
                    phoneme_id_map = load_phoneme_ids(ids_file)
            elif phonemes_txt.endswith(".json"):
                with open(phonemes_txt) as ids_file:
                    phoneme_id_map = json.load(ids_file)

        # check if model was trained for PiperTTS
        if VoiceConfig.is_piper(config):
            lang_code = lang_code or (config.get("language", {}).get("code") or
                         config.get("espeak", {}).get("voice"))
            phoneme_type_str = config.get("phoneme_type", PhonemeType.ESPEAK.value)
            if phoneme_type_str == "text":
                phoneme_type_str = PhonemeType.UNICODE.value

            # not configurable in piper
            config["pad"] =  DEFAULT_PAD_TOKEN
            config["blank"] = DEFAULT_BLANK_TOKEN
            config["bos"] = DEFAULT_BOS_TOKEN
            config["eos"] = DEFAULT_EOS_TOKEN

        # check if model was trained for Mimic3
        elif VoiceConfig.is_mimic3(config):
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

        # check if model was trained with Coqui
        # NOTE: cotovia is included here
        elif VoiceConfig.is_coqui_vits(config):
            if VoiceConfig.is_cotovia(config):
                phoneme_type_str = PhonemeType.COTOVIA.value
            else:
                phoneme_type_str = PhonemeType.GRAPHEMES.value

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
            lang_code=lang_code,
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
        print("Cotovia:", VoiceConfig.is_cotovia(config))
        print("Phoonx:", VoiceConfig.is_phoonnx(config))
        cfg = VoiceConfig.from_dict(config, phoneme_txts[idx])
        print(cfg)


