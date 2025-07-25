from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Dict

DEFAULT_NOISE_SCALE = 0.667
DEFAULT_LENGTH_SCALE = 1.0
DEFAULT_NOISE_W_SCALE = 0.8

try:
    from ovos_utils.log import LOG
except ImportError:
    import logging
    LOG = logging.getLogger(__name__)


class PhonemeType(str, Enum):
    ESPEAK = "espeak"
    BYT5 = "byt5"
    UNICODE = "unicode"
    COTOVIA = "cotovia"
    GRAPHEMES = "graphemes"


@dataclass
class VoiceConfig:
    """TTS model configuration"""

    num_symbols: int
    """Number of phonemes."""

    num_speakers: int
    """Number of speakers."""

    sample_rate: int
    """Sample rate of output audio."""

    lang_code: str
    """Name of espeak-ng voice or alphabet."""

    phoneme_id_map: Optional[Mapping[str, Sequence[int]]]
    """Phoneme -> [id,]. Used for phoneme-based models."""

    phoneme_type: PhonemeType
    """espeak, byt5, text, cotovia, or graphemes."""

    speaker_id_map: Mapping[str, int] = field(default_factory=dict)
    """Speaker -> id"""

    # Inference settings
    length_scale: float = DEFAULT_LENGTH_SCALE
    noise_scale: float = DEFAULT_NOISE_SCALE
    noise_w_scale: float = DEFAULT_NOISE_W_SCALE

    # fields for grapheme-based models
    characters_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    """Configuration for Graphemes class (characters, punctuations, pad)."""
    add_blank: bool = False
    """Whether to add blank tokens between characters for grapheme models."""
    use_eos_bos: bool = False
    """Whether to add BOS/EOS tokens for grapheme models."""


    @staticmethod
    def from_dict(config: dict[str, Any]) -> "VoiceConfig":
        """Load configuration from a dictionary."""
        inference = config.get("inference", {})
        lang_code = config.get("lang_code")

        if "phoneme_type" not in config:
            if "espeak" in config:
                LOG.debug(f"Autodetected phonemizer type: {PhonemeType.ESPEAK}")
                config["phoneme_type"] = PhonemeType.ESPEAK
            elif "characters" in config:
                LOG.debug(f"Autodetected phonemizer type: {PhonemeType.GRAPHEMES}")
                config["phoneme_type"] = PhonemeType.GRAPHEMES

        phoneme_type_str = config.get("phoneme_type", PhonemeType.BYT5.value)

        # piper TTS models compat
        if phoneme_type_str == "text":
            phoneme_type_str = PhonemeType.UNICODE.value
        elif phoneme_type_str == "espeak":
            lang_code = lang_code or config["espeak"].get("voice", "")

        phoneme_type = PhonemeType(phoneme_type_str)

        LOG.debug(f"phonemizer: {phoneme_type}")

        # Handle phoneme_id_map for phoneme-based models
        phoneme_id_map = config.get("phoneme_id_map")

        # Fields for grapheme-based models
        characters_config = config.get("characters")
        add_blank = config.get("add_blank", True)
        use_eos_bos = config.get("use_eos_bos", False)

        return VoiceConfig(
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
            characters_config=characters_config,
            add_blank=add_blank,
            use_eos_bos=use_eos_bos
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary."""
        config_dict = {
            "audio": {
                "sample_rate": self.sample_rate,
            },
            "espeak": { # Keep for compatibility, even if lang_code is elsewhere
                "voice": self.lang_code,
            },
            "phoneme_type": self.phoneme_type.value,
            "num_symbols": self.num_symbols,
            "num_speakers": self.num_speakers,
            "inference": {
                "noise_scale": self.noise_scale,
                "length_scale": self.length_scale,
                "noise_w": self.noise_w_scale,
            },
            "phoneme_id_map": self.phoneme_id_map,
            "speaker_id_map": self.speaker_id_map,
            # fields for grapheme models
            "characters": self.characters_config,
            "add_blank": self.add_blank,
            "use_eos_bos": self.use_eos_bos,
        }
        return config_dict


@dataclass
class SynthesisConfig:
    """Configuration for synthesis."""

    speaker_id: Optional[int] = None
    """Index of speaker to use (multi-speaker voices only)."""

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
