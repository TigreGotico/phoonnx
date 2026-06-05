"""
OptiSpeech ↔ phoonnx config bridge.

OptiSpeech embeds all of its config inside the ONNX model metadata (under the
``"inference"`` key) rather than in an external JSON file. This module turns
that embedded metadata into a phoonnx :class:`VoiceConfig` (with its tokenizer)
so the rest of the pipeline works unchanged.

Usage (from ``TTSVoice.load``)::

    meta = adapter.parse_onnx_meta(session)
    config = voice_config_from_optispeech_meta(meta)
"""
import logging
from typing import Any, Dict, Optional

from phoonnx.config import Alphabet, Engine, PhonemeType, VoiceConfig
from phoonnx.tokenizer import BlankBetween, TTSTokenizer, Vocabulary

LOG = logging.getLogger(__name__)

# OptiSpeech default special symbols
_OS_PAD = "_"
_OS_BOS = "^"
_OS_EOS = "$"


def _build_tokenizer(
    input_symbols: Dict[str, int],
    special_symbols: Dict[str, Any],
    text_processor: Dict[str, Any],
) -> TTSTokenizer:
    """Reproduce OptiSpeech's tokenization from the embedded metadata."""
    voc = Vocabulary(
        char2idx=dict(input_symbols),
        pad=_OS_PAD,
        blank=_OS_PAD,  # OptiSpeech uses PAD as the interspersed blank
        bos=special_symbols.get("bos", _OS_BOS),
        eos=special_symbols.get("eos", _OS_EOS),
    )
    add_blank = bool(text_processor.get("add_blank", True))
    add_bos_eos = bool(text_processor.get("add_bos_eos", True))
    return TTSTokenizer(
        voc,
        add_blank_char=add_blank,
        add_blank_word=False,
        use_eos_bos=add_bos_eos,
        blank_at_start=add_blank,
        blank_at_end=add_blank,
    )


def voice_config_from_optispeech_meta(
    meta: Dict[str, Any],
    *,
    lang_code: Optional[str] = None,
    phoneme_type: Optional[PhonemeType] = None,
) -> VoiceConfig:
    """
    Build a full :class:`VoiceConfig` from OptiSpeech's embedded ONNX metadata
    (the parsed ``inference`` JSON blob).
    """
    input_symbols = meta.get("input_symbols", {})
    special_symbols = meta.get("special_symbols", {})
    text_processor = meta.get("text_processor", {})
    inference_args = meta.get("inference_args", {})
    languages = meta.get("languages", []) or []
    speakers = meta.get("speakers", []) or []
    sample_rate = meta.get("sample_rate", 22050)

    if not lang_code and languages:
        lang_code = languages[0]

    # OptiSpeech's IPATokenizer phonemizes with espeak/piper_phonemize.
    if phoneme_type is None:
        tokenizer_name = text_processor.get("tokenizer", "ipa")
        phoneme_type = PhonemeType.GRAPHEMES if tokenizer_name == "arabic-buck" else PhonemeType.ESPEAK

    add_bos_eos = bool(text_processor.get("add_bos_eos", True))
    add_blank = bool(text_processor.get("add_blank", True))

    return VoiceConfig(
        tokenizer=_build_tokenizer(input_symbols, special_symbols, text_processor),
        num_symbols=len(input_symbols),
        num_speakers=max(len(speakers), 1),
        num_langs=max(len(languages), 1),
        sample_rate=sample_rate,
        lang_code=lang_code,
        phoneme_type=phoneme_type,
        alphabet=Alphabet.IPA,
        phonemizer_model=None,
        engine=Engine.OPTISPEECH,
        speaker_id_map={name: idx for idx, name in enumerate(speakers)} if speakers else {},
        add_diacritics=False,
        engine_params={
            "d_factor": inference_args.get("d_factor", 1.0),
            "p_factor": inference_args.get("p_factor", 1.0),
            "e_factor": inference_args.get("e_factor", 1.0),
        },
        blank_between=BlankBetween.TOKENS_AND_WORDS,
        blank_at_start=add_blank,
        blank_at_end=add_blank,
        pad_token=_OS_PAD,
        blank_token=_OS_PAD,
        bos_token=_OS_BOS if add_bos_eos else None,
        eos_token=_OS_EOS if add_bos_eos else None,
        word_sep_token=" ",
    )
