"""
OptiSpeech ↔ phoonnx config bridge.

OptiSpeech embeds all its config inside the ONNX model metadata (under
the ``"inference"`` key) rather than using an external JSON file.  This
module translates that embedded metadata into the phoonnx VoiceConfig /
TTSTokenizer / Vocabulary objects so that the rest of the phoonnx
pipeline (phonemization, tokenization, synthesis) works seamlessly.

Usage (from TTSVoice.load or model_manager)::

    from phoonnx.engines.optispeech_config import voice_config_from_optispeech_meta

    meta = adapter.parse_onnx_meta(session)
    config = voice_config_from_optispeech_meta(meta)
"""
import json
import logging
from typing import Any, Dict, Optional

from phoonnx.config import (
    Alphabet,
    Engine,
    PhonemeType,
    VoiceConfig,
)
from phoonnx.tokenizer import (
    BlankBetween,
    TTSTokenizer,
    Vocabulary,
    DEFAULT_PAD_TOKEN,
    DEFAULT_BOS_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_BLANK_WORD_TOKEN,
)

LOG = logging.getLogger(__name__)

# OptiSpeech default special symbols
_OS_PAD = "_"
_OS_BOS = "^"
_OS_EOS = "$"


def _build_vocabulary(input_symbols: Dict[str, int]) -> Vocabulary:
    """
    Convert OptiSpeech's ``input_symbols`` (symbol→id map) into a
    phoonnx ``Vocabulary``.

    The input_symbols dict comes straight from the ONNX metadata
    (``infer_params["input_symbols"]``).  It maps every IPA character
    (including specials like ``_``, ``^``, ``$``) to its integer ID.
    """
    # Invert to id→symbol for the Vocabulary constructor
    id_to_symbol = {v: k for k, v in input_symbols.items()}
    max_id = max(id_to_symbol.keys()) if id_to_symbol else 0
    # Build a dense list, using empty string for any gaps
    symbols = [""] * (max_id + 1)
    for idx, sym in id_to_symbol.items():
        symbols[idx] = sym
    return Vocabulary(symbols=symbols, symbol_to_id=dict(input_symbols))


def _build_tokenizer(
    input_symbols: Dict[str, int],
    special_symbols: Dict[str, Any],
    text_processor: Dict[str, Any],
) -> TTSTokenizer:
    """
    Build a phoonnx TTSTokenizer that reproduces OptiSpeech's
    tokenization behaviour.

    OptiSpeech tokenization options (from ``text_processor``):
      - ``add_blank``: intersperse blank (id 0) between tokens
      - ``add_bos_eos``: prepend BOS / append EOS
    """
    vocab = _build_vocabulary(input_symbols)

    # Resolve special token IDs
    pad_id = input_symbols.get(_OS_PAD, 0)
    bos_sym = special_symbols.get("bos", _OS_BOS)
    eos_sym = special_symbols.get("eos", _OS_EOS)

    add_blank = text_processor.get("add_blank", True)
    add_bos_eos = text_processor.get("add_bos_eos", True)

    if add_blank:
        blank_between = BlankBetween.TOKENS_AND_WORDS
    else:
        blank_between = BlankBetween.NONE

    return TTSTokenizer(
        vocab=vocab,
        blank_between=blank_between,
        blank_at_start=add_bos_eos,
        blank_at_end=add_bos_eos,
        pad_token=_OS_PAD,
        blank_token=_OS_PAD,  # OptiSpeech uses PAD as the intersperse blank
        bos_token=bos_sym if add_bos_eos else None,
        eos_token=eos_sym if add_bos_eos else None,
        word_sep_token=" ",
    )


def voice_config_from_optispeech_meta(
    meta: Dict[str, Any],
    *,
    lang_code: Optional[str] = None,
    phoneme_type: Optional[PhonemeType] = None,
) -> VoiceConfig:
    """
    Build a full ``VoiceConfig`` from OptiSpeech's embedded metadata.

    Parameters
    ----------
    meta : dict
        The parsed JSON from ``session.get_modelmeta().custom_metadata_map["inference"]``.
    lang_code : str, optional
        Override language code.  If not given, uses the first language
        listed in the metadata.
    phoneme_type : PhonemeType, optional
        Override phoneme type.  Defaults to ESPEAK (which is what
        OptiSpeech's IPATokenizer uses under the hood via piper_phonemize).

    Returns
    -------
    VoiceConfig
    """
    input_symbols = meta.get("input_symbols", {})
    special_symbols = meta.get("special_symbols", {})
    text_processor_cfg = meta.get("text_processor", {})
    inference_args = meta.get("inference_args", {})
    languages = meta.get("languages", [])
    speakers = meta.get("speakers", [])
    sample_rate = meta.get("sample_rate", 22050)

    # Resolve language
    if not lang_code and languages:
        lang_code = languages[0]

    # Resolve phoneme type from tokenizer name
    tokenizer_name = text_processor_cfg.get("tokenizer", "ipa")
    if phoneme_type is None:
        if tokenizer_name == "ipa":
            phoneme_type = PhonemeType.ESPEAK
        elif tokenizer_name == "arabic-buck":
            phoneme_type = PhonemeType.GRAPHEMES
        else:
            phoneme_type = PhonemeType.ESPEAK

    # Build tokenizer
    tokenizer = _build_tokenizer(input_symbols, special_symbols, text_processor_cfg)

    num_symbols = len(input_symbols)
    num_speakers = max(len(speakers), 1)

    return VoiceConfig(
        tokenizer=tokenizer,
        num_symbols=num_symbols,
        num_speakers=num_speakers,
        num_langs=max(len(languages), 1),
        sample_rate=sample_rate,
        lang_code=lang_code,
        phoneme_type=phoneme_type,
        alphabet=Alphabet.IPA,
        phonemizer_model=None,
        engine=Engine.OPTISPEECH,
        # OptiSpeech uses d/p/e factors, not noise/length/noise_w
        # We store sentinel values for the VITS-style fields
        noise_scale=0.0,
        length_scale=1.0,
        noise_w_scale=0.0,
        # Engine-specific params
        engine_params={
            "d_factor": inference_args.get("d_factor", 1.0),
            "p_factor": inference_args.get("p_factor", 1.0),
            "e_factor": inference_args.get("e_factor", 1.0),
        },
        speaker_id_map=(
            {name: idx for idx, name in enumerate(speakers)}
            if speakers else {}
        ),
        add_diacritics=False,
        # Tokenization matches OptiSpeech defaults
        blank_between=(
            BlankBetween.TOKENS_AND_WORDS
            if text_processor_cfg.get("add_blank", True)
            else BlankBetween.NONE
        ),
        blank_at_start=text_processor_cfg.get("add_bos_eos", True),
        blank_at_end=text_processor_cfg.get("add_bos_eos", True),
        pad_token=_OS_PAD,
        blank_token=_OS_PAD,
        bos_token=_OS_BOS if text_processor_cfg.get("add_bos_eos", True) else None,
        eos_token=_OS_EOS if text_processor_cfg.get("add_bos_eos", True) else None,
        word_sep_token=" ",
    )
