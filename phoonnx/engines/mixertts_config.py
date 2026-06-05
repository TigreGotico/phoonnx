"""
Mixer-TTS ↔ phoonnx config bridge.

Mixer-TTS takes IPA symbol ids (espeak) and uses a fixed symbol table
(``[pad] + punctuation + letters + IPA letters``). This builds a native phoonnx
:class:`VoiceConfig` from that ordered symbol list — the vocab is baked into the
mirrored config, so no Mixer-TTS code is needed at runtime.

The vocoder (the model's paired Vocos / HiFi-GAN) is supplied separately via
``engine_params['vocoder_path']`` / the voice-index ``vocoder_url``.
"""
from typing import Any, Dict, List

from phoonnx.config import Alphabet, Engine, PhonemeType, VoiceConfig
from phoonnx.tokenizer import BlankBetween, TTSTokenizer, Vocabulary

# Mixer-TTS pad symbol (models/symbols.py)
_MIXER_PAD = "$"


def voice_config_from_mixer(
    symbols: List[str],
    *,
    sample_rate: int = 22050,
    lang_code: str = "en",
) -> VoiceConfig:
    """Build a :class:`VoiceConfig` from Mixer-TTS's ordered symbol list."""
    char2idx = {s: i for i, s in enumerate(symbols)}
    pad = symbols[0] if symbols else _MIXER_PAD
    tok = TTSTokenizer(
        Vocabulary(char2idx=char2idx, pad=pad),
        add_blank_char=False, add_blank_word=False, use_eos_bos=False,
        blank_at_start=False, blank_at_end=False)
    return VoiceConfig(
        tokenizer=tok, num_symbols=len(char2idx), num_speakers=1, num_langs=1,
        sample_rate=sample_rate, lang_code=lang_code,
        phoneme_type=PhonemeType.ESPEAK, alphabet=Alphabet.IPA,
        phonemizer_model=None, engine=Engine.MIXERTTS, add_diacritics=False,
        blank_between=BlankBetween.TOKENS_AND_WORDS,
        blank_at_start=False, blank_at_end=False,
        pad_token=pad, blank_token=pad, bos_token=None, eos_token=None,
        word_sep_token=" ")
