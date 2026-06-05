"""
Coqui-TTS ↔ phoonnx config bridge.

Builds a phoonnx :class:`VoiceConfig` from a Coqui-TTS ``config.json``, used at
conversion time for any coqui acoustic model — GlowTTS, VITS, and FastPitch all
share coqui's tokenizer conventions. The tricky bits this reproduces exactly:

- **phonemizer** — coqui records which G2P produced the training phonemes; gruut
  and espeak emit different IPA, so we must phonemize with the same backend.
- **vocab order** — ``Graphemes``/``IPAPhonemes`` default ``is_sorted=True`` (the
  symbol set is sorted before id assignment); ``VitsCharacters`` instead keeps
  ``[pad] + punctuations + (graphemes + ipa) + [blank]`` unsorted with
  ``is_unique=False`` (no dedup; blank id = full-list length). Getting either
  wrong shifts ids and yields the right voice saying non-words.

The ``engine`` argument selects the target adapter (GlowTTS / VITS=coqui /
FastPitch); the tokenizer is identical across them.
"""
from typing import Any, Dict

from phoonnx.config import Alphabet, Engine, PhonemeType, VoiceConfig
from phoonnx.tokenizer import BlankBetween, TTSTokenizer, Vocabulary

_PHONEMIZER = {"gruut": PhonemeType.GRUUT, "espeak": PhonemeType.ESPEAK,
               "espeak-ng": PhonemeType.ESPEAK, "espeakng": PhonemeType.ESPEAK}


def voice_config_from_coqui(config: Dict[str, Any], *, lang_code: str,
                            engine: Engine = Engine.GLOWTTS) -> VoiceConfig:
    """Build a :class:`VoiceConfig` from a Coqui-TTS ``config.json``."""
    ch = config.get("characters", {})
    audio = config.get("audio", {})
    use_phonemes = bool(config.get("use_phonemes", False))
    _phon = _PHONEMIZER.get(str(config.get("phonemizer") or "").lower(), PhonemeType.ESPEAK)
    add_blank = bool(config.get("add_blank", False))
    use_eos_bos = bool(config.get("enable_eos_bos_chars", False))
    pad, eos, bos = ch.get("pad", "_"), ch.get("eos", "~"), ch.get("bos", "^")
    blank = ch.get("blank") or "<BLNK>"

    # VITS uses its own VitsCharacters, whose _create_vocab OVERRIDES the base sort:
    # [pad] + punctuations + (graphemes + ipa_characters, unsorted) + [blank], with
    # the blank interspersed at synthesis. is_unique=False -> NO dedup, char_to_id
    # keeps the LAST occurrence, num_chars counts the full list (incl. blank).
    if "Vits" in str(ch.get("characters_class") or ""):
        combined = list(ch.get("characters") or "") + list(ch.get("phonemes") or "")
        full = [pad] + list(ch.get("punctuations") or "") + combined + [blank]
        char2idx = {c: i for i, c in enumerate(full)}
        n_spk = config.get("num_speakers") or config.get("model_args", {}).get("num_speakers") or 1
        tok = TTSTokenizer(
            Vocabulary(char2idx=char2idx, pad=pad, blank=blank),
            add_blank_char=True, add_blank_word=False, use_eos_bos=False,
            blank_at_start=True, blank_at_end=True)
        return VoiceConfig(
            tokenizer=tok, num_symbols=len(full), num_speakers=max(int(n_spk), 1),
            num_langs=1, sample_rate=audio.get("sample_rate", 22050), lang_code=lang_code,
            phoneme_type=_phon if use_phonemes else PhonemeType.GRAPHEMES,
            alphabet=Alphabet.IPA if use_phonemes else Alphabet.UNICODE,
            phonemizer_model=None, engine=engine, add_diacritics=False,
            blank_between=BlankBetween.TOKENS_AND_WORDS, blank_at_start=True, blank_at_end=True,
            pad_token=pad, blank_token=blank, bos_token=None, eos_token=None, word_sep_token=" ")

    # Graphemes / IPAPhonemes: [pad, eos, bos, blank?] + sorted(symbols) + punctuations.
    # IPAPhonemes stores its IPA set in the `characters` field; some configs use
    # the separate `phonemes` field. For phoneme models prefer `phonemes`, else
    # fall back to `characters`.
    if use_phonemes:
        symbol_set = list(ch.get("phonemes") or ch.get("characters") or "")
    else:
        symbol_set = list(ch.get("characters") or "")
    if ch.get("is_unique", False):
        symbol_set = list(dict.fromkeys(symbol_set))
    if ch.get("is_sorted", True):
        symbol_set = sorted(symbol_set)
    specials = [pad, eos, bos] + ([blank] if add_blank else [])
    char2idx = {}
    for s in specials + symbol_set + list(ch.get("punctuations", "")):
        if s not in char2idx:
            char2idx[s] = len(char2idx)
    tok = TTSTokenizer(
        Vocabulary(char2idx=char2idx, pad=pad, bos=bos, eos=eos, blank=blank if add_blank else None),
        add_blank_char=add_blank, add_blank_word=False, use_eos_bos=use_eos_bos,
        blank_at_start=add_blank, blank_at_end=add_blank)
    return VoiceConfig(
        tokenizer=tok, num_symbols=len(char2idx), num_speakers=1, num_langs=1,
        sample_rate=audio.get("sample_rate", 22050), lang_code=lang_code,
        phoneme_type=_phon if use_phonemes else PhonemeType.GRAPHEMES,
        alphabet=Alphabet.IPA if use_phonemes else Alphabet.UNICODE,
        phonemizer_model=None, engine=engine, add_diacritics=False,
        blank_between=BlankBetween.TOKENS_AND_WORDS,
        blank_at_start=add_blank, blank_at_end=add_blank,
        pad_token=pad, blank_token=blank if add_blank else pad,
        bos_token=bos if use_eos_bos else None, eos_token=eos if use_eos_bos else None,
        word_sep_token=" ")
