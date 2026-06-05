"""
GlowTTS (Larynx) ↔ phoonnx config bridge.

Larynx GlowTTS voices ship a training ``config.json`` (audio + model params)
and a ``phonemes.txt`` symbol table (``id phoneme`` per line, gruut IPA). This
module turns those into a phoonnx :class:`VoiceConfig` whose tokenizer
reproduces Larynx's tokenization (gruut phonemes, blank-interspersed).

The vocoder (Larynx HiFi-GAN) is supplied separately via
``engine_params['vocoder_path']`` / the voice-index ``vocoder_url``.
"""
from typing import Any, Dict, Optional

from phoonnx.config import Alphabet, Engine, PhonemeType, VoiceConfig
from phoonnx.tokenizer import BlankBetween, TTSTokenizer, Vocabulary

_GLOW_PAD = "_"


def voice_config_from_coqui(config: Dict[str, Any], *, lang_code: str,
                            engine: Engine = Engine.GLOWTTS) -> VoiceConfig:
    """
    Build a :class:`VoiceConfig` from a Coqui-TTS GlowTTS ``config.json``.

    Coqui's ``BaseCharacters._create_vocab`` order is
    ``[pad, eos, bos, blank] + characters + punctuations`` (blank at id 3 when
    ``add_blank``). Graphemes when ``use_phonemes`` is false, otherwise espeak IPA.
    """
    ch = config.get("characters", {})
    audio = config.get("audio", {})
    use_phonemes = bool(config.get("use_phonemes", False))
    # coqui records which G2P produced the training phonemes; gruut and espeak
    # emit different IPA (e.g. "words" -> gruut "wˈɚdz" vs espeak "wˈɜːdz"), so we
    # must phonemize with the SAME backend or the model gets wrong ids.
    _PHONEMIZER = {"gruut": PhonemeType.GRUUT, "espeak": PhonemeType.ESPEAK,
                   "espeak-ng": PhonemeType.ESPEAK, "espeakng": PhonemeType.ESPEAK}
    _phon = _PHONEMIZER.get(str(config.get("phonemizer") or "").lower(), PhonemeType.ESPEAK)
    add_blank = bool(config.get("add_blank", False))
    use_eos_bos = bool(config.get("enable_eos_bos_chars", False))
    pad, eos, bos = ch.get("pad", "_"), ch.get("eos", "~"), ch.get("bos", "^")
    blank = ch.get("blank") or "<BLNK>"

    # VITS uses its own VitsCharacters, whose _create_vocab OVERRIDES the base sort:
    # [pad] + punctuations + (graphemes + ipa_characters, unsorted) + [blank], with
    # the blank interspersed at synthesis. Build that exact table.
    if "Vits" in str(ch.get("characters_class") or ""):
        combined = list(ch.get("characters", "")) + list(ch.get("phonemes", ""))
        full = [pad] + list(ch.get("punctuations", "")) + combined + [blank]
        # VitsCharacters is is_unique=False: NO dedup, char_to_id keeps the LAST
        # occurrence, and num_chars counts the full list (incl. the trailing blank).
        # Deduping shifts the blank id by one -> interspersed garbage.
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

    symbol_set = list(ch.get("phonemes", "") if use_phonemes else ch.get("characters", ""))
    # coqui's Graphemes/IPAPhonemes both default is_sorted=True: the symbol set is
    # sorted alphabetically *before* ids are assigned (config may override). Getting
    # this wrong shifts every id -> the right voice saying non-words.
    if ch.get("is_unique", False):
        symbol_set = list(dict.fromkeys(symbol_set))
    if ch.get("is_sorted", True):
        symbol_set = sorted(symbol_set)
    # coqui order: [pad, eos, bos, blank] + sorted(symbols) + punctuations (config order).
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


def voice_config_from_larynx(
    config: Dict[str, Any],
    phonemes_txt: str,
    *,
    lang_code: str,
    phoneme_type: PhonemeType = PhonemeType.GRUUT,
) -> VoiceConfig:
    """Build a :class:`VoiceConfig` from a Larynx GlowTTS config + phonemes.txt."""
    audio = config.get("audio", {})
    model = config.get("model", {})

    # phonemes.txt is "<id> <phoneme>" per line
    tok = TTSTokenizer.from_tokens_txt(phonemes_txt, id_first=True)
    # GlowTTS interleaves the blank (PAD, id 0) between symbols, no BOS/EOS.
    tok.add_blank_char = True
    tok.add_blank_word = False
    tok.use_eos_bos = False
    tok.blank_at_start = True
    tok.blank_at_end = True

    return VoiceConfig(
        tokenizer=tok,
        num_symbols=model.get("num_symbols", len(tok.vocabulary.char2idx)),
        num_speakers=max(model.get("n_speakers", 1), 1),
        num_langs=1,
        sample_rate=audio.get("sample_rate", 22050),
        lang_code=lang_code,
        phoneme_type=phoneme_type,
        alphabet=Alphabet.IPA,
        phonemizer_model=None,
        engine=Engine.GLOWTTS,
        add_diacritics=False,
        blank_between=BlankBetween.TOKENS_AND_WORDS,
        blank_at_start=True,
        blank_at_end=True,
        pad_token=_GLOW_PAD,
        blank_token=_GLOW_PAD,
        bos_token=None,
        eos_token=None,
        word_sep_token=" ",
    )
