"""
GlowTTS (Larynx) ↔ phoonnx config bridge.

Larynx GlowTTS voices ship a training ``config.json`` (audio + model params)
and a ``phonemes.txt`` symbol table (``id phoneme`` per line, gruut IPA). This
module turns those into a phoonnx :class:`VoiceConfig` whose tokenizer
reproduces Larynx's tokenization (gruut phonemes, blank-interspersed).

The vocoder (Larynx HiFi-GAN) is supplied separately via
``engine_params['vocoder_path']`` / the voice-index ``vocoder_url``.

The generic Coqui-TTS bridge (GlowTTS/VITS/FastPitch) lives in
:mod:`phoonnx.engines.coqui_config`; ``voice_config_from_coqui`` is re-exported
here for backwards compatibility.
"""
from typing import Any, Dict

from phoonnx.config import Alphabet, Engine, PhonemeType, VoiceConfig
from phoonnx.tokenizer import BlankBetween, TTSTokenizer, Vocabulary
from phoonnx.engines.coqui_config import voice_config_from_coqui  # noqa: F401  (re-export)

_GLOW_PAD = "_"


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
