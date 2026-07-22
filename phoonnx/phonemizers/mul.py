"""Compatibility shim — implementation lives in scriptconv.phonemizers.

Import paths and class identities are preserved for existing voice
configs and callers; construction through ``phoonnx.config.get_phonemizer``
injects phoonnx's text normalizer so behavior is unchanged.
"""
from scriptconv.phonemizers.mul import *  # noqa: F401,F403
from scriptconv.phonemizers.mul import (EspeakPhonemizer, EspeakError, EpitranPhonemizer,
    MisakiPhonemizer, MisakiEnPhonemizer, MisakiJaPhonemizer, MisakiZhPhonemizer,
    MisakiKoPhonemizer, MisakiViPhonemizer, GoruutPhonemizer, GruutPhonemizer,
    ByT5Phonemizer, CharsiuPhonemizer, TransphonePhonemizer)  # noqa: F401
