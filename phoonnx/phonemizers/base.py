"""Compatibility shim — implementation lives in scriptconv.phonemizers.

Import paths and class identities are preserved for existing voice
configs and callers; construction through ``phoonnx.config.get_phonemizer``
injects phoonnx's text normalizer so behavior is unchanged.
"""
from scriptconv.phonemizers.base import *  # noqa: F401,F403
from scriptconv.phonemizers.base import (BasePhonemizer, GraphemePhonemizer,
    UnicodeCodepointPhonemizer, TextChunks, RawPhonemizedChunks, PhonemizedChunks)  # noqa: F401
from langcodes import tag_distance  # noqa: F401
from quebra_frases import sentence_tokenize  # noqa: F401
from phoonnx.util import match_lang, normalize  # noqa: F401
from phoonnx.thirdparty.phonikud import PhonikudDiacritizer  # noqa: F401
