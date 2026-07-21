"""Compatibility shim — implementation lives in scriptconv.phonemizers.

Import paths and class identities are preserved for existing voice
configs and callers; construction through ``phoonnx.config.get_phonemizer``
injects phoonnx's text normalizer so behavior is unchanged.
"""
from scriptconv.phonemizers.mwl import MirandesePhonemizer, DIALECTS  # noqa: F401
