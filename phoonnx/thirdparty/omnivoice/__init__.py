"""Vendored helpers from `k2-fsa/OmniVoice <https://github.com/k2-fsa/OmniVoice>`_.

OmniVoice is Apache-2.0 (the code; the released weights are CC-BY-NC because of their
training data). Only the parts phoonnx needs at inference time are here, with torch,
torchaudio and pydub replaced by numpy so phoonnx keeps its dependency surface:

``duration``  the rule-based duration estimator that decides how many audio frames to
              generate. OmniVoice is non-autoregressive, so it cannot stop when it is
              done -- output length is chosen before decoding starts.
``text``      reference/target text joining, non-verbal tag tokenization, punctuation.
``audio``     sinc resampling, pydub-equivalent silence gating, fade and pad.
``lang_map``  the language name/ID table for the ``<|lang_start|>`` prompt slot.
"""
from phoonnx.thirdparty.omnivoice.audio import (fade_and_pad_audio, remove_silence,
                                                resample)
from phoonnx.thirdparty.omnivoice.duration import RuleDurationEstimator
from phoonnx.thirdparty.omnivoice.lang_map import (LANG_IDS, LANG_NAME_TO_ID, LANG_NAMES,
                                                   lang_display_name)
from phoonnx.thirdparty.omnivoice.text import (add_punctuation, combine_text,
                                               tokenize_with_nonverbal_tags)

__all__ = ["RuleDurationEstimator", "combine_text", "tokenize_with_nonverbal_tags",
           "add_punctuation", "resample", "remove_silence", "fade_and_pad_audio",
           "LANG_IDS", "LANG_NAMES", "LANG_NAME_TO_ID", "lang_display_name"]
