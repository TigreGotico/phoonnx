"""
Vendored pure-torch port of coqui-TTS ``ForwardTTS`` (FastPitch / SpeedySpeech).

Adapted from https://github.com/coqui-ai/TTS (``TTS/tts/models/forward_tts.py``
and the layers it uses), original code © Coqui GmbH, licensed under the
Mozilla Public License 2.0 (MPL-2.0). This vendored copy is self-contained:
it has no dependency on the ``TTS``/``coqui-tts`` package.

FastPitch and SpeedySpeech are both ``ForwardTTS`` configurations — the same
non-autoregressive text→mel model with per-token duration (and optionally
pitch/energy) predictors; SpeedySpeech simply drops the pitch predictor and
uses residual-conv encoder/decoder blocks instead of FFT transformer blocks.
"""
from phoonnx_train.fastpitch.model import ForwardTTS, ForwardTTSArgs

__all__ = ["ForwardTTS", "ForwardTTSArgs"]
