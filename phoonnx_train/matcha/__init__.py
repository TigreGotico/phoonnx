"""Vendored Matcha-TTS (MIT, https://github.com/shivammehta25/Matcha-TTS) ported to pytorch_lightning."""

# ``MatchaTTS`` pulls in the diffusers-backed decoder; keep it lazy so leaf
# utilities (e.g. the ``matcha.audio`` mel front-end shared with the Vocos
# vocoder trainer) stay importable without the full acoustic-model stack.
__all__ = ["MatchaTTS"]


def __getattr__(name):
    if name == "MatchaTTS":
        from phoonnx_train.matcha.lightning import MatchaTTS
        return MatchaTTS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
