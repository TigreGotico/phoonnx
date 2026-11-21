"""Rápido TTS: a flow-matching DiT over VoxCPM2 AudioVAE latents.

:mod:`.model` holds the acoustic network (text stage, aligner, velocity backbone),
:mod:`.audio_vae` the frozen VoxCPM2 AudioVAE v2 encoder and decoder, and
:mod:`.export_onnx` the four-graph export the ``rapido`` inference engine reads.
"""
