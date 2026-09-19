"""SuperTonic TTS training (Kim et al., Supertone Inc., "SupertonicTTS",
arXiv:2503.23108).

SuperTonic is a three-stage system, each stage trained independently:

1. a GAN-trained **speech autoencoder** (:mod:`.autoencoder`,
   :mod:`.discriminators`, :mod:`.losses`) that maps a waveform to a
   low-dimensional continuous latent and back;
2. a **text-to-latent** module (:mod:`.text_to_latent`) that maps character-level
   text plus a reference voice style to that latent by conditional flow matching;
3. a **duration predictor** (:mod:`.duration_predictor`) that estimates the total
   utterance length.

Supporting modules: :mod:`.config` (per-stage dataclasses + released-``tts.json``
loading), :mod:`.layers` (shared ConvNeXt / attention / pooling blocks),
:mod:`.latent_utils` (temporal compression + reference crops), :mod:`.text`
(character tokenizer), :mod:`.dataset` (filelist datasets), :mod:`.lightning`
(one LightningModule per stage), :mod:`.checkpointing` (atomic full checkpoints +
resume + grow-vocab load), :mod:`.import_onnx` (fine-tune from the released ONNX
weights) and :mod:`.export_onnx` (the four-graph inference contract).

Modules are written from scratch against the SuperTonic paper and the public
Supertone/supertonic-3 ONNX graphs; nothing here is copied from the unlicensed
community training repository.

Nothing is imported at package load time so the training-engine registry stays
importable without torch; each submodule pulls torch in on its own.
"""
