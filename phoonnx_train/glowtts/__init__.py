"""
GlowTTS training package.

Provenance / honesty note (see also each module's own docstring):

This is a **reimplementation from the published GlowTTS paper architecture**
(Kim et al. 2020, "Glow-TTS: A Generative Flow for Text-to-Speech via
Monotonic Alignment Search", https://arxiv.org/abs/2005.11129) and from
general knowledge of how GlowTTS-style implementations (including the
coqui-TTS / Larynx family that ``phoonnx.engines.glowtts`` already targets
for *inference*) are structured — it is **not a verified line-by-line port**
of coqui-TTS source. The coding agent that authored this module did not have
network access to fetch and diff against the actual coqui-TTS
(``TTS.tts.models.glow_tts`` / ``TTS.tts.layers.glow_tts``) source at
authoring time, so no such claim is made.

coqui-TTS is distributed under the Mozilla Public License 2.0 (MPL-2.0).
Because this code was reconstructed from the paper and general architectural
knowledge rather than copied from coqui-TTS, no MPL-2.0 file-level obligations
apply to these files; this note exists purely for attribution/provenance
transparency, per project policy for documenting library usage.

What *was* directly read and reused (verified, not inferred) from this
repository:

  - ``phoonnx_train/vits/attentions.py``  (``Encoder`` — Transformer text
    encoder block, shared verbatim by GlowTTS's own text encoder).
  - ``phoonnx_train/vits/modules.py``     (``LayerNorm``, ``ConvReluNorm``,
    ``WN`` — WaveNet-style affine-coupling conditioner).
  - ``phoonnx_train/vits/commons.py``     (``sequence_mask``,
    ``generate_path``, ``init_weights``).
  - ``phoonnx_train/vits/monotonic_align``  (compiled Cython
    ``maximum_path`` — Monotonic Alignment Search dynamic-programming
    kernel), reused unmodified rather than re-vendored.
  - ``phoonnx_train/vits/dataset.py`` + ``mel_processing.py``  (dataset /
    collation / mel-spectrogram extraction pipeline), reused unmodified —
    GlowTTS only needs mel features, no waveform.
  - ``phoonnx/engines/glowtts.py``  (the *inference*-side ONNX contract this
    engine's ``export_onnx`` must satisfy — input names ``input`` /
    ``input_lengths`` / ``scales`` and a ``[B, n_mels, T]`` mel output).

Everything GlowTTS-specific (the duration predictor, the invertible flow
decoder assembled from activation-normalization + invertible 1x1 conv +
affine coupling, Monotonic Alignment Search glue, the MLE training loss, and
the two-stage ONNX export) is new code in this package, reconstructed from
the paper.

Fidelity audit (added when this engine was ported onto the dev engine
registry): the training math was reviewed against the original reference
implementation, jaywalnut310/glow-tts (MIT, the authors' official repo):

  - MAS prior ``neg_cent`` decomposition, ActNorm (with data-dependent
    init), grouped invertible 1x1 conv (``n_split=4``, QR-orthogonal init,
    det-sign fix), zero-initialized affine-coupling projection, and
    squeeze/unsqueeze (``n_sqz=2``) all match the reference structure.
  - MLE loss = per-element-normalized NLL minus log-det plus the
    ``0.5*log(2*pi)`` constant; duration loss = MSE on log-durations
    normalized by text length — both as in the reference.
  - Optimizer: plain Adam, betas (0.9, 0.98), eps 1e-9, Noam schedule with
    4000 warmup steps; the schedule is re-normalized so ``learning_rate``
    is the post-warmup peak (reference peak with dim=192 is ~1.14e-3).
  - Known deviation: the reference squeezes per-sample lengths to
    ``n_sqz`` multiples inside the flow; here the batch mel axis is
    trimmed to an ``n_sqz`` multiple before the decoder and per-sample
    masks are subsampled in ``_squeeze`` — equivalent masking, the loss
    normalization uses the same masked element count.
"""
