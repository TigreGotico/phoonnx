"""MOSS-TTS-Nano training / finetuning pipeline (vendored, self-contained).

MOSS-TTS-Nano is a ~100M-parameter **global-local (RQ-Transformer) codec LM**: a
12-layer GPT-2-style backbone with RoPE consumes rows of ``1 + n_vq`` tokens and emits
one *global* hidden state per frame; a 1-layer *local* transformer then walks the
``n_vq + 1`` channels of that frame autoregressively (text-continuation channel first,
then the 16 RVQ codebooks) to produce the frame's tokens. See arXiv:2603.18090 for the
architecture; this package implements it in-repo rather than depending on the upstream
package, so phoonnx never needs ``trust_remote_code`` or the upstream repo at runtime.

Modules
-------
``config``       vendored configuration dataclasses (mirrors the upstream ``config.json``)
``model``        the vendored backbone + local transformer
``dataset``      JSONL of pre-encoded RVQ codes -> packed ``[T, n_vq + 1]`` rows
``lightning``    :class:`~pytorch_lightning.LightningModule` wrapper (loss, optim, sched)
``prepare_data`` CLI: audio manifest -> JSONL with codes from the frozen ONNX codec
``warmstart``    explicit state-dict mapping from the upstream safetensors checkpoint
``export_onnx``  checkpoint -> the multi-graph ONNX layout ``phoonnx.engines.mosstts`` uses
"""

from phoonnx_train.mosstts.config import GPT2DecoderConfig, MossTTSNanoConfig

__all__ = ["GPT2DecoderConfig", "MossTTSNanoConfig"]
