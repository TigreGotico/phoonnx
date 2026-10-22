"""Vendored yl4579/StyleTTS2 training package (MIT).

Faithful copy of the upstream model, loss, and data code with imports made
package-relative and the compiled ``monotonic_align`` extension replaced by
a pure-numpy port (``monotonic.py``).  The Lightning training recipe that
drives it lives in ``phoonnx_train.engines.styletts2``.

Lineage: https://github.com/yl4579/StyleTTS2 (© 2023 Aaron (Yinghao) Li, MIT
license). Paper: Li et al., "StyleTTS 2: Towards Human-Level Text-to-Speech
through Style Diffusion and Adversarial Training with Large Speech Language
Models" (NeurIPS 2023, https://arxiv.org/abs/2306.07691).
"""
from phoonnx_train.styletts2.models import (
    build_model,
    load_ASR_models,
    load_F0_models,
    load_checkpoint,
)
from phoonnx_train.styletts2.utils import (
    get_data_path_list,
    length_to_mask,
    log_norm,
    recursive_munch,
)

__all__ = [
    "build_model",
    "load_ASR_models",
    "load_F0_models",
    "load_checkpoint",
    "get_data_path_list",
    "length_to_mask",
    "log_norm",
    "recursive_munch",
]
