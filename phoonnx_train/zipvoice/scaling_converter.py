"""Convert scaled training modules to export-friendly equivalents, so the
model can be traced for ONNX export, as in the Zipformer recipes (Yao et
al., arXiv:2310.11230). Ported from k2-fsa/ZipVoice (Apache-2.0).

This file replaces various modules in a model.
Specifically, ActivationBalancer is replaced with an identity operator;
Whiten is also replaced with an identity operator;
BasicNorm is replaced by a module with `exp` removed.
"""

import copy
from typing import List

import torch
import torch.nn as nn

from phoonnx_train.zipvoice.scaling import (
    Balancer,
    Dropout3,
    SwooshL,
    SwooshLOnnx,
    SwooshR,
    SwooshROnnx,
    Whiten,
)
from phoonnx_train.zipvoice.zipformer import CompactRelPositionalEncoding


# Copied from https://pytorch.org/docs/1.9.0/_modules/torch/nn/modules/module.html#Module.get_submodule  # noqa
# get_submodule was added to nn.Module at v1.9.0
def get_submodule(model, target):
    if target == "":
        return model
    atoms: List[str] = target.split(".")
    mod: torch.nn.Module = model
    for item in atoms:
        if not hasattr(mod, item):
            raise AttributeError(
                mod._get_name() + " has no " "attribute `" + item + "`"
            )
        mod = getattr(mod, item)
        if not isinstance(mod, torch.nn.Module):
            raise AttributeError("`" + item + "` is not " "an nn.Module")
    return mod


def convert_scaled_to_non_scaled(
    model: nn.Module,
    inplace: bool = False,
    is_pnnx: bool = False,
    is_onnx: bool = False,
):
    """
    Args:
      model:
        The model to be converted.
      inplace:
        If True, the input model is modified inplace.
        If False, the input model is copied and we modify the copied version.
      is_pnnx:
        True if we are going to export the model for PNNX.
      is_onnx:
        True if we are going to export the model for ONNX.
    Return:
      Return a model without scaled layers.
    """
    if not inplace:
        model = copy.deepcopy(model)

    d = {}
    for name, m in model.named_modules():
        if isinstance(m, (Balancer, Dropout3, Whiten)):
            d[name] = nn.Identity()
        elif is_onnx and isinstance(m, SwooshR):
            d[name] = SwooshROnnx()
        elif is_onnx and isinstance(m, SwooshL):
            d[name] = SwooshLOnnx()
        elif is_onnx and isinstance(m, CompactRelPositionalEncoding):
            # We want to recreate the positional encoding vector when
            # the input changes, so we have to use torch.jit.script()
            # to replace torch.jit.trace()
            d[name] = torch.jit.script(m)

    for k, v in d.items():
        if "." in k:
            parent, child = k.rsplit(".", maxsplit=1)
            setattr(get_submodule(model, parent), child, v)
        else:
            setattr(model, k, v)

    return model
