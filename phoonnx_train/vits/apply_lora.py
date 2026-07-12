import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm, remove_weight_norm

from .lora import LoRAConv1d, LoRAConvTranspose1d, LoRALinear
from .lora_config import LoRAConfig
if TYPE_CHECKING:  # models pulls in the compiled monotonic_align extension
    from .models import SynthesizerTrn

_LOGGER = logging.getLogger("phoonnx_train.lora")


def _unwrap_weight_norm(module: nn.Module) -> nn.Module:
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        try:
            return remove_weight_norm(module)
        except ValueError:
            pass
    return module


def _strip_weight_norm_recursive(model: nn.Module) -> None:
    for name, child in list(model.named_children()):
        _strip_weight_norm_recursive(child)
        if hasattr(child, 'weight_g') or hasattr(child, 'weight_v'):
            try:
                new_mod = remove_weight_norm(child)
                setattr(model, name, new_mod)
            except (ValueError, AttributeError):
                pass


def apply_lora(model: "SynthesizerTrn", config: LoRAConfig) -> List[str]:
    _strip_weight_norm_recursive(model)
    replaced = []
    for module_name in config.target_modules:
        submodule = getattr(model, module_name, None)
        if submodule is None:
            _LOGGER.warning("Target module '%s' not found in model, skipping", module_name)
            continue
        count = _apply_lora_to_module(submodule, config, f"model.{module_name}")
        replaced.extend(count)
        _LOGGER.info(
            "Applied LoRA (rank=%d) to '%s': %d layers adapted",
            config.rank, module_name, len(count),
        )
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAConv1d, LoRAConvTranspose1d)):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    _LOGGER.info(
        "LoRA applied: %d trainable / %d total params (%.2f%%)",
        trainable, total, 100.0 * trainable / total if total > 0 else 0,
    )
    return replaced


def _apply_lora_to_module(
    module: nn.Module, config: LoRAConfig, prefix: str
) -> List[str]:
    replaced = []
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}"
        if isinstance(child, (LoRALinear, LoRAConv1d, LoRAConvTranspose1d)):
            continue
        if isinstance(child, nn.Conv1d):
            if child.weight.shape[0] >= config.rank and child.weight.shape[1] >= config.rank:
                lora_layer = LoRAConv1d(
                    child,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
                setattr(module, name, lora_layer)
                replaced.append(full_name)
        elif isinstance(child, nn.ConvTranspose1d):
            if child.weight.shape[0] >= config.rank and child.weight.shape[1] >= config.rank:
                lora_layer = LoRAConvTranspose1d(
                    child,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
                setattr(module, name, lora_layer)
                replaced.append(full_name)
        elif isinstance(child, nn.Linear):
            if child.in_features >= config.rank and child.out_features >= config.rank:
                lora_layer = LoRALinear(
                    child,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
                setattr(module, name, lora_layer)
                replaced.append(full_name)
        else:
            replaced.extend(_apply_lora_to_module(child, config, full_name))
    return replaced


def merge_lora(model: "SynthesizerTrn") -> None:
    _merge_lora_recursive(model)


def _merge_lora_recursive(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            setattr(module, name, child.merge())
        elif isinstance(child, LoRAConv1d):
            setattr(module, name, child.merge())
        elif isinstance(child, LoRAConvTranspose1d):
            setattr(module, name, child.merge())
        else:
            _merge_lora_recursive(child)


def get_lora_state_dict(model: "SynthesizerTrn") -> Dict[str, torch.Tensor]:
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv1d, LoRAConvTranspose1d)):
            state[f"{name}.lora_A"] = module.lora_A.data.cpu()
            state[f"{name}.lora_B"] = module.lora_B.data.cpu()
    return state


def load_lora_adapter(model: "SynthesizerTrn", state_dict: Dict[str, torch.Tensor]) -> None:
    lora_names = set()
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv1d, LoRAConvTranspose1d)):
            lora_names.add(name)

    loaded = 0
    for key, tensor in state_dict.items():
        key_clean = key.removeprefix("model_g.")
        if key_clean.endswith(".lora_A"):
            module_name = key_clean[: -len(".lora_A")]
            module = _get_submodule(model, module_name)
            if module is not None and hasattr(module, "lora_A"):
                module.lora_A.data = tensor.to(module.lora_A.device)
                loaded += 1
        elif key_clean.endswith(".lora_B"):
            module_name = key_clean[: -len(".lora_B")]
            module = _get_submodule(model, module_name)
            if module is not None and hasattr(module, "lora_B"):
                module.lora_B.data = tensor.to(module.lora_B.device)
                loaded += 1

    _LOGGER.info("Loaded %d LoRA weight tensors", loaded)


def _get_submodule(module: nn.Module, target: str) -> Optional[nn.Module]:
    atoms = target.split(".")
    mod = module
    for atom in atoms:
        if not hasattr(mod, atom):
            return None
        mod = getattr(mod, atom)
    return mod


def count_parameters(model: "SynthesizerTrn") -> Tuple[int, int, float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total > 0 else 0
    return trainable, total, pct