import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper, TensorProto

from phoonnx.config import VoiceConfig
from phoonnx.util import LOG

_LOGGER = logging.getLogger("phoonnx.lora_runtime")


def get_lora_config_from_path(lora_path: Union[str, Path]) -> Optional[Dict]:
    config_path = Path(lora_path)
    if config_path.is_dir():
        config_path = config_path / "lora_config.json"
    elif config_path.suffix == ".pt":
        config_path = config_path.parent / "lora_config.json"
    elif config_path.suffix == ".json":
        pass
    else:
        return None

    if not config_path.exists():
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_lora_weights(lora_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    lora_path = Path(lora_path)
    if lora_path.is_dir():
        lora_path = lora_path / "lora_adapter.pt"

    import torch
    state = torch.load(str(lora_path), map_location="cpu", weights_only=True)

    weights = {}
    if isinstance(state, dict):
        for key, value in state.items():
            if "lora_A" in key or "lora_B" in key:
                if hasattr(value, "numpy"):
                    weights[key] = value.numpy()
                elif isinstance(value, np.ndarray):
                    weights[key] = value
                else:
                    weights[key] = np.array(value)
    return weights


def apply_lora_to_onnx_graph(
    onnx_model: onnx.ModelProto,
    lora_weights: Dict[str, np.ndarray],
    lora_config: Dict,
) -> onnx.ModelProto:
    rank = lora_config.get("rank", 8)
    alpha = lora_config.get("alpha", 16.0)
    scaling = alpha / rank

    graph = onnx_model.graph
    name_counter = [0]

    def unique_name(prefix: str) -> str:
        name_counter[0] += 1
        return f"{prefix}_{name_counter[0]}"

    initializer_map = {init.name: init for init in graph.initializer}

    nodes_to_add = []
    initializers_to_add = []

    for weight_key, weight_value in lora_weights.items():
        if ".lora_B" not in weight_key:
            continue

        lora_a_key = weight_key.replace(".lora_B", ".lora_A")
        if lora_a_key not in lora_weights:
            continue

        lora_b = lora_weights[weight_key]
        lora_a = lora_weights[lora_a_key]

        base_name = weight_key.replace(".lora_B", "")
        base_name = base_name.replace("model_g.", "")

        is_conv1d = len(lora_a.shape) == 3
        is_linear = len(lora_a.shape) == 2

        conv_name = _find_matching_conv_node(graph, base_name)
        if conv_name is None:
            _LOGGER.debug("Could not find ONNX node for LoRA key '%s', skipping", weight_key)
            continue

        conv_node = None
        weight_input_name = None
        for node in graph.node:
            if node.output[0] == conv_name or conv_name in node.output:
                conv_node = node
                weight_input_name = node.input[1]
                break

        if conv_node is None or weight_input_name is None:
            continue

        conv_output_name = conv_name
        lora_output_name = unique_name(f"{base_name}_lora_out")
        merged_output_name = unique_name(f"{base_name}_merged")

        if is_linear:
            b_init = numpy_helper.from_array(lora_b.astype(np.float32), name=unique_name("lora_b"))
            a_init = numpy_helper.from_array(lora_a.astype(np.float32), name=unique_name("lora_a"))
            initializers_to_add.extend([b_init, a_init])

            matmul_node = onnx.helper.make_node(
                "MatMul",
                inputs=[conv_node.input[0], a_init.name],
                outputs=[unique_name("lora_a_out")],
                name=unique_name("lora_matmul_a"),
            )
            nodes_to_add.append(matmul_node)

            matmul_b_node = onnx.helper.make_node(
                "MatMul",
                inputs=[matmul_node.output[0], b_init.name],
                outputs=[lora_output_name],
                name=unique_name("lora_matmul_b"),
            )
            nodes_to_add.append(matmul_b_node)

        elif is_conv1d:
            b_init = numpy_helper.from_array(lora_b.astype(np.float32), name=unique_name("lora_b"))
            a_init = numpy_helper.from_array(lora_a.astype(np.float32), name=unique_name("lora_a"))
            initializers_to_add.extend([b_init, a_init])

            a_mat = lora_a.reshape(lora_a.shape[0], -1)
            b_mat = lora_b.reshape(lora_b.shape[0], -1)
            delta = (b_mat @ a_mat).reshape(lora_b.shape[0], lora_a.shape[1], lora_a.shape[2])
            delta_scaled = (delta * scaling).astype(np.float32)

            delta_init = numpy_helper.from_array(delta_scaled, name=unique_name("lora_delta"))
            initializers_to_add.append(delta_init)

            conv_lora_node = onnx.helper.make_node(
                "Conv",
                inputs=[conv_node.input[0], delta_init.name],
                outputs=[lora_output_name],
                name=unique_name("lora_conv"),
                kernel_shape=[delta.shape[2]],
                pads=list(conv_node.attribute.get("pads", [0, 0, 0, 0])
                          if any(a.name == "pads" for a in conv_node.attribute)
                          else _infer_pads(conv_node, lora_a.shape[2])),
                strides=list(conv_node.attribute.get("strides", [1]
                          if any(a.name == "strides" for a in conv_node.attribute)
                          else [1])),
                dilations=list(conv_node.attribute.get("dilations", [1]
                              if any(a.name == "dilations" for a in conv_node.attribute)
                              else [1])),
                group=1,
            )
            nodes_to_add.append(conv_lora_node)
        else:
            continue

        scale_init = numpy_helper.from_array(
            np.array(scaling, dtype=np.float32), name=unique_name("lora_scale")
        )
        initializers_to_add.append(scale_init)

        scale_node = onnx.helper.make_node(
            "Mul",
            inputs=[lora_output_name, scale_init.name],
            outputs=[unique_name("lora_scaled")],
            name=unique_name("lora_scale_mul"),
        )
        nodes_to_add.append(scale_node)

        add_node = onnx.helper.make_node(
            "Add",
            inputs=[conv_output_name, scale_node.output[0]],
            outputs=[merged_output_name],
            name=unique_name("lora_add"),
        )
        nodes_to_add.append(add_node)

        for downstream_node in graph.node:
            for i, inp_name in enumerate(downstream_node.input):
                if inp_name == conv_output_name:
                    downstream_node.input[i] = merged_output_name

    graph.node.extend(nodes_to_add)
    graph.initializer.extend(initializers_to_add)

    return onnx_model


def _find_matching_conv_node(graph, base_name: str) -> Optional[str]:
    for node in graph.node:
        for output in node.output:
            if base_name in output:
                return output
    for init in graph.initializer:
        if base_name in init.name:
            return init.name
    return None


def _infer_pads(conv_node, kernel_size: int) -> list:
    for attr in conv_node.attribute:
        if attr.name == "pads":
            return list(attr.ints)
    pad = kernel_size // 2
    return [pad, 0, pad, 0]


def merge_lora_onnx(
    base_onnx_path: Union[str, Path],
    lora_weights: Dict[str, np.ndarray],
    lora_config: Dict,
    output_path: Union[str, Path],
) -> Path:
    onnx_model = onnx.load(str(base_onnx_path))
    merged_model = apply_lora_to_onnx_graph(onnx_model, lora_weights, lora_config)
    onnx.save(merged_model, str(output_path))
    _LOGGER.info("Saved merged ONNX model with LoRA to %s", output_path)
    return Path(output_path)


def apply_lora_weight_overlay(
    session: onnxruntime.InferenceSession,
    lora_weights: Dict[str, np.ndarray],
    lora_config: Dict,
) -> onnxruntime.InferenceSession:
    rank = lora_config.get("rank", 8)
    alpha = lora_config.get("alpha", 16.0)
    scaling = alpha / rank

    model_meta = session.get_modelmeta()
    graph_def = model_meta.graph_def

    _LOGGER.warning(
        "Weight-overlay LoRA requires re-creating the InferenceSession. "
        "For production use, prefer merge_lora_onnx() for a pre-merged model."
    )
    raise NotImplementedError(
        "Runtime weight overlay LoRA is not yet supported. "
        "Use merge_lora_onnx() to produce a pre-merged ONNX model, "
        "then load it with TTSVoice.load() as usual."
    )