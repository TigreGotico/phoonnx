"""Add a ``pace`` input to an exported Coqui ForwardTTS graph.

ForwardTTS scales the predicted durations by ``length_scale``:
``(exp(log_dur) - 1) * mask * length_scale``. The tracer bakes that scale into
the graph as a constant, so the exported model has no speed control. This
divides the constant by a new ``pace`` input (float32 ``[1]``, >1 faster), the
input that phoonnx's FastPitch adapter feeds.

Usage: python add_pace.py in.onnx out.onnx
"""
import sys

import onnx
from onnx import TensorProto, helper, numpy_helper


def add_pace_input(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return ``model`` with its duration scale divided by a ``pace`` input."""
    graph = model.graph
    if any(i.name == "pace" for i in graph.input):
        return model
    producer = {out: node for node in graph.node for out in node.output}
    scalars = {node.output[0] for node in graph.node
               if node.op_type == "Constant" and numpy_helper.to_array(node.attribute[0].t).size == 1}
    scalars |= {i.name for i in graph.initializer if numpy_helper.to_array(i).size == 1}

    def made_by(name: str, op_type: str) -> bool:
        return name in producer and producer[name].op_type == op_type

    # Mul(Mul(Sub(Exp(log_dur), 1), mask), length_scale)
    matches = [
        node for node in graph.node
        if node.op_type == "Mul" and node.input[1] in scalars
        and made_by(node.input[0], "Mul")
        and made_by(producer[node.input[0]].input[0], "Sub")
        and made_by(producer[producer[node.input[0]].input[0]].input[0], "Exp")
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one length_scale Mul after the duration Exp, found {len(matches)}")
    scale_mul = matches[0]
    scaled = scale_mul.input[1] + "_over_pace"
    divide = helper.make_node("Div", [scale_mul.input[1], "pace"], [scaled], name="/pace/Div")
    graph.node.insert(list(graph.node).index(scale_mul), divide)
    scale_mul.input[1] = scaled
    graph.input.append(helper.make_tensor_value_info("pace", TensorProto.FLOAT, [1]))
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    onnx.save(add_pace_input(onnx.load(sys.argv[1])), sys.argv[2])
