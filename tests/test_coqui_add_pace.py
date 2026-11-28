"""add_pace.py: a pace input divides the duration scale of a Coqui ForwardTTS graph."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, helper, numpy_helper

_SCRIPT = Path(__file__).parent.parent / "scripts" / "conversion" / "coqui_fastpitch_export" / "add_pace.py"
spec = importlib.util.spec_from_file_location("add_pace", _SCRIPT)
add_pace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(add_pace)


def _constant(name, value):
    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(np.array(value, dtype=np.float32)))


def _duration_graph(length_scale=1.0):
    """The traced ForwardTTS duration step: (exp(log_dur) - 1) * mask * length_scale."""
    nodes = [
        _constant("one", 1.0), _constant("mask", np.ones((1, 1, 4))), _constant("scale", length_scale),
        helper.make_node("Exp", ["log_dur"], ["exp"]),
        helper.make_node("Sub", ["exp", "one"], ["dur"]),
        helper.make_node("Mul", ["dur", "mask"], ["masked"]),
        helper.make_node("Mul", ["masked", "scale"], ["durations"]),
    ]
    graph = helper.make_graph(
        nodes, "durations",
        [helper.make_tensor_value_info("log_dur", TensorProto.FLOAT, [1, 1, 4])],
        [helper.make_tensor_value_info("durations", TensorProto.FLOAT, [1, 1, 4])])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])


def _run(model, **feeds):
    session = onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, {k: np.asarray(v, dtype=np.float32) for k, v in feeds.items()})[0]


class TestAddPace(unittest.TestCase):
    log_dur = np.log(np.array([[[3.0, 5.0, 9.0, 2.0]]]))  # durations 2, 4, 8, 1

    def test_pace_divides_the_durations(self):
        model = add_pace.add_pace_input(_duration_graph())
        self.assertIn("pace", [i.name for i in model.graph.input])
        np.testing.assert_allclose(_run(model, log_dur=self.log_dur, pace=[1.0]), [[[2, 4, 8, 1]]], rtol=1e-5)
        np.testing.assert_allclose(_run(model, log_dur=self.log_dur, pace=[2.0]), [[[1, 2, 4, 0.5]]], rtol=1e-5)
        np.testing.assert_allclose(_run(model, log_dur=self.log_dur, pace=[0.5]), [[[4, 8, 16, 2]]], rtol=1e-5)

    def test_an_exported_length_scale_is_kept(self):
        model = add_pace.add_pace_input(_duration_graph(length_scale=1.5))
        np.testing.assert_allclose(_run(model, log_dur=self.log_dur, pace=[1.0]), [[[3, 6, 12, 1.5]]], rtol=1e-5)

    def test_a_graph_with_pace_is_left_as_it_is(self):
        once = add_pace.add_pace_input(_duration_graph())
        twice = add_pace.add_pace_input(onnx.load_from_string(once.SerializeToString()))
        self.assertEqual(twice.SerializeToString(), once.SerializeToString())

    def test_a_graph_without_the_duration_step_is_refused(self):
        graph = helper.make_graph(
            [helper.make_node("Exp", ["x"], ["y"])], "plain",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
        with self.assertRaisesRegex(ValueError, "found 0"):
            add_pace.add_pace_input(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)]))


if __name__ == "__main__":
    unittest.main()
