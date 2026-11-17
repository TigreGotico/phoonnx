"""The pure-Python reader of external-data names, checked against ``onnx``.

``onnx.external_data_helper`` walks every tensor of a model with the protobuf
runtime. The reader must find the same file names in the same graphs, without
it, and must reject a file that is not an ONNX model.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.external_data_helper import _get_all_tensors, set_external_data, uses_external_data

from phoonnx._onnx_external import external_data_locations


def _external(name, location, value=1.0):
    tensor = numpy_helper.from_array(np.full(2, value, dtype=np.float32), name=name)
    set_external_data(tensor, location, offset=0, length=len(tensor.raw_data))
    tensor.ClearField("raw_data")
    return tensor


def _value(name):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [2])


def _nested_model():
    """External data in every place ``onnx`` looks: an initializer, a
    subgraph initializer, a node attribute tensor in a subgraph, and a node
    attribute tensor in a model function."""
    then_graph = helper.make_graph(
        [helper.make_node("Constant", [], ["C"], value=_external("C", "then_const.bin")),
         helper.make_node("Add", ["C", "T"], ["out"])],
        "then", [], [_value("out")], [_external("T", "then_init.bin")])
    else_graph = helper.make_graph(
        [helper.make_node("Identity", ["E"], ["out"])],
        "else", [], [_value("out")], [_external("E", "model.onnx_data")])
    function = helper.make_function(
        "local", "F", ["x"], ["y"],
        [helper.make_node("Constant", [], ["k"], value=_external("k", "function.bin")),
         helper.make_node("Add", ["x", "k"], ["y"])],
        [helper.make_opsetid("", 13)])
    graph = helper.make_graph(
        [helper.make_node("If", ["cond"], ["branch"], then_branch=then_graph,
                          else_branch=else_graph),
         helper.make_node("F", ["branch"], ["Y"], domain="local")],
        "g", [helper.make_tensor_value_info("cond", TensorProto.BOOL, [])],
        [_value("Y")],
        [_external("W", "model.onnx_data"), _external("W1", "model.onnx_data_1")])
    return helper.make_model(graph, functions=[function],
                             opset_imports=[helper.make_opsetid("", 13),
                                            helper.make_opsetid("local", 1)])


class TestExternalDataLocations(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.dir = Path(self._dir.name)

    def write(self, name, data):
        path = self.dir / name
        path.write_bytes(data)
        return path

    def oracle(self, model):
        locations = []
        for tensor in _get_all_tensors(model):
            if uses_external_data(tensor):
                locations += [e.value for e in tensor.external_data if e.key == "location"]
        return list(dict.fromkeys(locations))

    def test_the_names_onnx_finds_in_a_nested_model(self):
        model = _nested_model()
        path = self.write("model.onnx", model.SerializeToString())
        expected = self.oracle(onnx.load(str(path), load_external_data=False))
        self.assertEqual(sorted(expected),
                         ["function.bin", "model.onnx_data", "model.onnx_data_1",
                          "then_const.bin", "then_init.bin"])
        found = external_data_locations(path)
        self.assertEqual(len(found), len(set(found)), "each name comes once")
        self.assertEqual(sorted(found), sorted(expected))

    def test_a_graph_with_its_weights_inside_names_nothing(self):
        weight = numpy_helper.from_array(np.ones((256, 256), dtype=np.float32), name="W")
        graph = helper.make_graph([helper.make_node("Add", ["X", "W"], ["Y"])], "g",
                                  [helper.make_tensor_value_info("X", TensorProto.FLOAT, [256, 256])],
                                  [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [256, 256])],
                                  [weight])
        path = self.write("model.onnx", helper.make_model(graph).SerializeToString())
        self.assertEqual(external_data_locations(path), [])

    def test_a_file_that_is_not_an_onnx_model_is_rejected(self):
        graph = _nested_model().SerializeToString()
        cases = {
            "empty": b"",
            "json": b'{"model": "model.onnx"}',
            "truncated graph": graph[:len(graph) // 2],
            # a valid protobuf message with no graph field
            "no graph": helper.make_opsetid("", 13).SerializeToString(),
        }
        for name, data in cases.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    external_data_locations(self.write(name, data))

    def test_a_missing_file_raises_oserror(self):
        with self.assertRaises(OSError):
            external_data_locations(self.dir / "absent.onnx")

    def test_a_varint_is_at_most_ten_bytes(self):
        model = _nested_model().SerializeToString()
        # an unknown field 15 of wire type 0: 2**64 - 1 takes 10 bytes
        ten = b"\x78" + b"\xff" * 9 + b"\x01"
        eleven = b"\x78" + b"\xff" * 10 + b"\x01"
        expected = external_data_locations(self.write("plain.onnx", model))
        self.assertEqual(external_data_locations(self.write("ten.onnx", model + ten)), expected)
        with self.assertRaisesRegex(ValueError, "varint"):
            external_data_locations(self.write("eleven.onnx", model + eleven))

    def test_a_graph_nested_too_deeply_is_rejected(self):
        def field(number, payload):
            return _encode_varint(number << 3 | 2) + _encode_varint(len(payload)) + payload

        payload = b""
        for _ in range(1500):
            # graph -> node (1) -> attribute (5) -> graph (6)
            payload = field(1, field(5, field(6, payload)))
        data = field(7, payload)
        with self.assertRaisesRegex(ValueError, "too deeply"):
            external_data_locations(self.write("deep.onnx", data))

    def test_tensors_and_graphs_list_attributes_are_walked(self):
        def sub(name, location):
            return helper.make_graph(
                [helper.make_node("Identity", [name], [name + "_o"])], name, [],
                [_value(name + "_o")], [_external(name, location)])

        node = helper.make_node("Custom", [], ["y"], domain="local")
        node.attribute.append(helper.make_attribute("ts", [_external("a", "tensors_list.bin")]))
        node.attribute.append(helper.make_attribute("gs", [sub("g", "graphs_list.bin")]))
        graph = helper.make_graph([node], "g", [], [_value("y")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        path = self.write("lists.onnx", model.SerializeToString())
        self.assertEqual(sorted(external_data_locations(path)),
                         ["graphs_list.bin", "tensors_list.bin"])


def _encode_varint(value: int) -> bytes:
    out = bytearray()
    while value > 0x7F:
        out.append(value & 0x7F | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


if __name__ == "__main__":
    unittest.main()
