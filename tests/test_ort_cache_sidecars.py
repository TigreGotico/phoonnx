"""The optimized-graph cache for a graph that keeps its weights in a sidecar file.

The graph holds ``y = x + W`` with ``W`` of 1024 floats in ``model.onnx_data``.
The expected outputs are the weights written by the test itself.
"""
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.external_data_helper import set_external_data

from phoonnx import providers

CPU = ["CPUExecutionProvider"]
SIZE = 1024
INPUT = {"x": np.zeros(SIZE, dtype=np.float32)}


def _write_weights(model: Path, value: float, mtime_ns: int = None) -> None:
    sidecar = Path(f"{model}_data")
    np.full(SIZE, value, dtype=np.float32).tofile(sidecar)
    if mtime_ns is not None:
        os.utime(sidecar, ns=(mtime_ns, mtime_ns))


@pytest.fixture
def model(tmp_path, monkeypatch):
    monkeypatch.setenv(providers.SHARE_SESSIONS_ENV_VAR, "0")
    path = tmp_path / "model.onnx"
    weights = numpy_helper.from_array(np.zeros(SIZE, dtype=np.float32), name="W")
    set_external_data(weights, "model.onnx_data", offset=0, length=SIZE * 4)
    weights.ClearField("raw_data")
    weights.data_location = TensorProto.EXTERNAL
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [SIZE])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [SIZE])
    graph = helper.make_graph([helper.make_node("Add", ["x", "W"], ["y"])], "g", [x], [y], [weights])
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    proto.ir_version = 8
    path.write_bytes(proto.SerializeToString())
    _write_weights(path, 1.0)
    return path


def _output(session) -> float:
    values = session.run(None, INPUT)[0]
    assert np.all(values == values[0])
    return float(values[0])


def test_a_graph_with_a_sidecar_loads_from_the_cache(model, tmp_path):
    cache = tmp_path / "cache"
    assert _output(providers._build_session(model, CPU, None, cache)) == 1.0
    with patch.object(providers.LOG, "warning") as warn:
        session = providers._build_session(model, CPU, None, cache)
    assert not warn.called, warn.call_args_list
    assert _output(session) == 1.0


def test_the_cache_key_follows_the_sidecar(model, tmp_path):
    before = providers._cache_key_path(model, CPU, tmp_path)
    stat = os.stat(f"{model}_data")
    _write_weights(model, 5.0, mtime_ns=stat.st_mtime_ns + 1_000_000_000)
    assert os.stat(f"{model}_data").st_size == stat.st_size
    assert providers._cache_key_path(model, CPU, tmp_path) != before


def test_the_cache_key_follows_the_cpu(model, tmp_path):
    keys = []
    for flags in ("sse4_2 avx2", "sse4_2 avx512f"):
        cpuinfo = tmp_path / "cpuinfo"
        cpuinfo.write_text(f"processor\t: 0\nflags\t\t: {flags}\n")
        with patch.object(providers, "_CPUINFO", str(cpuinfo), create=True):
            keys.append(providers._cache_key_path(model, CPU, tmp_path))
    assert keys[0] != keys[1]


def test_the_copy_is_renamed_into_place(model, tmp_path):
    cache = tmp_path / "cache"
    final = providers._cache_key_path(model, CPU, cache)
    written_to = []
    real = providers.onnxruntime.InferenceSession

    def record(path, sess_options=None, **kwargs):
        written_to.append(sess_options.optimized_model_filepath)
        return real(path, sess_options=sess_options, **kwargs)

    with patch.object(providers.onnxruntime, "InferenceSession", side_effect=record):
        providers._build_session(model, CPU, None, cache)
    assert written_to and written_to[0] != str(final)
    assert final.is_file()
    assert sorted(entry.name for entry in cache.iterdir()) == sorted([final.name, f"{final.name}.data"])
