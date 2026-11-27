import json
import struct

import pytest

from phoonnx._safetensors import load_file


def _file(path, header, blob):
    head = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head + blob)
    return path


# (dtype, struct format, values, expected values); BF16 0x3FC0 is 1.5, 0xC000 is -2.0
CASES = [
    ("F64", "<2d", [1.25, -3.5], [1.25, -3.5]),
    ("F32", "<2f", [0.5, -7.0], [0.5, -7.0]),
    ("F16", "<2e", [1.5, -0.25], [1.5, -0.25]),
    ("I64", "<2q", [2 ** 40, -5], [2 ** 40, -5]),
    ("I32", "<2i", [70000, -1], [70000, -1]),
    ("I8", "<2b", [-128, 127], [-128, 127]),
    ("U8", "<2B", [0, 255], [0, 255]),
    ("BOOL", "<2?", [True, False], [True, False]),
    ("BF16", "<2H", [0x3FC0, 0xC000], [1.5, -2.0]),
]


@pytest.mark.parametrize("dtype,fmt,raw,expected", CASES, ids=[c[0] for c in CASES])
def test_reads_each_dtype(tmp_path, dtype, fmt, raw, expected):
    blob = b"\0" * 3 + struct.pack(fmt, *raw)  # a leading pad proves offsets are honoured
    path = _file(tmp_path / "t.safetensors", {
        "__metadata__": {"format": "pt"},
        "x": {"dtype": dtype, "shape": [2, 1], "data_offsets": [3, len(blob)]}}, blob)
    tensors = load_file(path)
    assert list(tensors) == ["x"]
    assert tensors["x"].shape == (2, 1)
    assert tensors["x"].ravel().tolist() == expected


def test_bf16_becomes_float32(tmp_path):
    path = _file(tmp_path / "t.safetensors",
                 {"x": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}, struct.pack("<H", 0x3FC0))
    assert str(load_file(path)["x"].dtype) == "float32"


@pytest.mark.parametrize("data", [
    b"\x01\x00",                                         # shorter than 8 bytes
    struct.pack("<Q", 999) + b"{}",                      # header length past the end
    struct.pack("<Q", 4) + b"nope",                      # header is not JSON
], ids=["short", "long-header", "bad-json"])
def test_bad_header_raises(tmp_path, data):
    path = tmp_path / "t.safetensors"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        load_file(path)


def test_out_of_range_offsets_raise(tmp_path):
    path = _file(tmp_path / "t.safetensors",
                 {"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}, b"\0" * 4)
    with pytest.raises(ValueError, match="out of range"):
        load_file(path)


def test_warm_start_reads_bf16_into_a_writable_tensor(tmp_path):
    torch = pytest.importorskip("torch")
    from phoonnx_train.mosstts.warmstart import load_source_state_dict

    path = _file(tmp_path / "model.safetensors",
                 {"w": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}, struct.pack("<2H", 0x3FC0, 0xC000))
    state, _ = load_source_state_dict(path)
    assert state["w"].dtype == torch.float32
    assert state["w"].tolist() == [1.5, -2.0]
    state["w"].add_(1.0)
