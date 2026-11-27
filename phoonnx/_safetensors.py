"""Read a safetensors file into numpy arrays.

The format is an 8-byte little-endian header length, a JSON header that maps each
tensor name to ``dtype``, ``shape`` and ``data_offsets`` (relative to the end of the
header), then the raw little-endian tensor bytes.
"""
import json
from pathlib import Path
from typing import Dict, Union

import numpy as np

_DTYPES = {
    "F64": "<f8", "F32": "<f4", "F16": "<f2",
    "I64": "<i8", "I32": "<i4", "I16": "<i2", "I8": "i1",
    "U8": "u1", "BOOL": "?",
}


def load_file(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Return ``{name: array}`` for every tensor in the file. BF16 becomes float32."""
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ValueError(f"{path}: file is shorter than the 8-byte header length")
    header_len = int.from_bytes(data[:8], "little")
    if 8 + header_len > len(data):
        raise ValueError(f"{path}: header length {header_len} is past the end of the file")
    try:
        header = json.loads(data[8:8 + header_len])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: header is not valid JSON") from exc
    if not isinstance(header, dict):
        raise ValueError(f"{path}: header is not a JSON object")

    body = memoryview(data)[8 + header_len:]
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype, shape, (start, end) = info["dtype"], info["shape"], info["data_offsets"]
        if not 0 <= start <= end <= len(body):
            raise ValueError(f"{path}: {name} offsets {start}..{end} are out of range")
        raw = body[start:end]
        if dtype == "BF16":
            array = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
        elif dtype in _DTYPES:
            array = np.frombuffer(raw, _DTYPES[dtype])
        else:
            raise ValueError(f"{path}: {name} has unsupported dtype {dtype}")
        if array.size != int(np.prod(shape)):
            raise ValueError(f"{path}: {name} has {array.size} values for shape {shape}")
        tensors[name] = array.reshape(shape)
    return tensors
