"""Read the external-data file names of an ONNX graph, in pure Python.

A graph can keep the bytes of a tensor in another file. Such a tensor names
the file in its ``external_data`` entry ``location``, relative to the folder of
the graph, and onnxruntime reads the file from there. A downloader needs these
names to put every file beside the graph: exporters do not agree on them
(``model.onnx_data``, ``model.onnx_data_1``, ``model_int8.onnx.data``).

The graph is a protobuf ``ModelProto``. The ``onnx`` package reads it, but it is
not a dependency of phoonnx, and only a few messages can hold a tensor. The
walk below reads those messages and skips every other field by its length, so
the bytes of the weights are never copied. It visits the tensors that
``onnx.external_data_helper`` visits: the initializers of the graph and of its
subgraphs, and the tensor attributes of every node, also in model functions.

The field numbers come from ``onnx.proto``.
"""
import functools
import mmap
import os
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

_MODEL_GRAPH, _MODEL_FUNCTIONS = 7, 25
_GRAPH_NODE, _GRAPH_INITIALIZER = 1, 5
_FUNCTION_NODE = 7
_NODE_ATTRIBUTE = 5
_ATTRIBUTE_TENSORS = (5, 10)  # t, tensors
_ATTRIBUTE_GRAPHS = (6, 11)   # g, graphs
_TENSOR_EXTERNAL_DATA = 13
_ENTRY_KEY, _ENTRY_VALUE = 1, 2


def _varint(buf, index: int) -> Tuple[int, int]:
    """Read one base-128 varint of at most 10 bytes, returning its value and
    the next index."""
    value = shift = 0
    while True:
        byte = buf[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, index
        shift += 7
        if shift > 63:
            raise ValueError("a protobuf varint is longer than 10 bytes")


def _fields(buf, start: int, end: int) -> Iterator[Tuple[int, int, int]]:
    """Yield ``(field number, start, end)`` of each length-delimited field of
    the message in ``buf[start:end]``, skipping the other fields."""
    index = start
    while index < end:
        key, index = _varint(buf, index)
        wire = key & 7
        if wire == 0:
            _, index = _varint(buf, index)
        elif wire == 1:
            index += 8
        elif wire == 5:
            index += 4
        elif wire == 2:
            length, index = _varint(buf, index)
            if index + length > end:
                raise ValueError("a protobuf field runs past its message")
            yield key >> 3, index, index + length
            index += length
        else:
            raise ValueError(f"protobuf wire type {wire}")
    if index != end:
        raise ValueError("a protobuf field runs past its message")


def _tensor_locations(buf, start: int, end: int) -> Iterator[str]:
    for number, entry_start, entry_end in _fields(buf, start, end):
        if number != _TENSOR_EXTERNAL_DATA:
            continue
        entry = {key: buf[s:e] for key, s, e in _fields(buf, entry_start, entry_end)}
        if entry.get(_ENTRY_KEY) == b"location":
            yield entry.get(_ENTRY_VALUE, b"").decode("utf-8")


def _node_locations(buf, start: int, end: int) -> Iterator[str]:
    for number, attr_start, attr_end in _fields(buf, start, end):
        if number != _NODE_ATTRIBUTE:
            continue
        for kind, s, e in _fields(buf, attr_start, attr_end):
            if kind in _ATTRIBUTE_TENSORS:
                yield from _tensor_locations(buf, s, e)
            elif kind in _ATTRIBUTE_GRAPHS:
                yield from _graph_locations(buf, s, e, _GRAPH_NODE, _GRAPH_INITIALIZER)


def _graph_locations(buf, start: int, end: int, node: int,
                     initializer: Optional[int] = None) -> Iterator[str]:
    """Locations in a ``GraphProto``, or in a ``FunctionProto``, which has
    nodes under another field number and no initializers."""
    for number, s, e in _fields(buf, start, end):
        if number == initializer:
            yield from _tensor_locations(buf, s, e)
        elif number == node:
            yield from _node_locations(buf, s, e)


def external_data_locations(path: Union[str, Path]) -> Tuple[str, ...]:
    """The external-data file names the ONNX graph at ``path`` refers to.

    Each name comes once, in the order of the first tensor that names it. A
    graph that keeps all its weights inside gives an empty tuple. The names are
    given as written in the graph: this function does not check them. The
    answer is kept per file version.

    Raises:
        OSError: the file cannot be read.
        ValueError: the file is not an ONNX model.
    """
    real_path = os.path.realpath(path)
    stat = os.stat(real_path)
    return _read_locations(real_path, stat.st_size, stat.st_mtime_ns, stat.st_ino)


@functools.lru_cache(maxsize=1024)
def _read_locations(path: str, size: int, mtime_ns: int, inode: int) -> Tuple[str, ...]:
    """``size``, ``mtime_ns`` and ``inode`` only key the cache."""
    with open(path, "rb") as handle, \
            mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        locations: List[str] = []
        has_graph = False
        try:
            for number, start, end in _fields(buf, 0, len(buf)):
                if number == _MODEL_GRAPH:
                    has_graph = True
                    locations += _graph_locations(buf, start, end, _GRAPH_NODE,
                                                  _GRAPH_INITIALIZER)
                elif number == _MODEL_FUNCTIONS:
                    locations += _graph_locations(buf, start, end, _FUNCTION_NODE)
        except IndexError as exc:
            raise ValueError(f"'{path}' ends inside a protobuf field") from exc
        except RecursionError as exc:
            raise ValueError(f"'{path}' nests graphs too deeply") from exc
    if not has_graph:
        raise ValueError(f"'{path}' is not an ONNX model: it has no graph")
    return tuple(dict.fromkeys(locations))
