"""The VITS text-encoder input built from a phone feature table.

A ``features.json`` file (format ``phoonnx-input-features/1``) gives a feature
vector for each id of a voice's ``phoneme_id_map``, or ``null`` for an id that
is not a phone. Two input layers read it:

``residual``
    A linear projection of the features, plus one learned vector per id.
``composition``
    A linear projection of the features, plus a learned vector for the base
    symbol of the id and one for each of its marks.

Every learned table starts at zero. A phone that never occurs in training gets
no gradient in its own rows, so it keeps the projection of its features, which
places it near the trained phones that share them. An id with ``null`` features
gets only the learned part.

Export replaces either layer with a plain embedding table of the same values
(:meth:`FeatureInput.table`), so an exported voice runs as any other.
"""
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from torch import nn

from phoonnx_train.segments import ATTACHING, TIES

FORMAT = "phoonnx-input-features/1"
INPUT_LAYERS = ("ids", "residual", "composition")


def map_sha256(phoneme_id_map: Mapping[str, Any]) -> str:
    """The sha256 that ties a feature table to the map it was made from."""
    text = json.dumps(phoneme_id_map, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_input_features(path: Path, phoneme_id_map: Mapping[str, Any], num_symbols: int) -> Dict[str, Any]:
    """Read *path* and check it against the voice's map.

    Returns plain lists, which a checkpoint keeps in its hyperparameters:
    ``features`` (one row per id, zeros for a ``null`` id), ``has_vector`` and
    ``symbols``.
    """
    table = json.loads(Path(path).read_text(encoding="utf-8"))
    if table.get("format") != FORMAT:
        raise ValueError(f"{path}: format is {table.get('format')!r}, expected {FORMAT!r}")
    expected = map_sha256(phoneme_id_map)
    if table.get("phoneme_id_map_sha256") != expected:
        raise ValueError(
            f"{path} was made from another phoneme_id_map: its sha256 is "
            f"{table.get('phoneme_id_map_sha256')}, the dataset config's map gives {expected}. "
            "Generate the table again from this config.")
    width = len(table["feature_names"])
    vectors = {int(i): row for i, row in table["vectors"].items()}
    map_ids = {i for v in phoneme_id_map.values() for i in (v if isinstance(v, list) else [v])}
    missing = sorted(map_ids - vectors.keys())
    if missing:
        raise ValueError(f"{path} has no entry for ids {missing}")
    outside = sorted(i for i in vectors if not 0 <= i < num_symbols)
    if outside:
        raise ValueError(f"{path} has ids {outside} outside the {num_symbols} model symbols")

    features = [[0.0] * width for _ in range(num_symbols)]
    has_vector = [False] * num_symbols
    symbols = [""] * num_symbols
    for i, row in vectors.items():
        symbols[i] = row["symbol"]
        if row["vector"] is not None:
            if len(row["vector"]) != width:
                raise ValueError(f"{path}: id {i} has {len(row['vector'])} values, expected {width}")
            features[i] = [float(v) for v in row["vector"]]
            has_vector[i] = True
    return {"features": features, "has_vector": has_vector, "symbols": symbols,
            "source": table.get("source"), "feature_names": table["feature_names"]}


def split_marks(symbol: str) -> Tuple[str, List[str]]:
    """The base of *symbol* and its marks, in canonical decomposition.

    A mark is a combining mark other than a tie bar, or an attaching modifier
    letter: ``ɐ̃`` gives ``ɐ`` and U+0303, ``tʲ`` gives ``t`` and ``ʲ``, and
    ``d͡ʒ`` is a base with no marks. A symbol made only of marks is its own base.
    """
    chars = unicodedata.normalize("NFD", symbol)
    marks = [c for c in chars
             if c in ATTACHING or (unicodedata.category(c) == "Mn" and c not in TIES)]
    base = "".join(c for c in chars if c not in marks)
    return (base, marks) if base else (chars, [])


class FeatureInput(nn.Module):
    """Feature projection plus a learned part, in place of ``nn.Embedding``."""

    def __init__(self, input_features: Dict[str, Any], hidden_channels: int, mode: str):
        super().__init__()
        if mode not in ("residual", "composition"):
            raise ValueError(f"input layer {mode!r} does not read features")
        self.mode = mode
        features = torch.tensor(input_features["features"], dtype=torch.float32)
        self.register_buffer("features", features, persistent=False)
        self.proj = nn.Linear(features.size(1), hidden_channels)
        n_vocab = features.size(0)
        if mode == "residual":
            self.residual = nn.Embedding(n_vocab, hidden_channels)
            nn.init.zeros_(self.residual.weight)
            return
        parts = [split_marks(s) for s in input_features["symbols"]]
        bases = sorted({base for base, _ in parts})
        # index 0 is "no mark": a fixed zero row that pads the mark lists
        marks = ["", *sorted({m for _, ms in parts for m in ms})]
        width = max(1, max(len(ms) for _, ms in parts))
        self.register_buffer("base_ids", torch.tensor([bases.index(b) for b, _ in parts]), persistent=False)
        self.register_buffer("mark_ids", torch.tensor(
            [[marks.index(m) for m in ms] + [0] * (width - len(ms)) for _, ms in parts]), persistent=False)
        self.base = nn.Embedding(len(bases), hidden_channels)
        self.marks = nn.Embedding(len(marks), hidden_channels, padding_idx=0)
        nn.init.zeros_(self.base.weight)
        nn.init.zeros_(self.marks.weight)

    def table(self) -> torch.Tensor:
        """The input vector of every id, ``[n_vocab, hidden]``."""
        learned = (self.residual.weight if self.mode == "residual"
                   else self.base(self.base_ids) + self.marks(self.mark_ids).sum(1))
        return self.proj(self.features) + learned

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "residual":
            return self.proj(self.features[x]) + self.residual(x)
        return (self.proj(self.features[x]) + self.base(self.base_ids[x])
                + self.marks(self.mark_ids[x]).sum(-2))


def make_input_layer(input_layer: str, input_features: Optional[Dict[str, Any]],
                     hidden_channels: int) -> Optional[FeatureInput]:
    """The feature layer for *input_layer*, or None for the plain id embedding."""
    if input_layer not in INPUT_LAYERS:
        raise ValueError(f"input layer {input_layer!r}: choose one of {', '.join(INPUT_LAYERS)}")
    if input_layer == "ids":
        if input_features is not None:
            raise ValueError("a feature table needs input layer 'residual' or 'composition'")
        return None
    if input_features is None:
        raise ValueError(f"input layer {input_layer!r} needs a features.json table")
    return FeatureInput(input_features, hidden_channels, input_layer)
