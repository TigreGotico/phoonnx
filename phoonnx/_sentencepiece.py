"""A reader for SentencePiece ``tokenizer.model`` files, in pure Python.

The Pocket TTS engine needs two things from a SentencePiece model: turn text
into ids and turn ids back into text. That is a small enough surface to
implement directly, which keeps a compiled dependency out of the install for
the seven Pocket TTS voices.

Unigram and BPE vocabularies are implemented, with or without byte fallback,
and with any compiled normalizer (``identity``, ``nmt_nfkc`` and the other
precompiled charsmaps). MOSS-TTS-Nano uses a BPE model with ``nmt_nfkc``.
Anything else raises
:class:`UnsupportedSentencePieceModel`, so a voice with an exotic
configuration fails loudly instead of being silently encoded the wrong way. A
wrong id is worse than a missing package here: it makes plausible but wrong
audio rather than an error.

The file is a protobuf ``ModelProto``. It is a flat message, so a few dozen
lines of varint reading replace the protobuf runtime as well.
"""
import heapq
import struct
from typing import Dict, List, Optional, Sequence, Tuple

from phoonnx.errors import UnsupportedSentencePieceModel

# Unigram scores are single-precision floats in the file, and SentencePiece
# adds them up in single precision too. Ties in the Viterbi search are broken
# by node order, so a sum that rounds differently can pick a different — and
# equally scoring, but differently spelled — segmentation. The arithmetic here
# is kept in float32 for that reason.
_F32 = struct.Struct("<f")

# SentencePiece charges unknown characters this much below the worst real
# piece. The constant is kUnkPenalty in unigram_model.cc.
_UNK_PENALTY = 10.0

# The renamed space. SentencePiece writes a space as U+2581 so that a piece
# can carry the word boundary that produced it.
_SPACE = "▁"
_SPACE_BYTES = _SPACE.encode("utf-8")

# ModelProto.SentencePiece.Type
_NORMAL, _UNKNOWN, _CONTROL, _USER_DEFINED, _UNUSED, _BYTE = 1, 2, 3, 4, 5, 6

# TrainerSpec.ModelType
_MODEL_TYPES = {1: "UNIGRAM", 2: "BPE", 3: "WORD", 4: "CHAR"}


def _read_varint(blob: bytes, index: int) -> Tuple[int, int]:
    """Read one base-128 varint, returning its value and the next index."""
    value = 0
    shift = 0
    while True:
        byte = blob[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, index
        shift += 7


def _fields(blob: bytes):
    """Yield ``(field_number, wire_type, value)`` for one protobuf message.

    Length-delimited fields come back as raw bytes, to be decoded by the
    caller that knows whether they hold a string or a nested message.
    """
    index = 0
    while index < len(blob):
        key, index = _read_varint(blob, index)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, index = _read_varint(blob, index)
            yield number, wire, value
        elif wire == 2:
            length, index = _read_varint(blob, index)
            yield number, wire, blob[index:index + length]
            index += length
        elif wire == 5:
            yield number, wire, _F32.unpack_from(blob, index)[0]
            index += 4
        elif wire == 1:
            yield number, wire, struct.unpack_from("<d", blob, index)[0]
            index += 8
        else:
            raise UnsupportedSentencePieceModel(f"protobuf wire type {wire}")


def _f32(value: float) -> float:
    """Round a Python float to what a C++ ``float`` would hold."""
    return _F32.unpack(_F32.pack(value))[0]


class SentencePieceProcessor:
    """A ``tokenizer.model`` reader with the calls phoonnx makes.

    The class name, ``Load``, ``Encode`` and ``Decode`` match the
    ``sentencepiece`` package, so it drops into the same call sites.
    """

    def __init__(self, model_file: Optional[str] = None):
        self._source = "<unloaded>"
        self._pieces: List[str] = []
        self._scores: List[float] = []
        self._types: List[int] = []
        self._piece_to_id: Dict[str, int] = {}
        # Only the pieces the lattice may match: NORMAL and USER_DEFINED.
        # Byte, control and unknown pieces are reachable by id but never by
        # matching their name in the text, exactly as SentencePiece keeps
        # them out of its trie.
        self._matchable: Dict[bytes, Tuple[int, float]] = {}
        self._max_match = 0
        self._unk_id = 0
        self._unk_score = 0.0
        self._byte_fallback = False
        self._byte_ids: Dict[int, int] = {}
        self._add_dummy_prefix = True
        self._remove_extra_whitespaces = True  # the protobuf default
        self._model_type = 1
        # The compiled normalizer: a darts-clone double-array trie over UTF-8
        # input, and the NUL-terminated replacement strings it points into.
        self._trie: Tuple[int, ...] = ()
        self._replacements = b""
        # BPE merges look pieces up by text: NORMAL pieces only.
        self._bpe_pieces: Dict[str, Tuple[int, float]] = {}
        self._unk_surface = " ⁇ "
        if model_file is not None:
            self.Load(model_file)

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: str) -> "SentencePieceProcessor":
        return cls(path)

    def Load(self, model_file: str) -> None:
        """Parse a ``.model`` file, refusing anything unsupported."""
        with open(model_file, "rb") as handle:
            self.LoadFromSerializedProto(handle.read(), source=str(model_file))

    load = Load

    def LoadFromSerializedProto(self, blob: bytes,
                                source: str = "<bytes>") -> None:
        self._source = source
        trainer: Optional[bytes] = None
        normalizer: Optional[bytes] = None
        for number, wire, value in _fields(blob):
            if number == 1 and wire == 2:          # repeated SentencePiece
                self._add_piece(value)
            elif number == 2 and wire == 2:        # TrainerSpec
                trainer = value
            elif number == 3 and wire == 2:        # NormalizerSpec
                normalizer = value
        if not self._pieces:
            raise UnsupportedSentencePieceModel(
                f"{self._source}: the model carries no pieces")
        self._read_trainer(trainer or b"")
        self._read_normalizer(normalizer or b"")
        # The unknown score is fixed by the worst NORMAL piece, so an unknown
        # character always loses to any real segmentation that covers it.
        normal = [score for score, kind in zip(self._scores, self._types)
                  if kind == _NORMAL]
        if not normal:
            raise UnsupportedSentencePieceModel(
                f"{self._source}: the model carries no NORMAL pieces")
        self._unk_score = _f32(min(normal) - _UNK_PENALTY)
        if self._byte_fallback and len(self._byte_ids) != 256:
            raise UnsupportedSentencePieceModel(
                f"{self._source}: byte_fallback is set but the vocabulary has "
                f"{len(self._byte_ids)} of the 256 <0xNN> byte pieces")
        # User-defined pieces skip the normalizer and start as one BPE symbol.
        # Neither is implemented, and UNUSED pieces need the BPE resegment step.
        if _USER_DEFINED in self._types and (self._trie or self._model_type == 2):
            raise UnsupportedSentencePieceModel(
                f"{self._source}: user-defined pieces with a compiled "
                f"normalizer or a BPE model are not implemented")
        if self._model_type == 2 and _UNUSED in self._types:
            raise UnsupportedSentencePieceModel(
                f"{self._source}: a BPE model with UNUSED pieces is not "
                f"implemented")

    def _add_piece(self, blob: bytes) -> None:
        piece, score, kind = "", 0.0, _NORMAL
        for number, _wire, value in _fields(blob):
            if number == 1:
                piece = value.decode("utf-8", "replace")
            elif number == 2:
                score = _f32(value)
            elif number == 3:
                kind = value
        index = len(self._pieces)
        self._pieces.append(piece)
        self._scores.append(score)
        self._types.append(kind)
        self._piece_to_id.setdefault(piece, index)
        if kind == _NORMAL:
            self._bpe_pieces.setdefault(piece, (index, score))
        if kind in (_NORMAL, _USER_DEFINED):
            key = piece.encode("utf-8")
            self._matchable[key] = (index, score)
            self._max_match = max(self._max_match, len(key))
        elif kind == _UNKNOWN:
            self._unk_id = index
        elif kind == _BYTE:
            # Byte pieces are spelled "<0x0A>" and are only ever reached
            # through the fallback path, never by matching that text.
            if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
                try:
                    self._byte_ids[int(piece[3:5], 16)] = index
                except ValueError:
                    pass

    def _read_trainer(self, blob: bytes) -> None:
        model_type = 1
        treat_whitespace_as_suffix = False
        for number, _wire, value in _fields(blob):
            if number == 3:
                model_type = value
            elif number == 24:
                treat_whitespace_as_suffix = bool(value)
            elif number == 35:
                self._byte_fallback = bool(value)
            elif number == 44:
                self._unk_surface = value.decode("utf-8", "replace")
        if model_type not in (1, 2):
            raise UnsupportedSentencePieceModel(
                f"{self._source}: this reader implements UNIGRAM and BPE "
                f"models, and the file declares "
                f"{_MODEL_TYPES.get(model_type, model_type)!r}")
        self._model_type = model_type
        if treat_whitespace_as_suffix:
            raise UnsupportedSentencePieceModel(
                f"{self._source}: treat_whitespace_as_suffix is set, which "
                f"this reader does not implement")

    def _read_normalizer(self, blob: bytes) -> None:
        charsmap = b""
        escape_whitespaces = True
        for number, _wire, value in _fields(blob):
            if number == 2:
                charsmap = value
            elif number == 3:
                self._add_dummy_prefix = bool(value)
            elif number == 4:
                self._remove_extra_whitespaces = bool(value)
            elif number == 5:
                escape_whitespaces = bool(value)
        if not escape_whitespaces:
            raise UnsupportedSentencePieceModel(
                f"{self._source}: escape_whitespaces is unset, which this "
                f"reader does not implement")
        if charsmap:
            size = struct.unpack_from("<I", charsmap)[0] if len(charsmap) >= 4 else -1
            if size <= 0 or size % 4 or 4 + size > len(charsmap):
                raise UnsupportedSentencePieceModel(
                    f"{self._source}: the compiled charsmap is malformed")
            self._trie = struct.unpack_from(f"<{size // 4}I", charsmap, 4)
            self._replacements = charsmap[4 + size:]

    # ------------------------------------------------------------------
    # vocabulary
    # ------------------------------------------------------------------
    def GetPieceSize(self) -> int:
        return len(self._pieces)

    def IdToPiece(self, index: int) -> str:
        return self._pieces[int(index)]

    def PieceToId(self, piece: str) -> int:
        return self._piece_to_id.get(piece, self._unk_id)

    def unk_id(self) -> int:
        return self._unk_id

    # ------------------------------------------------------------------
    # encoding
    # ------------------------------------------------------------------
    def Encode(self, text: str) -> List[int]:
        """Turn text into ids, the way ``sentencepiece`` does it.

        The text is normalized, cut into pieces by a Viterbi search over the
        piece scores, and any character no piece covers is spelled out as its
        UTF-8 bytes.
        """
        if isinstance(text, (list, tuple)):
            return [self.Encode(item) for item in text]
        normalized = self._normalize(text)
        if not normalized:
            return []
        if self._model_type == 2:
            segments = self._bpe(normalized.decode("utf-8"))
        else:
            segments = [(normalized[begin:end], index)
                        for begin, end, index in self._viterbi(normalized)]
        ids: List[int] = []
        for surface, index in segments:
            if index != self._unk_id:
                ids.append(index)
            elif self._byte_fallback:
                ids.extend(self._byte_ids[byte] for byte in surface)
            elif not ids or ids[-1] != self._unk_id:
                # A run of unknown pieces becomes one unknown id.
                ids.append(index)
        return ids

    encode = Encode

    def EncodeAsIds(self, text: str) -> List[int]:
        return self.Encode(text)

    def EncodeAsPieces(self, text: str) -> List[str]:
        return [self._pieces[index] for index in self.Encode(text)]

    def _normalize(self, text: str) -> bytes:
        """Normalize ``text`` to UTF-8, as normalizer.cc ``Normalize`` does.

        Each step replaces the longest charsmap match at the current place,
        or keeps one character. Spaces become U+2581, a dummy prefix space is
        added, and extra spaces are dropped when the model asks for it.
        """
        data = text.encode("utf-8")
        squeeze = self._remove_extra_whitespaces
        position = 0
        if squeeze:
            while position < len(data):
                piece, length = self._normalize_prefix(data, position)
                if piece != b" ":
                    break
                position += length
        if position == len(data):
            return b""
        out = bytearray(_SPACE_BYTES if self._add_dummy_prefix else b"")
        is_prev_space = squeeze
        while position < len(data):
            piece, length = self._normalize_prefix(data, position)
            position += length
            if is_prev_space:
                piece = piece.lstrip(b" ")
            if piece:
                out += piece.replace(b" ", _SPACE_BYTES)
                is_prev_space = squeeze and piece.endswith(b" ")
        if squeeze:
            while out.endswith(_SPACE_BYTES):
                del out[-len(_SPACE_BYTES):]
        return bytes(out)

    def _normalize_prefix(self, data: bytes, position: int) -> Tuple[bytes, int]:
        """The replacement for the longest charsmap match at ``position``.

        The search is darts-clone ``commonPrefixSearch``.
        """
        trie = self._trie
        if trie:
            node = trie[0] >> 10 << ((trie[0] & 0x200) >> 6)
            match = None
            for end in range(position, len(data)):
                node ^= data[end]
                unit = trie[node]
                if unit & 0x800000FF != data[end]:
                    break
                node ^= unit >> 10 << ((unit & 0x200) >> 6)
                if unit & 0x100:
                    match = end + 1, trie[node] & 0x7FFFFFFF
            if match:
                start = match[1]
                return (self._replacements[start:self._replacements.index(0, start)],
                        match[0] - position)
        length = _utf8_len(data, position)
        return data[position:position + length], length

    def _bpe(self, text: str) -> List[Tuple[bytes, int]]:
        """BPE segmentation of normalized ``text``, as ``(surface, piece id)``.

        Start from single characters and merge the adjacent pair whose joined
        piece scores best, the leftmost pair on a tie, as bpe_model.cc does.
        """
        pieces = self._bpe_pieces
        symbols: List[Optional[str]] = list(text)
        following = list(range(1, len(symbols) + 1))
        following[-1] = -1
        preceding = list(range(-1, len(symbols) - 1))
        agenda: List[Tuple[float, int, int, str]] = []

        def push(left: int, right: int) -> None:
            if left < 0 or right < 0:
                return
            joined = symbols[left] + symbols[right]
            found = pieces.get(joined)
            if found is not None:
                heapq.heappush(agenda, (-found[1], left, right, joined))

        for left in range(len(symbols) - 1):
            push(left, left + 1)
        while agenda:
            _score, left, right, joined = heapq.heappop(agenda)
            # A pair whose symbols changed after it was queued is stale.
            if following[left] != right or symbols[left] is None \
                    or symbols[right] is None \
                    or symbols[left] + symbols[right] != joined:
                continue
            symbols[left], symbols[right] = joined, None
            following[left] = following[right]
            if following[left] >= 0:
                preceding[following[left]] = left
            push(preceding[left], left)
            push(left, following[left])

        unk = (self._unk_id, 0.0)
        return [(symbol.encode("utf-8"), pieces.get(symbol, unk)[0])
                for symbol in symbols if symbol is not None]

    def _viterbi(self, blob: bytes) -> List[Tuple[int, int, int]]:
        """Best segmentation of ``blob``, as ``(begin, end, piece id)``.

        Nodes are built in the same order SentencePiece builds them — left to
        right, shortest match first, the unknown node last — because ties are
        broken by that order and not by anything in the scores.
        """
        length = len(blob)
        begin_nodes: List[List[int]] = [[] for _ in range(length + 1)]
        end_nodes: List[List[int]] = [[] for _ in range(length + 1)]
        starts: List[int] = []
        ends: List[int] = []
        piece_ids: List[int] = []
        scores: List[float] = []

        position = 0
        while position < length:
            char_len = _utf8_len(blob, position)
            has_single = False
            for match_len in range(1, min(self._max_match, length - position) + 1):
                found = self._matchable.get(blob[position:position + match_len])
                if found is None:
                    continue
                if match_len == char_len:
                    has_single = True
                starts.append(position)
                ends.append(position + match_len)
                piece_ids.append(found[0])
                scores.append(found[1])
                node = len(starts) - 1
                begin_nodes[position].append(node)
                end_nodes[position + match_len].append(node)
            if not has_single:
                # No piece covers this character on its own, so it can only be
                # crossed as an unknown.
                starts.append(position)
                ends.append(position + char_len)
                piece_ids.append(self._unk_id)
                scores.append(self._unk_score)
                node = len(starts) - 1
                begin_nodes[position].append(node)
                end_nodes[position + char_len].append(node)
            position += char_len

        # -1 stands for the begin-of-sentence node, which scores zero.
        end_nodes[0].append(-1)
        backtrace: List[float] = [0.0] * len(starts)
        previous: List[int] = [-1] * len(starts)

        for position in range(length + 1):
            for node in begin_nodes[position]:
                best_node = None
                best_score = 0.0
                for left in end_nodes[position]:
                    score = _f32((0.0 if left < 0 else backtrace[left])
                                 + scores[node])
                    if best_node is None or score > best_score:
                        best_node, best_score = left, score
                if best_node is None:
                    raise UnsupportedSentencePieceModel(
                        f"{self._source}: no path through the lattice; the "
                        f"vocabulary cannot cover this text")
                previous[node] = best_node
                backtrace[node] = best_score

        # The end-of-sentence node scores zero, so it simply picks the best
        # node that reaches the end of the text.
        best_node = None
        best_score = 0.0
        for left in end_nodes[length]:
            score = 0.0 if left < 0 else backtrace[left]
            if best_node is None or score > best_score:
                best_node, best_score = left, score

        path: List[Tuple[int, int, int]] = []
        node = best_node
        while node is not None and node >= 0:
            path.append((starts[node], ends[node], piece_ids[node]))
            node = previous[node]
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # decoding
    # ------------------------------------------------------------------
    def Decode(self, ids: Sequence[int]) -> str:
        """Turn ids back into text, the way ``sentencepiece`` does it.

        Byte pieces are gathered up and decoded together, so a character split
        across several of them comes back whole. Control pieces contribute
        nothing, an unknown id contributes the model's unknown surface, and
        the dummy prefix added at encode time is taken off again.
        """
        if ids and isinstance(ids[0], (list, tuple)):
            return [self.Decode(item) for item in ids]
        out: List[str] = []
        pending: bytearray = bytearray()
        first = True
        for index in ids:
            index = int(index)
            if not 0 <= index < len(self._pieces):
                raise IndexError(f"piece id {index} out of range")
            kind = self._types[index]
            if kind == _BYTE and self._byte_fallback:
                pending.append(int(self._pieces[index][3:5], 16))
                continue
            if pending:
                out.append(_utf8_to_text(bytes(pending)))
                pending.clear()
                first = False
            if kind == _CONTROL:
                continue
            if kind == _UNKNOWN:
                text = self._unk_surface
            else:
                text = self._pieces[index]
                if first and self._add_dummy_prefix and text.startswith(_SPACE):
                    text = text[1:]
                text = text.replace(_SPACE, " ")
            # With remove_extra_whitespaces the text start lasts until
            # something is written, so every leading U+2581 piece is dropped.
            first = first and not text and self._remove_extra_whitespaces
            out.append(text)
        if pending:
            out.append(_utf8_to_text(bytes(pending)))
        return "".join(out)

    decode = Decode

    def DecodeIds(self, ids: Sequence[int]) -> str:
        return self.Decode(ids)


def _utf8_to_text(data: bytes) -> str:
    """Decode bytes the way SentencePiece decodes a run of byte pieces.

    A byte that starts no valid UTF-8 character is replaced on its own, one
    U+FFFD per bad byte. Python's ``errors="replace"`` collapses a truncated
    multi-byte sequence into a single U+FFFD instead, which would give a
    different string than the reference implementation.
    """
    out: List[str] = []
    index = 0
    while index < len(data):
        width = _utf8_len(data, index)
        chunk = data[index:index + width]
        if len(chunk) == width:
            try:
                out.append(chunk.decode("utf-8"))
                index += width
                continue
            except UnicodeDecodeError:
                pass
        out.append("\ufffd")
        index += 1
    return "".join(out)


def _utf8_len(blob: bytes, index: int) -> int:
    """Length in bytes of the UTF-8 character starting at ``index``."""
    lead = blob[index]
    if lead < 0x80:
        return 1
    if lead >> 5 == 0b110:
        return 2
    if lead >> 4 == 0b1110:
        return 3
    if lead >> 3 == 0b11110:
        return 4
    return 1


def load(path: str) -> SentencePieceProcessor:
    """Read a ``tokenizer.model`` file."""
    return SentencePieceProcessor(path)
