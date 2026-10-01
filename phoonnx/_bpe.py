"""A reader for HuggingFace ``tokenizer.json`` files, in pure Python.

phoonnx needs four things from a tokenizer: load a ``tokenizer.json``, encode
text to ids, decode ids back to text, and look a token up by name. That is a
small enough surface to
implement directly, which keeps a compiled dependency out of the install for
the handful of voices that carry a BPE vocab.

Only what those files actually use is implemented — BPE models, the
normalizers and pre-tokenizers they declare. Anything else raises
:class:`UnsupportedTokenizer`, so a voice with an exotic configuration fails
loudly and can fall back to the ``tokenizers`` package rather than being
silently encoded the wrong way.
"""
import heapq
import json
import sys
import re
import unicodedata
from functools import lru_cache
from typing import Dict, List, Optional, Tuple


class UnsupportedTokenizer(Exception):
    """The tokenizer.json declares a component this reader does not implement."""


@lru_cache(maxsize=1)
def _byte_to_unicode() -> Dict[int, str]:
    """The GPT-2 byte encoder table used by ByteLevel.

    Every byte maps to a printable character, so arbitrary bytes survive a
    round trip through text. The table is fixed by the algorithm.
    """
    printable = (list(range(ord("!"), ord("~") + 1))
                 + list(range(ord("\xa1"), ord("\xac") + 1))
                 + list(range(ord("\xae"), ord("\xff") + 1)))
    mapped = printable[:]
    spare = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapped.append(256 + spare)
            spare += 1
    return {b: chr(c) for b, c in zip(printable, mapped)}


@lru_cache(maxsize=1)
def _unicode_to_byte() -> Dict[str, int]:
    """The GPT-2 byte decoder table: the inverse of :func:`_byte_to_unicode`."""
    return {char: byte for byte, char in _byte_to_unicode().items()}


@lru_cache(maxsize=None)
def _category_ranges(prop: str) -> str:
    """Character ranges for a Unicode property, as class body text.

    ``re`` has no ``\\p{L}``, so the property is expanded into explicit ranges
    built from :mod:`unicodedata`. The result goes inside a character class.
    """
    pieces: List[str] = []
    start = None
    previous = None
    for code in range(sys.maxunicode + 1):
        category = unicodedata.category(chr(code))
        if category.startswith(prop) if len(prop) == 1 else category == prop:
            if start is None:
                start = code
            previous = code
        elif start is not None:
            pieces.append((start, previous))
            start = None
    if start is not None:
        pieces.append((start, previous))
    out = []
    for low, high in pieces:
        if low == high:
            out.append(re.escape(chr(low)))
        else:
            out.append(f"{re.escape(chr(low))}-{re.escape(chr(high))}")
    return "".join(out)


_PROPERTY_RE = re.compile(r"\\([pP])\{(\w+)\}")


def _translate_regex(pattern: str) -> str:
    """Rewrite a Rust-flavoured regex into one :mod:`re` accepts.

    Only ``\\p{...}`` and ``\\P{...}`` differ in the patterns these
    vocabularies declare. Everything else is common syntax.
    """
    if "\\p{" not in pattern and "\\P{" not in pattern:
        return pattern
    out = []
    index = 0
    in_class = False
    while index < len(pattern):
        char = pattern[index]
        if char == "\\":
            match = _PROPERTY_RE.match(pattern, index)
            if match:
                negated = match.group(1) == "P"
                body = _category_ranges(match.group(2))
                if not body:
                    raise UnsupportedTokenizer(f"regex property {match.group(0)!r}")
                if in_class:
                    if negated:
                        raise UnsupportedTokenizer(
                            f"negated {match.group(0)!r} inside a character class")
                    out.append(body)
                else:
                    out.append(("[^" if negated else "[") + body + "]")
                index = match.end()
                continue
            out.append(pattern[index:index + 2])
            index += 2
            continue
        if char == "[" and not in_class:
            in_class = True
        elif char == "]" and in_class:
            in_class = False
        out.append(char)
        index += 1
    return "".join(out)


def _normalize(text: str, spec: Optional[dict]) -> str:
    if not spec:
        return text
    kind = spec.get("type")
    if kind == "Sequence":
        for step in spec.get("normalizers", []):
            text = _normalize(text, step)
        return text
    if kind in ("NFC", "NFD", "NFKC", "NFKD"):
        return unicodedata.normalize(kind, text)
    if kind == "Lowercase":
        return text.lower()
    if kind == "Strip":
        left = spec.get("strip_left", True)
        right = spec.get("strip_right", True)
        if left:
            text = text.lstrip()
        if right:
            text = text.rstrip()
        return text
    if kind == "Replace":
        pattern = spec.get("pattern", {})
        content = spec.get("content", "")
        if "String" in pattern:
            return text.replace(pattern["String"], content)
        if "Regex" in pattern:
            return re.sub(pattern["Regex"], content, text)
        raise UnsupportedTokenizer(f"Replace pattern {pattern!r}")
    if kind == "Prepend":
        return spec.get("prepend", "") + text
    raise UnsupportedTokenizer(f"normalizer {kind!r}")


_WHITESPACE_RE = re.compile(r"\w+|[^\w\s]+")

# The split GPT-2 applies inside ByteLevel when ``use_regex`` is on.
_GPT2_SPLIT = (r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+"
               r"| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")


@lru_cache(maxsize=128)
def _compile(pattern: str):
    try:
        return re.compile(_translate_regex(pattern))
    except re.error as error:
        raise UnsupportedTokenizer(f"regex {pattern!r}: {error}") from error


def _split(text: str, regex: str, behavior: str) -> List[str]:
    """Cut text on a regex, keeping the matches where the behaviour says to.

    ``removed`` drops the matched separators. ``isolated`` keeps each match as
    its own piece. The ``merged_*`` behaviours glue the match onto the
    neighbouring piece.
    """
    compiled = _compile(regex)
    pieces: List[str] = []
    cursor = 0
    for match in compiled.finditer(text):
        if match.start() == match.end():
            continue
        before = text[cursor:match.start()]
        found = match.group()
        cursor = match.end()
        if behavior == "removed":
            if before:
                pieces.append(before)
        elif behavior == "merged_with_previous":
            if pieces and not before:
                pieces[-1] += found
            else:
                pieces.append(before + found)
        elif behavior == "merged_with_next":
            if before:
                pieces.append(before)
            pieces.append(found)
        else:  # isolated, and the default
            if before:
                pieces.append(before)
            pieces.append(found)
    tail = text[cursor:]
    if tail:
        if behavior == "merged_with_next" and pieces:
            pieces[-1] += tail
        else:
            pieces.append(tail)
    if behavior == "merged_with_next":
        # A match belongs to the piece that follows it; fold them now.
        folded: List[str] = []
        pending = ""
        for piece in pieces:
            if compiled.fullmatch(piece):
                pending += piece
            else:
                folded.append(pending + piece)
                pending = ""
        if pending:
            folded.append(pending)
        pieces = folded
    return [p for p in pieces if p]


def _pre_tokenize(text: str, spec: Optional[dict]) -> List[str]:
    """Split text into the pieces the BPE model is applied to."""
    if not spec:
        return [text]
    kind = spec.get("type")
    if kind == "Sequence":
        pieces = [text]
        for step in spec.get("pretokenizers", []) or spec.get("pre_tokenizers", []):
            nxt: List[str] = []
            for piece in pieces:
                nxt.extend(_pre_tokenize(piece, step))
            pieces = nxt
        return pieces
    if kind == "Whitespace":
        return _WHITESPACE_RE.findall(text)
    if kind == "WhitespaceSplit":
        return text.split()
    if kind == "ByteLevel":
        if spec.get("add_prefix_space") and text and not text.startswith(" "):
            text = " " + text
        table = _byte_to_unicode()
        pieces = ([m.group() for m in _compile(_GPT2_SPLIT).finditer(text)]
                  if spec.get("use_regex", True) else [text])
        return ["".join(table[b] for b in piece.encode("utf-8"))
                for piece in pieces]
    if kind == "Split":
        pattern = spec.get("pattern", {})
        if "Regex" in pattern:
            regex = pattern["Regex"]
        elif "String" in pattern:
            regex = re.escape(pattern["String"])
        else:
            raise UnsupportedTokenizer(f"Split pattern {pattern!r}")
        if spec.get("invert"):
            raise UnsupportedTokenizer("inverted Split")
        return _split(text, regex, (spec.get("behavior") or "").lower())
    if kind == "Punctuation":
        return [p for p in re.findall(r"\w+|[^\w\s]", text) if p]
    if kind == "Metaspace":
        # The reader implements one shape: prepend the replacement, do not split
        # on it. `split` and `prepend_scheme` both move token boundaries, so a
        # vocab that sets them differently is declined rather than guessed at.
        unknown = set(spec) - {"type", "replacement", "add_prefix_space",
                               "prepend_scheme", "split"}
        if unknown:
            raise UnsupportedTokenizer(f"Metaspace options {sorted(unknown)}")
        if spec.get("split", False):
            raise UnsupportedTokenizer("Metaspace split=True")
        scheme = spec.get("prepend_scheme")
        if scheme is not None and scheme != "always":
            raise UnsupportedTokenizer(f"Metaspace prepend_scheme={scheme!r}")
        replacement = spec.get("replacement", "▁")
        if spec.get("add_prefix_space", True) and not text.startswith(" "):
            text = " " + text
        return [text.replace(" ", replacement)]
    raise UnsupportedTokenizer(f"pre_tokenizer {kind!r}")


class _BPE:
    """The byte-pair merge loop over one pre-tokenized piece."""

    def __init__(self, vocab: Dict[str, int], merges, unk: Optional[str],
                 continuing_prefix: str = "", ignore_merges: bool = False):
        self.vocab = vocab
        self.unk = unk
        self.continuing_prefix = continuing_prefix
        # Llama-family vocabs set this: a pre-token that is already a token is
        # taken whole, and the merges are never consulted for it.
        self.ignore_merges = ignore_merges
        self.ranks: Dict[Tuple[str, str], int] = {}
        for index, merge in enumerate(merges):
            if isinstance(merge, str):
                left, _, right = merge.partition(" ")
            else:
                left, right = merge
            self.ranks[(left, right)] = index
        self._cache: Dict[str, List[str]] = {}

    def tokenize(self, piece: str) -> List[str]:
        """Merge one pre-token down to its subwords.

        Rescanning the whole piece after every merge costs O(n^2), and a
        pre-token is not always short: the pre-tokenizers these vocabularies
        declare keep a run of letters whole, so a paragraph of Chinese or
        Japanese — written without spaces — arrives here as a single piece of
        thousands of characters. That is a request-length stall on a TTS
        server, so the merges are driven from a heap over a linked list
        instead: each merge only reconsiders the two pairs it created.
        """
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        if self.ignore_merges and piece in self.vocab:
            self._cache[piece] = [piece]
            return [piece]
        symbols = list(piece)
        if not symbols:
            return []
        if self.continuing_prefix:
            symbols = [symbols[0]] + [self.continuing_prefix + s for s in symbols[1:]]
        count = len(symbols)
        if count > 1 and self.ranks:
            # `symbols` is read as a linked list from here on: a merge rewrites
            # the left slot and unlinks the right one, so no element ever moves.
            following = list(range(1, count + 1))
            following[-1] = -1
            preceding = list(range(-1, count - 1))
            alive = [True] * count

            # (rank, left slot) orders the merges exactly as the full rescan did:
            # lowest rank first, and leftmost among equal ranks. The strings are
            # carried along so a pair invalidated by an earlier merge is
            # recognised when it surfaces, rather than being re-applied.
            heap = [(rank, i, i + 1, symbols[i], symbols[i + 1])
                    for i in range(count - 1)
                    for rank in (self.ranks.get((symbols[i], symbols[i + 1])),)
                    if rank is not None]
            heapq.heapify(heap)

            while heap:
                rank, left_at, right_at, left, right = heapq.heappop(heap)
                if not alive[left_at] or not alive[right_at]:
                    continue
                if following[left_at] != right_at:
                    continue
                if symbols[left_at] != left or symbols[right_at] != right:
                    continue
                symbols[left_at] = left + right
                alive[right_at] = False
                after = following[right_at]
                following[left_at] = after
                if after != -1:
                    preceding[after] = left_at
                before = preceding[left_at]
                if before != -1:
                    rank = self.ranks.get((symbols[before], symbols[left_at]))
                    if rank is not None:
                        heapq.heappush(heap, (rank, before, left_at,
                                              symbols[before], symbols[left_at]))
                if after != -1:
                    rank = self.ranks.get((symbols[left_at], symbols[after]))
                    if rank is not None:
                        heapq.heappush(heap, (rank, left_at, after,
                                              symbols[left_at], symbols[after]))

            merged = []
            at = 0
            while at != -1:
                merged.append(symbols[at])
                at = following[at]
            symbols = merged
        self._cache[piece] = symbols
        return symbols

    def to_ids(self, tokens: List[str]) -> List[int]:
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            elif self.unk is not None and self.unk in self.vocab:
                ids.append(self.vocab[self.unk])
        return ids


# Every key a BPE model may carry, and the only value of each one this reader
# reproduces. A key outside this table, or a value outside it, changes what the
# ids come out as — so the vocab is declined and `tokenizers` takes over.
# Reading the file and ignoring the switch is the one outcome to avoid: it makes
# wrong audio instead of an error.
_BPE_HANDLED = {
    "byte_fallback": (False, None),
    "dropout": (None, 0, 0.0),
    "fuse_unk": (False, None),
    "end_of_word_suffix": (None, ""),
}
_BPE_READ = {"type", "vocab", "merges", "unk_token", "continuing_subword_prefix",
             "ignore_merges"}


def _check_bpe_options(model: dict) -> None:
    """Refuse a BPE model whose options this reader does not implement.

    ``byte_fallback`` is the one that bites: with it on, a character outside
    the vocab becomes a run of ``<0xNN>`` byte tokens, and without it the same
    character becomes ``<unk>``. Both encode; only one is right. The Indic
    Parler prompt vocabulary sets it, along with ``fuse_unk``.
    """
    for key, allowed in _BPE_HANDLED.items():
        if key in model and model[key] not in allowed:
            raise UnsupportedTokenizer(
                f"BPE option {key}={model[key]!r} (this reader implements "
                f"{key}={allowed[0]!r} only)")
    unknown = set(model) - _BPE_READ - set(_BPE_HANDLED)
    if unknown:
        raise UnsupportedTokenizer(
            f"unrecognised BPE options {sorted(unknown)}: they may change the "
            f"ids, and this reader would ignore them")


def _wordpiece_cleanup(text: str) -> str:
    """Undo the spacing WordPiece tokenization puts around punctuation."""
    for old, new in ((" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","),
                     (" ' ", "'"), (" n't", "n't"), (" 'm", "'m"),
                     (" do not", " don't"), (" 's", "'s"), (" 've", "'ve"),
                     (" 're", "'re")):
        text = text.replace(old, new)
    return text


def _byte_level_decode(tokens: List[str]) -> List[str]:
    """Read the GPT-2 byte characters back into the bytes they stand for.

    A token whose characters are not all in the table is passed through as its
    own UTF-8 bytes, and the whole run is decoded once at the end — a character
    can be split across two tokens, so decoding token by token would break it.
    """
    table = _unicode_to_byte()
    raw = bytearray()
    for token in tokens:
        try:
            raw.extend(table[char] for char in token)
        except KeyError:
            raw.extend(token.encode("utf-8"))
    return [raw.decode("utf-8", "replace")]


def _decode_chain(tokens: List[str], spec: Optional[dict]) -> List[str]:
    """One step of the decoder, token list in and token list out.

    This mirrors ``tokenizers``' ``decode_chain``: a decoder rewrites the list
    of tokens, and the pieces are joined with nothing at the end.
    """
    if not spec:
        return tokens
    kind = spec.get("type")
    if kind == "Sequence":
        for step in spec.get("decoders") or []:
            tokens = _decode_chain(tokens, step)
        return tokens
    if kind == "ByteLevel":
        return _byte_level_decode(tokens)
    if kind == "Metaspace":
        replacement = spec.get("replacement", "▁")
        scheme = spec.get("prepend_scheme", "always")
        if scheme not in ("always", "first", "never"):
            raise UnsupportedTokenizer(f"Metaspace prepend_scheme={scheme!r}")
        out = []
        for index, token in enumerate(tokens):
            if index == 0 and scheme != "never":
                token = "".join("" if c == replacement else c for c in token)
            else:
                token = token.replace(replacement, " ")
            out.append(token)
        return out
    if kind == "WordPiece":
        prefix = spec.get("prefix", "##")
        cleanup = spec.get("cleanup", True)
        out = []
        for index, token in enumerate(tokens):
            if index != 0:
                if token.startswith(prefix):
                    token = token.replace(prefix, "", 1)
                else:
                    token = " " + token
            out.append(_wordpiece_cleanup(token) if cleanup else token)
        return out
    if kind == "Replace":
        pattern = spec.get("pattern", {})
        content = spec.get("content", "")
        if "String" in pattern:
            return [t.replace(pattern["String"], content) for t in tokens]
        if "Regex" in pattern:
            # Refused rather than approximated. The reference implementation
            # compiles the pattern with the Rust regex crate and inserts the
            # replacement literally, so Python's `re` differs twice over: it
            # expands backreferences in the replacement, and it rejects (or
            # worse, reinterprets) Rust-only syntax such as \p{L} and POSIX
            # classes. Both produce plausible text that is quietly wrong,
            # which nobody notices in synthesized speech. No shipped
            # vocabulary uses a Regex Replace decoder.
            raise UnsupportedTokenizer(
                f"a Replace decoder with a regular-expression pattern "
                f"({pattern['Regex']!r}) is not implemented by the built-in "
                f"tokenizer.json reader; install the `tokenizers` package "
                f"(`pip install tokenizers`) to decode with this vocabulary")
        raise UnsupportedTokenizer(f"Replace pattern {pattern!r}")
    if kind == "Strip":
        content = spec.get("content", " ")
        start = int(spec.get("start", 0) or 0)
        stop = int(spec.get("stop", 0) or 0)
        out = []
        for token in tokens:
            begin = 0
            while begin < start and begin < len(token) and token[begin] == content:
                begin += 1
            end = 0
            while (end < stop and end < len(token) - begin
                   and token[len(token) - 1 - end] == content):
                end += 1
            out.append(token[begin:len(token) - end])
        return out
    if kind == "Fuse":
        return ["".join(tokens)]
    if kind == "ByteFallback":
        out: List[str] = []
        pending = bytearray()

        def flush() -> None:
            if not pending:
                return
            try:
                out.append(pending.decode("utf-8"))
            except UnicodeDecodeError:
                out.extend("�" * len(pending))
            pending.clear()

        for token in tokens:
            byte = None
            if len(token) == 6 and token.startswith("<0x") and token.endswith(">"):
                try:
                    byte = int(token[3:5], 16)
                except ValueError:
                    byte = None
            if byte is None:
                flush()
                out.append(token)
            else:
                pending.append(byte)
        flush()
        return out
    raise UnsupportedTokenizer(
        f"decoder {kind!r} is not implemented by the built-in tokenizer.json "
        f"reader; install the `tokenizers` package (`pip install tokenizers`) "
        f"to decode with this vocabulary")


class Encoding:
    """The subset of the ``tokenizers`` Encoding that phoonnx reads."""

    def __init__(self, ids: List[int], tokens: List[str]):
        self.ids = ids
        self.tokens = tokens


class Tokenizer:
    """A ``tokenizer.json`` reader with the calls phoonnx makes."""

    def __init__(self, spec: dict, source: str = "tokenizer.json"):
        self._source = source
        model = spec.get("model") or {}
        if model.get("type") != "BPE":
            raise UnsupportedTokenizer(
                f"{self._source}: this reader implements BPE, and the file "
                f"declares a {model.get('type')!r} model. Install the "
                f"`tokenizers` package to read it — for Indic Parler, that is "
                f"`pip install phoonnx[indic-parler]`.")
        _check_bpe_options(model)
        self._normalizer = spec.get("normalizer")
        self._pre_tokenizer = spec.get("pre_tokenizer")
        self._model_vocab: Dict[str, int] = dict(model.get("vocab") or {})
        self._vocab: Dict[str, int] = dict(self._model_vocab)
        self._added = {t["content"]: t["id"] for t in spec.get("added_tokens", [])}
        self._special = {t["content"] for t in spec.get("added_tokens", [])
                         if t.get("special")}
        self._vocab.update(self._added)
        self._decoder = spec.get("decoder")
        # Checked here rather than at the first decode. The caller falls back
        # to the `tokenizers` package when this reader refuses a file, and
        # that fallback only exists around loading — a decoder that raised
        # lazily would load fine, encode fine, and then fail in the middle of
        # synthesis on a machine where `tokenizers` could have decoded it.
        _decode_chain([""], self._decoder)
        self._by_id: Optional[Dict[int, str]] = None
        self._bpe = _BPE(self._vocab, model.get("merges") or [],
                         model.get("unk_token"),
                         model.get("continuing_subword_prefix") or "",
                         bool(model.get("ignore_merges")))
        self._post = spec.get("post_processor")
        # Added tokens are matched before anything else touches the text, so a
        # special token is never split by the pre-tokenizer.
        self._added_re = (re.compile("(" + "|".join(
            sorted((re.escape(t) for t in self._added), key=len, reverse=True)) + ")")
            if self._added else None)

    @classmethod
    def from_file(cls, path: str) -> "Tokenizer":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(json.load(handle), source=str(path))

    def get_vocab(self, with_added_tokens: bool = True) -> Dict[str, int]:
        if with_added_tokens:
            return dict(self._vocab)
        return dict(self._model_vocab)

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        return len(self.get_vocab(with_added_tokens))

    def token_to_id(self, token: str) -> Optional[int]:
        return self._vocab.get(token)

    def id_to_token(self, index: int) -> Optional[str]:
        if self._by_id is None:
            # First name wins, which is the order a scan of the vocab found.
            by_id: Dict[int, str] = {}
            for token, value in self._vocab.items():
                by_id.setdefault(value, token)
            self._by_id = by_id
        return self._by_id.get(index)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Turn ids back into text, the way ``tokenizers`` does it.

        An id outside the vocabulary has no token to contribute, so it is
        dropped rather than guessed at. The tokens then run through whatever
        the file declares in its ``decoder`` section; a decoder this reader
        does not implement raises :class:`UnsupportedTokenizer` instead of
        returning text that would be subtly wrong.
        """
        tokens: List[str] = []
        for index in ids:
            token = self.id_to_token(int(index))
            if token is None:
                continue
            if skip_special_tokens and token in self._special:
                continue
            tokens.append(token)
        if not self._decoder:
            # No decoder declared: the reference implementation joins on a space.
            return " ".join(tokens)
        return "".join(_decode_chain(tokens, self._decoder))

    def encode(self, text: str, add_special_tokens: bool = True) -> Encoding:
        ids: List[int] = []
        tokens: List[str] = []
        for chunk in (self._added_re.split(text) if self._added_re else [text]):
            if not chunk:
                continue
            if chunk in self._added:
                ids.append(self._added[chunk])
                tokens.append(chunk)
                continue
            normalized = _normalize(chunk, self._normalizer)
            # Match added tokens again: a normalizer that rewrites spaces to
            # "[SPACE]" creates them, and the pre-tokenizer would otherwise
            # split that into "[", "SPACE", "]".
            for part in (self._added_re.split(normalized)
                         if self._added_re else [normalized]):
                if not part:
                    continue
                if part in self._added:
                    ids.append(self._added[part])
                    tokens.append(part)
                    continue
                for piece in _pre_tokenize(part, self._pre_tokenizer):
                    if not piece:
                        continue
                    if piece in self._added:
                        ids.append(self._added[piece])
                        tokens.append(piece)
                        continue
                    merged = self._bpe.tokenize(piece)
                    tokens.extend(merged)
                    ids.extend(self._bpe.to_ids(merged))
        if not add_special_tokens:
            return Encoding(ids, tokens)
        return self._apply_post(ids, tokens)

    def _apply_post(self, ids: List[int], tokens: List[str]) -> Encoding:
        """Wrap the sequence in whatever special tokens the post-processor adds."""
        return _post_process(Encoding(ids, tokens), self._post)


def _post_process(encoding: "Encoding", spec: Optional[dict]) -> "Encoding":
    if not spec:
        return encoding
    kind = spec.get("type")
    if kind == "ByteLevel":
        # ByteLevel post-processing only adjusts offsets, never the ids.
        return encoding
    if kind == "Sequence":
        for step in spec.get("processors") or []:
            encoding = _post_process(encoding, step)
        return encoding
    if kind != "TemplateProcessing":
        raise UnsupportedTokenizer(f"post_processor {kind!r}")
    specials = spec.get("special_tokens") or {}
    out_ids: List[int] = []
    out_tokens: List[str] = []
    for item in spec.get("single") or []:
        if "Sequence" in item:
            out_ids.extend(encoding.ids)
            out_tokens.extend(encoding.tokens)
        elif "SpecialToken" in item:
            entry = specials.get(item["SpecialToken"]["id"]) or {}
            out_ids.extend(entry.get("ids") or [])
            out_tokens.extend(entry.get("tokens") or [])
    return Encoding(out_ids, out_tokens)
