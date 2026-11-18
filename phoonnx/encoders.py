"""The step between the text frontend and the engine: text or symbols to the ids a model reads.

An :class:`Encoder` does not run a neural network. A graph named ``encoder.onnx`` is part of the
model and stays in the adapter. Two encoders cover the engines of the package:

* :class:`SymbolEncoder` reads the symbol chunks of the frontend and gives them to the voice's
  vocabulary tokenizer.
* :class:`TextEncoder` gives the staged text of each chunk to the engine's own tokenizer, such as a
  BPE with a language prefix.

``adapter.make_encoder`` builds the encoder of a voice.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple

from scriptconv.frontend import Stages, TextFrontend

from phoonnx.config import SynthesisConfig

#: The request fields that hold one value per phoneme id, as a phonemizer's side streams fill them.
SIDE_STREAMS = ("language_ids", "stress")


@dataclass(frozen=True)
class Utterance:
    """The model input for one engine call, from :meth:`TTSVoice.prepare`."""

    text: Optional[str]
    """The text of this engine call's chunk, from ``Segment.text``. ``""`` when
    the previous utterance's text covers this chunk, and None when it is
    unknown."""

    symbols: Optional[Tuple[str, ...]]
    """The phonemes or characters that the tokenizer read. None for an engine
    that reads text, whose tokenizer turns the text into ids itself."""

    ids: Tuple[int, ...]
    """The token ids that the engine reads."""

    streams: Mapping[str, Sequence[int]] = field(default_factory=dict)
    """Values a phonemizer gives beside its symbols, one per id, keyed by the
    ``AdapterSynthesisRequest`` field they fill, such as ``language_ids``."""


class Encoder(Protocol):
    def encode(self, text: str, config: SynthesisConfig) -> Iterator[Utterance]:
        """The utterances of *text*, one per engine call. An utterance with no ids is not yielded."""

    def encode_symbols(self, symbols: Sequence[str], config: SynthesisConfig) -> Utterance:
        """The utterance of symbols that are already phonemic, with no frontend pass."""


def stages_of(config: SynthesisConfig) -> Stages:
    """The text stages that *config* sets. A value that *config* leaves unset takes the frontend's own."""
    return Stages(diacritize=config.add_diacritics, diacritizer_model=config.diacritizer_model,
                  chunk=config.chunk)


@dataclass(frozen=True)
class SymbolEncoder:
    """Frontend symbol chunks, tokenized by the voice's vocabulary tokenizer."""

    frontend: TextFrontend
    tokenizer: Any

    def encode(self, text: str, config: SynthesisConfig) -> Iterator[Utterance]:
        for segment in self.frontend.segments(text, stages_of(config)):
            if not segment.phonemes:
                continue
            ids = self.tokenizer.tokenize(list(segment.phonemes))
            if ids:
                streams = {name: list(values) for name, values in segment.streams.items()
                           if name in SIDE_STREAMS}
                yield Utterance(text=segment.text, symbols=tuple(segment.phonemes), ids=tuple(ids),
                                streams=streams)

    def encode_symbols(self, symbols: Sequence[str], config: SynthesisConfig) -> Utterance:
        return Utterance(text=None, symbols=tuple(symbols), ids=tuple(self.tokenizer.tokenize(list(symbols))))


@dataclass(frozen=True)
class TextEncoder:
    """The staged text of each frontend chunk, tokenized by the engine's own tokenizer.

    ``tokenize(text, config)`` returns one id list per engine call that the text needs.

    With ``whole_text`` the encoder gives ``tokenize`` the whole staged text in one call, as the
    ``encode_text`` of an adapter expects, and that adapter cuts the text itself.
    """

    frontend: TextFrontend
    tokenize: Callable[[str, SynthesisConfig], List[List[int]]]
    whole_text: bool = False

    def encode(self, text: str, config: SynthesisConfig) -> Iterator[Utterance]:
        stages = stages_of(config)
        if self.whole_text:
            prepared = self.frontend.prepare(text, stages)
            id_lists = self.tokenize(prepared, config)
            chunk_text = prepared if len(id_lists) == 1 else None
            for ids in id_lists:
                if ids:
                    yield Utterance(text=chunk_text, symbols=None, ids=tuple(ids))
            return
        for segment in self.frontend.segments(text, stages):
            for ids in self.tokenize(segment.text, config):
                if ids:
                    yield Utterance(text=segment.text, symbols=None, ids=tuple(ids))

    def encode_symbols(self, symbols: Sequence[str], config: SynthesisConfig) -> Utterance:
        """A text engine reads symbols as the text they spell."""
        text = "".join(symbols)
        ids = [i for chunk in self.tokenize(text, config) for i in chunk]
        return Utterance(text=text, symbols=tuple(symbols), ids=tuple(ids))
