"""In-context (infilling) pair construction for ZipVoice-style training.

ZipVoice has no speaker encoder: cloning is *in-context* — the reference
audio + its transcription are prepended to the target and the model learns
to continue that voice (see ``phoonnx/engines/zipvoice.py`` for the runtime
side of the same convention: ``prompt_tokens`` + ``prompt_features_len``
feeding the text encoder, ``speech_condition`` holding the prompt mel).

Training therefore does not consume utterances one at a time; it consumes
**(reference, target) pairs**. This module builds those pairs from the
utterance list phoonnx_train's ``preprocess.py`` already produces (text +
phoneme ids + normalized audio path), with no dependency on the actual
Zipformer backbone, so it is independently testable.

Two pairing strategies, mirroring how upstream ZipVoice builds its infilling
targets:

- **split**: cut a single utterance into a reference prefix and a target
  suffix (frame-based ratio). Every utterance long enough on its own yields
  one pair.
- **cross**: pair consecutive utterances from the *same speaker* — one as
  reference, the next as target. Needs multi-utterance speakers but gives
  the model reference/target pairs with genuinely different content, closer
  to inference-time cloning.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class UtteranceInfo:
    """Minimal utterance fields needed to build in-context pairs.

    Deliberately narrower than ``phoonnx_train.preprocess.Utterance`` — only
    what pairing needs — so this module has no import-time dependency on the
    preprocessing pipeline or torch.
    """

    utt_id: str
    tokens: Sequence[int]
    num_frames: int
    speaker: Optional[str] = None


@dataclass
class InContextPair:
    """A (reference, target) training pair for the CFM objective."""

    ref_utt_id: str
    ref_tokens: Sequence[int]
    ref_frames: int
    target_utt_id: str
    target_tokens: Sequence[int]
    target_frames: int
    speaker: Optional[str] = None
    strategy: str = "split"


def _split_pair(u: UtteranceInfo, ref_frac: float) -> Optional[InContextPair]:
    ref_frames = int(round(u.num_frames * ref_frac))
    ref_frames = max(1, min(ref_frames, u.num_frames - 1))
    target_frames = u.num_frames - ref_frames
    if ref_frames < 1 or target_frames < 1:
        return None
    # token split proportional to the frame split — tokens have no explicit
    # frame alignment here, so this is an approximation preprocessing
    # accepts as "close enough" for reference chunking (matches how upstream
    # ZipVoice treats unaligned text/audio boundaries for infilling).
    n_tok = len(u.tokens)
    tok_split = max(1, min(n_tok - 1, int(round(n_tok * ref_frac)))) if n_tok > 1 else n_tok
    return InContextPair(
        ref_utt_id=u.utt_id,
        ref_tokens=list(u.tokens[:tok_split]),
        ref_frames=ref_frames,
        target_utt_id=u.utt_id,
        target_tokens=list(u.tokens[tok_split:]) or list(u.tokens),
        target_frames=target_frames,
        speaker=u.speaker,
        strategy="split",
    )


def build_in_context_pairs(
    utterances: Sequence[UtteranceInfo],
    strategy: str = "split",
    ref_frac: float = 0.3,
    min_frames: int = 2,
) -> List[InContextPair]:
    """Build (reference, target) pairs from a list of utterances.

    Args:
        utterances: utterances to pair, in dataset order.
        strategy: ``"split"`` (single-utterance ref/target cut) or
            ``"cross"`` (consecutive same-speaker utterances).
        ref_frac: for ``"split"``, the fraction of frames used as the
            reference prefix. Must be in ``(0, 1)``.
        min_frames: utterances shorter than this (in frames) are skipped —
            too short to yield a non-empty reference and target.

    Returns:
        A list of :class:`InContextPair`. Utterances that cannot form a
        valid pair (too short, or no speaker-mate for ``"cross"``) are
        silently skipped, never raise.

    Raises:
        ValueError: for an unknown ``strategy`` or an ``ref_frac`` outside
            ``(0, 1)``.
    """
    if strategy not in ("split", "cross"):
        raise ValueError(f"unknown strategy {strategy!r}, expected 'split' or 'cross'")
    if not (0.0 < ref_frac < 1.0):
        raise ValueError(f"ref_frac must be in (0, 1), got {ref_frac}")

    usable = [u for u in utterances if u.num_frames >= min_frames]

    if strategy == "split":
        pairs = []
        for u in usable:
            pair = _split_pair(u, ref_frac)
            if pair is not None:
                pairs.append(pair)
        return pairs

    # strategy == "cross": pair consecutive utterances of the same speaker
    by_speaker: dict = {}
    for u in usable:
        by_speaker.setdefault(u.speaker, []).append(u)

    pairs = []
    for _, utts in by_speaker.items():
        for a, b in zip(utts, utts[1:]):
            pairs.append(InContextPair(
                ref_utt_id=a.utt_id, ref_tokens=list(a.tokens), ref_frames=a.num_frames,
                target_utt_id=b.utt_id, target_tokens=list(b.tokens), target_frames=b.num_frames,
                speaker=a.speaker, strategy="cross",
            ))
    return pairs
