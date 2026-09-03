"""Dataset + collation for MOSS-TTS-Nano finetuning.

Input is a JSONL produced by :mod:`phoonnx_train.mosstts.prepare_data`, one record per
utterance::

    {"audio": "wavs/0001.wav",          # informational after preprocessing
     "text": "Bom dia.",
     "language": "pt",                  # optional
     "ref_audio": "wavs/ref.wav",       # optional
     "audio_codes": [[c0, ..., c15], ...],       # [frames, n_vq], REQUIRED
     "ref_audio_codes": [[c0, ..., c15], ...]}   # optional

Audio never touches this module — the codec is frozen and runs once, offline. Each
record is packed into a ``[T, n_vq + 1]`` row matrix:

* text rows put a SentencePiece id in column 0 and ``audio_pad_token_id`` in the rest,
* audio rows put a *slot* token in column 0 (user slot for the reference clip, assistant
  slot for the target) and the frame's RVQ codes in columns ``1..n_vq``.

Labels are the row matrix shifted left by one, with everything before the assistant turn
masked to ``-100``, so the loss only sees the audio the model is asked to generate (plus
the ``audio_end`` token that terminates it).

There is no delay pattern: MOSS-TTS-Nano predicts a whole frame per timestep through its
local transformer, so channels stay time-aligned (see :mod:`phoonnx_train.mosstts.model`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset

from phoonnx_train.mosstts.config import MossTTSNanoConfig

# The chat template MOSS-TTS-Nano was pretrained with (upstream ``prompting.py``).
USER_ROLE_PREFIX = "user\n"
USER_TEMPLATE_REFERENCE_PREFIX = "<user_inst>\n- Reference(s):\n"
USER_TEMPLATE_SUFFIX = "\n</user_inst>"
ASSISTANT_TURN_PREFIX = "\n"
ASSISTANT_ROLE_PREFIX = "assistant\n"

#: Optional metadata lines, in the order the template expects them.
OPTIONAL_MESSAGE_FIELDS = (
    ("instruction", "Instruction"),
    ("tokens", "Tokens"),
    ("quality", "Quality"),
    ("sound_event", "Sound Event"),
    ("ambient_sound", "Ambient Sound"),
    ("language", "Language"),
)

IGNORE_INDEX = -100


class SentencePieceTextTokenizer:
    """Thin wrapper over the checkpoint's ``tokenizer.model``.

    Upstream wraps the same SentencePiece model in a ``PreTrainedTokenizer`` subclass;
    nothing in the training path needs that, and avoiding it keeps ``transformers`` out
    of phoonnx's dependency set.
    """

    def __init__(self, model_file: Union[str, Path]) -> None:
        import sentencepiece as spm

        self.model_file = str(model_file)
        self.sp = spm.SentencePieceProcessor(model_file=self.model_file)

    def encode(self, text: str) -> List[int]:
        return [int(token) for token in self.sp.encode(text, out_type=int)]

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(self.sp.decode([int(token) for token in token_ids]))

    @property
    def vocab_size(self) -> int:
        return int(self.sp.get_piece_size())


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
    return records


def dump_jsonl(records: Iterable[Dict[str, Any]], path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    return path


def _as_code_matrix(value: Any, field_name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.ndim != 2:
        raise ValueError(f"`{field_name}` must have shape [frames, n_vq], got {tuple(tensor.shape)}")
    return tensor.cpu().contiguous()


class MossTTSNanoDataset(Dataset):
    """Pre-encoded JSONL -> packed rows, ready for :class:`MossTTSNanoCollator`."""

    def __init__(
        self,
        records: Iterable[Dict[str, Any]],
        tokenizer: SentencePieceTextTokenizer,
        config: MossTTSNanoConfig,
        max_length: int = 1024,
        prompt_style: str = "inference",
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = int(max_length)
        if prompt_style not in {"inference", "finetuning"}:
            raise ValueError(f"prompt_style must be 'inference' or 'finetuning', got {prompt_style!r}")
        self.prompt_style = prompt_style
        if self.max_length < 8:
            raise ValueError("max_length must be >= 8")
        if not self.records:
            raise ValueError("dataset is empty")

    @classmethod
    def from_jsonl(
        cls,
        paths: Union[str, Path, Sequence[Union[str, Path]]],
        tokenizer: SentencePieceTextTokenizer,
        config: MossTTSNanoConfig,
        max_length: int = 1024,
        prompt_style: str = "inference",
    ) -> "MossTTSNanoDataset":
        if isinstance(paths, (str, Path)):
            paths = [paths]
        records: List[Dict[str, Any]] = []
        for path in paths:
            records.extend(load_jsonl(path))
        return cls(
            records,
            tokenizer=tokenizer,
            config=config,
            max_length=max_length,
            prompt_style=prompt_style,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.build_example(self.records[index], index=index)

    # ------------------------------------------------------------------
    # row builders
    # ------------------------------------------------------------------
    def text_rows(self, token_ids: Sequence[int]) -> torch.Tensor:
        rows = torch.full(
            (len(token_ids), self.config.row_width),
            int(self.config.audio_pad_token_id),
            dtype=torch.long,
        )
        if len(token_ids):
            rows[:, 0] = torch.as_tensor(list(token_ids), dtype=torch.long)
        return rows

    def audio_rows(self, codes: torch.Tensor, slot_token_id: int) -> torch.Tensor:
        rows = torch.full(
            (int(codes.shape[0]), self.config.row_width),
            int(self.config.audio_pad_token_id),
            dtype=torch.long,
        )
        if rows.shape[0]:
            rows[:, 0] = int(slot_token_id)
            rows[:, 1:] = codes
        return rows

    def _pad_codes_to_width(self, codes: torch.Tensor, field_name: str, index: int) -> torch.Tensor:
        target = self.config.n_vq
        source = int(codes.shape[1])
        if source > target:
            raise ValueError(
                f"record {index}: `{field_name}` has n_vq={source}, model expects at most {target}"
            )
        if source == target:
            return codes
        padded = torch.full(
            (int(codes.shape[0]), target), int(self.config.audio_pad_token_id), dtype=torch.long
        )
        if source:
            padded[:, :source] = codes
        return padded

    def _metadata_suffix(self, record: Dict[str, Any]) -> str:
        lines = [""]
        for field_name, display_name in OPTIONAL_MESSAGE_FIELDS:
            value = record.get(field_name)
            lines.append(f"- {display_name}:")
            lines.append("None" if value in (None, "") else str(value))
        lines.append("- Text:")
        lines.append(str(record["text"]))
        return "\n".join(lines)

    def build_prompt_rows(
        self,
        record: Dict[str, Any],
        reference_codes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Everything before the target audio: instruction, optional reference, text.

        .. note::

           This follows the checkpoint's own ``prompting.py`` — including the
           ``im_end`` token that closes the user turn. Upstream's ``finetuning/dataset.py``
           drops that token, which silently desynchronises finetunes from the prompt the
           inference path (and :class:`phoonnx.engines.mosstts.MossTTSNanoAdapter`)
           actually builds. Pass ``prompt_style="finetuning"`` to reproduce the upstream
           finetuning variant instead.
        """
        encode = self.tokenizer.encode
        cfg = self.config
        sections = [
            self.text_rows(
                [cfg.im_start_token_id]
                + encode(USER_ROLE_PREFIX)
                + encode(USER_TEMPLATE_REFERENCE_PREFIX)
            )
        ]
        suffix_text = self._metadata_suffix(record)

        if reference_codes is None:
            sections.append(self.text_rows(encode("None" + suffix_text)))
        else:
            sections.append(self.text_rows([cfg.audio_start_token_id]))
            sections.append(self.audio_rows(reference_codes, cfg.audio_user_slot_token_id))
            sections.append(
                self.text_rows([cfg.audio_end_token_id] + encode(suffix_text))
            )

        assistant_prefix = encode(USER_TEMPLATE_SUFFIX)
        if self.prompt_style == "inference":
            assistant_prefix = assistant_prefix + [cfg.im_end_token_id]
        assistant_prefix = (
            assistant_prefix
            + encode(ASSISTANT_TURN_PREFIX)
            + [cfg.im_start_token_id]
            + encode(ASSISTANT_ROLE_PREFIX)
            + [cfg.audio_start_token_id]
        )
        sections.append(self.text_rows(assistant_prefix))
        return torch.cat(sections, dim=0)

    # ------------------------------------------------------------------
    def build_example(self, record: Dict[str, Any], index: int = 0) -> Dict[str, torch.Tensor]:
        if "text" not in record or not str(record["text"]).strip():
            raise ValueError(f"record {index} has no non-empty `text`")
        if record.get("audio_codes") is None:
            raise ValueError(f"record {index} has no `audio_codes` — run prepare_data.py first")

        target_codes = self._pad_codes_to_width(
            _as_code_matrix(record["audio_codes"], "audio_codes"), "audio_codes", index
        )
        if target_codes.shape[0] == 0:
            raise ValueError(f"record {index} has zero audio frames")

        reference_codes = None
        if record.get("ref_audio_codes") is not None:
            reference_codes = self._pad_codes_to_width(
                _as_code_matrix(record["ref_audio_codes"], "ref_audio_codes"), "ref_audio_codes", index
            )
        elif record.get("ref_audio") is not None:
            raise ValueError(
                f"record {index} has `ref_audio` but no `ref_audio_codes` — run prepare_data.py "
                "first so training never touches the codec"
            )

        prompt_rows = self.build_prompt_rows(record, reference_codes)
        target_rows = self.audio_rows(target_codes, self.config.audio_assistant_slot_token_id)
        end_rows = self.text_rows([self.config.audio_end_token_id])
        rows = torch.cat([prompt_rows, target_rows, end_rows], dim=0)

        prompt_length = int(prompt_rows.shape[0])
        if prompt_length >= self.max_length:
            raise ValueError(
                f"record {index}: prompt length {prompt_length} >= max_length {self.max_length}; "
                "raise --max-length or shorten the text / reference clip"
            )
        if rows.shape[0] > self.max_length:
            rows = rows[: self.max_length]
        if rows.shape[0] < 2:
            raise ValueError(f"record {index}: packed sequence is too short ({rows.shape[0]})")

        return {
            "rows": rows,
            "seq_len": torch.tensor(int(rows.shape[0]), dtype=torch.long),
            "prompt_length": torch.tensor(prompt_length, dtype=torch.long),
        }


class MossTTSNanoCollator:
    """Pad a batch of packed row matrices and build shifted, masked labels.

    Sequences are padded to the batch maximum rather than to ``max_length``; the
    attention mask makes this exactly equivalent to upstream's fixed-length padding while
    avoiding the wasted compute on short batches.
    """

    def __init__(self, config: MossTTSNanoConfig) -> None:
        self.config = config

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("empty batch")
        batch_size = len(batch)
        total_length = max(int(item["seq_len"]) for item in batch)
        if total_length < 2:
            raise ValueError("every sequence in the batch is shorter than 2 rows")
        width = self.config.row_width

        rows = torch.full((batch_size, total_length, width), int(self.config.audio_pad_token_id), dtype=torch.long)
        rows[:, :, 0] = int(self.config.pad_token_id)
        mask = torch.zeros((batch_size, total_length), dtype=torch.bool)
        loss_mask = torch.zeros((batch_size, total_length - 1), dtype=torch.bool)

        for batch_index, item in enumerate(batch):
            seq_len = int(item["seq_len"])
            prompt_length = int(item["prompt_length"])
            rows[batch_index, :seq_len, :] = item["rows"]
            mask[batch_index, :seq_len] = True
            # the row at prompt_length - 1 is the one that *predicts* the first audio frame
            loss_mask[batch_index, prompt_length - 1 : seq_len - 1] = True

        labels = rows[:, 1:, :].clone()
        labels = labels.masked_fill(~loss_mask.unsqueeze(-1), IGNORE_INDEX)
        labels = labels.masked_fill(~mask[:, 1:].unsqueeze(-1), IGNORE_INDEX)
        # padded audio columns of text rows are not targets
        labels[:, :, 1:] = labels[:, :, 1:].masked_fill(
            labels[:, :, 1:] == int(self.config.audio_pad_token_id), IGNORE_INDEX
        )

        return {
            "input_ids": rows[:, :-1, :].contiguous(),
            "attention_mask": mask[:, :-1].contiguous(),
            "labels": labels.contiguous(),
        }


__all__ = [
    "MossTTSNanoDataset",
    "MossTTSNanoCollator",
    "SentencePieceTextTokenizer",
    "load_jsonl",
    "dump_jsonl",
    "IGNORE_INDEX",
]
