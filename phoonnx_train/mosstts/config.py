"""Configuration classes for the vendored MOSS-TTS-Nano trainer.

These mirror the fields of the upstream ``config.json`` (``model_type:
moss_tts_nano``) so :mod:`phoonnx_train.mosstts.warmstart` can load an upstream
checkpoint without ``transformers`` or ``trust_remote_code``. Only the fields the
vendored model actually reads are kept; everything else in the upstream JSON is
``transformers`` boilerplate.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# The GPT-2 knobs the vendored decoder honours. Anything else in the upstream
# ``gpt2_config`` block is transformers plumbing and is dropped on load.
_GPT2_FIELDS = (
    "vocab_size",
    "n_positions",
    "n_embd",
    "n_layer",
    "n_head",
    "n_inner",
    "activation_function",
    "resid_pdrop",
    "embd_pdrop",
    "attn_pdrop",
    "layer_norm_epsilon",
    "initializer_range",
    "scale_attn_weights",
    "scale_attn_by_inverse_layer_idx",
    "position_embedding_type",
    "rope_base",
    "pad_token_id",
)


@dataclass
class GPT2DecoderConfig:
    """GPT-2 decoder geometry. MOSS-TTS-Nano uses RoPE instead of learned positions."""

    vocab_size: int = 16384
    n_positions: int = 32768
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: Optional[int] = 3072
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    scale_attn_weights: bool = True
    scale_attn_by_inverse_layer_idx: bool = False
    position_embedding_type: str = "rope"
    rope_base: float = 10000.0
    pad_token_id: int = 3

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd={self.n_embd} must be divisible by n_head={self.n_head}")
        if self.position_embedding_type not in {"absolute", "rope"}:
            raise ValueError(f"unsupported position_embedding_type={self.position_embedding_type!r}")
        if self.activation_function not in {"gelu_new", "gelu", "relu", "silu"}:
            raise ValueError(f"unsupported activation_function={self.activation_function!r}")

    @property
    def hidden_size(self) -> int:
        return int(self.n_embd)

    @property
    def num_attention_heads(self) -> int:
        return int(self.n_head)

    @property
    def head_dim(self) -> int:
        return int(self.n_embd) // int(self.n_head)

    @property
    def inner_size(self) -> int:
        return int(self.n_inner or 4 * self.n_embd)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GPT2DecoderConfig":
        return cls(**{k: v for k, v in payload.items() if k in _GPT2_FIELDS})


@dataclass
class MossTTSNanoConfig:
    """Full MOSS-TTS-Nano configuration (global backbone + local transformer).

    ``audio_pad_token_id`` deliberately sits *outside* every codebook: a text row
    carries it in all ``n_vq`` audio columns and the model masks those embeddings out.
    """

    gpt2: GPT2DecoderConfig = field(default_factory=GPT2DecoderConfig)
    n_vq: int = 16
    audio_codebook_sizes: List[int] = field(default_factory=lambda: [1024] * 16)
    audio_pad_token_id: int = 1024
    pad_token_id: int = 3
    im_start_token_id: int = 4
    im_end_token_id: int = 5
    audio_start_token_id: int = 6
    audio_end_token_id: int = 7
    audio_user_slot_token_id: int = 8
    audio_assistant_slot_token_id: int = 9
    local_transformer_layers: int = 1
    initializer_range: float = 0.02
    audio_tokenizer_sample_rate: int = 48000
    audio_tokenizer_downsample_rate: int = 3840
    audio_tokenizer_channels: int = 2
    attn_implementation: str = "sdpa"
    local_transformer_attn_implementation: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.gpt2, dict):
            self.gpt2 = GPT2DecoderConfig.from_dict(self.gpt2)
        self.n_vq = int(self.n_vq)
        if self.n_vq <= 0:
            raise ValueError("n_vq must be > 0")
        self.audio_codebook_sizes = [int(size) for size in self.audio_codebook_sizes]
        if len(self.audio_codebook_sizes) != self.n_vq:
            raise ValueError(
                f"audio_codebook_sizes must have length n_vq={self.n_vq}, "
                f"got {len(self.audio_codebook_sizes)}"
            )
        if any(size <= 0 for size in self.audio_codebook_sizes):
            raise ValueError("audio_codebook_sizes must be positive")
        if self.audio_pad_token_id < max(self.audio_codebook_sizes):
            raise ValueError(
                "audio_pad_token_id must be >= max(audio_codebook_sizes) so pad stays "
                f"outside every codebook (got {self.audio_pad_token_id})"
            )
        if self.local_transformer_layers <= 0:
            raise ValueError("local_transformer_layers must be > 0")
        if self.local_transformer_attn_implementation is None:
            self.local_transformer_attn_implementation = self.attn_implementation

    # ------------------------------------------------------------------
    @property
    def hidden_size(self) -> int:
        return self.gpt2.hidden_size

    @property
    def row_width(self) -> int:
        """Width of one input row: the text/slot column plus ``n_vq`` codebook columns."""
        return self.n_vq + 1

    @property
    def frames_per_second(self) -> float:
        return self.audio_tokenizer_sample_rate / float(self.audio_tokenizer_downsample_rate)

    def local_gpt2(self) -> GPT2DecoderConfig:
        """Geometry of the local transformer: same width, ``local_transformer_layers`` deep."""
        payload = asdict(self.gpt2)
        payload["n_layer"] = int(self.local_transformer_layers)
        payload["n_positions"] = self.row_width
        return GPT2DecoderConfig.from_dict(payload)

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["gpt2"] = asdict(self.gpt2)
        return payload

    def save_json(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MossTTSNanoConfig":
        payload = dict(payload)
        gpt2_payload = payload.pop("gpt2", None) or payload.pop("gpt2_config", None) or {}
        known = {f for f in cls.__dataclass_fields__ if f != "gpt2"}
        kwargs = {k: v for k, v in payload.items() if k in known}
        return cls(gpt2=GPT2DecoderConfig.from_dict(gpt2_payload), **kwargs)

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> "MossTTSNanoConfig":
        """Load either a vendored config or an upstream HF ``config.json``."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model_type = payload.get("model_type")
        if model_type is not None and model_type != "moss_tts_nano":
            raise ValueError(f"expected model_type='moss_tts_nano', got {model_type!r}")
        architecture = payload.get("model_architecture")
        if architecture is not None and architecture != "global_local_transformer":
            raise ValueError(
                "this trainer only implements the global-local (RQ-Transformer) scheme, "
                f"but the checkpoint declares model_architecture={architecture!r}"
            )
        return cls.from_dict(payload)

    @classmethod
    def from_pretrained_dir(cls, path: Union[str, Path]) -> "MossTTSNanoConfig":
        return cls.from_json_file(Path(path) / "config.json")

    # ------------------------------------------------------------------
    @staticmethod
    def tiny(n_vq: int = 4, codebook_size: int = 32, vocab_size: int = 64) -> "MossTTSNanoConfig":
        """A few-thousand-parameter config for tests and smoke runs."""
        return MossTTSNanoConfig(
            gpt2=GPT2DecoderConfig(
                vocab_size=vocab_size,
                n_positions=256,
                n_embd=16,
                n_layer=2,
                n_head=2,
                n_inner=32,
            ),
            n_vq=n_vq,
            audio_codebook_sizes=[codebook_size] * n_vq,
            audio_pad_token_id=codebook_size,
            local_transformer_layers=1,
            attn_implementation="eager",
        )


@dataclass
class TrainingConfig:
    """Optimizer / scheduler defaults, taken from upstream ``finetuning/sft.py``."""

    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    warmup_steps: int = 0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0
    #: ``"1,32"`` in upstream shorthand: text head weight 1, total audio weight 32 split
    #: evenly across the ``n_vq`` audio heads.
    channelwise_loss_weight: str = "1,32"
    max_length: int = 1024
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_workers: int = 0
    seed: int = 42


__all__ = ["GPT2DecoderConfig", "MossTTSNanoConfig", "TrainingConfig"]
