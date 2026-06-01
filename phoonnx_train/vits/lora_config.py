from dataclasses import dataclass
from typing import Tuple


SCOPE_PRESETS = {
    "generator-only": {
        "rank": 4,
        "alpha": 8.0,
        "target_modules": ("dec",),
    },
    "full-acoustic": {
        "rank": 8,
        "alpha": 16.0,
        "target_modules": ("dec", "enc_q", "flow", "dp"),
    },
    "aggressive": {
        "rank": 16,
        "alpha": 32.0,
        "target_modules": ("dec", "enc_q", "flow", "dp", "enc_p"),
    },
}

VALID_TARGET_MODULES = ("dec", "enc_q", "flow", "dp", "enc_p")


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: Tuple[str, ...] = ("dec", "enc_q", "flow", "dp")

    @staticmethod
    def from_preset(preset: str) -> "LoRAConfig":
        if preset not in SCOPE_PRESETS:
            raise ValueError(
                f"Unknown LoRA scope preset '{preset}'. "
                f"Available: {', '.join(SCOPE_PRESETS.keys())}"
            )
        p = SCOPE_PRESETS[preset]
        return LoRAConfig(
            rank=p["rank"],
            alpha=p["alpha"],
            dropout=0.0,
            target_modules=tuple(p["target_modules"]),
        )

    def __post_init__(self):
        for m in self.target_modules:
            if m not in VALID_TARGET_MODULES:
                raise ValueError(
                    f"Invalid target module '{m}'. "
                    f"Valid modules: {', '.join(VALID_TARGET_MODULES)}"
                )
        if self.rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be > 0, got {self.alpha}")