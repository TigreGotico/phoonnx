from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DisentangleConfig:
    enabled: bool = False
    ref_enc_hidden_channels: int = 256
    ref_enc_n_layers: int = 3
    ref_enc_kernel_size: int = 3
    ref_enc_stride: int = 2
    ref_enc_n_gru_layers: int = 1
    timbre_dim: int = 0
    artic_dim: int = 0
    prosody_dim: int = 0
    lambda_mi: float = 0.1
    lambda_cycle: float = 1.0
    lambda_kl_dis: float = 0.01
    emotion_labels: Optional[List[str]] = None

    def __post_init__(self):
        if self.timbre_dim == 0:
            self.timbre_dim = self.ref_enc_hidden_channels
        if self.artic_dim == 0:
            self.artic_dim = self.ref_enc_hidden_channels
        if self.prosody_dim == 0:
            self.prosody_dim = self.ref_enc_hidden_channels
