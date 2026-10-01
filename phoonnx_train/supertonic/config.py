"""Architecture and hyper-parameter configuration for the SuperTonic TTS stack.

SuperTonic (Kim et al., Supertone Inc., "SupertonicTTS", arXiv:2503.23108) is a
three-stage system:

* a GAN-trained **speech autoencoder** that turns a waveform into a low
  dimensional continuous latent sequence and back,
* a **text-to-latent** module that maps character-level text plus a reference
  voice style to that latent via conditional flow matching, and
* a **duration predictor** that estimates the total utterance length.

The dataclasses below carry the shape/size knobs for each stage. The defaults are
small enough to build and forward on CPU in a test; the values published for the
released ``Supertone/supertonic-3`` model are read from its ``tts.json`` by
:func:`load_model_config` when that file is supplied, so training at the released
scale never requires editing this module.

All sequence tensors use the ``(batch, channels, time)`` layout at module
boundaries, matching the ONNX graphs of the released model.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


def _dig(raw: dict, dotted: str, fallback: Any) -> Any:
    """Fetch ``raw["a"]["b"]["c"]`` for ``dotted="a.b.c"``, or ``fallback``."""
    cur: Any = raw
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return fallback
        cur = cur[part]
    return cur


@dataclass
class AutoencoderConfig:
    """Speech autoencoder (spectrogram -> latent -> waveform)."""

    sample_rate: int = 22050
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    concat_linear_spec: bool = True
    hidden_dim: int = 64
    ffn_dim: int = 128
    latent_dim: int = 24
    encoder_layers: int = 3
    encoder_kernel: int = 7
    decoder_layers: int = 4
    decoder_kernel: int = 7
    decoder_dilations: tuple = (1, 2, 4, 1)
    head_dim: int = 128

    @property
    def input_dim(self) -> int:
        linear_bins = (self.n_fft // 2 + 1) if self.concat_linear_spec else 0
        return self.n_mels + linear_bins


@dataclass
class TextToLatentConfig:
    """Conditional-flow-matching text-to-latent module."""

    latent_dim: int = 24
    compress_factor: int = 6
    normalizer_scale: float = 0.25
    batch_expand: int = 2
    prob_text_uncond: float = 0.05
    prob_both_uncond: float = 0.04
    sigma_min: float = 1e-8

    char_dim: int = 64
    convnext_ffn: int = 128
    convnext_layers: int = 2
    convnext_kernel: int = 5
    self_attn_layers: int = 2
    self_attn_heads: int = 2
    self_attn_ffn: int = 128
    self_attn_window: int = 4

    style_dim: int = 64
    style_convnext_layers: int = 2
    style_convnext_ffn: int = 128
    style_convnext_kernel: int = 5
    n_style: int = 8
    style_heads: int = 2
    prompt_heads: int = 2

    vf_dim: int = 64
    vf_ffn: int = 128
    vf_kernel: int = 5
    vf_blocks: int = 2
    vf_dilations: tuple = (1, 2)
    vf_final_layers: int = 2
    vf_text_heads: int = 2
    vf_style_heads: int = 2
    rotary_base: float = 10000.0
    rotary_scale: float = 10.0
    time_dim: int = 32
    time_hidden: int = 128

    @property
    def compressed_dim(self) -> int:
        return self.latent_dim * self.compress_factor


@dataclass
class DurationPredictorConfig:
    """Utterance-level duration predictor."""

    latent_dim: int = 24
    compress_factor: int = 6
    normalizer_scale: float = 1.0

    char_dim: int = 32
    convnext_ffn: int = 64
    convnext_layers: int = 2
    convnext_kernel: int = 5
    self_attn_layers: int = 1
    self_attn_heads: int = 2
    self_attn_ffn: int = 64
    self_attn_window: int = 4

    style_dim: int = 32
    style_convnext_layers: int = 2
    style_convnext_ffn: int = 64
    n_style: int = 4
    style_value_dim: int = 16
    style_heads: int = 2

    predictor_hidden: int = 64

    @property
    def compressed_dim(self) -> int:
        return self.latent_dim * self.compress_factor


@dataclass
class SuperTonicConfig:
    ae: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    ttl: TextToLatentConfig = field(default_factory=TextToLatentConfig)
    dp: DurationPredictorConfig = field(default_factory=DurationPredictorConfig)
    vocab_size: int = 256

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str) -> None:
        Path(path).write_text(self.to_json())


def load_model_config(path: Optional[str]) -> SuperTonicConfig:
    """Build a config, overlaying values from a released ``tts.json`` if given.

    Unknown/absent keys keep the dataclass defaults, so a partial or a
    differently-shaped file still yields a usable config.
    """
    cfg = SuperTonicConfig()
    if not path:
        return cfg
    raw = json.loads(Path(path).read_text())

    ae = cfg.ae
    ae.sample_rate = _dig(raw, "ae.sample_rate", ae.sample_rate)
    ae.n_fft = _dig(raw, "ae.encoder.spec_processor.n_fft", ae.n_fft)
    ae.hop_length = _dig(raw, "ae.encoder.spec_processor.hop_length", ae.hop_length)
    ae.win_length = _dig(raw, "ae.encoder.spec_processor.win_length", ae.win_length)
    ae.n_mels = _dig(raw, "ae.encoder.spec_processor.n_mels", ae.n_mels)
    ae.hidden_dim = _dig(raw, "ae.encoder.hdim", ae.hidden_dim)
    ae.ffn_dim = _dig(raw, "ae.encoder.intermediate_dim", ae.ffn_dim)
    ae.latent_dim = _dig(raw, "ae.ldim", ae.latent_dim)
    ae.encoder_layers = _dig(raw, "ae.encoder.num_layers", ae.encoder_layers)
    ae.encoder_kernel = _dig(raw, "ae.encoder.ksz", ae.encoder_kernel)
    ae.decoder_layers = _dig(raw, "ae.decoder.num_layers", ae.decoder_layers)
    ae.decoder_kernel = _dig(raw, "ae.decoder.ksz", ae.decoder_kernel)
    dils = _dig(raw, "ae.decoder.dilation_lst", None)
    if dils is not None:
        ae.decoder_dilations = tuple(dils)
    ae.head_dim = _dig(raw, "ae.decoder.head.hdim", ae.head_dim)
    measured = _dig(raw, "ae.encoder.idim", None)
    if measured is not None:
        ae.concat_linear_spec = measured != ae.n_mels

    ttl = cfg.ttl
    ttl.latent_dim = _dig(raw, "ttl.latent_dim", ttl.latent_dim)
    ttl.compress_factor = _dig(raw, "ttl.chunk_compress_factor", ttl.compress_factor)
    ttl.normalizer_scale = _dig(raw, "ttl.normalizer.scale", ttl.normalizer_scale)
    ttl.prob_text_uncond = _dig(raw, "ttl.uncond_masker.prob_text_uncond", ttl.prob_text_uncond)
    ttl.sigma_min = _dig(raw, "ttl.flow_matching.sig_min", ttl.sigma_min)
    ttl.char_dim = _dig(raw, "ttl.text_encoder.text_embedder.char_emb_dim", ttl.char_dim)
    ttl.self_attn_heads = _dig(raw, "ttl.text_encoder.attn_encoder.n_heads", ttl.self_attn_heads)
    ttl.n_style = _dig(raw, "ttl.style_encoder.style_token_layer.n_style", ttl.n_style)
    ttl.style_dim = _dig(raw, "ttl.style_encoder.convnext.idim", ttl.style_dim)
    ttl.vf_dim = _dig(raw, "ttl.vector_field.proj_in.odim", ttl.vf_dim)
    ttl.vf_blocks = _dig(raw, "ttl.vector_field.main_blocks.n_blocks", ttl.vf_blocks)
    ttl.rotary_base = _dig(raw, "ttl.vector_field.main_blocks.text_cond_layer.rotary_base", ttl.rotary_base)
    ttl.rotary_scale = _dig(raw, "ttl.vector_field.main_blocks.text_cond_layer.rotary_scale", ttl.rotary_scale)
    ttl.time_dim = _dig(raw, "ttl.vector_field.time_encoder.time_dim", ttl.time_dim)

    dp = cfg.dp
    dp.latent_dim = _dig(raw, "dp.latent_dim", dp.latent_dim)
    dp.compress_factor = _dig(raw, "dp.chunk_compress_factor", dp.compress_factor)
    dp.normalizer_scale = _dig(raw, "dp.normalizer.scale", dp.normalizer_scale)
    dp.char_dim = _dig(raw, "dp.sentence_encoder.char_emb_dim", dp.char_dim)
    dp.n_style = _dig(raw, "dp.style_encoder.style_token_layer.n_style", dp.n_style)
    dp.style_value_dim = _dig(raw, "dp.style_encoder.style_token_layer.style_value_dim", dp.style_value_dim)
    dp.style_dim = _dig(raw, "dp.style_encoder.convnext.idim", dp.style_dim)
    dp.predictor_hidden = _dig(raw, "dp.predictor.hdim", dp.predictor_hidden)

    return cfg


def tiny_config(vocab_size: int = 64) -> SuperTonicConfig:
    """A minimal config sized for fast CPU forward/backward smoke tests."""
    cfg = SuperTonicConfig(vocab_size=vocab_size)
    cfg.ae = AutoencoderConfig(
        sample_rate=16000, n_fft=256, win_length=256, hop_length=64, n_mels=32,
        hidden_dim=16, ffn_dim=32, latent_dim=8, encoder_layers=2, decoder_layers=2,
        decoder_dilations=(1, 2), head_dim=32,
    )
    cfg.ttl = TextToLatentConfig(
        latent_dim=8, compress_factor=2, char_dim=16, convnext_ffn=32, convnext_layers=1,
        self_attn_layers=1, self_attn_heads=2, self_attn_ffn=32, style_dim=16, n_style=4,
        style_convnext_layers=1, style_convnext_ffn=32, vf_dim=16, vf_ffn=32, vf_blocks=1,
        vf_dilations=(1,), vf_final_layers=1, vf_text_heads=2, vf_style_heads=2,
        time_dim=16, time_hidden=32, batch_expand=1,
    )
    cfg.dp = DurationPredictorConfig(
        latent_dim=8, compress_factor=2, char_dim=16, convnext_ffn=32, convnext_layers=1,
        self_attn_layers=1, self_attn_heads=2, self_attn_ffn=32, style_dim=16, n_style=4,
        style_value_dim=8, style_convnext_layers=1, style_convnext_ffn=32, predictor_hidden=32,
    )
    return cfg
