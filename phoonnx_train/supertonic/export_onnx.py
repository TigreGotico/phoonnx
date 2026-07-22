"""Export trained SuperTonic modules to the four-graph ONNX contract the phoonnx
``supertonic`` inference engine consumes.

The four graphs and their exact input names (matching the official ``helper.py``):

* ``duration_predictor.onnx`` : ``text_ids``, ``style_dp``, ``text_mask`` -> ``duration``
* ``text_encoder.onnx``       : ``text_ids``, ``style_ttl``, ``text_mask`` -> ``text_emb``
* ``vector_estimator.onnx``   : ``noisy_latent``, ``text_emb``, ``style_ttl``,
  ``text_mask``, ``latent_mask``, ``current_step``, ``total_step`` -> ``latent``
* ``vocoder.onnx``            : ``latent`` -> ``wav``

The exported duration-predictor and text-encoder graphs take the *pre-pooled*
style tokens directly (``style_dp`` / ``style_ttl``), so the style encoders are
not part of any graph — the inference engine supplies per-voice style JSONs. The
vector-estimator graph folds one Euler integration step (``current_step`` ->
``current_step+1``) so the engine's loop can feed the output straight back in.
The vocoder graph decompresses + denormalises the latent internally before the
waveform decoder.

Alongside the graphs it writes ``tts.json`` (the runtime config the engine
reads) and ``unicode_indexer.json`` (the code-point -> id table).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from phoonnx_train.supertonic.config import SuperTonicConfig
from phoonnx_train.supertonic.text import CharTokenizer

_OPSET = 17


def _export_kwargs() -> Dict[str, Any]:
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        return {"dynamo": False}
    return {}


class _DurationOnnx(torch.nn.Module):
    def __init__(self, dp):
        super().__init__()
        self.dp = dp

    def forward(self, text_ids, style_dp, text_mask):
        text_emb = self.dp.text_encoder(text_ids, text_mask)
        style_flat = style_dp.reshape(style_dp.shape[0], -1)
        return self.dp.log_duration_from_embeddings(text_emb, style_flat).exp()


class _TextEncoderOnnx(torch.nn.Module):
    def __init__(self, ttl):
        super().__init__()
        self.text_encoder = ttl.text_encoder

    def forward(self, text_ids, style_ttl, text_mask):
        return self.text_encoder(text_ids, text_mask, style_ttl)


class _VectorEstimatorOnnx(torch.nn.Module):
    def __init__(self, ttl):
        super().__init__()
        self.vf = ttl.vector_field

    def forward(self, noisy_latent, text_emb, style_ttl, text_mask, latent_mask,
                current_step, total_step):
        t = current_step / total_step
        v = self.vf(noisy_latent, t, text_emb, style_ttl, latent_mask, text_mask)
        dt = (1.0 / total_step).view(-1, 1, 1)
        return (noisy_latent + v * dt) * latent_mask


class _VocoderOnnx(torch.nn.Module):
    def __init__(self, ae, compress_factor, latent_dim, scale):
        super().__init__()
        self.ae = ae
        self.k = compress_factor
        self.ld = latent_dim
        self.scale = scale

    def forward(self, latent):
        from phoonnx_train.supertonic.latent_utils import decompress_and_denormalize
        raw = decompress_and_denormalize(self.ae, latent, self.k, self.ld, self.scale)
        return self.ae.decoder(raw)


def _write(module, args, path, input_names, output_names, dynamic_axes):
    torch.onnx.export(
        module, args, str(path), opset_version=_OPSET, input_names=input_names,
        output_names=output_names, dynamic_axes=dynamic_axes, **_export_kwargs())


def export_all(output_dir: str, *, config: SuperTonicConfig, tokenizer: CharTokenizer,
               autoencoder, text_to_latent, duration_predictor) -> Dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    autoencoder.eval(); text_to_latent.eval(); duration_predictor.eval()

    ttl, dp, ae = config.ttl, config.dp, config.ae
    if not ae.latent_dim == ttl.latent_dim == dp.latent_dim:
        raise ValueError("latent_dim disagrees across the stages "
                         f"(ae={ae.latent_dim}, ttl={ttl.latent_dim}, dp={dp.latent_dim}); "
                         "the exported graphs would reshape the latent inconsistently")
    b, t_text, t_lat = 1, 6, 4
    n_ttl, n_dp = ttl.n_style, dp.n_style
    text_ids = torch.randint(1, max(2, config.vocab_size), (b, t_text), dtype=torch.long)
    text_mask = torch.ones(b, 1, t_text)
    latent_mask = torch.ones(b, 1, t_lat)
    style_ttl = torch.randn(b, n_ttl, ttl.style_dim)
    style_dp = torch.randn(b, n_dp, dp.style_value_dim)
    noisy = torch.randn(b, ttl.compressed_dim, t_lat)
    text_emb = torch.randn(b, ttl.char_dim, t_text)

    with torch.no_grad():
        dp_path = out / "duration_predictor.onnx"
        _write(_DurationOnnx(duration_predictor), (text_ids, style_dp, text_mask), dp_path,
               ["text_ids", "style_dp", "text_mask"], ["duration"],
               {"text_ids": {0: "B", 1: "T"}, "style_dp": {0: "B"}, "text_mask": {0: "B", 2: "T"}})

        te_path = out / "text_encoder.onnx"
        _write(_TextEncoderOnnx(text_to_latent), (text_ids, style_ttl, text_mask), te_path,
               ["text_ids", "style_ttl", "text_mask"], ["text_emb"],
               {"text_ids": {0: "B", 1: "T"}, "style_ttl": {0: "B"},
                "text_mask": {0: "B", 2: "T"}, "text_emb": {0: "B", 2: "T"}})

        current_step = torch.tensor([1.0])
        total_step = torch.tensor([8.0])
        ve_path = out / "vector_estimator.onnx"
        _write(_VectorEstimatorOnnx(text_to_latent),
               (noisy, text_emb, style_ttl, text_mask, latent_mask, current_step, total_step),
               ve_path,
               ["noisy_latent", "text_emb", "style_ttl", "text_mask", "latent_mask",
                "current_step", "total_step"], ["latent"],
               {"noisy_latent": {0: "B", 2: "L"}, "text_emb": {0: "B", 2: "T"},
                "style_ttl": {0: "B"}, "text_mask": {0: "B", 2: "T"},
                "latent_mask": {0: "B", 2: "L"}, "current_step": {0: "B"},
                "total_step": {0: "B"}, "latent": {0: "B", 2: "L"}})

        voc_path = out / "vocoder.onnx"
        _write(_VocoderOnnx(autoencoder, ttl.compress_factor, ae.latent_dim, ttl.normalizer_scale),
               (noisy,), voc_path, ["latent"], ["wav"],
               {"latent": {0: "B", 2: "L"}, "wav": {0: "B", 1: "T"}})

    (out / "tts.json").write_text(json.dumps({
        "ae": {"sample_rate": ae.sample_rate, "base_chunk_size": ae.hop_length},
        "ttl": {"chunk_compress_factor": ttl.compress_factor, "latent_dim": ae.latent_dim,
                "normalizer": {"scale": ttl.normalizer_scale}},
        "dp": {"chunk_compress_factor": dp.compress_factor, "latent_dim": ae.latent_dim,
               "normalizer": {"scale": dp.normalizer_scale}},
        "engine": "supertonic",
    }, indent=2))
    (out / "unicode_indexer.json").write_text(json.dumps(tokenizer.to_indexer_list()))

    return {"duration_predictor": dp_path, "text_encoder": te_path,
            "vector_estimator": ve_path, "vocoder": voc_path,
            "tts": out / "tts.json", "unicode_indexer": out / "unicode_indexer.json"}


def _load_stage(checkpoint: Optional[str], stage: str):
    if checkpoint is None:
        return None, None, None
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    found = ckpt.get("supertonic_stage")
    if found is not None and found != stage:
        raise ValueError(f"{checkpoint} is a {found!r} checkpoint but was passed as the {stage!r} one")
    from phoonnx_train.supertonic.lightning import _build_config
    cfg = _build_config(ckpt.get("supertonic_config"))
    tok = CharTokenizer.from_dict(ckpt.get("supertonic_tokenizer") or {})
    return ckpt, cfg, tok


def export_from_checkpoints(output_dir: str, *, autoencoder_ckpt: str,
                            text_to_latent_ckpt: str, duration_predictor_ckpt: str) -> Dict[str, Path]:
    """Rebuild all three modules from their Lightning checkpoints and export."""
    from phoonnx_train.supertonic.autoencoder import SpeechAutoencoder
    from phoonnx_train.supertonic.duration_predictor import DurationPredictor
    from phoonnx_train.supertonic.text_to_latent import TextToLatentModel

    ae_ckpt, ae_cfg, _ = _load_stage(autoencoder_ckpt, "autoencoder")
    ttl_ckpt, ttl_cfg, tok = _load_stage(text_to_latent_ckpt, "text_to_latent")
    dp_ckpt, dp_cfg, _ = _load_stage(duration_predictor_ckpt, "duration_predictor")
    cfg = ttl_cfg or dp_cfg or ae_cfg

    ae = SpeechAutoencoder(cfg.ae)
    _load_prefixed(ae, ae_ckpt["state_dict"], "generator.", autoencoder_ckpt)
    ttl = TextToLatentModel(cfg.ttl, cfg.vocab_size)
    _load_prefixed(ttl, ttl_ckpt["state_dict"], "model.", text_to_latent_ckpt)
    dp = DurationPredictor(cfg.dp, cfg.vocab_size)
    _load_prefixed(dp, dp_ckpt["state_dict"], "model.", duration_predictor_ckpt)

    return export_all(output_dir, config=cfg, tokenizer=tok,
                      autoencoder=ae, text_to_latent=ttl, duration_predictor=dp)


def _load_prefixed(module, state_dict, prefix, source):
    picked = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    missing = set(module.state_dict()) - set(picked)
    if missing:
        raise ValueError(
            f"{source} does not contain weights for {type(module).__name__} under {prefix!r}: "
            f"{len(missing)} missing key(s), e.g. {sorted(missing)[:3]}")
    module.load_state_dict(picked, strict=False)
