"""StyleTTS2 checkpoint -> phoonnx ONNX export.

Generalization of ``scripts/conversion/styletts2/export_bsc.py`` working
against the vendored package instead of an upstream clone.  Produces the
two-graph contract the phoonnx ``StyleTTS2Adapter`` consumes:

    model.onnx          tokens(int64) + style(1, 2*style_dim) + speed(1) -> waveform
    style_encoder.onnx  waveform(1, T)                       -> ref_p, ref_s

The style-diffusion sampler is bypassed on the synthesis path — the style is
an input, computed from reference audio by ``style_encoder.onnx``.

The exportability rewrites (all faithful for batch=1):
  * InstanceNorm1d/2d -> manual normalization (dynamic channel);
  * TextEncoder / DurationEncoder -> no pack_padded_sequence (bakes length);
  * PL-BERT called without attention_mask (SDPA masking bakes length);
  * dynamic all-False text mask + arange(sum(dur)) alignment expansion.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

_LOG = logging.getLogger(__name__)

OPSET_VERSION = 17


def _instance_norm_forward(self, x):
    dims = tuple(range(2, x.dim()))
    mean = x.mean(dims, keepdim=True)
    var = x.var(dims, keepdim=True, unbiased=False)
    xn = (x - mean) / torch.sqrt(var + self.eps)
    if getattr(self, "affine", False):
        shape = [1, -1] + [1] * (x.dim() - 2)
        xn = xn * self.weight.view(*shape) + self.bias.view(*shape)
    return xn


def _text_encoder_forward(self, x, input_lengths, m):
    x = self.embedding(x).transpose(1, 2)
    m = m.to(input_lengths.device).unsqueeze(1)
    x = x.masked_fill(m, 0.0)
    for c in self.cnn:
        x = c(x).masked_fill(m, 0.0)
    x = x.transpose(1, 2)
    self.lstm.flatten_parameters()
    x, _ = self.lstm(x)
    x = x.transpose(-1, -2)
    return x.masked_fill(m, 0.0)


def _duration_encoder_forward(self, x, style, text_lengths, m):
    from phoonnx_train.styletts2 import models as _M
    masks = m.to(text_lengths.device)
    x = x.permute(2, 0, 1)
    s = style.expand(x.shape[0], x.shape[1], -1)
    x = torch.cat([x, s], axis=-1)
    x = x.masked_fill(masks.unsqueeze(-1).transpose(0, 1), 0.0)
    x = x.transpose(0, 1).transpose(-1, -2)
    for block in self.lstms:
        if isinstance(block, _M.AdaLayerNorm):
            x = block(x.transpose(-1, -2), style).transpose(-1, -2)
            x = torch.cat([x, s.permute(1, -1, 0)], axis=1)
            x = x.masked_fill(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
        else:
            x = x.transpose(-1, -2)
            block.flatten_parameters()
            x, _ = block(x)
            x = x.transpose(-1, -2)
    return x.transpose(-1, -2)


def _apply_export_patches() -> None:
    from phoonnx_train.styletts2 import models as _M
    torch.nn.InstanceNorm1d.forward = _instance_norm_forward
    torch.nn.InstanceNorm2d.forward = _instance_norm_forward
    _M.TextEncoder.forward = _text_encoder_forward
    _M.DurationEncoder.forward = _duration_encoder_forward


class Synth(torch.nn.Module):
    """tokens + style + speed -> waveform (diffusion sampler bypassed)."""

    def __init__(self, model, style_dim: int, hifigan_shift: bool):
        super().__init__()
        self.m = model
        self.style_dim = style_dim
        self.hifigan_shift = hifigan_shift

    def forward(self, tokens, style, speed):
        m = self.m
        acoustic = style[:, :self.style_dim]
        prosodic = style[:, self.style_dim:]
        L = torch.tensor([tokens.shape[-1]])
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        t_en = m.text_encoder(tokens, L, mask)
        bert_dur = m.bert(tokens)
        d_en = m.bert_encoder(bert_dur).transpose(-1, -2)

        d = m.predictor.text_encoder(d_en, prosodic, L, mask)
        x, _ = m.predictor.lstm(d)
        duration = torch.sigmoid(m.predictor.duration_proj(x)).sum(dim=-1) / speed
        pred_dur = torch.round(duration.squeeze()).clamp(min=1).long()

        ends = torch.cumsum(pred_dur, dim=0)
        frames = torch.arange(ends[-1])
        aln = ((frames[None, :] >= (ends - pred_dur)[:, None]) &
               (frames[None, :] < ends[:, None])).float()

        en = d.transpose(-1, -2) @ aln[None]
        if self.hifigan_shift:
            en = torch.cat([en[:, :, :1], en[:, :, :-1]], dim=2)
        F0_pred, N_pred = m.predictor.F0Ntrain(en, prosodic)
        asr = t_en @ aln[None]
        if self.hifigan_shift:
            asr = torch.cat([asr[:, :, :1], asr[:, :, :-1]], dim=2)
        return m.decoder(asr, F0_pred, N_pred, acoustic).squeeze()


class MelSpectrogram(torch.nn.Module):
    """yl4579 to_mel replica as conv1d-DFT so it ONNX-exports (no STFT op)."""

    def __init__(self, n_mels: int = 80, sample_rate: int = 24000):
        super().__init__()
        import torchaudio
        n_fft, win, self.hop, self.n_fft = 2048, 1200, 300, 2048
        w = torch.nn.functional.pad(
            torch.hann_window(win),
            ((n_fft - win) // 2, n_fft - win - (n_fft - win) // 2))
        k = torch.arange(n_fft // 2 + 1).unsqueeze(1).float()
        n = torch.arange(n_fft).unsqueeze(0).float()
        ang = 2 * np.pi * k * n / n_fft
        self.register_buffer("cos", (torch.cos(ang) * w).unsqueeze(1))
        self.register_buffer("sin", (-torch.sin(ang) * w).unsqueeze(1))
        self.register_buffer("fb", torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=win,
            hop_length=self.hop, n_mels=n_mels).mel_scale.fb.t().contiguous())

    def forward(self, x):
        x = torch.nn.functional.pad(
            x.unsqueeze(1), (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        p = (torch.nn.functional.conv1d(x, self.cos, stride=self.hop) ** 2
             + torch.nn.functional.conv1d(x, self.sin, stride=self.hop) ** 2)
        return ((torch.log(1e-5 + torch.matmul(self.fb, p)) + 4) / 4).unsqueeze(1)


def _fix_negative_transpose_perms(path: Path) -> None:
    """torch can emit Transpose perms with negative indices under dynamic
    rank; onnxruntime rejects them."""
    import onnx
    m = onnx.load(str(path))
    changed = 0
    for node in m.graph.node:
        if node.op_type != "Transpose":
            continue
        for attr in node.attribute:
            if attr.name == "perm" and any(v < 0 for v in attr.ints):
                r = len(attr.ints)
                attr.ints[:] = [v % r for v in attr.ints]
                changed += 1
    if changed:
        onnx.save(m, str(path))


def _load_config(config_path: Path) -> Dict[str, Any]:
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix in (".yml", ".yaml"):
        import yaml
        return yaml.safe_load(text)
    return json.loads(text)


def export_styletts2_onnx(
    checkpoint_path: Path,
    config_path: Path,
    output_dir: Path,
    sample_rate: int = 24000,
    phoneme_id_map: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> Path:
    """Export a trained StyleTTS2 checkpoint (upstream 'net' layout or a
    Lightning .ckpt saved by StyleTTS2Module) to model.onnx +
    style_encoder.onnx + config.json."""
    from phoonnx_train.engines.styletts2 import StyleTTS2Config

    cfg_raw = _load_config(config_path)
    scfg = StyleTTS2Config(**{k: v for k, v in cfg_raw.get("styletts2", cfg_raw).items()
                              if k in StyleTTS2Config.__dataclass_fields__})

    _apply_export_patches()

    from munch import munchify

    from phoonnx_train.styletts2.models import build_model
    from phoonnx_train.engines.styletts2 import (_build_pitch_extractor,
                                                 _build_plbert,
                                                 _build_text_aligner)

    mp = munchify(scfg.resolved_model_params())
    text_aligner = _build_text_aligner(scfg, mp.n_mels, mp.n_token)
    pitch_extractor = _build_pitch_extractor(scfg)
    plbert = _build_plbert(scfg, mp.n_token)
    model = build_model(mp, text_aligner, pitch_extractor, plbert)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net = state.get("net")
    if net is None and "state_dict" in state:
        # Lightning layout: nets.<key>.<param>
        net = {}
        for k, v in state["state_dict"].items():
            if k.startswith("nets."):
                key, rest = k[len("nets."):].split(".", 1)
                net.setdefault(key, {})[rest] = v
    if not net:
        raise ValueError(f"No model weights found in {checkpoint_path}")
    for k in model:
        if k in net:
            sd = {(kk[7:] if kk.startswith("module.") else kk): vv
                  for kk, vv in net[k].items()}
            model[k].load_state_dict(sd, strict=False)
        model[k].eval()
        model[k].requires_grad_(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    synth = Synth(model, mp.style_dim,
                  hifigan_shift=(mp.decoder.type == "hifigan")).eval()
    torch.manual_seed(0)
    tokens = torch.randint(1, mp.n_token, (1, 40), dtype=torch.long)
    style = torch.randn(1, 2 * mp.style_dim)
    speed = torch.tensor([1.0])

    model_path = output_dir / "model.onnx"
    torch.onnx.export(synth, (tokens, style, speed), str(model_path),
                      input_names=["tokens", "style", "speed"],
                      output_names=["waveform"],
                      dynamic_axes={"tokens": {1: "n"}, "waveform": {0: "s"}},
                      opset_version=OPSET_VERSION, dynamo=False)

    class Enc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mel = MelSpectrogram(n_mels=mp.n_mels, sample_rate=sample_rate)
            self.style = model.style_encoder
            self.pred = model.predictor_encoder

        def forward(self, wav):
            mm = self.mel(wav)
            return self.style(mm), self.pred(mm)

    enc_path = output_dir / "style_encoder.onnx"
    torch.onnx.export(Enc().eval(), torch.randn(1, sample_rate * 2), str(enc_path),
                      input_names=["waveform"], output_names=["ref_p", "ref_s"],
                      dynamic_axes={"waveform": {1: "s"}},
                      opset_version=OPSET_VERSION, dynamo=False)

    _fix_negative_transpose_perms(model_path)
    _fix_negative_transpose_perms(enc_path)

    out_cfg: Dict[str, Any] = {
        "engine": "styletts2",
        "sample_rate": sample_rate,
        "style_dim": int(mp.style_dim),
        "num_symbols": int(mp.n_token),
    }
    if phoneme_id_map:
        out_cfg["phoneme_id_map"] = phoneme_id_map
    (output_dir / "config.json").write_text(
        json.dumps(out_cfg, ensure_ascii=False, indent=2))

    _LOG.info("StyleTTS2 exported to %s", model_path)
    return model_path
