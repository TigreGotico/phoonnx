#!/usr/bin/env python3
"""Export the XCodec2 codec used by Llasa to ONNX.

XCodec2 (HKUSTAudio) has a single 65,536-entry finite-scalar-quantisation (FSQ)
codebook at 50 tokens/s and 16 kHz. Llasa emits those token ids directly, so
phoonnx needs the **decoder** for every synthesis and the **encoder** only to
tokenise a user-supplied reference clip for voice cloning.

Two graphs are produced:

``decoder.onnx``
    ``codes`` [batch, 1, frames] int64 -> ``audio`` [batch, samples] float32

``encoder.onnx`` (``--encoder``)
    ``input_features`` [batch, frames, 160] float32 (w2v-BERT front end) and
    ``audio`` [batch, 1, samples] float32 -> ``codes`` [batch, frames] int64

The upstream checkpoint was saved with the pre-2.1 ``weight_norm`` parameter
names (``weight_g`` / ``weight_v``); modern torch registers them under
``parametrizations.weight.original0`` / ``original1``. The encoder weights are
remapped on load, otherwise ``CodecEnc`` silently keeps its random init.

Usage::

    python export_xcodec2.py --output out/xcodec2 [--encoder]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import torch
from torch import nn


def load_xcodec2(repo: str = "HKUSTAudio/xcodec2"):
    from huggingface_hub import snapshot_download

    snap = snapshot_download(
        repo,
        allow_patterns=["*.py", "*.json", "model.safetensors", "vq/*"],
        ignore_patterns=["**/__pycache__/*"],
    )
    sys.path.insert(0, snap)
    from modeling_xcodec2 import XCodec2Model  # noqa: E402  (needs sys.path)

    model = XCodec2Model.from_pretrained(snap).eval()
    _restore_weight_norm(model, snap)
    return model


def patch_istft(model) -> None:
    """Swap the complex ISTFT head for the traceable real-valued one."""
    model.generator.head = RealISTFTHead(model.generator.head).eval()


def _restore_weight_norm(model: nn.Module, snap: str) -> None:
    """Copy the checkpoint's ``weight_g``/``weight_v`` into torch's parametrisations."""
    from safetensors.torch import load_file

    sd = load_file(glob.glob(snap + "/model.safetensors")[0])
    live = dict(model.named_parameters())
    fixed = 0
    for key, value in sd.items():
        if ".parametrizations.weight.original" not in key:
            continue
        # CodecEnc.x.parametrizations.weight.original0 -> CodecEnc.x.weight_g
        base, suffix = key.split(".parametrizations.weight.original")
        target = f"{base}.weight_{'g' if suffix == '0' else 'v'}"
        if target in live and live[target].shape == value.shape:
            live[target].data.copy_(value)
            fixed += 1
    print(f"restored {fixed} weight_norm tensors")


class RealISTFTHead(nn.Module):
    """Drop-in ``ISTFTHead`` that avoids complex tensors, which ONNX cannot trace.

    The upstream head builds ``mag * exp(i*phase)`` and calls ``torch.fft.irfft``
    plus ``F.fold``. Both are re-expressed with real arithmetic:

    * the inverse real FFT becomes two constant cosine/sine basis matmuls;
    * the overlap-add ``fold`` becomes a ``conv_transpose1d`` with an identity
      kernel, which is the same sum of shifted frames.

    The result is bit-comparable to the complex path (see ``parity_codec.py``).
    """

    def __init__(self, head):
        super().__init__()
        self.out = head.out
        istft = head.istft
        n_fft, hop, win = istft.n_fft, istft.hop_length, istft.win_length
        self.n_fft, self.hop_length, self.win_length = n_fft, hop, win
        self.pad = (win - hop) // 2

        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]
        n = torch.arange(n_fft, dtype=torch.float64)[None, :]
        weight = torch.full((n_fft // 2 + 1, 1), 2.0, dtype=torch.float64)
        weight[0] = 1.0
        if n_fft % 2 == 0:
            weight[-1] = 1.0
        angle = 2 * torch.pi * k * n / n_fft
        self.register_buffer("cos_basis", (torch.cos(angle) * weight / n_fft).float())
        self.register_buffer("sin_basis", (torch.sin(angle) * weight / n_fft).float())
        self.register_buffer("window", istft.window.clone())
        self.register_buffer("ola_kernel", torch.eye(win).unsqueeze(1))

    def _overlap_add(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv_transpose1d(frames, self.ola_kernel,
                                                    stride=self.hop_length)

    def forward(self, x: torch.Tensor):
        x_pred = self.out(x).transpose(1, 2)
        mag, phase = x_pred.chunk(2, dim=1)
        mag = torch.clip(torch.exp(mag), max=1e2)
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)

        # irfft along the frequency axis, as a real matmul over the basis
        ifft = torch.einsum("kn,bkt->bnt", self.cos_basis, real) \
            - torch.einsum("kn,bkt->bnt", self.sin_basis, imag)
        ifft = ifft * self.window[None, :, None]

        y = self._overlap_add(ifft)[:, 0, self.pad:-self.pad]
        window_sq = (self.window ** 2)[None, :, None].expand(1, -1, ifft.shape[-1])
        envelope = self._overlap_add(window_sq)[0, 0, self.pad:-self.pad]
        return (y / envelope).unsqueeze(1), x_pred


class DecoderWrapper(nn.Module):
    """``codes`` -> waveform, mirroring ``XCodec2Model.decode_code``.

    ``ResidualFSQ.get_output_from_indices`` goes through ``einops``, which bakes
    the traced frame count into the graph. The lookup is a single FSQ level with
    an implicit 65,536 x 8 codebook, so it is replaced by an embedding gather
    followed by the quantiser's own ``project_out`` — identical arithmetic, but
    the frame axis stays dynamic.
    """

    def __init__(self, model):
        super().__init__()
        self.generator = model.generator
        self.fc_post_a = model.fc_post_a
        quantizer = model.generator.quantizer
        fsq = quantizer.layers[0]
        self.register_buffer("codebook", fsq.implicit_codebook.detach().clone().float())
        self.project_out = quantizer.project_out

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        emb = torch.nn.functional.embedding(codes[:, 0, :], self.codebook)
        emb = self.project_out(emb)                       # [batch, frames, 2048]
        emb = self.fc_post_a(emb)                         # [batch, frames, 1024]
        return self.generator(emb, vq=False)[0]


class EncoderWrapper(nn.Module):
    """``input_features`` + waveform -> codes, mirroring ``XCodec2Model.encode_code``."""

    def __init__(self, model):
        super().__init__()
        self.semantic_model = model.semantic_model
        self.semantic_encoder = model.SemanticEncoder_module
        self.codec_enc = model.CodecEnc
        self.fc_prior = model.fc_prior
        self.generator = model.generator

    def forward(self, input_features: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        hidden = self.semantic_model(input_features, output_hidden_states=True).hidden_states[16]
        semantic = self.semantic_encoder(hidden.transpose(1, 2))
        vq_emb = self.codec_enc(audio).transpose(1, 2)
        n = torch.minimum(torch.tensor(vq_emb.shape[-1]), torch.tensor(semantic.shape[-1]))
        vq_emb = vq_emb[:, :, :n]
        semantic = semantic[:, :, :n]
        concat = torch.cat([semantic, vq_emb], dim=1)
        concat = self.fc_prior(concat.transpose(1, 2)).transpose(1, 2)
        return self.generator(concat, vq=True)[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--encoder", action="store_true", help="also export the encoder")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model = load_xcodec2()
    patch_istft(model)

    dec = DecoderWrapper(model).eval()
    codes = torch.randint(0, 65536, (1, 1, 40), dtype=torch.int64)
    with torch.no_grad():
        torch.onnx.export(
            dec, (codes,), str(out / "decoder.onnx"),
            input_names=["codes"], output_names=["audio"],
            dynamic_axes={"codes": {0: "batch", 2: "frames"},
                          "audio": {0: "batch", 1: "samples"}},
            opset_version=args.opset, do_constant_folding=True, dynamo=False,
        )
    print("wrote", out / "decoder.onnx")

    if args.encoder:
        enc = EncoderWrapper(model).eval()
        feats = torch.randn(1, 100, 160)
        audio = torch.randn(1, 1, 16000)
        with torch.no_grad():
            torch.onnx.export(
                enc, (feats, audio), str(out / "encoder.onnx"),
                input_names=["input_features", "audio"], output_names=["codes"],
                dynamic_axes={"input_features": {0: "batch", 1: "feat_frames"},
                              "audio": {0: "batch", 2: "samples"},
                              "codes": {0: "batch", 1: "frames"}},
                opset_version=args.opset, do_constant_folding=True, dynamo=False,
            )
        print("wrote", out / "encoder.onnx")


if __name__ == "__main__":
    main()
