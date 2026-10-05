"""ONNX export wrapper around the ito (Lokutor) acoustic model.

Wraps a loaded ``ito.Front`` / ``ito.Vocoder`` pair into a single static
graph: ``input`` (int64 token ids, [1, L]) + ``speed`` (float32, [1]) ->
``wav`` (+ ``durations`` / ``mel`` / ``f0`` extras). Everything is
whole-utterance inference — the streaming windowing of the chip engine is
an execution strategy, not different math, so the exported graph computes
the same audio as ``ito.synth.Chain.infer_full``.

Only the ONNX-unfriendly pieces are re-expressed here (they are standard
DSP/indexing ops, written for this exporter):

* the text encoder's pack/unpack (a no-op at batch 1, no padding);
* length regulation as an explicit cumsum/gather (batch 1);
* the prosody net in float32 (the ito reference runs it in float64 to make
  its *streaming* checks exact; a single whole-utterance pass has no such
  stability requirement — see the exporter's parity report);
* the harmonic source (kept in float64: phase accumulation and sine are
  exact, matching the reference);
* STFT / iSTFT as matmul with a fixed DFT basis plus windowed
  overlap-add via strided transposed convolution.

The ito package (GPL-3.0) is an import-time dependency of this developer
tool only; no ito code is vendored and the exported artifacts carry only
weights (CC BY-NC-SA 4.0, Lokutor) and this exporter's graph.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def snr_db(a: torch.Tensor, b: torch.Tensor) -> float:
    """Signal-to-noise ratio of ``a`` against reference ``b``, in dB."""
    a = a.double().reshape(-1)
    b = b.double().reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    noise = (a - b).pow(2).sum()
    power = b.pow(2).sum()
    if noise == 0:
        return float("inf")
    return float(10 * torch.log10(power / noise))


class ItoOnnxGraph(nn.Module):
    """Single-graph export of ito's front + vocoder (batch size 1)."""

    def __init__(self, front, voc, style, noise, phase0: float = 0.25):
        super().__init__()
        self.front, self.voc = front, voc
        self.hop = voc.hop
        self.sr = voc.src.sr
        self.n_fft = voc.n_fft
        self.register_buffer("style", style.float().reshape(1, -1))
        # deterministic excitation noise: tiled to the utterance length at
        # run time (the reference draws fresh N(0,1) noise per sentence;
        # a fixed tiled buffer is distribution-identical and reproducible)
        self.register_buffer("noise", noise.float().reshape(1, -1))
        self.register_buffer("phase0", torch.full((1,), phase0, dtype=torch.float64))
        # harmonic overtone indices k = 1..n_harm (float64, like the source)
        self.register_buffer("harm_k",
                             torch.arange(1, voc.src.n_harm + 1, dtype=torch.float64).reshape(1, -1, 1))

        # DFT bases for the source STFT and the iSTFT (float64 for accuracy;
        # frames are cast to float32 after the transform, like the reference).
        # Stored pre-transposed so no Transpose node feeds a MatMul — ORT's
        # MatMulTransposeFusion emits a float-only fused kernel.
        n = self.n_fft
        nb = n // 2 + 1
        kk = torch.arange(nb, dtype=torch.float64).reshape(-1, 1)
        nn_ = torch.arange(n, dtype=torch.float64).reshape(1, -1)
        arg = 2 * math.pi * kk * nn_ / n
        self.register_buffer("dft_cos_t", torch.cos(arg).transpose(0, 1))   # [n, nb]
        self.register_buffer("dft_sin_t", torch.sin(arg).transpose(0, 1))   # [n, nb]
        # irfft: x[n] = (1/n) * sum_k c_k * (ReX_k cos - ImX_k sin),
        # with c_0 = c_{n/2} = 1 and c_k = 2 otherwise
        c = torch.full((nb,), 2.0, dtype=torch.float64)
        c[0] = 1.0
        c[-1] = 1.0
        self.register_buffer("idft_cos_t", (c.reshape(-1, 1) * torch.cos(arg) / n).transpose(0, 1))  # [n, nb]
        self.register_buffer("idft_sin_t", (c.reshape(-1, 1) * torch.sin(arg) / n).transpose(0, 1))  # [n, nb]

    # ---- text side (batch 1, no padding) ----
    def encode(self, tokens: torch.Tensor):
        """tokens [1, L] int64 -> (h [1, C, L], s [1, sd], logd [1, L])."""
        front = self.front
        L = tokens.size(1)
        tmask = torch.ones(1, 1, L, device=tokens.device)
        s = front.style_proj(self.style)
        h = front.emb(tokens).transpose(1, 2) * tmask
        for layer in front.enc:
            h = layer(h, tmask)
        if front.text_rnn:
            # batch 1 with no padding: the reference's packed sequence is the
            # input itself, so the GRU runs directly
            out, _ = front.rnn(h.transpose(1, 2))
            h = (h + front.rnn_proj(out).transpose(1, 2)) * tmask
        x = h
        for layer in front.dur_layers:
            x = layer(x, s, tmask)
        logd = front.dur_out(x)[:, 0]
        return h, s, logd

    def regulate(self, h: torch.Tensor, dur: torch.Tensor) -> torch.Tensor:
        """h [1, C, L], dur [1, L] -> hf [1, C + 2 (+1), T] (batch 1)."""
        front = self.front
        d = dur[0]                                   # [L] int64
        bounds = torch.cumsum(d, 0)                  # [L]
        T = bounds[-1]                              # scalar int64 tensor
        ar = torch.arange(T, device=h.device)        # [T]
        # token index of frame t: the count of durations already spent
        idx = (bounds.reshape(-1, 1) <= ar.reshape(1, -1)).sum(0)
        start = bounds - d
        pos = (ar - start[idx]).to(h.dtype) / d[idx].to(h.dtype)
        log_dur = torch.log(d[idx].to(h.dtype)) / 3.0
        cols = [h[0][:, idx][None]]
        cols.append(pos.reshape(1, 1, -1))
        cols.append(log_dur.reshape(1, 1, -1))
        if front.sent_pos:
            cols.append((ar.to(h.dtype) / T.to(h.dtype)).reshape(1, 1, -1))
        return torch.cat(cols, 1)

    # ---- frame side ----
    def curves_and_mel(self, hf, s):
        """Prosody net (float32) + mel head, mirroring Front.prosody/mel_head."""
        front = self.front
        x = front.pros_in(hf)
        for layer in front.pros:
            x = layer(x, s)
        pred = front.pros_out(x)
        v = (pred[:, 1] > 0).to(hf.dtype)
        lf0n = pred[:, 0] * v
        en = pred[:, 2]
        f0hz = torch.exp(pred[:, 0] * front.stats[1] + front.stats[0]) * v
        dec = torch.cat([hf, lf0n.unsqueeze(1), v.unsqueeze(1), en.unsqueeze(1)], 1)
        y = front.mel_in(dec)
        for layer in front.mel_blocks:
            y = layer(y, s)
        mel = front.mel_out(y) * front.mel_std.reshape(1, -1, 1) + front.mel_mean.reshape(1, -1, 1)
        return mel, f0hz

    # ---- harmonic source (float64, as the reference) ----
    def source_signal(self, f0hz: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """f0hz [1, T] float32 -> excitation [1, T*hop] float32."""
        src = self.voc.src
        hop, sr = self.hop, self.sr
        T = T if isinstance(T, torch.Tensor) else torch.tensor(T)
        n = torch.arange(T * hop, device=f0hz.device, dtype=torch.float64)
        pos = ((n + 0.5) / hop - 0.5).clamp(0, T.to(torch.float64) - 1)
        i0 = pos.floor().to(torch.long)
        i1 = (i0 + 1).clamp(max=T - 1)
        w = pos - i0.to(torch.float64)
        f = f0hz.to(torch.float64)
        f = f[:, i0] * (1 - w) + f[:, i1] * w             # [1, N] Hz, linear interp
        uv = (f > 1.0).to(torch.float64)
        ph = self.phase0.reshape(-1, 1) + torch.cumsum(f / sr, -1)
        ph = ph - ph.floor()
        keep = (self.harm_k * f.unsqueeze(1) < sr / 2).to(torch.float64)  # [1, K, N]
        harm = (torch.sin(2 * math.pi * self.harm_k * ph.unsqueeze(1)) * keep).sum(1)
        h = harm.to(torch.float32) / src.n_harm
        noise = self.tile_noise(T * hop)
        uvf = uv.to(torch.float32)
        return (src.amp * h * uvf + src.noise * noise * uvf
                + (src.amp / 3) * noise * (1 - uvf))

    def tile_noise(self, n_samples: torch.Tensor) -> torch.Tensor:
        """The fixed noise buffer, tiled to n_samples. -> [1, n]."""
        m = self.noise.size(1)
        idx = torch.arange(n_samples, device=self.noise.device) % m
        return self.noise[:, idx]

    # ---- source STFT / vocoder iSTFT as basis matmuls ----
    def source_feats(self, s: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Excitation [1, T*hop] -> [1, n_fft + 2, T] real/imag, the ito
        vocoder's time base (frame t of the analysis, `[..., :T]`).

        Mirrors HarmonicSource.feats_full: hop//2 of zero padding on both
        ends, then a centered STFT — i.e. n_fft//2 of *reflect* padding
        (torch.stft's center=True default) and frames at stride hop. The
        framing is an explicit gather (no unfold: dynamic input sizes are
        not exportable that way).
        """
        hop, n_fft = self.hop, self.n_fft
        sp = F.pad(s, (hop // 2, hop // 2))
        refl = F.pad(sp, (n_fft // 2, n_fft // 2), mode="reflect")
        starts = torch.arange(T, device=s.device) * hop                    # [T]
        offs = torch.arange(n_fft, device=s.device)                       # [n_fft]
        frames = refl[0][starts.reshape(-1, 1) + offs.reshape(1, -1)]     # [T, n_fft]
        framed = frames.to(torch.float64).unsqueeze(0) * self.voc.src.window.to(torch.float64).reshape(1, 1, -1)
        re = torch.matmul(framed, self.dft_cos_t)               # [1, T, nb]
        im = -torch.matmul(framed, self.dft_sin_t)              # [1, T, nb]
        return torch.cat([re.transpose(1, 2), im.transpose(1, 2)], 1).to(s.dtype)

    def istft_ola(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Spectral head output [1, n_fft + 2, T] -> wav [1, T*hop]."""
        voc = self.voc
        hop, n_fft = self.hop, self.n_fft
        nb = n_fft // 2 + 1
        # complex spectrum (magnitude, phase) -> real/imag, then the
        # real-ifft basis: x[n] = sum_k c_k (ReX_k cos - ImX_k sin) / n.
        # The polar step stays float32 (as the reference's complex_spec);
        # the basis matmul runs in float64 for accuracy — ORT ships no
        # double Cos kernel.
        mag = torch.exp(x[:, :nb]).clamp(max=1e2)
        pha = x[:, nb:]
        re_x = (mag * torch.cos(pha)).to(torch.float64)
        im_x = (mag * torch.sin(pha)).to(torch.float64)
        td = (torch.matmul(self.idft_cos_t, re_x)
              - torch.matmul(self.idft_sin_t, im_x))  # [1, n_fft, T]
        frames = td.to(x.dtype) * voc.window.reshape(1, -1, 1)

        # overlap-add of windowed frames and of the squared window, as a
        # strided transposed convolution: input channel c carries sample c
        # of each frame, and the identity kernel places it at offset c from
        # every t*hop — one op per accumulation buffer, no explicit loop
        place = torch.eye(n_fft, device=x.device, dtype=x.dtype).reshape(n_fft, 1, n_fft)
        ybuf = F.conv_transpose1d(frames, place, stride=hop)        # [1, 1, (T-1)*hop + n_fft]
        w2 = (voc.window * voc.window).reshape(1, 1, n_fft).to(x.dtype)
        ones = torch.ones(1, 1, frames.size(2), device=x.device, dtype=x.dtype)
        env = F.conv_transpose1d(ones, w2, stride=hop)              # [1, 1, (T-1)*hop + n_fft]
        y = ybuf / env
        o0 = torch.tensor(hop // 2 + n_fft // 2, device=x.device)
        o1 = o0 + T * hop
        return y.reshape(-1)[o0:o1].reshape(1, -1)

    def forward(self, input: torch.Tensor, speed: torch.Tensor):
        """input [1, L] int64 (leading pad included), speed [1] float32.

        Returns (wav [1, T*hop] float32, durations [1, L] int64,
                 mel [1, n_mels, T] float32, f0 [1, T] float32).
        """
        h, s, logd = self.encode(input)
        dur = (torch.round(torch.exp(logd) / speed).clamp(min=1)).to(torch.long)
        hf = self.regulate(h, dur)
        T = torch.cumsum(dur[0], 0)[-1]
        mel, f0hz = self.curves_and_mel(hf, s)
        f0hz = self.voc.clean_f0(f0hz)
        xin = self.voc.feats(mel, f0hz)
        smp = self.source_signal(f0hz, T)
        hfeat = self.source_feats(smp, T)
        spec = self.voc.spec(xin, None, hfeat)
        wav = self.istft_ola(spec, T)
        return wav, dur, mel, f0hz
