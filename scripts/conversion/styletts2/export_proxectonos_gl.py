"""ProxectoNos Galician StyleTTS2 (Celtia / Brais) -> phoonnx ONNX.

Exports the yl4579-architecture StyleTTS2 checkpoints published by Proxecto Nos
(https://huggingface.co/proxectonos) into the two-graph contract the phoonnx
``StyleTTS2Adapter`` consumes -- the same contract ``export_bsc.py`` emits:

    model.onnx          tokens(int64) + style(1,256) + speed(1)  ->  waveform
    style_encoder.onnx  waveform(1,T @24kHz)                     ->  ref_p[128], ref_s[128]
    style.bin           float32[1,256] default speaker style (see below)

Both voices are SINGLE-speaker (``multispeaker: false``), so a reference clip is
optional here: ``style.bin`` holds the speaker's own style, averaged over clips
from the published ``proxectonos/Nos_{Celtia,Brais}-GL`` test splits, and the
voice renders with it out of the box. Passing a reference clip still works
through ``style_encoder.onnx`` (zero-shot cloning).

Differences from the BSC export -- all read off the checkpoints, not assumed:
  * decoder is **istftnet**, not hifigan.  Two consequences: (a) no hifigan
    frame-shift on ``en``/``asr`` (upstream ``inference.py`` does not apply it),
    and (b) ``torch.stft``/``torch.istft`` inside ``Modules/istftnet.py`` have no
    ONNX op, so ``TorchSTFT`` is replaced by a conv1d-DFT forward transform and a
    conv_transpose1d overlap-add inverse (exact, not an approximation --
    ``_assert_stft_equivalence`` proves both directions and the round-trip
    against ``torch.stft``/``torch.istft`` at export time, n_fft=20).
  * vocabulary is the 69-symbol Galician Cotovia phoneset from
    ``phoneme_token_maps.json`` -- NOT the yl4579 178-symbol espeak-IPA set.
  * PL-BERT is the ALBERT-style PL-BERT (verified: the checkpoint's ``bert``
    keys are ``encoder.albert_layer_groups.*``), config from
    ``Models/galician/PLBERT/config.yml``.  The weights come from the StyleTTS2
    checkpoint itself, so the 1 GB ``step_1000000.t7`` is never needed.
    (``proxectonos/PL-ModernBERT-gl`` is a *newer, separate* Galician PL-BERT; it
    is NOT what these two checkpoints embed.)
  * the aligner/pitch-extractor are the repo's own Galician ASR + JDC; they are
    off the inference path and only shape ``build_model``.

The remaining yl4579 ONNX gotchas (InstanceNorm, pack_padded_sequence, eager
PL-BERT attention, dynamic text mask, ``module.`` prefix stripping, negative
Transpose perms) are reused from ``export_bsc`` -- import it, do not re-derive.

Prereqs (CPU is fine -- tracing only):
    pip install torch torchaudio onnx onnxruntime huggingface_hub transformers munch pyyaml
Run from inside a checkout of the voice repo (it ships models.py + Modules/ + Utils/):
    huggingface-cli download proxectonos/Nos_StyleTTS2-Celtia-GL --local-dir celtia
    cd celtia && python /path/to/export_proxectonos_gl.py celtia /out/proxectonos-gl-celtia
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# export_bsc installs the shared yl4579 ONNX monkeypatches at import time and
# pulls models.py out of the CWD checkout; import it before touching models.
from gl_vocab import build_phoneme_id_map  # noqa: E402
from export_bsc import (  # noqa: E402
    STYLE_DIM,
    SR,
    _Mel,
    _fix_negative_transpose_perms,
)
import models as _M  # noqa: E402
from models import build_model, load_ASR_models  # noqa: E402
from munch import munchify  # noqa: E402
import yaml  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

VOICES = {
    # name: (hf repo, checkpoint path inside the repo)
    "celtia": ("proxectonos/Nos_StyleTTS2-Celtia-GL",
               "Models/galician/celtia/epoch_2nd_00057.pth",
               "Models/galician/celtia/config.yml"),
    "brais": ("proxectonos/Nos_StyleTTS2-Brais-GL",
              "Models/galician/brais/epoch_2nd_00057.pth",
              "Models/galician/brais/config.yml"),
}

# Published recordings of each speaker; the shipped default style is averaged
# over the first clips of the test split.
STYLE_REFS = {
    "celtia": "proxectonos/Nos_Celtia-GL",
    "brais": "proxectonos/Nos_Brais-GL",
}

# ---------------------------------------------------------------------------
# ONNX-exportable STFT / iSTFT for the istftnet decoder
# ---------------------------------------------------------------------------


class _OnnxSTFT(torch.nn.Module):
    """Drop-in replacement for ``Modules.istftnet.TorchSTFT``.

    ``torch.stft``/``torch.istft`` have no ONNX lowering, so both directions are
    rebuilt from primitives that do:

    * forward: reflect-pad (``center=True``) then a conv1d against the windowed
      cos/sin DFT basis -> magnitude/phase, exactly ``torch.stft``;
    * inverse: hermitian-extended real IDFT as a matmul, window, overlap-add via
      ``conv_transpose1d`` with an identity kernel, divided by the window-square
      OLA envelope, then the ``center=True`` trim.

    n_fft here is 20, so the DFT matrices are tiny and the rewrite costs nothing.
    """

    def __init__(self, filter_length: int, hop_length: int, win_length: int):
        super().__init__()
        self.n_fft = int(filter_length)
        self.hop = int(hop_length)
        self.win_length = int(win_length)
        n_fft, win = self.n_fft, self.win_length
        w = torch.hann_window(win, periodic=True)
        if win < n_fft:  # torch.stft centre-pads a short window
            pad = n_fft - win
            w = torch.nn.functional.pad(w, (pad // 2, pad - pad // 2))
        k = torch.arange(n_fft // 2 + 1).unsqueeze(1).float()
        n = torch.arange(n_fft).unsqueeze(0).float()
        ang = 2 * np.pi * k * n / n_fft
        # forward basis: [n_bins, 1, n_fft] conv1d kernels
        self.register_buffer("fwd_cos", (torch.cos(ang) * w).unsqueeze(1), persistent=False)
        self.register_buffer("fwd_sin", (-torch.sin(ang) * w).unsqueeze(1), persistent=False)
        # inverse basis: hermitian weights so the one-sided spectrum reconstructs
        # the real frame; [n_bins, n_fft]
        wk = torch.full((n_fft // 2 + 1, 1), 2.0)
        wk[0] = 1.0
        if n_fft % 2 == 0:
            wk[-1] = 1.0
        self.register_buffer("inv_cos", (wk * torch.cos(ang)) / n_fft, persistent=False)
        self.register_buffer("inv_sin", (wk * torch.sin(ang)) / n_fft, persistent=False)
        self.register_buffer("window", w, persistent=False)
        self.register_buffer("eye", torch.eye(n_fft).unsqueeze(1), persistent=False)
        self.register_buffer("win_sq", (w * w).reshape(1, n_fft, 1), persistent=False)

    def transform(self, x):
        pad = self.n_fft // 2
        xp = torch.nn.functional.pad(x.unsqueeze(1), (pad, pad), mode="reflect")
        re = torch.nn.functional.conv1d(xp, self.fwd_cos, stride=self.hop)
        im = torch.nn.functional.conv1d(xp, self.fwd_sin, stride=self.hop)
        mag = torch.sqrt(re * re + im * im + 1e-12)
        phase = torch.atan2(im, re)
        return mag, phase

    def inverse(self, magnitude, phase):
        re = magnitude * torch.cos(phase)
        im = magnitude * torch.sin(phase)
        # [B, n_bins, T] -> [B, n_fft, T] real frames
        frames = (self.inv_cos.t() @ re) - (self.inv_sin.t() @ im)
        frames = frames * self.window.reshape(1, -1, 1)
        y = torch.nn.functional.conv_transpose1d(frames, self.eye, stride=self.hop)
        env = torch.nn.functional.conv_transpose1d(
            self.win_sq.expand(frames.shape[0], -1, frames.shape[-1]),
            self.eye, stride=self.hop)
        y = y / (env + 1e-11)
        pad = self.n_fft // 2
        return y[..., pad:-pad]

    def forward(self, x):
        mag, phase = self.transform(x)
        return self.inverse(mag, phase)


def _assert_stft_equivalence():
    """Prove ``_OnnxSTFT`` reproduces ``torch.stft``/``torch.istft`` exactly,
    before ``_patch_istftnet`` swaps it in for every Decoder built afterwards.

    n_fft is 20 (istftnet's own filter length), so the whole check is cheap
    enough to run on every export: random and structured signals, several
    lengths, forward vs ``torch.stft``, inverse vs ``torch.istft`` fed the
    torch spectrum directly (not round-tripped, to isolate the inverse), and
    a pure forward->inverse round-trip against the raw input.
    """
    n_fft, hop, win = 20, 5, 20
    stft = _OnnxSTFT(n_fft, hop, win)
    torch.manual_seed(0)
    gens = {
        "random": lambda n: torch.randn(2, n),
        "sine": lambda n: torch.sin(torch.linspace(0, 40 * np.pi, n)).unsqueeze(0).repeat(2, 1) * 0.7,
        "impulse": lambda n: torch.nn.functional.pad(torch.ones(2, 1), (0, n - 1)),
        "silence": lambda n: torch.zeros(2, n),
    }
    for length in (37, 80, 401):
        for name, gen in gens.items():
            x = gen(length)

            ref = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                              window=stft.window, center=True, return_complex=True,
                              pad_mode="reflect")
            ref_mag, ref_phase = ref.abs(), ref.angle()
            mag, phase = stft.transform(x)
            k = min(mag.shape[-1], ref_mag.shape[-1])

            mag_err = (mag[..., :k] - ref_mag[..., :k]).abs().max().item()
            re_err = (mag[..., :k] * torch.cos(phase[..., :k]) -
                      ref_mag[..., :k] * torch.cos(ref_phase[..., :k])).abs().max().item()
            im_err = (mag[..., :k] * torch.sin(phase[..., :k]) -
                      ref_mag[..., :k] * torch.sin(ref_phase[..., :k])).abs().max().item()
            assert mag_err < 1e-4, \
                f"STFT magnitude diverges from torch.stft ({name}, n={length}): {mag_err:.2e}"
            assert max(re_err, im_err) < 1e-4, \
                f"STFT complex value diverges from torch.stft ({name}, n={length}): re={re_err:.2e} im={im_err:.2e}"

            y_ref = torch.istft(ref_mag * torch.exp(1j * ref_phase), n_fft=n_fft,
                                 hop_length=hop, win_length=win, window=stft.window,
                                 center=True, length=length)
            y = stft.inverse(mag, phase).squeeze(1)
            k2 = min(y.shape[-1], y_ref.shape[-1])
            inv_err = (y[..., :k2] - y_ref[..., :k2]).abs().max().item()
            assert inv_err < 1e-4, \
                f"iSTFT diverges from torch.istft ({name}, n={length}): {inv_err:.2e}"

            rt = stft(x).squeeze(1)
            k3 = min(rt.shape[-1], x.shape[-1])
            rt_err = (rt[..., :k3] - x[..., :k3]).abs().max().item()
            assert rt_err < 1e-4, \
                f"STFT round-trip diverges from input ({name}, n={length}): {rt_err:.2e}"
    print("STFT_ASSERT ok: forward vs torch.stft, inverse vs torch.istft, and "
          "round-trip all match within 1e-4 (n_fft=20)", flush=True)


def _patch_istftnet():
    """Swap TorchSTFT for the ONNX-exportable one, before any Decoder is built."""
    _assert_stft_equivalence()
    from Modules import istftnet as _istft
    _istft.TorchSTFT = _OnnxSTFT


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _load_plbert(plbert_cfg_path):
    """ALBERT-style PL-BERT shell; the weights come from the StyleTTS2 checkpoint.

    Eager attention + no attention_mask, as in export_bsc: SDPA's mask handling
    bakes the sequence length into the graph.
    """
    from transformers import AlbertConfig, AlbertModel

    class CustomAlbert(AlbertModel):
        def forward(self, *a, **k):
            return super().forward(*a, **k).last_hidden_state

    cfg = yaml.safe_load(open(plbert_cfg_path))
    return CustomAlbert(AlbertConfig(**cfg["model_params"], attn_implementation="eager"))


def load(voice):
    repo, ckpt_name, cfg_name = VOICES[voice]
    _patch_istftnet()
    ckpt = hf_hub_download(repo, ckpt_name)
    cfg = yaml.safe_load(open(hf_hub_download(repo, cfg_name)))
    assert cfg["model_params"]["decoder"]["type"] == "istftnet", \
        "Synth assumes the istftnet decoder (no hifigan frame-shift)"
    assert cfg["model_params"]["multispeaker"] is False

    text_aligner = load_ASR_models(hf_hub_download(repo, "Models/galician/ASR/epoch_00080.pth"),
                                   hf_hub_download(repo, "Models/galician/ASR/config.yml"))
    # Utils/JDC/bst.t7 is not published with these repos; the pitch extractor is
    # off the inference path and its weights come from the checkpoint anyway, so
    # build the bare module rather than chasing the upstream yl4579 blob.
    from Utils.JDC.model import JDCNet
    pitch_extractor = JDCNet(num_class=1, seq_len=192)
    plbert = _load_plbert(hf_hub_download(repo, "Models/galician/PLBERT/config.yml"))

    model = build_model(munchify(cfg["model_params"]), text_aligner, pitch_extractor, plbert)
    net = torch.load(ckpt, map_location="cpu")["net"]
    for k in model:
        if k in net:
            from collections import OrderedDict
            # DataParallel-saved -> "module."-prefixed keys; without stripping,
            # every module silently loads ZERO weights (style ~1e18, noise out).
            stripped = OrderedDict((kk[7:] if kk.startswith("module.") else kk, vv)
                                   for kk, vv in net[k].items())
            missing, unexpected = model[k].load_state_dict(stripped, strict=False)
            if k in ("bert", "bert_encoder", "text_encoder", "predictor",
                     "decoder", "style_encoder", "predictor_encoder", "diffusion"):
                assert not [m for m in missing if "position_ids" not in m], \
                    f"{k}: missing weights {missing[:5]}"
        model[k].eval()
        model[k].requires_grad_(False)
    return model, cfg


# ---------------------------------------------------------------------------
# synthesis graph
# ---------------------------------------------------------------------------


class Synth(torch.nn.Module):
    """tokens + style(1,256) + speed -> waveform.

    Upstream ``LFinference`` with the diffusion sampler bypassed (the style is an
    input). ``style[:, :128]`` is the acoustic style handed to the decoder,
    ``style[:, 128:]`` the prosodic style handed to the predictor -- the same
    ``cat([style_encoder(mel), predictor_encoder(mel)])`` order upstream's
    ``compute_style`` produces. No hifigan frame-shift: this is an istftnet
    decoder and upstream does not shift.
    """

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, tokens, style, speed):
        m = self.m
        acoustic = style[:, :STYLE_DIM]
        prosodic = style[:, STYLE_DIM:]
        L = torch.tensor([tokens.shape[-1]])
        # single unpadded sequence -> all-False mask, built from tokens so the
        # length stays DYNAMIC (length_to_mask would bake arange(N) in).
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        t_en = m.text_encoder(tokens, L, mask)
        bert_dur = m.bert(tokens)          # no attention_mask (eager, see _load_plbert)
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
        F0_pred, N_pred = m.predictor.F0Ntrain(en, prosodic)
        asr = t_en @ aln[None]
        out = m.decoder(asr, F0_pred, N_pred, acoustic).squeeze()
        return out[..., :-50]      # upstream trims the decoder's tail click


class Enc(torch.nn.Module):
    """waveform -> ref_p (acoustic) ++ ref_s (prosodic), for zero-shot cloning."""

    def __init__(self, model):
        super().__init__()
        self.mel = _Mel()
        self.style = model.style_encoder
        self.pred = model.predictor_encoder

    def forward(self, wav):
        mm = self.mel(wav)
        return self.style(mm), self.pred(mm)


# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------


def build_vocab(repo):
    """The model's own 69-symbol phoneset, from the training-time token table.

    ``build_phoneme_id_map`` (shared with the test-suite) also folds in the
    multi-character Cotovia surface forms, so a raw
    ``CotoviaPhonemizer(model="stress", alphabet=COTOVIA)`` string tokenises to
    exactly the ids the checkpoint was trained with.
    """
    tm = json.load(open(hf_hub_download(repo, "Utils/ASR/AuxiliaryASR/phoneme_token_maps.json")))
    return build_phoneme_id_map(tm)


def write_config(out, repo, voice):
    vocab = build_vocab(repo)
    cfg = {
        "phoonnx_version": "1.0",
        "engine": "styletts2",
        "phoneme_type": "cotovia",
        "alphabet": "cotovia",
        "phonemizer_model": "stress",
        "lang_code": "gl",
        "audio": {"sample_rate": SR},
        "num_symbols": 69,
        "num_speakers": 1,
        "num_langs": 1,
        "speaker_id_map": {},
        "lang_id_map": {},
        "add_diacritics": False,
        "inference": {"length_scale": 1.0, "noise_scale": 0.667, "noise_w": 0.8},
        "phoneme_id_map": vocab,
        # Pad with "X" (id 0) -- the token the checkpoints were TRAINED with.
        # Upstream is inconsistent with itself here:
        #   meldataset.py:113,135  text.insert(0, 0); text.append(0)   <- training
        #   inference.py:76        tokens.insert(0, textcleaner([" "])[0])  <- id 1
        # The word separator (id 1) is a trained speech symbol, so a leading id 1
        # makes the model speak an extra syllable: on 52 Galician sentences,
        # scored with OpenVoiceOS/Nos_ASR-wav2vec2-xls-r-300m-gl-onnx, id 1 costs
        # Brais 0.196 WER against 0.122 for id 0 and Celtia 0.180 against 0.159.
        # Training also appends the pad; a trailing pad measured no better
        # (Brais 0.125, Celtia 0.176), so keep the start-only padding the adapter
        # already applies to every plain StyleTTS2 voice.
        "pad": "X",
        "blank": None,
        "bos": None,
        "eos": None,
        "add_blank_char": False,
        "add_blank_word": False,
        "use_eos_bos": False,
        "blank_at_start": False,
        "blank_at_end": False,
        "word_sep_token": " ",
        "blank_between": "tokens_and_words",
    }
    json.dump(cfg, open(f"{out}/config.json", "w"), ensure_ascii=False, indent=2)
    return vocab


# ---------------------------------------------------------------------------
# default style from the speaker's own recordings
# ---------------------------------------------------------------------------


def default_style(model, voice):
    """Average ``compute_style`` over a fixed set of the speaker's own clips.

    Upstream ``inference.py`` conditions on reference recordings (its
    ``normal_reference``) and only *blends* the style-diffusion prior on top.
    The prior alone is not a usable style -- sampled without a reference it
    diverges (|style| ~ 1e4) -- so the shipped default is the reference style
    itself, averaged over the first clips of the voice's published test split.
    """
    import io
    import numpy as _np
    import soundfile as sf
    from huggingface_hub import hf_hub_download, list_repo_files

    ds = STYLE_REFS[voice]
    wavs = sorted(f for f in list_repo_files(ds, repo_type="dataset")
                  if f.startswith("audio/test/") and f.endswith(".wav"))[:8]
    mel = _Mel()
    styles = []
    with torch.no_grad():
        for w in wavs:
            audio, sr = sf.read(hf_hub_download(ds, w, repo_type="dataset"), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(1)
            x = torch.from_numpy(_np.ascontiguousarray(audio)).unsqueeze(0)
            if sr != SR:
                # the published clips are 16 kHz; the model runs at 24 kHz. The
                # resampled reference has no energy above 8 kHz, so the default
                # style is slightly darker than a native 24 kHz reference would
                # give -- pass your own 24 kHz clip to synthesize() to beat it.
                import torchaudio
                x = torchaudio.functional.resample(x, sr, SR)
            mm = mel(x)
            styles.append(torch.cat([model.style_encoder(mm), model.predictor_encoder(mm)], dim=1))
    return torch.cat(styles, 0).mean(0, keepdim=True)


# ---------------------------------------------------------------------------


def export(voice, out):
    os.makedirs(out, exist_ok=True)
    torch.set_grad_enabled(False)
    repo = VOICES[voice][0]
    model, _ = load(voice)
    vocab = write_config(out, repo, voice)

    style = default_style(model, voice)
    style.numpy().astype(np.float32).tofile(f"{out}/style.bin")
    print(f"STYLE abs-mean={float(style.abs().mean()):.4f}", flush=True)
    assert float(style.abs().mean()) < 100, "style blew up -> weights did not load"

    synth = Synth(model).eval()
    synth.requires_grad_(False)
    torch.manual_seed(0)
    tokens = torch.randint(1, 69, (1, 40), dtype=torch.long)
    speed = torch.tensor([1.0])
    torch.onnx.export(synth, (tokens, style, speed), f"{out}/model.onnx",
                      input_names=["tokens", "style", "speed"], output_names=["waveform"],
                      dynamic_axes={"tokens": {1: "n"}, "waveform": {0: "s"}},
                      opset_version=17, dynamo=False)
    torch.onnx.export(Enc(model).eval(), torch.randn(1, SR * 2), f"{out}/style_encoder.onnx",
                      input_names=["waveform"], output_names=["ref_p", "ref_s"],
                      dynamic_axes={"waveform": {1: "s"}}, opset_version=17, dynamo=False)
    _fix_negative_transpose_perms(f"{out}/model.onnx")
    _fix_negative_transpose_perms(f"{out}/style_encoder.onnx")

    # --- parity -------------------------------------------------------------
    # Read the correlation as a phase-agreement number, not a pass/fail gate.
    # The istftnet source module draws a random initial phase per harmonic plus
    # gaussian excitation noise, but that alone is worth about 0.998: rerunning
    # either framework against itself on identical inputs gives corr 0.998
    # (torch) and 0.998 (onnxruntime). Brais lands near 0.94 against torch
    # because its phase agreement DECAYS along the utterance (0.97 in the first
    # decile down to 0.86 in the last) while its magnitude spectrum still
    # matches at 0.97 -- a phase-domain drift in the low-F0 sine source, not a
    # spectral defect. It is inaudible to ASR: on 25 Galician sentences through
    # identical token sequences, torch scores 0.101 WER and this ONNX graph
    # 0.106. Celtia's higher F0 keeps it at 0.9996.
    import onnxruntime as ort
    sess = ort.InferenceSession(f"{out}/model.onnx", providers=["CPUExecutionProvider"])

    def feed(t):
        return {"tokens": t.numpy().astype(np.int64),
                "style": style.numpy().astype(np.float32),
                "speed": speed.numpy().astype(np.float32)}

    mel = _Mel()
    for n in (40, 61):
        t = torch.randint(1, 69, (1, n), dtype=torch.long)
        torch.manual_seed(0)
        with torch.no_grad():
            ref = synth(t, style, speed).numpy()
        got = sess.run(None, feed(t))[0].reshape(-1)
        k = min(len(got), len(ref))
        corr = float(np.corrcoef(got[:k], ref[:k])[0, 1])
        with torch.no_grad():
            m1 = mel(torch.from_numpy(ref[:k]).unsqueeze(0))
            m2 = mel(torch.from_numpy(got[:k]).unsqueeze(0))
        print(f"PARITY_LEN{n} corr={corr:.4f} mel_l1={float((m1 - m2).abs().mean()):.4f} "
              f"onnx_len={len(got)} torch_len={len(ref)}", flush=True)

    # --- style encoder parity ----------------------------------------------
    wav = torch.randn(1, SR * 2) * 0.05
    enc = Enc(model).eval()
    with torch.no_grad():
        rp, rs = enc(wav)
    esess = ort.InferenceSession(f"{out}/style_encoder.onnx", providers=["CPUExecutionProvider"])
    grp, grs = esess.run(None, {"waveform": wav.numpy().astype(np.float32)})
    print("ENCODER max_abs_err ref_p={:.2e} ref_s={:.2e}".format(
        float(np.abs(grp - rp.numpy()).max()), float(np.abs(grs - rs.numpy()).max())), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("voice", choices=sorted(VOICES))
    ap.add_argument("out")
    a = ap.parse_args()
    export(a.voice, a.out)
