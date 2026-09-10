"""Proxecto Nós StyleTTS2 (Galician)  ->  phoonnx single-style ONNX.

Exports the two Galician StyleTTS2 voices published by Proxecto Nós --
``proxectonos/Nos_StyleTTS2-Brais-GL`` (male) and
``proxectonos/Nos_StyleTTS2-Celtia-GL`` (female), both Apache-2.0 -- into the
contract the phoonnx ``StyleTTS2Adapter`` consumes:

    model.onnx          tokens(int64) + style(1,256) + speed(1)  ->  waveform
    style_encoder.onnx  waveform(1,T @24kHz)                     ->  ref_p[128], ref_s[128]
    <voice>.bin         the baked 256-d style of that speaker (float32)
    config.json         phoneme_id_map + engine/alphabet metadata

Same lineage as ``export_bsc.py`` (yl4579 StyleTTS2, diffusion sampler bypassed and
the style supplied as a graph input), so the four ONNX-export rewrites documented
there apply unchanged: manual InstanceNorm, pack_padded_sequence-free Text/Duration
encoders, eager-attention PL-BERT without an attention mask, and a dynamic all-False
text mask. Beyond those, the Nós models differ from the BSC ones in five ways:

1. **iSTFTNet decoder, not hifigan.** ``torch.stft``/``torch.istft`` build complex
   tensors, which the exporter cannot represent ("Unknown number type: complex").
   ``TorchSTFT.transform``/``.inverse`` are therefore replaced with real-arithmetic
   equivalents (conv1d DFT forward; IDFT matmul + overlap-add inverse with the
   window-square normalisation torch.istft applies). Verified against the torch ops
   to ~2e-6 max abs error. The hifigan one-frame shift that ``export_bsc.py`` bakes
   in is *not* applied here -- the yl4579 istftnet path does not use it.
2. **Single speaker.** There is no multispeaker id and no zero-shot cloning: the
   style is computed once from a reference clip of that speaker (the Nós
   ``compute_style``: style_encoder ++ predictor_encoder over the trimmed 24 kHz mel)
   and shipped as a ``.bin``, exactly like the HiTZ Basque StyleTTS2 voices.
   The diffusion style sampler is bypassed entirely, so synthesis is deterministic
   up to the NSF source's own noise.
3. **PL-BERT and the aligner ASR ship inside the model repo** (``Models/galician/``),
   not in separate HF repos, and the PL-BERT is an Albert with a 69-symbol phoneme
   vocab (the README's "PL-ModernBERT" wording notwithstanding -- the shipped
   ``config.yml`` is an AlbertConfig).
4. **Cotovía phoneme set, not espeak IPA.** The 69-symbol vocab is derived from the
   repo's ``phoneme_token_maps.json`` in insertion order, then the
   ``text_utils_gal`` special tokens are appended -- reproducing ``TextCleanerGal``
   exactly. Token 0 is the unknown symbol ``X`` and the sequence is prefixed with a
   single blank (the id of ``" "``), which is the Nós convention, not the yl4579
   ``$``-padding.
5. **The pitch extractor checkpoint (``Utils/JDC/bst.t7``) is not shipped.** It is
   off the inference path (F0 comes from ``predictor.F0Ntrain``), so a freshly
   initialised ``JDCNet`` is passed to ``build_model`` purely to satisfy its
   signature; its weights are then overwritten by the 2nd-stage checkpoint anyway.

Prereqs (CPU is fine -- tracing only):
    pip install torch torchaudio onnx onnxruntime huggingface_hub transformers \
                munch pyyaml librosa soundfile scipy
Run:
    python export_nos_gl.py brais  /out/nos-gl-brais
    python export_nos_gl.py celtia /out/nos-gl-celtia

Parity (torch <-> onnxruntime waveform correlation, same style and tokens):
  * celtia  0.9998 at 59 tokens, 0.9999 at 18 -- above the >=0.99 bar.
  * brais   0.94   at 59 tokens, 0.93 at 18, reproducible across runs, with the
    onnxruntime waveform ~20% down on RMS. The stochastic floor (two torch runs, or
    two onnxruntime runs, of the same input) is 0.9975, so this is a real
    divergence, not the NSF source noise; it survives zeroing that noise. Since the
    graph, architecture and export code are byte-identical between the two voices
    and only the weights differ, it is a weight-dependent numerical sensitivity
    somewhere in the iSTFTNet/NSF decoder that is still unexplained. **Do not ship
    the brais export until this is understood.**
"""
import sys, os, json, glob, re, argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# Nós/yl4579 checkpoints pickle non-tensor globals; torch>=2.6 defaults weights_only=True.
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
from munch import munchify
from huggingface_hub import snapshot_download, hf_hub_download

STYLE_DIM = 128
SR = 24000

VOICES = {
    # voice: (model_repo, dataset_repo, reference_clip)
    "brais": ("proxectonos/Nos_StyleTTS2-Brais-GL", "proxectonos/Nos_Brais-GL",
              "audio/train_1/brais-norm-00001.wav"),   # the Nós "normal_reference"
    "celtia": ("proxectonos/Nos_StyleTTS2-Celtia-GL", "proxectonos/Nos_Celtia-GL",
               "audio/test/norm_nos_gl_celtia_01832.wav"),
}

# ---------------------------------------------------------------------------
# ONNX-exportable STFT/iSTFT (iSTFTNet decoder)
# ---------------------------------------------------------------------------

def _stft_basis(n_fft, window):
    k = torch.arange(n_fft // 2 + 1).unsqueeze(1).float()
    n = torch.arange(n_fft).unsqueeze(0).float()
    ang = 2 * np.pi * k * n / n_fft
    return (torch.cos(ang) * window).unsqueeze(1), (-torch.sin(ang) * window).unsqueeze(1)


def _stft_transform(self, x):
    """torch.stft (center=True, reflect pad, onesided) without complex tensors."""
    n_fft, hop = self.filter_length, self.hop_length
    w = self.window.to(x.device)
    cos, sin = _stft_basis(n_fft, w)
    xp = F.pad(x.unsqueeze(1), (n_fft // 2, n_fft // 2), mode="reflect")
    re = F.conv1d(xp, cos.to(x.device), stride=hop)
    im = F.conv1d(xp, sin.to(x.device), stride=hop)
    return torch.sqrt(re ** 2 + im ** 2 + 1e-12), torch.atan2(im, re)


def _stft_inverse(self, magnitude, phase):
    """torch.istft without complex tensors: IDFT matmul + windowed overlap-add."""
    n_fft, hop = self.filter_length, self.hop_length
    w = self.window.to(magnitude.device)
    re = magnitude * torch.cos(phase)
    im = magnitude * torch.sin(phase)
    k = torch.arange(n_fft // 2 + 1, device=magnitude.device).unsqueeze(0).float()
    n = torch.arange(n_fft, device=magnitude.device).unsqueeze(1).float()
    c = torch.full((1, n_fft // 2 + 1), 2.0, device=magnitude.device)
    c[0, 0] = c[0, -1] = 1.0                       # DC and Nyquist are not mirrored
    ang = 2 * np.pi * k * n / n_fft
    frames = torch.matmul((c * torch.cos(ang)) / n_fft, re) + \
             torch.matmul((-c * torch.sin(ang)) / n_fft, im)
    frames = frames * w.view(1, -1, 1)
    eye = torch.eye(n_fft, device=magnitude.device).unsqueeze(1)
    y = F.conv_transpose1d(frames, eye, stride=hop)
    wsq = F.conv_transpose1d((w ** 2).view(1, -1, 1) * torch.ones_like(frames), eye, stride=hop)
    y = y / (wsq + 1e-11)
    return y[:, :, n_fft // 2: n_fft // 2 + (frames.shape[-1] - 1) * hop]


# ---------------------------------------------------------------------------
# yl4579 modules made exportable (see export_bsc.py for the rationale)
# ---------------------------------------------------------------------------

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
    return x.transpose(-1, -2).masked_fill(m, 0.0)


def _duration_encoder_forward(self, x, style, text_lengths, m, _M):
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


class Synth(torch.nn.Module):
    """tokens + style(1,256) + speed -> waveform (diffusion sampler bypassed).

    ``style[:, :128]`` is the acoustic style (decoder), ``style[:, 128:]`` the
    prosodic one (predictor) -- the ordering the Nós ``compute_style`` produces and
    the phoonnx styletts2 speaker encoder emits. ``shift`` applies the one-frame
    hifigan alignment shift; it stays off for the istftnet decoder.
    """

    def __init__(self, model, shift=False):
        super().__init__()
        self.m = model
        self.shift = shift

    def forward(self, tokens, style, speed):
        m = self.m
        acoustic, prosodic = style[:, :STYLE_DIM], style[:, STYLE_DIM:]
        L = torch.tensor([tokens.shape[-1]])
        mask = torch.zeros_like(tokens, dtype=torch.bool)   # keeps the length dynamic
        t_en = m.text_encoder(tokens, L, mask)
        d_en = m.bert_encoder(m.bert(tokens)).transpose(-1, -2)
        d = m.predictor.text_encoder(d_en, prosodic, L, mask)
        x, _ = m.predictor.lstm(d)
        duration = torch.sigmoid(m.predictor.duration_proj(x)).sum(dim=-1) / speed
        pred_dur = torch.round(duration.squeeze()).clamp(min=1).long()
        ends = torch.cumsum(pred_dur, dim=0)
        frames = torch.arange(ends[-1])
        aln = ((frames[None, :] >= (ends - pred_dur)[:, None]) &
               (frames[None, :] < ends[:, None])).float()
        en = d.transpose(-1, -2) @ aln[None]
        asr = t_en @ aln[None]
        if self.shift:
            en = torch.cat([en[:, :, :1], en[:, :, :-1]], dim=2)
            asr = torch.cat([asr[:, :, :1], asr[:, :, :-1]], dim=2)
        F0_pred, N_pred = m.predictor.F0Ntrain(en, prosodic)
        return m.decoder(asr, F0_pred, N_pred, acoustic).squeeze()


class Mel(torch.nn.Module):
    """yl4579 to_mel as a conv1d-DFT so the style encoder exports (no STFT op)."""

    def __init__(self):
        super().__init__()
        import torchaudio
        n_fft, win, self.hop, self.n_fft = 2048, 1200, 300, 2048
        w = F.pad(torch.hann_window(win), ((n_fft - win) // 2, n_fft - win - (n_fft - win) // 2))
        cos, sin = _stft_basis(n_fft, w)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.register_buffer("fb", torchaudio.transforms.MelSpectrogram(
            n_fft=n_fft, win_length=win, hop_length=self.hop, n_mels=80).mel_scale.fb.t().contiguous())

    def forward(self, x):
        x = F.pad(x.unsqueeze(1), (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        p = F.conv1d(x, self.cos, stride=self.hop) ** 2 + F.conv1d(x, self.sin, stride=self.hop) ** 2
        return ((torch.log(1e-5 + torch.matmul(self.fb, p)) + 4) / 4).unsqueeze(1)


def fix_negative_transpose_perms(path):
    """torch's exporter can emit Transpose perms with negative indices; ORT rejects
    them. rank == len(perm) for a Transpose, so v % len(perm) is the fix."""
    import onnx
    m = onnx.load(path)
    changed = 0
    for node in m.graph.node:
        if node.op_type != "Transpose":
            continue
        for attr in node.attribute:
            if attr.name == "perm" and any(v < 0 for v in attr.ints):
                attr.ints[:] = [v % len(attr.ints) for v in attr.ints]
                changed += 1
    if changed:
        onnx.save(m, path)


# ---------------------------------------------------------------------------
# Cotovía tokenisation (reproduces TextCleanerGal + Nós clean_output)
# ---------------------------------------------------------------------------

_SPECIALS = list('!"(),./:;?[]{}¡ª°´º¿\'')
_DIGRAPHS = {"rr": "R", "ll": "Z", "nh": "N", "ch": "C", "ao": "O", "tS": "W"}
_ACUTE = {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú",
          "A": "Á", "E": "É", "I": "Í", "O": "Ó", "U": "Ú"}


def symbols(repo_dir):
    """The 69-symbol vocab, in TextCleanerGal order (json insertion order, then
    the special tokens that are not already present)."""
    tokmap = json.load(open(os.path.join(repo_dir, "phoneme_token_maps.json")))
    syms = []
    for v in tokmap.values():
        if v["phoneme"] not in syms:
            syms.append(v["phoneme"])
    for s in _SPECIALS:
        if s not in syms:
            syms.append(s)
    return syms


def clean_output(out):
    """Nós ``Utils/ASR/AuxiliaryASR/phonemize.clean_output``: fold the ``V^`` stress
    marker into an acute-accented vowel and the multi-char phonemes into single
    symbols."""
    out = re.sub(r"([aeiouAEIOU])\^", lambda m: _ACUTE[m.group(1)],
                 out.replace("-", "").lstrip("\t "))

    def norm(word):
        o, i = [], 0
        while i < len(word):
            if i + 1 < len(word) and word[i:i + 2].lower() in _DIGRAPHS:
                o.append(_DIGRAPHS[word[i:i + 2].lower()])
                i += 2
                continue
            o.append(word[i])
            i += 1
        return "".join(o)

    return " ".join(norm(w) for w in out.strip().split())


def phonemize(text):
    """Cotovía with the stressed vowel marked (``tra=2``), then Nós' clean_output."""
    from pycotovia import Phonemizer
    return clean_output(Phonemizer().phonemize(text, tra=2))


def tokenize(text, syms):
    sym2id = {s: i for i, s in enumerate(syms)}
    ps = phonemize(text)
    ids = [sym2id[" "]] + [sym2id.get(c, sym2id["X"]) for c in ps]   # leading blank
    return ps, ids


# ---------------------------------------------------------------------------

def load(repo_dir, voice):
    from models import build_model, load_ASR_models
    import models as _M
    from Utils.JDC.model import JDCNet
    import Modules.istftnet as _IS
    from transformers import AlbertConfig, AlbertModel

    _IS.TorchSTFT.transform = _stft_transform
    _IS.TorchSTFT.inverse = _stft_inverse
    torch.nn.InstanceNorm1d.forward = _instance_norm_forward
    torch.nn.InstanceNorm2d.forward = _instance_norm_forward
    _M.TextEncoder.forward = _text_encoder_forward
    _M.DurationEncoder.forward = lambda s, x, st, tl, m: _duration_encoder_forward(s, x, st, tl, m, _M)

    cfg = yaml.safe_load(open(f"{repo_dir}/Models/galician/{voice}/config.yml"))
    text_aligner = load_ASR_models(f"{repo_dir}/Models/galician/ASR/epoch_00080.pth",
                                  f"{repo_dir}/Models/galician/ASR/config.yml")
    pitch_extractor = JDCNet(num_class=1, seq_len=192)   # off the inference path

    class CustomAlbert(AlbertModel):
        def forward(self, *a, **k):
            return super().forward(*a, **k).last_hidden_state

    pcfg = yaml.safe_load(open(f"{repo_dir}/Models/galician/PLBERT/config.yml"))
    bert = CustomAlbert(AlbertConfig(**pcfg["model_params"], attn_implementation="eager"))

    model = build_model(munchify(cfg["model_params"]), text_aligner, pitch_extractor, bert)
    ckpt = sorted(glob.glob(f"{repo_dir}/Models/galician/{voice}/epoch_2nd_*.pth"))[-1]
    params = torch.load(ckpt, map_location="cpu")["net"]
    for k in model:
        if k in params:
            try:
                model[k].load_state_dict(params[k])
            except Exception:
                # DataParallel-saved -> "module."-prefixed keys; without stripping,
                # every module silently loads ZERO weights.
                model[k].load_state_dict(
                    OrderedDict((kk[7:] if kk.startswith("module.") else kk, vv)
                                for kk, vv in params[k].items()), strict=False)
        model[k].eval()
        model[k].requires_grad_(False)
    return model, cfg


def compute_style(model, wav_path):
    """Nós ``compute_style``: trimmed 24 kHz mel -> style_encoder ++ predictor_encoder."""
    import librosa, torchaudio
    wave, _ = librosa.load(wav_path, sr=SR)
    wave, _ = librosa.effects.trim(wave, top_db=30)
    to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
    mel = (torch.log(1e-5 + to_mel(torch.from_numpy(wave).float()).unsqueeze(0)) + 4) / 4
    return torch.cat([model.style_encoder(mel.unsqueeze(1)),
                      model.predictor_encoder(mel.unsqueeze(1))], dim=1)


SAMPLE = "Este é un sistema de conversión de texto a voz en lingua galega."
SAMPLE_SHORT = "Bo día, como estás?"


def export(voice, out):
    import soundfile as sf
    import onnxruntime as ort

    repo, dataset, ref_clip = VOICES[voice]
    # only the model code + checkpoints; skipping Utils/cotovia keeps this at a few
    # hundred MB instead of ~4.6 GB of phonemizer data we do not use (phonemization
    # goes through pycotovia).
    repo_dir = snapshot_download(repo, allow_patterns=[
        "*.py", "Modules/**", "Utils/ASR/*", "Utils/JDC/*", "Utils/PLBERT/*",
        "Models/galician/**", "phoneme_token_maps.json", "Configs/*",
    ])
    ref_wav = hf_hub_download(dataset, ref_clip, repo_type="dataset")
    sys.path.insert(0, repo_dir)
    os.chdir(repo_dir)                     # the Nós repo is a StyleTTS2 clone
    os.makedirs(out, exist_ok=True)
    torch.set_grad_enabled(False)

    model, cfg = load(repo_dir, voice)
    decoder_type = cfg["model_params"]["decoder"]["type"]
    style = compute_style(model, ref_wav)
    syms = symbols(repo_dir)
    assert len(syms) == cfg["model_params"]["n_token"], (len(syms), cfg["model_params"]["n_token"])

    ps, ids = tokenize(SAMPLE, syms)
    print(f"phonemes: {ps}", flush=True)
    tokens = torch.LongTensor(ids).unsqueeze(0)
    speed = torch.tensor([1.0])
    synth = Synth(model, shift=(decoder_type == "hifigan")).eval()
    ref = synth(tokens, style, speed).numpy()
    sf.write(f"{out}/sample_torch.wav", ref, SR)

    torch.onnx.export(synth, (tokens, style, speed), f"{out}/model.onnx",
                      input_names=["tokens", "style", "speed"], output_names=["waveform"],
                      dynamic_axes={"tokens": {1: "n"}, "waveform": {0: "s"}},
                      opset_version=17, dynamo=False)

    class Enc(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.mel, s.style, s.pred = Mel(), model.style_encoder, model.predictor_encoder

        def forward(s, wav):
            mm = s.mel(wav)
            return s.style(mm), s.pred(mm)

    torch.onnx.export(Enc().eval(), torch.randn(1, SR * 2), f"{out}/style_encoder.onnx",
                      input_names=["waveform"], output_names=["ref_p", "ref_s"],
                      dynamic_axes={"waveform": {1: "s"}}, opset_version=17, dynamo=False)

    fix_negative_transpose_perms(f"{out}/model.onnx")
    fix_negative_transpose_perms(f"{out}/style_encoder.onnx")
    style.numpy().astype(np.float32).tofile(f"{out}/{voice}.bin")

    json.dump({
        "phoonnx_version": "1.0",
        "engine": "styletts2",
        "phoneme_type": "cotovia",
        "alphabet": "cotovia",
        "lang_code": "gl-ES",
        "audio": {"sample_rate": SR},
        "num_symbols": len(syms),
        "num_speakers": 1,
        "num_langs": 1,
        "speaker_id_map": {},
        "lang_id_map": {},
        "add_diacritics": False,
        "inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8},
        "phoneme_id_map": {s: i for i, s in enumerate(syms)},
    }, open(f"{out}/config.json", "w"), ensure_ascii=False, indent=2)

    sess = ort.InferenceSession(f"{out}/model.onnx", providers=["CPUExecutionProvider"])
    feed = {"tokens": tokens.numpy().astype(np.int64),
            "style": style.numpy().astype(np.float32),
            "speed": speed.numpy().astype(np.float32)}
    got = sess.run(None, feed)[0].reshape(-1)
    sf.write(f"{out}/sample_onnx.wav", got, SR)
    n = min(len(got), len(ref))
    print(f"PARITY corr={float(np.corrcoef(got[:n], ref[:n])[0, 1]):.4f} "
          f"dur={len(got)/SR:.3f}s peak={np.abs(got).max():.4f} "
          f"rms={np.sqrt((got ** 2).mean()):.4f}", flush=True)

    # the stochastic floor: the NSF source draws its own noise on every run
    again = sess.run(None, feed)[0].reshape(-1)
    n2 = min(len(again), len(got))
    print(f"NOISE_FLOOR corr={float(np.corrcoef(again[:n2], got[:n2])[0, 1]):.4f}", flush=True)

    # dynamic-length smoke test
    _, ids2 = tokenize(SAMPLE_SHORT, syms)
    t2 = torch.LongTensor(ids2).unsqueeze(0)
    g2 = sess.run(None, {**feed, "tokens": t2.numpy().astype(np.int64)})[0].reshape(-1)
    r2 = synth(t2, style, speed).numpy()
    m2 = min(len(g2), len(r2))
    print(f"PARITY_LEN{len(ids2)} corr={float(np.corrcoef(g2[:m2], r2[:m2])[0, 1]):.4f} "
          f"onnx_len={len(g2)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("voice", choices=sorted(VOICES))
    ap.add_argument("out")
    a = ap.parse_args()
    export(a.voice, a.out)
