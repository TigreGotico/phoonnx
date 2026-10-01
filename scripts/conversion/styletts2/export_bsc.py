"""BSC-LT StyleTTS2 multispeaker  ->  phoonnx zero-shot-cloning ONNX.

Exports the yl4579 StyleTTS2 checkpoints published by the Barcelona Supercomputing
Center (Spanish / Catalan multispeaker) into the single-graph contract the phoonnx
``StyleTTS2Adapter`` consumes:

    model.onnx          tokens(int64) + style(1,256) + speed(1)  ->  waveform
    style_encoder.onnx  waveform(1,T @24kHz)                     ->  ref_p[128], ref_s[128]

These are **style-diffusion** multispeaker models with no shipped reference audio, so
they are wired as **zero-shot cloning** voices (``bsc/es-styletts2``,
``bsc/ca-styletts2``): the caller supplies a reference clip, ``style_encoder.onnx``
turns it into the 256-d style (``ref_p`` ++ ``ref_s`` = acoustic ++ prosodic), and
``model.onnx`` renders speech in that voice. The diffusion sampler is therefore
bypassed on the synthesis path -- the style comes in as an input (as in DDATT's
``final_simp.onnx``). ``style[:, :128]`` is the acoustic style (decoder),
``style[:, 128:]`` the prosodic style (predictor).

Vocab / phonemizer: the standard yl4579 178-symbol espeak-IPA set (same map as
``ddatt/en-styletts2``); es/ca are phonemized with espeak. The emitted config.json is
that map with ``lang_code`` set to the language.

Making the yl4579 graph ONNX-exportable required four faithful, batch=1-only rewrites,
all applied here as monkeypatches (see inline notes):
  * InstanceNorm1d/2d -> manual normalization (F.instance_norm needs a static channel);
  * TextEncoder/DurationEncoder -> drop pack_padded_sequence (bakes the token length);
  * PL-BERT (Albert) -> eager attention + no attention_mask (SDPA masking bakes length);
  * dynamic all-False text mask + torch.arange(sum(dur)) so length stays dynamic.
Plus: checkpoints are DataParallel-saved (``module.``-prefixed keys) -> strip the
prefix or every module silently loads zero weights.

Prereqs (CPU is fine -- tracing only):
    pip install torch torchaudio onnx onnxruntime huggingface_hub transformers munch pyyaml
    git clone https://github.com/yl4579/StyleTTS2   # provides models.py + Utils/
Run from inside the StyleTTS2 clone:
    python /path/to/export_bsc.py es /out/bsc-es-styletts2
    python /path/to/export_bsc.py ca /out/bsc-ca-styletts2

Parity: torch<->onnxruntime waveform correlation >= 0.99 at two token lengths, and a
dynamic-length smoke test (no shape error at a length other than the export length).
The definitive check is downstream ASR intelligibility of a real cloned synthesis.
"""
import sys, os, json, argparse
import numpy as np
import torch, yaml
# yl4579/BSC checkpoints pickle non-tensor globals; torch>=2.6 defaults weights_only=True.
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
from munch import munchify
from huggingface_hub import hf_hub_download, snapshot_download

sys.path.insert(0, os.getcwd())  # StyleTTS2 repo
from models import build_model, load_ASR_models, load_F0_models
import models as _M

STYLE_DIM = 128
SR = 24000

MODELS = {
    # lang: (model_repo, ckpt_name|None, config_name|None, plbert_repo, plbert_cfg, plbert_step)
    "es": ("BSC-LT/styletts2-spanish-multispeaker",
           "styletts2-spanish-multispeaker_2nd_phase_epoch_86.pth",
           "styletts2-spanish-multispeaker_2nd_phase_config.yml",
           "BSC-LT/PL-BERT-wp-es", "config_es.yml", "step_1000000_es.t7"),
    "ca": ("BSC-LT/styletts2-catalan-multispeaker", None, None,
           "BSC-LT/PL-BERT-wp-ca", "config_cat.yml", "step_1000000_cat.t7"),
}


def _instance_norm_forward(self, x):
    """Manual instance-norm — F.instance_norm won't ONNX-export with a dynamic channel.
    Normalizes per (sample, channel) over spatial dims; applies affine if present."""
    dims = tuple(range(2, x.dim()))
    mean = x.mean(dims, keepdim=True)
    var = x.var(dims, keepdim=True, unbiased=False)
    xn = (x - mean) / torch.sqrt(var + self.eps)
    if getattr(self, "affine", False):
        shape = [1, -1] + [1] * (x.dim() - 2)
        xn = xn * self.weight.view(*shape) + self.bias.view(*shape)
    return xn


def _text_encoder_forward(self, x, input_lengths, m):
    """yl4579 TextEncoder without pack_padded_sequence (which bakes the token length).
    batch=1, unpadded -> packing is a no-op; run the LSTM on the full sequence."""
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
    """yl4579 DurationEncoder without pack_padded_sequence (see _text_encoder_forward)."""
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


torch.nn.InstanceNorm1d.forward = _instance_norm_forward
torch.nn.InstanceNorm2d.forward = _instance_norm_forward
_M.TextEncoder.forward = _text_encoder_forward
_M.DurationEncoder.forward = _duration_encoder_forward


def load_plbert(log_dir):
    """yl4579 load_plbert, but eager-attention (SDPA masking bakes the sequence length)
    and guarded for checkpoints lacking embeddings.position_ids."""
    from collections import OrderedDict
    from transformers import AlbertConfig, AlbertModel

    class CustomAlbert(AlbertModel):
        def forward(self, *a, **k):
            return super().forward(*a, **k).last_hidden_state

    cfg = yaml.safe_load(open(os.path.join(log_dir, "config.yml")))
    bert = CustomAlbert(AlbertConfig(**cfg["model_params"], attn_implementation="eager"))
    ckpts = [f for f in os.listdir(log_dir) if f.startswith("step_")]
    it = sorted(int(f.split("_")[-1].split(".")[0]) for f in ckpts)[-1]
    net = torch.load(f"{log_dir}/step_{it}.t7", map_location="cpu")["net"]
    sd = OrderedDict()
    for k, v in net.items():
        name = k[7:]
        if name.startswith("encoder."):
            sd[name[8:]] = v
    sd.pop("embeddings.position_ids", None)
    bert.load_state_dict(sd, strict=False)
    return bert


def _setup_plbert(pl_repo, cfg_name, step_name):
    pb = snapshot_download(pl_repo)
    d = f"Utils/PLBERT_{cfg_name.split('_')[-1].split('.')[0]}"  # e.g. Utils/PLBERT_es
    os.makedirs(d, exist_ok=True)
    import shutil
    shutil.copy(f"{pb}/{cfg_name}", f"{d}/config.yml")
    shutil.copy(f"{pb}/{step_name}", f"{d}/step_1000000.t7")
    return d


def load(lang):
    repo, ckpt_name, cfg_name, pl_repo, pl_cfg, pl_step = MODELS[lang]
    if ckpt_name is None or cfg_name is None:
        from huggingface_hub import list_repo_files
        files = list(list_repo_files(repo))
        ckpt_name = ckpt_name or next(f for f in files if f.endswith(".pth"))
        cfg_name = cfg_name or next(f for f in files if "2nd_phase_config" in f and f.endswith(".yml"))
    ckpt = hf_hub_download(repo, ckpt_name)
    cfg = yaml.safe_load(open(hf_hub_download(repo, cfg_name)))
    pl_dir = _setup_plbert(pl_repo, pl_cfg, pl_step)

    # text_aligner / pitch_extractor are OFF the inference path; use the bundled English
    # yl4579 modules just to satisfy build_model (their weights are unused downstream).
    text_aligner = load_ASR_models("Utils/ASR/epoch_00080.pth", "Utils/ASR/config.yml")
    pitch_extractor = load_F0_models("Utils/JDC/bst.t7")
    plbert = load_plbert(pl_dir)

    model = build_model(munchify(cfg["model_params"]), text_aligner, pitch_extractor, plbert)
    assert cfg["model_params"]["decoder"]["type"] == "hifigan", "Synth bakes the hifigan frame-shift"
    net = torch.load(ckpt, map_location="cpu")["net"]
    for k in model:
        if k in net:
            try:
                model[k].load_state_dict(net[k])
            except Exception:
                # checkpoints are DataParallel-saved -> keys are "module."-prefixed;
                # without stripping it every module loads ZERO weights (silent!).
                from collections import OrderedDict
                stripped = OrderedDict((kk[7:] if kk.startswith("module.") else kk, vv)
                                       for kk, vv in net[k].items())
                model[k].load_state_dict(stripped, strict=False)
        model[k].eval()
        model[k].requires_grad_(False)   # Munch isn't an nn.Module -> clear grad per-submodule
    return model, cfg


class Synth(torch.nn.Module):
    """tokens + style(1,256) + speed -> waveform; yl4579 inference with the diffusion
    sampler bypassed (style supplied as input). hifigan decoder -> frame-shift on en/asr."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, tokens, style, speed):
        m = self.m
        acoustic = style[:, :STYLE_DIM]      # style_encoder ref -> decoder
        prosodic = style[:, STYLE_DIM:]      # predictor_encoder ref -> predictor
        L = torch.tensor([tokens.shape[-1]])
        # single unpadded sequence -> all-False mask; build it from tokens so the length
        # stays DYNAMIC (length_to_mask would bake arange(N) as a constant).
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        t_en = m.text_encoder(tokens, L, mask)
        bert_dur = m.bert(tokens)            # no attention_mask (see load_plbert)
        d_en = m.bert_encoder(bert_dur).transpose(-1, -2)

        d = m.predictor.text_encoder(d_en, prosodic, L, mask)
        x, _ = m.predictor.lstm(d)
        duration = torch.sigmoid(m.predictor.duration_proj(x)).sum(dim=-1) / speed
        pred_dur = torch.round(duration.squeeze()).clamp(min=1).long()

        ends = torch.cumsum(pred_dur, dim=0)
        frames = torch.arange(ends[-1])
        aln = ((frames[None, :] >= (ends - pred_dur)[:, None]) &
               (frames[None, :] < ends[:, None])).float()          # [n_tok, total]

        en = d.transpose(-1, -2) @ aln[None]
        en = torch.cat([en[:, :, :1], en[:, :, :-1]], dim=2)        # hifigan shift
        F0_pred, N_pred = m.predictor.F0Ntrain(en, prosodic)
        asr = t_en @ aln[None]
        asr = torch.cat([asr[:, :, :1], asr[:, :, :-1]], dim=2)     # hifigan shift
        return m.decoder(asr, F0_pred, N_pred, acoustic).squeeze()


class _Mel(torch.nn.Module):
    """yl4579 to_mel replica as a conv1d-DFT so it ONNX-exports (no STFT op)."""

    def __init__(self):
        super().__init__()
        import torchaudio
        n_fft, win, self.hop, self.n_fft = 2048, 1200, 300, 2048
        w = torch.nn.functional.pad(torch.hann_window(win), ((n_fft-win)//2, n_fft-win-(n_fft-win)//2))
        k = torch.arange(n_fft//2+1).unsqueeze(1).float(); n = torch.arange(n_fft).unsqueeze(0).float()
        ang = 2*np.pi*k*n/n_fft
        self.register_buffer("cos", (torch.cos(ang)*w).unsqueeze(1))
        self.register_buffer("sin", (-torch.sin(ang)*w).unsqueeze(1))
        self.register_buffer("fb", torchaudio.transforms.MelSpectrogram(
            n_fft=n_fft, win_length=win, hop_length=self.hop, n_mels=80).mel_scale.fb.t().contiguous())

    def forward(self, x):
        x = torch.nn.functional.pad(x.unsqueeze(1), (self.n_fft//2, self.n_fft//2), mode="reflect")
        p = torch.nn.functional.conv1d(x, self.cos, stride=self.hop)**2 + \
            torch.nn.functional.conv1d(x, self.sin, stride=self.hop)**2
        return ((torch.log(1e-5+torch.matmul(self.fb, p))+4)/4).unsqueeze(1)


def _fix_negative_transpose_perms(path):
    """torch's exporter can emit Transpose perms with negative indices (e.g. {1,-1,0})
    from transpose(-1,-2) under dynamic rank; onnxruntime rejects them. Rewrite v<0 to
    v % len(perm) (rank == len(perm) for a Transpose)."""
    import onnx
    m = onnx.load(path)
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
        onnx.save(m, path)


def export(lang, out):
    os.makedirs(out, exist_ok=True)
    torch.set_grad_enabled(False)
    model, _ = load(lang)
    synth = Synth(model).eval(); synth.requires_grad_(False)
    torch.manual_seed(0)
    tokens = torch.randint(1, 178, (1, 40), dtype=torch.long)
    style = torch.randn(1, 2 * STYLE_DIM)
    speed = torch.tensor([1.0])
    with torch.no_grad():
        ref = synth(tokens, style, speed).numpy()
    torch.onnx.export(synth, (tokens, style, speed), f"{out}/model.onnx",
                      input_names=["tokens", "style", "speed"], output_names=["waveform"],
                      dynamic_axes={"tokens": {1: "n"}, "waveform": {0: "s"}},
                      opset_version=17, dynamo=False)

    class Enc(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.mel = _Mel(); s.style = model.style_encoder; s.pred = model.predictor_encoder
        def forward(s, wav):
            mm = s.mel(wav); return s.style(mm), s.pred(mm)
    torch.onnx.export(Enc().eval(), torch.randn(1, SR*2), f"{out}/style_encoder.onnx",
                      input_names=["waveform"], output_names=["ref_p", "ref_s"],
                      dynamic_axes={"waveform": {1: "s"}}, opset_version=17, dynamo=False)

    _fix_negative_transpose_perms(f"{out}/model.onnx")
    _fix_negative_transpose_perms(f"{out}/style_encoder.onnx")

    ddatt = json.load(open(hf_hub_download("OpenVoiceOS/phoonnx-styletts2", "ddatt-en-styletts2/config.json")))
    ddatt["lang_code"] = lang
    json.dump(ddatt, open(f"{out}/config.json", "w"), ensure_ascii=False, indent=2)

    import onnxruntime as ort
    s = ort.InferenceSession(f"{out}/model.onnx", providers=["CPUExecutionProvider"])
    feed = lambda t: {"tokens": t.numpy().astype(np.int64), "style": style.numpy().astype(np.float32),
                      "speed": speed.numpy().astype(np.float32)}
    got = s.run(None, feed(tokens))[0].reshape(-1)
    n = min(len(got), len(ref))
    print(f"PARITY corr={float(np.corrcoef(got[:n], ref[:n])[0, 1]):.4f}", flush=True)
    t2 = torch.randint(1, 178, (1, 61), dtype=torch.long)          # dynamic-length smoke test
    g2 = s.run(None, feed(t2))[0].reshape(-1)
    with torch.no_grad():
        r2 = synth(t2, style, speed).numpy()
    m2 = min(len(g2), len(r2))
    print(f"PARITY_LEN61 corr={float(np.corrcoef(g2[:m2], r2[:m2])[0, 1]):.4f} onnx_len={len(g2)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lang", choices=sorted(MODELS))
    ap.add_argument("out")
    export(*vars(ap.parse_args()).values())
