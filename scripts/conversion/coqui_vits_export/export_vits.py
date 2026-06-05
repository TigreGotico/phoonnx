"""Standalone coqui VITS -> ONNX exporter (no coqui-tts package dependency).
Vendors the pure-torch layers + replicates Vits.inference. Output ONNX matches
phoonnx's existing VitsAdapter (input/input_lengths/scales -> waveform)."""
import sys, json, argparse
sys.path.insert(0, "/tmp/vitsexport")
import torch
from networks import TextEncoder, ResidualCouplingBlocks
from sdp import StochasticDurationPredictor
from glow_tts.duration_predictor import DurationPredictor
from hifigan_generator import HifiganGenerator
from helpers import sequence_mask, generate_path


def g(a, k, d):
    v = a.get(k)
    return v if v is not None else d


class VitsExport(torch.nn.Module):
    def __init__(self, a, num_chars, num_speakers, num_languages):
        super().__init__()
        hc = g(a, "hidden_channels", 192)
        self.use_sdp = g(a, "use_sdp", True)
        self.use_spk = g(a, "use_speaker_embedding", False) and num_speakers > 1
        self.use_lang = g(a, "use_language_embedding", False) and num_languages > 1
        self.cond_dp_spk = g(a, "condition_dp_on_speaker", True)
        spk_dim = g(a, "speaker_embedding_channels", 256) if self.use_spk else 0
        lang_dim = g(a, "embedded_language_dim", 4) if self.use_lang else 0
        self.spk_dim, self.lang_dim = spk_dim, lang_dim
        self.text_encoder = TextEncoder(num_chars, hc, hc,
            g(a, "hidden_channels_ffn_text_encoder", 768), g(a, "num_heads_text_encoder", 2),
            g(a, "num_layers_text_encoder", 6), g(a, "kernel_size_text_encoder", 3),
            g(a, "dropout_p_text_encoder", 0.1), language_emb_dim=lang_dim)
        self.flow = ResidualCouplingBlocks(hc, hc, kernel_size=g(a, "kernel_size_flow", 5),
            dilation_rate=g(a, "dilation_rate_flow", 1), num_layers=g(a, "num_layers_flow", 4),
            cond_channels=spk_dim)
        if self.use_sdp:
            self.duration_predictor = StochasticDurationPredictor(hc, 192, 3,
                g(a, "dropout_p_duration_predictor", 0.5), 4,
                cond_channels=spk_dim if self.cond_dp_spk else 0, language_emb_dim=lang_dim)
        else:
            self.duration_predictor = DurationPredictor(hc, 256, 3,
                g(a, "dropout_p_duration_predictor", 0.5), cond_channels=spk_dim, language_emb_dim=lang_dim)
        self.waveform_decoder = HifiganGenerator(hc, 1, g(a, "resblock_type_decoder", "1"),
            g(a, "resblock_dilation_sizes_decoder", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
            g(a, "resblock_kernel_sizes_decoder", [3, 7, 11]),
            g(a, "upsample_kernel_sizes_decoder", [16, 16, 4, 4]),
            g(a, "upsample_initial_channel_decoder", 512),
            g(a, "upsample_rates_decoder", [8, 8, 2, 2]), inference_padding=0,
            cond_channels=spk_dim, conv_pre_weight_norm=False, conv_post_weight_norm=False, conv_post_bias=False)
        if self.use_spk:
            self.emb_g = torch.nn.Embedding(num_speakers, spk_dim)
        if self.use_lang:
            self.emb_l = torch.nn.Embedding(num_languages, lang_dim)

    def forward(self, x, x_lengths, scales, sid=None, langid=None):
        noise_scale, length_scale, noise_scale_dp = scales[0], scales[1], scales[2]
        g_ = self.emb_g(sid).unsqueeze(-1) if (self.use_spk and sid is not None) else None
        lang_emb = self.emb_l(langid).unsqueeze(-1) if (self.use_lang and langid is not None) else None
        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=lang_emb)
        if self.use_sdp:
            logw = self.duration_predictor(x, x_mask, g=g_ if self.cond_dp_spk else None,
                                           reverse=True, noise_scale=noise_scale_dp, lang_emb=lang_emb)
        else:
            logw = self.duration_predictor(x, x_mask, g=g_ if self.cond_dp_spk else None, lang_emb=lang_emb)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype).unsqueeze(1)
        attn_mask = x_mask * y_mask.transpose(1, 2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1).transpose(1, 2))
        m_p = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g_, reverse=True)
        return self.waveform_decoder((z * y_mask), g=g_)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--export", action="store_true")
    A = ap.parse_args()
    try:
        import json5; cfg = json5.load(open(A.config, encoding="utf-8"))
    except Exception:
        cfg = json.load(open(A.config))
    a = cfg.get("model_args") or cfg
    sd = torch.load(A.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    emb = next((k for k in sd if k.endswith("text_encoder.emb.weight")), None)
    num_chars = sd[emb].shape[0]
    num_spk = sd["emb_g.weight"].shape[0] if "emb_g.weight" in sd else 1
    num_lang = sd["emb_l.weight"].shape[0] if "emb_l.weight" in sd else 1
    model = VitsExport(a, num_chars, num_spk, num_lang).eval()
    miss, unexp = model.load_state_dict(sd, strict=False)
    miss = [m for m in miss if not m.startswith("posterior_encoder")]
    print(f"load: missing(non-posterior)={len(miss)} unexpected={len([u for u in unexp if not u.startswith('posterior_encoder') and 'disc' not in u])}")
    for m in model.modules():
        if hasattr(m, "remove_weight_norm"):
            try: m.remove_weight_norm()
            except Exception: pass
    x = torch.randint(1, num_chars - 1, (1, 22)); xl = torch.tensor([22])
    scales = torch.tensor([0.667, 1.0, 0.8])
    args = (x, xl, scales) + ((torch.tensor([0]),) if num_spk > 1 else ())
    with torch.no_grad():
        o = model(*args)
    print(f"forward OK -> audio {tuple(o.shape)}")
    if A.export:
        names = ["input", "input_lengths", "scales"] + (["sid"] if num_spk > 1 else [])
        dyn = {"input": {0: "b", 1: "t"}, "input_lengths": {0: "b"}, "output": {0: "b", 2: "tw"}}
        torch.onnx.export(model, args, A.out, input_names=names, output_names=["output"],
                          dynamic_axes=dyn, opset_version=15, dynamo=False)
        print("exported ->", A.out)
