"""Standalone coqui GlowTTS -> ONNX exporter (no coqui-tts package dependency).
Vendors only the pure-torch Encoder/Decoder + helpers, replicates GlowTTS.inference,
and exports an ONNX matching the Larynx contract (input/input_lengths/scales -> mel)
so it works with phoonnx's existing GlowTTSAdapter."""
import sys, json, argparse
sys.path.insert(0, "/tmp/glowexport")
import torch
from glow_tts.encoder import Encoder
from glow_tts.decoder import Decoder
from helpers import sequence_mask, generate_path


def compute_outputs(attn, o_mean, o_log_scale):
    y_mean = torch.matmul(attn.squeeze(1).transpose(1, 2), o_mean.transpose(1, 2)).transpose(1, 2)
    y_log_scale = torch.matmul(attn.squeeze(1).transpose(1, 2), o_log_scale.transpose(1, 2)).transpose(1, 2)
    return y_mean, y_log_scale


class GlowExport(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.encoder = Encoder(c["num_chars"], c["out_channels"], c["hidden_channels_enc"],
                               c["hidden_channels_dp"], c["encoder_type"], c["encoder_params"],
                               dropout_p_dp=c.get("dropout_p_dp", 0.1), mean_only=c.get("mean_only", True),
                               use_prenet=c.get("use_encoder_prenet", True), c_in_channels=c.get("c_in_channels", 0))
        self.decoder = Decoder(c["out_channels"], c["hidden_channels_dec"], c["kernel_size_dec"],
                               c["dilation_rate"], c["num_flow_blocks_dec"], c["num_block_layers"],
                               dropout_p=c.get("dropout_p_dec", 0.05), num_splits=c.get("num_splits", 4),
                               num_squeeze=c.get("num_squeeze", 2), sigmoid_scale=c.get("sigmoid_scale", False),
                               c_in_channels=c.get("c_in_channels", 0))

    def forward(self, x, x_lengths, scales):
        noise_scale, length_scale = scales[0], scales[1]
        o_mean, o_log_scale, o_dur_log, x_mask = self.encoder(x, x_lengths, g=None)
        w = (torch.exp(o_dur_log) - 1) * x_mask * length_scale
        w_ceil = torch.clamp_min(torch.ceil(w), 1)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
        y_mean, y_log_scale = compute_outputs(attn, o_mean, o_log_scale)
        z = (y_mean + torch.exp(y_log_scale) * torch.randn_like(y_mean) * noise_scale) * y_mask
        y, _ = self.decoder(z, y_mask, g=None, reverse=True)
        return y


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    try:
        import json5
        raw = json5.load(open(a.config, encoding="utf-8"))
    except Exception:
        raw = json.load(open(a.config))
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    # derive num_chars from the embedding; default the standard glow arch params
    emb = next((k for k in sd if k.endswith("encoder.emb.weight")), None)
    DEF = dict(out_channels=80, hidden_channels_enc=192, hidden_channels_dp=256,
               hidden_channels_dec=192, encoder_type="rel_pos_transformer",
               encoder_params={"kernel_size": 3, "dropout_p": 0.1, "num_layers": 6,
                               "num_heads": 2, "hidden_channels_ffn": 768, "input_length": None},
               mean_only=True, use_encoder_prenet=True, dropout_p_dp=0.1, kernel_size_dec=5,
               dilation_rate=1, num_flow_blocks_dec=12, num_block_layers=4, num_splits=4,
               num_squeeze=2, sigmoid_scale=False, c_in_channels=0, dropout_p_dec=0.05)
    cfg = {k: (raw.get(k) if raw.get(k) is not None else v) for k, v in DEF.items()}
    cfg["num_chars"] = sd[emb].shape[0] if emb else raw.get("num_chars")
    cfg["out_channels"] = raw.get("audio", {}).get("num_mels", cfg["out_channels"])
    model = GlowExport(cfg).eval()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded: missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:3]: print("  missing sample:", missing[:3])
    if unexpected[:3]: print("  unexpected sample:", unexpected[:3])
    with torch.no_grad():
        x = torch.randint(1, cfg["num_chars"] - 1, (1, 24)); xl = torch.tensor([24])
        scales = torch.tensor([0.667, 1.0])
        mel = model(x, xl, scales)
    print(f"forward OK -> mel {tuple(mel.shape)} (expect [1, {cfg['out_channels']}, T])")
    if a.export:
        # pre-invert the flow 1x1 convs so the reverse pass is a plain matmul
        # (avoids the un-exportable aten::linalg_inv at trace time)
        for m in model.modules():
            if type(m).__name__ == "InvConvNear":
                m.store_inverse()  # only the 1x1 convs need a precomputed matrix inverse
        torch.onnx.export(model, (x, xl, scales), a.out,
                          input_names=["input", "input_lengths", "scales"], output_names=["output"],
                          dynamic_axes={"input": {0: "b", 1: "t"}, "input_lengths": {0: "b"},
                                        "output": {0: "b", 2: "tmel"}},
                          opset_version=15, dynamo=False)
        print("exported ->", a.out)
