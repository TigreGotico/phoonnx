"""StyleTTS2 style encoder -> ONNX  (wav -> ref_p[128], ref_s[128]).

Stitches an ONNX-friendly mel front end (yl4579 MelSpectrogram replica) onto the
DDATT style_encoder + predictor_encoder, so a reference clip yields the prosody/
acoustic style vectors that condition the (un-baked) StyleTTS2 cloning decoder.
"""
import sys
import torch, torch.nn as nn, torchaudio, onnx, numpy as np
from onnx import compose
from huggingface_hub import hf_hub_download


class StyleTTS2Mel(nn.Module):
    """Replica of yl4579 to_mel: MelSpectrogram(n_fft=2048,win=1200,hop=300,n_mels=80,
    default sr=16000 basis) on 24 kHz audio, then (log(1e-5+x)+4)/4 — conv1d STFT so
    it's ONNX-exportable."""
    def __init__(self):
        super().__init__()
        n_fft, win, self.hop = 2048, 1200, 300
        self.n_fft = n_fft
        w = nn.functional.pad(torch.hann_window(win), ((n_fft - win) // 2, n_fft - win - (n_fft - win) // 2))
        k = torch.arange(n_fft // 2 + 1).unsqueeze(1).float(); n = torch.arange(n_fft).unsqueeze(0).float()
        ang = 2 * 3.141592653589793 * k * n / n_fft
        self.register_buffer("cos", (torch.cos(ang) * w).unsqueeze(1))
        self.register_buffer("sin", (-torch.sin(ang) * w).unsqueeze(1))
        ta = torchaudio.transforms.MelSpectrogram(n_fft=n_fft, win_length=win, hop_length=self.hop, n_mels=80)
        self.register_buffer("mel_fb", ta.mel_scale.fb.t().contiguous())

    def forward(self, x):                                   # x: [1, T] @24kHz
        x = nn.functional.pad(x.unsqueeze(1), (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        power = nn.functional.conv1d(x, self.cos, stride=self.hop) ** 2 + \
                nn.functional.conv1d(x, self.sin, stride=self.hop) ** 2
        mel = torch.matmul(self.mel_fb, power)
        mel = (torch.log(1e-5 + mel) + 4) / 4
        return mel.unsqueeze(1)                             # [1, 1, 80, T]


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/styletts2_se.onnx"
    R = "DDATT/StyleTTS2-ONNX-Cpp"
    sty_m = onnx.load(hf_hub_download(R, "style_encoder_simp.onnx"))
    prd_m = onnx.load(hf_hub_download(R, "predictor_encoder_simp.onnx"))
    opset = sty_m.opset_import[0].version  # match the DDATT graphs (conv1d-DFT mel needs no STFT op)
    torch.onnx.export(StyleTTS2Mel().eval(), torch.randn(1, 24000 * 2), "/tmp/st2_mel.onnx",
                      input_names=["waveform"], output_names=["mel"],
                      dynamic_axes={"waveform": {0: "b", 1: "samples"}}, opset_version=opset, dynamo=False)
    mel = compose.add_prefix(onnx.load("/tmp/st2_mel.onnx"), "m_")
    sty = compose.add_prefix(sty_m, "se_")
    prd = compose.add_prefix(prd_m, "pe_")
    mel.ir_version = sty.ir_version  # align for compose
    g = compose.merge_models(mel, sty, io_map=[("m_mel", "se_x")])
    g.graph.output.append(onnx.helper.make_tensor_value_info("m_mel", onnx.TensorProto.FLOAT, [1, 1, 80, "t"]))
    g = compose.merge_models(g, prd, io_map=[("m_mel", "pe_x")])
    ren = {"m_waveform": "waveform", "se_ref_p": "ref_p", "pe_ref_s": "ref_s"}
    for vi in list(g.graph.input) + list(g.graph.output):
        if vi.name in ren: vi.name = ren[vi.name]
    for nd in g.graph.node:
        nd.input[:] = [ren.get(x, x) for x in nd.input]; nd.output[:] = [ren.get(x, x) for x in nd.output]
    keep = [o for o in g.graph.output if o.name in ("ref_p", "ref_s")]; del g.graph.output[:]; g.graph.output.extend(keep)
    onnx.checker.check_model(g); onnx.save(g, out)
    print("exported ->", out)
