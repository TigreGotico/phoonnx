"""ZipVoice VocosFbank mel -> ONNX (waveform@24kHz -> log-mel[1,100,T]).

torchaudio's MelSpectrogram uses a complex STFT torch.onnx can't export, so the STFT
is a conv1d against the windowed real/imag DFT basis (power=1 -> magnitude), with the
mel filterbank lifted verbatim from torchaudio. Matches VocosFbank: log(clamp(x, 1e-7)).
"""
import sys
import torch, torch.nn as nn, torchaudio


class ZipVoiceMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_fft, self.hop = 1024, 256
        win = torch.hann_window(1024)
        k = torch.arange(513).unsqueeze(1).float(); n = torch.arange(1024).unsqueeze(0).float()
        ang = 2 * 3.141592653589793 * k * n / 1024
        self.register_buffer("cos", (torch.cos(ang) * win).unsqueeze(1))
        self.register_buffer("sin", (-torch.sin(ang) * win).unsqueeze(1))
        ta = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100, center=True, power=1)
        self.register_buffer("mel_fb", ta.mel_scale.fb.t().contiguous())

    def forward(self, x):                                       # [1, T] @24kHz
        x = nn.functional.pad(x.unsqueeze(1), (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        mag = torch.sqrt(nn.functional.conv1d(x, self.cos, stride=self.hop) ** 2
                         + nn.functional.conv1d(x, self.sin, stride=self.hop) ** 2 + 1e-9)
        return torch.matmul(self.mel_fb, mag).clamp(min=1e-7).log()   # [1, 100, T]


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/zipvoice_mel.onnx"
    torch.onnx.export(ZipVoiceMel().eval(), torch.randn(1, 24000 * 2), out,
                      input_names=["waveform"], output_names=["mel"],
                      dynamic_axes={"waveform": {0: "b", 1: "samples"}}, opset_version=17, dynamo=False)
    print("exported ->", out)
