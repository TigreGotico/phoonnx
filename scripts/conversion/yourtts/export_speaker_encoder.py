"""Standalone coqui ResNet speaker-encoder -> ONNX (no coqui-tts dependency).

Vendors the inference path of TTS.encoder.models.resnet.ResNetSpeakerEncoder.
The ONNX takes a 16 kHz mono waveform (1, T) and returns the L2-normalised 512-d
speaker d-vector that conditions YourTTS — enabling runtime zero-shot cloning:
``reference.wav -> speaker_encoder.onnx -> d_vector -> YourTTSAdapter``.

Usage: python export_speaker_encoder.py <model_se.pth> <config_se.json> <out.onnx>
"""
import sys
import json

import torch
import torch.nn as nn
import torchaudio


class PreEmphasis(nn.Module):
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.register_buffer("flipped_filter", torch.tensor([-coef, 1.0]).view(1, 1, 2))

    def forward(self, x):
        x = nn.functional.pad(x.unsqueeze(1), (1, 0), "reflect")
        return nn.functional.conv1d(x, self.flipped_filter).squeeze(1)


class OnnxMel(nn.Module):
    """ONNX-exportable replica of PreEmphasis + torchaudio MelSpectrogram.

    torchaudio's MelSpectrogram uses a complex STFT that torch.onnx can't export, so
    the STFT is done as a conv1d against the windowed real/imag DFT basis, and the mel
    filterbank is lifted verbatim from a torchaudio MelSpectrogram so the output is
    numerically identical.
    """

    def __init__(self, audio):
        super().__init__()
        self.preemph = PreEmphasis(audio["preemphasis"])
        n_fft, win_len, self.hop = audio["fft_size"], audio["win_length"], audio["hop_length"]
        self.n_fft = n_fft
        win = torch.hamming_window(win_len)                      # periodic, matches torchaudio
        pad = (n_fft - win_len) // 2
        win = nn.functional.pad(win, (pad, n_fft - win_len - pad))
        k = torch.arange(n_fft // 2 + 1).unsqueeze(1).float()
        n = torch.arange(n_fft).unsqueeze(0).float()
        ang = 2 * 3.141592653589793 * k * n / n_fft
        self.register_buffer("cos", (torch.cos(ang) * win).unsqueeze(1))
        self.register_buffer("sin", (-torch.sin(ang) * win).unsqueeze(1))
        ta = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio["sample_rate"], n_fft=n_fft, win_length=win_len,
            hop_length=self.hop, window_fn=torch.hamming_window, n_mels=audio["num_mels"])
        self.register_buffer("mel_fb", ta.mel_scale.fb.t().contiguous())  # [n_mels, n_freq]

    def forward(self, x):                                        # x: [B, T]
        x = self.preemph(x).unsqueeze(1)
        x = nn.functional.pad(x, (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        real = nn.functional.conv1d(x, self.cos, stride=self.hop)
        imag = nn.functional.conv1d(x, self.sin, stride=self.hop)
        power = real ** 2 + imag ** 2                           # [B, n_freq, frames]
        return torch.matmul(self.mel_fb, power)                 # [B, n_mels, frames]


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True),
                                nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.bn1(self.relu(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResNetSpeakerEncoder(nn.Module):
    def __init__(self, audio_config, input_dim=64, proj_dim=512, layers=(3, 4, 6, 3),
                 num_filters=(32, 64, 128, 256), encoder_type="ASP", log_input=True):
        super().__init__()
        self.encoder_type, self.input_dim, self.log_input = encoder_type, input_dim, log_input
        self.conv1 = nn.Conv2d(1, num_filters[0], 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.inplanes = num_filters[0]
        self.layer1 = self._layer(num_filters[0], layers[0])
        self.layer2 = self._layer(num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._layer(num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._layer(num_filters[3], layers[3], stride=(2, 2))
        self.instancenorm = nn.InstanceNorm1d(input_dim)
        self.torch_spec = OnnxMel(audio_config)
        outmap = input_dim // 8
        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[3] * outmap, 128, 1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, num_filters[3] * outmap, 1), nn.Softmax(dim=2))
        out_dim = num_filters[3] * outmap * (2 if encoder_type == "ASP" else 1)
        self.fc = nn.Linear(out_dim, proj_dim)

    def _layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                                       nn.BatchNorm2d(planes))
        ls = [SEBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        ls += [SEBasicBlock(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*ls)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.torch_spec(x)
        if self.log_input:
            x = (x + 1e-6).log()
        # InstanceNorm1d(affine=False, eps=1e-5) spelled out — the symbolic
        # instance_norm op is rejected by the exporter, this is numerically identical.
        mean = x.mean(dim=2, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=2, keepdim=True)
        x = ((x - mean) / torch.sqrt(var + 1e-5)).unsqueeze(1)
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = x.reshape(x.size(0), -1, x.size(-1))
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-5))
        x = self.fc(torch.cat((mu, sg), 1))
        return torch.nn.functional.normalize(x, p=2, dim=1)


if __name__ == "__main__":
    ckpt, cfg_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    cfg = json.load(open(cfg_path))
    a = cfg["audio"]
    audio = {"preemphasis": a["preemphasis"], "sample_rate": a["sample_rate"],
             "fft_size": a["fft_size"], "win_length": a["win_length"],
             "hop_length": a["hop_length"], "num_mels": a["num_mels"]}
    model = ResNetSpeakerEncoder(audio, proj_dim=cfg["model_params"]["proj_dim"], log_input=True).eval()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"load: missing={len(miss)} unexpected={len(unexp)}")
    x = torch.randn(1, 16000 * 3)
    with torch.no_grad():
        ref = model(x)
    print(f"torch d-vector: {tuple(ref.shape)} norm={ref.norm().item():.3f}")
    torch.onnx.export(model, x, out, input_names=["waveform"], output_names=["d_vector"],
                      dynamic_axes={"waveform": {0: "b", 1: "samples"}}, opset_version=17, dynamo=False)
    print("exported ->", out)
