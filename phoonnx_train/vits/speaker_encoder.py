# Mozilla Public License 2.0 — this file is adapted from coqui-ai/TTS
# (TTS/encoder/models/resnet.py + TTS/encoder/models/base_encoder.py) and
# remains under the MPL-2.0; see https://mozilla.org/MPL/2.0/.
#
# H/ASP ResNet speaker encoder (https://arxiv.org/abs/2009.14153, adapted
# from https://github.com/clovaai/voxceleb_trainer), the reference speaker
# encoder for YourTTS. This differentiable torch implementation exists so
# the speaker-consistency loss can backpropagate through the generated
# waveform at training time (the ONNX encoder in
# ``phoonnx.engines.speaker_encoders`` covers inference/preprocessing).
# It loads the same ``model_se.pth.tar`` checkpoint the ONNX encoder was
# converted from.
"""Differentiable H/ASP ResNet speaker encoder for the YourTTS
speaker-consistency loss (wav in → 512-d d-vector out)."""
from typing import Any, Dict, Optional

import torch
import torchaudio
from torch import nn

# audio front-end of the released YourTTS speaker encoder (config_se.json)
HASP_AUDIO_CONFIG: Dict[str, Any] = {
    "sample_rate": 16000,
    "fft_size": 512,
    "win_length": 400,
    "hop_length": 160,
    "num_mels": 64,
    "preemphasis": 0.97,
}


class PreEmphasis(nn.Module):
    def __init__(self, coefficient=0.97):
        super().__init__()
        self.coefficient = coefficient
        self.register_buffer(
            "filter",
            torch.FloatTensor([-self.coefficient, 1.0]).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        assert len(x.size()) == 2
        x = torch.nn.functional.pad(x.unsqueeze(1), (1, 0), "reflect")
        return torch.nn.functional.conv1d(x, self.filter).squeeze(1)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNetSpeakerEncoder(nn.Module):
    """H/ASP without batch normalization in the speaker embedding."""

    def __init__(
        self,
        input_dim=64,
        proj_dim=512,
        layers=(3, 4, 6, 3),
        num_filters=(32, 64, 128, 256),
        encoder_type="ASP",
        log_input=True,
        audio_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.input_dim = input_dim
        self.log_input = log_input
        self.audio_config = dict(HASP_AUDIO_CONFIG, **(audio_config or {}))
        self.proj_dim = proj_dim

        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1,
                               padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])

        self.inplanes = num_filters[0]
        self.layer1 = self.create_layer(SEBasicBlock, num_filters[0], layers[0])
        self.layer2 = self.create_layer(SEBasicBlock, num_filters[1], layers[1],
                                        stride=(2, 2))
        self.layer3 = self.create_layer(SEBasicBlock, num_filters[2], layers[2],
                                        stride=(2, 2))
        self.layer4 = self.create_layer(SEBasicBlock, num_filters[3], layers[3],
                                        stride=(2, 2))

        self.instancenorm = nn.InstanceNorm1d(input_dim)

        self.torch_spec = nn.Sequential(
            PreEmphasis(self.audio_config["preemphasis"]),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self.audio_config["sample_rate"],
                n_fft=self.audio_config["fft_size"],
                win_length=self.audio_config["win_length"],
                hop_length=self.audio_config["hop_length"],
                window_fn=torch.hamming_window,
                n_mels=self.audio_config["num_mels"],
            ),
        )

        outmap_size = int(self.input_dim / 8)

        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[3] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_filters[3] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

        if self.encoder_type == "SAP":
            out_dim = num_filters[3] * outmap_size
        elif self.encoder_type == "ASP":
            out_dim = num_filters[3] * outmap_size * 2
        else:
            raise ValueError("Undefined encoder")

        self.fc = nn.Linear(out_dim, proj_dim)

        self._init_layers()

    def _init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def create_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, l2_norm=True):
        """x: waveform ``(N, T)`` at ``audio_config['sample_rate']`` →
        d-vector ``(N, proj_dim)``. Differentiable."""
        x = self.torch_spec(x)
        if self.log_input:
            x = (x + 1e-6).log()
        x = self.instancenorm(x).unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.size()[0], -1, x.size()[-1])

        w = self.attention(x)

        if self.encoder_type == "SAP":
            x = torch.sum(x * w, dim=2)
        else:  # ASP
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt(
                (torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-5))
            x = torch.cat((mu, sg), 1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        if l2_norm:
            x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x


def load_hasp_speaker_encoder(checkpoint_path: str) -> ResNetSpeakerEncoder:
    """Build the H/ASP encoder and load a ``model_se.pth.tar``-style
    checkpoint (the released YourTTS speaker encoder)."""
    encoder = ResNetSpeakerEncoder()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    # the released checkpoint carries the (unused here) torch_spec buffers
    # and criterion weights — tolerate them
    model_state = encoder.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in model_state and v.shape == model_state[k].shape}
    encoder.load_state_dict(filtered, strict=False)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder
