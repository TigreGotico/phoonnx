import numpy as np
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import read

MAX_WAV_VALUE = 32768.0

# Magnitude-floor epsilon added under the sqrt when converting the complex STFT
# to a magnitude spectrogram. The vocos/matcha lineage pairs a 1e-9 STFT floor
# with the 1e-5 spectral-compression clip; VITS/HiFi-GAN instead uses a larger
# 1e-6 STFT floor (see vits/mel_processing.py's VITS_STFT_EPS). Keep this at
# 1e-9 to preserve matcha numerics exactly.
MEL_STFT_EPS = 1e-9


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window  # pylint: disable=global-statement
    # Key the mel_basis cache on the FULL filterbank-defining parameter set,
    # not fmax alone: two configs sharing fmax but differing in fmin / n_mels /
    # n_fft / sample_rate produce different filterbanks and must not alias.
    mel_key = (n_fft, num_mels, sampling_rate, fmin, fmax, str(y.device))
    win_key = (win_size, str(y.device))
    if mel_key not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[mel_key] = torch.from_numpy(mel).float().to(y.device)
    if win_key not in hann_window:
        hann_window[win_key] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[win_key],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + MEL_STFT_EPS)

    spec = torch.matmul(mel_basis[mel_key], spec)
    spec = spectral_normalize_torch(spec)

    return spec
