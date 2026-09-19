import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0

# Magnitude-floor epsilon added under the sqrt when converting the complex STFT
# to a magnitude spectrogram. VITS/HiFi-GAN pairs a 1e-6 STFT floor with the
# 1e-5 spectral-compression clip (dynamic_range_compression_torch); the vocos
# lineage instead pairs a smaller 1e-9 STFT floor (see matcha/audio.py's
# MEL_STFT_EPS). Keep this at 1e-6 to preserve VITS numerics exactly.
VITS_STFT_EPS = 1e-6


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).type_as(y)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[wnsize_dtype_device],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + VITS_STFT_EPS)

    return spec


def _mel_basis_key(n_fft, num_mels, sampling_rate, fmin, fmax, tensor):
    # Key the mel_basis cache on the FULL filterbank-defining parameter set,
    # not fmax alone: two configs sharing fmax but differing in fmin / n_mels /
    # n_fft / sample_rate produce different filterbanks and must not alias to
    # the same cached tensor.
    return (n_fft, num_mels, sampling_rate, fmin, fmax,
            str(tensor.dtype), str(tensor.device))


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    key = _mel_basis_key(n_fft, num_mels, sampling_rate, fmin, fmax, spec)
    if key not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[key] = torch.from_numpy(mel).type_as(spec)
    spec = torch.matmul(mel_basis[key], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    key = _mel_basis_key(n_fft, num_mels, sampling_rate, fmin, fmax, y)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if key not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[key] = torch.from_numpy(mel).type_as(y)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).type_as(y)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)
    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[wnsize_dtype_device],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + VITS_STFT_EPS)

    spec = torch.matmul(mel_basis[key], spec)
    spec = spectral_normalize_torch(spec)

    return spec
