"""Small numpy-only audio effects behind SSML: pitch, whisper, soft voice, compression, breaths.

A VITS voice has no pitch or phonation controls, so these work on the finished samples
(float32 mono in [-1, 1]). They are deliberately simple and dependency-free (numpy only):

* :func:`pitch_shift` moves the pitch by semitones and keeps the duration (resample, then
  WSOLA time-stretch back). The formants move with the pitch, so it suits the modest shifts SSML
  asks for (a few semitones) and sounds like a different speaker beyond about +-6.
* :func:`whisper` re-synthesizes the signal with random STFT phase, which keeps the spectral
  envelope (the words) and drops the voicing (the pitch): an approximation of whispered speech.
* :func:`soft` is a quieter, breathier, duller voice (``phonation="soft"``).
* :func:`compress` evens out the loudness (``<amazon:effect name="drc">``).
* :func:`breath` and :func:`beep` make the short sounds between sentences and over an expletive.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def _as_float(audio: np.ndarray) -> np.ndarray:
    """Mono float32 samples; NaN and infinity (a bad model output) become silence and full scale."""
    return np.nan_to_num(np.asarray(audio, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=-1.0)


def resample(audio: np.ndarray, ratio: float) -> np.ndarray:
    """`audio` read `ratio` times faster (pitch and speed both change). Linear interpolation."""
    audio = _as_float(audio)
    if audio.size < 2 or abs(ratio - 1.0) < 1e-4:
        return audio
    new_length = max(2, int(round(audio.size / ratio)))
    positions = np.linspace(0, audio.size - 1, new_length)
    return np.interp(positions, np.arange(audio.size), audio).astype(np.float32)


def time_stretch(audio: np.ndarray, factor: float, sample_rate: int) -> np.ndarray:
    """Stretch `audio` to `factor` times its length (> 1 slower) without changing its pitch (WSOLA)."""
    audio = _as_float(audio)
    if audio.size < 64 or abs(factor - 1.0) < 1e-3:
        return audio
    frame = max(256, int(0.04 * sample_rate) // 2 * 2)
    hop = frame // 2
    tolerance = frame // 4
    window = np.hanning(frame).astype(np.float32)
    step = max(1, sample_rate // 11025)
    out_length = int(audio.size * factor)
    padded = np.concatenate([np.zeros(tolerance, np.float32), audio,
                             np.zeros(frame + 2 * tolerance, np.float32)])
    out = np.zeros(out_length + 2 * frame, np.float32)
    norm = np.zeros_like(out)
    position = 0.0          # where the next frame is wanted in the input
    previous = 0            # start (in padded) of the frame placed last
    for index in range(0, out_length // hop + 1):
        wanted = int(round(index * hop / factor)) + tolerance
        if index == 0:
            start = wanted
        else:
            # The input that would naturally have followed the previous frame is the template;
            # slide the new frame around `wanted` to where it lines up best with it.
            template = padded[previous + hop: previous + hop + frame]
            low, high = max(0, wanted - tolerance), min(padded.size - frame, wanted + tolerance)
            if high <= low or template.size < frame:
                start = min(wanted, padded.size - frame)
            else:
                # Compare every `step`-th sample: the best overlap is a matter of waveform shape,
                # and this cuts the cost of the search (the whole effect) by the same factor.
                candidates = np.lib.stride_tricks.sliding_window_view(
                    padded[low: high + frame], frame)[:, ::step]
                start = low + int(np.argmax(candidates @ template[::step]))
        segment = padded[start: start + frame]
        if segment.size < frame:
            segment = np.pad(segment, (0, frame - segment.size))
        at = index * hop
        out[at: at + frame] += segment * window
        norm[at: at + frame] += window
        previous = start
        position += hop
        if at >= out_length:
            break
    norm[norm < 1e-3] = 1.0
    return (out / norm)[:out_length].astype(np.float32)


def pitch_shift(audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    """`audio` raised by `semitones` (negative lowers it), at its original duration."""
    audio = _as_float(audio)
    if abs(semitones) < 0.05 or audio.size < 64:
        return audio
    ratio = 2.0 ** (float(np.clip(semitones, -12.0, 12.0)) / 12.0)
    # Reading faster by `ratio` raises the pitch and shortens the clip; stretch it back.
    shifted = time_stretch(resample(audio, ratio), ratio, sample_rate)
    if shifted.size > audio.size:
        shifted = shifted[: audio.size]
    else:
        shifted = np.pad(shifted, (0, audio.size - shifted.size))
    return np.clip(shifted, -1.0, 1.0)


def _stft(audio: np.ndarray, size: int, hop: int) -> np.ndarray:
    window = np.hanning(size).astype(np.float32)
    padded = np.pad(audio, (size, size))
    count = 1 + (padded.size - size) // hop
    frames = np.lib.stride_tricks.sliding_window_view(padded, size)[::hop][:count] * window
    return np.fft.rfft(frames, axis=1)


def _istft(spectrum: np.ndarray, size: int, hop: int, length: int) -> np.ndarray:
    window = np.hanning(size).astype(np.float32)
    frames = np.fft.irfft(spectrum, n=size, axis=1).astype(np.float32) * window
    count, parts = frames.shape[0], size // hop       # `size` is a whole number of hops
    out = np.zeros((count + parts - 1) * hop, np.float32)
    norm = np.zeros_like(out)
    squared = np.broadcast_to(window ** 2, frames.shape)
    for j in range(parts):  # overlap-add, one hop-wide slice of every frame at a time
        at = j * hop
        out[at: at + count * hop] += frames[:, j * hop: (j + 1) * hop].reshape(-1)
        norm[at: at + count * hop] += squared[:, j * hop: (j + 1) * hop].reshape(-1)
    norm[norm < 1e-6] = 1.0
    return (out / norm)[size: size + length]


def _box_smooth(values: np.ndarray, width: int) -> np.ndarray:
    """A moving average of `width` bins along the last axis, all rows at once (edges use fewer bins)."""
    half = width // 2
    padded = np.concatenate([np.zeros((values.shape[0], half + 1), values.dtype), values,
                             np.zeros((values.shape[0], half), values.dtype)], axis=1)
    cumulative = np.cumsum(padded, axis=1, dtype=np.float64)
    total = cumulative[:, width:] - cumulative[:, :-width]
    counts = np.minimum(np.arange(values.shape[1]) + half + 1, values.shape[1]) - np.maximum(
        np.arange(values.shape[1]) - half, 0)
    return (total[:, : values.shape[1]] / counts).astype(np.float32)


def whisper(audio: np.ndarray, sample_rate: int, seed: int | None = 0) -> np.ndarray:
    """An approximation of whispered speech: the spectral envelope with random phase."""
    audio = _as_float(audio)
    if audio.size < 512:
        return audio
    size = 1024 if sample_rate >= 16000 else 512
    hop = size // 4
    spectrum = _stft(audio, size, hop)
    # Smooth the magnitude across ~350 Hz so the harmonics (the pitch) are gone and only the
    # envelope (the vowels and consonants) is left; random phase then makes it noise.
    width = max(3, int(round(350.0 / (sample_rate / size))) | 1)
    magnitude = _box_smooth(np.abs(spectrum), width)
    rng = np.random.default_rng(seed)
    phase = rng.uniform(-np.pi, np.pi, spectrum.shape).astype(np.float32)
    noisy = magnitude * np.exp(1j * phase)
    out = _istft(noisy, size, hop, audio.size)
    # Keep the loudness of the input: whispering is not quieter on the page, only breathier.
    scale = (np.sqrt(np.mean(audio ** 2)) + _EPS) / (np.sqrt(np.mean(out ** 2)) + _EPS)
    return np.clip(out * min(scale, 8.0), -1.0, 1.0).astype(np.float32)


def soft(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """A soft, slightly breathy voice: quieter, duller, with a little whisper mixed in."""
    audio = _as_float(audio)
    if audio.size < 512:
        return np.clip(audio * 0.6, -1.0, 1.0)
    # One-pole low-pass at roughly 4 kHz, as a truncated impulse response (no per-sample loop).
    alpha = float(np.exp(-2.0 * np.pi * 4000.0 / sample_rate))
    taps = max(8, int(np.ceil(np.log(1e-4) / np.log(max(alpha, 1e-6)))))
    response = ((1.0 - alpha) * alpha ** np.arange(taps)).astype(np.float32)
    dull = np.convolve(audio, response)[: audio.size].astype(np.float32)
    mixed = 0.8 * dull + 0.2 * whisper(audio, sample_rate)
    return np.clip(mixed * 0.6, -1.0, 1.0).astype(np.float32)


def compress(audio: np.ndarray, sample_rate: int, threshold_db: float = -26.0,
             ratio: float = 3.0, window_ms: float = 20.0) -> np.ndarray:
    """Dynamic range compression, then make-up gain back to the input's peak."""
    audio = _as_float(audio)
    if audio.size < 64:
        return audio
    block = max(16, int(sample_rate * window_ms / 1000))
    count = int(np.ceil(audio.size / block))
    padded = np.pad(audio, (0, count * block - audio.size))
    level = np.sqrt(np.mean(padded.reshape(count, block) ** 2, axis=1)) + _EPS
    level_db = 20.0 * np.log10(level)
    over = np.maximum(level_db - threshold_db, 0.0)
    gain_db = -over * (1.0 - 1.0 / ratio)
    # Smooth the gain so blocks do not click.
    gain_db = np.convolve(np.pad(gain_db, 2, mode="edge"), np.ones(5) / 5.0, mode="valid")
    gain = np.repeat(10.0 ** (gain_db / 20.0), block)[: audio.size].astype(np.float32)
    out = audio * gain
    peak_in, peak_out = float(np.max(np.abs(audio))), float(np.max(np.abs(out)))
    if peak_out > _EPS:
        out = out * min(peak_in / peak_out, 4.0)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def breath(sample_rate: int, duration_ms: int = 350, volume: float = 1.0, seed: int | None = 1) -> np.ndarray:
    """A short in-breath: band-limited noise with a slow rise and a quicker fall."""
    length = max(1, int(sample_rate * duration_ms / 1000))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(length).astype(np.float32)
    # Take the lows and highs off: roughly 300 Hz - 3 kHz.
    spectrum = np.fft.rfft(noise)
    frequencies = np.fft.rfftfreq(length, 1.0 / sample_rate)
    spectrum[(frequencies < 300) | (frequencies > 3000)] = 0
    noise = np.fft.irfft(spectrum, n=length).astype(np.float32)
    envelope = np.concatenate([np.linspace(0, 1, int(length * 0.6)) ** 2,
                               np.linspace(1, 0, length - int(length * 0.6))])
    noise = noise * envelope.astype(np.float32)
    peak = float(np.max(np.abs(noise))) + _EPS
    return (noise / peak * 0.08 * float(np.clip(volume, 0.0, 3.0))).astype(np.float32)


def beep(sample_rate: int, duration_ms: int = 350, frequency: float = 1000.0) -> np.ndarray:
    """A censor bleep."""
    length = max(1, int(sample_rate * duration_ms / 1000))
    t = np.arange(length, dtype=np.float32) / sample_rate
    tone = 0.25 * np.sin(2.0 * np.pi * frequency * t)
    fade = min(length // 8, int(sample_rate * 0.01))
    if fade:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        tone[:fade] *= ramp
        tone[-fade:] *= ramp[::-1]
    return tone.astype(np.float32)
