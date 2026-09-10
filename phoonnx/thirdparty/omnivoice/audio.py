#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Audio front end and post-processing vendored from ``k2-fsa/OmniVoice``
(``omnivoice/utils/audio.py``), rewritten on numpy so phoonnx needs neither torch nor
torchaudio.

Two pieces have to match upstream, because both feed the codec:

* :func:`resample` reproduces ``torchaudio.functional.resample``'s default
  ``sinc_interp_hann`` kernel. The Higgs semantic encoder runs at 16 kHz while the
  acoustic one runs at 24 kHz, and a different anti-alias filter moves about 3 % of the
  reference codec codes. The unmasking loop amplifies that: with scipy's default
  ``resample_poly`` kernel the generated codes agree with upstream only 5 % of the time,
  and with this kernel they agree 100 %.
* :func:`remove_silence` gates on dBFS the way upstream does. Upstream runs pydub, whose
  splitter has behaviour that is hard to reproduce exactly (it merges overlapping ranges
  at their midpoint and lets a negative start index wrap), so pydub drives this function
  when it is installed and a numpy re-implementation takes over when it is not. The
  fallback agrees on which parts are speech; it can place a boundary a few tens of
  milliseconds differently, which changes the trim of a cloning reference but not the
  voice it carries.
"""
import math
from typing import List, Tuple

import numpy as np

MAX_AMPLITUDE = 32768.0
"""pydub measures dBFS against a full-scale 16-bit sample, so we do too."""


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def _sinc_kernel(orig_freq: int, new_freq: int, lowpass_filter_width: int = 6,
                 rolloff: float = 0.99) -> Tuple[np.ndarray, int, int, int]:
    """Build torchaudio's default ``sinc_interp_hann`` resampling kernel."""
    g = math.gcd(int(orig_freq), int(new_freq))
    orig_freq, new_freq = int(orig_freq) // g, int(new_freq) // g
    base_freq = min(orig_freq, new_freq) * rolloff
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    idx = np.arange(-width, width + orig_freq, dtype=np.float64)[None, :]
    i = np.arange(new_freq, dtype=np.float64)[:, None]
    t = np.clip((-i / new_freq + idx / orig_freq) * base_freq,
                -lowpass_filter_width, lowpass_filter_width)
    window = np.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    t = t * math.pi
    kernel = np.where(t == 0, 1.0, np.sin(t) / np.where(t == 0, 1.0, t)) * window
    return (kernel * (base_freq / orig_freq)).astype(np.float32), width, orig_freq, new_freq


def resample(waveform: np.ndarray, orig_freq: int, new_freq: int,
             lowpass_filter_width: int = 6, rolloff: float = 0.99) -> np.ndarray:
    """Band-limited sinc resampling of a 1-D waveform, matching torchaudio's default."""
    w = np.asarray(waveform, np.float32).reshape(-1)
    if orig_freq == new_freq:
        return w
    kernel, width, orig, new = _sinc_kernel(orig_freq, new_freq, lowpass_filter_width, rolloff)
    length = w.shape[-1]
    padded = np.pad(w, (width, width + orig))
    num = (len(padded) - kernel.shape[1]) // orig + 1
    windows = np.lib.stride_tricks.as_strided(
        padded, shape=(num, kernel.shape[1]),
        strides=(padded.strides[0] * orig, padded.strides[0]))
    # one row per output phase, then interleave the phases back into time order
    out = (windows @ kernel.T).T.reshape(-1, order="F")
    target = int(math.ceil(new * length / orig))
    return np.ascontiguousarray(out[:target], dtype=np.float32)


# ---------------------------------------------------------------------------
# Silence handling (pydub semantics on numpy)
# ---------------------------------------------------------------------------

def _int16(audio: np.ndarray) -> np.ndarray:
    """(C, T) float in [-1, 1] -> mono int16 as pydub would quantize it."""
    a = np.asarray(audio, np.float32)
    if a.ndim == 2:
        a = a.mean(axis=0)
    return (a * MAX_AMPLITUDE).clip(-32768, 32767).astype(np.int16).astype(np.float64)


def _rms_per_ms(samples: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cumulative sums of x and x^2 on a 1 ms grid, for O(1) window RMS queries."""
    spm = sample_rate / 1000.0
    n_ms = int(len(samples) / spm)
    edges = (np.arange(n_ms + 1) * spm).astype(np.int64)
    csum = np.concatenate([[0.0], np.cumsum(samples.astype(np.float64) ** 2)])
    return csum, edges


def _window_rms(csum: np.ndarray, edges: np.ndarray, a_ms: int, b_ms: int) -> float:
    a, b = edges[max(0, a_ms)], edges[min(len(edges) - 1, b_ms)]
    if b <= a:
        return 0.0
    return math.sqrt(max(0.0, (csum[b] - csum[a]) / (b - a)))


def _db_to_amp(db: float) -> float:
    return (10 ** (db / 20.0)) * MAX_AMPLITUDE


def _detect_silence(csum, edges, total_ms, min_silence_len, thresh_amp, seek_step):
    """pydub ``detect_silence``: [start_ms, end_ms] runs quieter than the threshold."""
    if total_ms < min_silence_len:
        return []
    starts = [i for i in range(0, total_ms - min_silence_len + 1, seek_step)
              if _window_rms(csum, edges, i, i + min_silence_len) <= thresh_amp]
    if not starts:
        return []
    ranges, cur_start, prev = [], starts[0], starts[0]
    for s in starts[1:]:
        if s > prev + seek_step:
            ranges.append([cur_start, prev + min_silence_len])
            cur_start = s
        prev = s
    ranges.append([cur_start, prev + min_silence_len])
    return ranges


def _detect_nonsilent(csum, edges, total_ms, min_silence_len, thresh_amp, seek_step):
    silent = _detect_silence(csum, edges, total_ms, min_silence_len, thresh_amp, seek_step)
    if not silent:
        return [[0, total_ms]]
    if silent[0][0] == 0 and silent[-1][1] >= total_ms and len(silent) == 1:
        return []
    out, prev_end = [], 0
    if silent[0][0] == 0:
        prev_end = silent[0][1]
        silent = silent[1:]
    for start, end in silent:
        out.append([prev_end, start])
        prev_end = end
    if prev_end < total_ms:
        out.append([prev_end, total_ms])
    return [r for r in out if r[1] > r[0]]


def _leading_silence_ms(csum, edges, total_ms, thresh_amp, chunk_ms=10) -> int:
    trim = 0
    while trim < total_ms and _window_rms(csum, edges, trim, trim + chunk_ms) < thresh_amp:
        trim += chunk_ms
    return min(trim, total_ms)


def _remove_silence_pydub(audio, sampling_rate, mid_sil, lead_sil, trail_sil,
                          silence_threshold):
    """Upstream's exact path: pydub ``split_on_silence`` + edge trimming."""
    from pydub import AudioSegment
    from pydub.silence import split_on_silence, detect_leading_silence

    a = np.asarray(audio, np.float32)
    if a.ndim == 1:
        a = a[None, :]
    audio_int = (a * MAX_AMPLITUDE).clip(-32768, 32767).astype(np.int16)
    if audio_int.shape[0] > 1:
        audio_int = audio_int.T.flatten()
    wave = AudioSegment(data=audio_int.tobytes(), sample_width=2,
                        frame_rate=sampling_rate, channels=a.shape[0])

    if mid_sil > 0:
        segs = split_on_silence(wave, min_silence_len=mid_sil, silence_thresh=silence_threshold,
                                keep_silence=mid_sil, seek_step=10)
        wave = AudioSegment.silent(duration=0, frame_rate=sampling_rate)
        for seg in segs:
            wave += seg

    start = max(0, detect_leading_silence(wave, silence_threshold=silence_threshold) - lead_sil)
    wave = wave[start:].reverse()
    start = max(0, detect_leading_silence(wave, silence_threshold=silence_threshold) - trail_sil)
    wave = wave[start:].reverse()

    data = np.array(wave.get_array_of_samples()).astype(np.float32) / MAX_AMPLITUDE
    if wave.channels == 1:
        return data[None, :]
    return data.reshape(-1, wave.channels).T


def remove_silence(audio: np.ndarray, sampling_rate: int, mid_sil: int = 300,
                   lead_sil: int = 100, trail_sil: int = 300,
                   silence_threshold: float = -50.0) -> np.ndarray:
    """Drop internal silences longer than ``mid_sil`` ms and trim the edges.

    Args:
        audio: ``(C, T)`` float array in [-1, 1].
        sampling_rate: sample rate of ``audio``.
        mid_sil: internal-silence threshold in ms; 0 skips the internal pass.
        lead_sil: leading silence kept, in ms.
        trail_sil: trailing silence kept, in ms.
        silence_threshold: gate in dBFS.

    Returns:
        ``(C, T')`` float array.
    """
    try:
        return _remove_silence_pydub(audio, sampling_rate, mid_sil, lead_sil,
                                     trail_sil, silence_threshold)
    except ImportError:
        pass

    a = np.asarray(audio, np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if a.shape[-1] == 0:
        return a
    quant = _int16(a)
    csum, edges = _rms_per_ms(quant, sampling_rate)
    total_ms = len(edges) - 1
    thresh = _db_to_amp(silence_threshold)
    spm = sampling_rate / 1000.0

    def cut(ms_ranges):
        parts = [a[:, int(s * spm):int(e * spm)] for s, e in ms_ranges]
        parts = [p for p in parts if p.shape[-1] > 0]
        return np.concatenate(parts, axis=-1) if parts else a[:, :0]

    if mid_sil > 0:
        ranges = _detect_nonsilent(csum, edges, total_ms, mid_sil, thresh, seek_step=10)
        ranges = [[max(0, s - mid_sil), e + mid_sil] for s, e in ranges]
        a = cut(ranges)
        if a.shape[-1] == 0:
            return a
        quant = _int16(a)
        csum, edges = _rms_per_ms(quant, sampling_rate)
        total_ms = len(edges) - 1

    start = max(0, _leading_silence_ms(csum, edges, total_ms, thresh) - lead_sil)
    rev_csum, rev_edges = _rms_per_ms(quant[::-1], sampling_rate)
    end_trim = max(0, _leading_silence_ms(rev_csum, rev_edges, total_ms, thresh) - trail_sil)
    end = max(start, total_ms - end_trim)
    return a[:, int(start * spm):int(end * spm)]


def fade_and_pad_audio(audio: np.ndarray, pad_duration: float = 0.1,
                       fade_duration: float = 0.1, sample_rate: int = 24000) -> np.ndarray:
    """Fade the edges and pad with silence, so playback does not click.

    Vendored unchanged from upstream (it was already numpy).
    """
    if audio.shape[-1] == 0:
        return audio
    fade_samples = int(fade_duration * sample_rate)
    pad_samples = int(pad_duration * sample_rate)
    processed = audio.copy()

    if fade_samples > 0:
        k = min(fade_samples, processed.shape[-1] // 2)
        if k > 0:
            processed[..., :k] = processed[..., :k] * np.linspace(0, 1, k, dtype=np.float32)
            processed[..., -k:] = processed[..., -k:] * np.linspace(1, 0, k, dtype=np.float32)

    if pad_samples > 0:
        silence = np.zeros((processed.shape[0], pad_samples), dtype=processed.dtype)
        processed = np.concatenate([silence, processed, silence], axis=-1)
    return processed
