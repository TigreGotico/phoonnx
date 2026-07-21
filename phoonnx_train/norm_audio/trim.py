from typing import Optional, Tuple

import numpy as np

from .vad import SileroVoiceActivityDetector


def trim_silence(
    audio_array: np.ndarray,
    detector: SileroVoiceActivityDetector,
    threshold: float = 0.2,
    samples_per_chunk=480,
    sample_rate=16000,
    keep_chunks_before: int = 2,
    keep_chunks_after: int = 2,
) -> Tuple[float, Optional[float]]:
    """Returns the offset/duration of trimmed audio in seconds"""
    offset_sec: float = 0.0
    duration_sec: Optional[float] = None
    first_chunk: Optional[int] = None
    last_chunk: Optional[int] = None
    seconds_per_chunk: float = samples_per_chunk / sample_rate

    # Clear any recurrent state left over from a previous utterance so this
    # trim depends only on this clip — reused detectors would otherwise make
    # the survivor set order-dependent (and thus run-dependent).
    detector.reset()

    num_chunks = (len(audio_array) + samples_per_chunk - 1) // samples_per_chunk

    # Determine main block of speech. Every chunk, including the final
    # (possibly short) tail chunk, is scored so a speech onset near the end
    # of the clip is never missed.
    for chunk_idx in range(num_chunks):
        start = chunk_idx * samples_per_chunk
        chunk = audio_array[start:start + samples_per_chunk]
        prob = detector(chunk, sample_rate=sample_rate)
        is_speech = prob >= threshold

        if is_speech:
            if first_chunk is None:
                # First speech
                first_chunk = chunk_idx
            # Last speech so far (a lone speech chunk keeps first == last)
            last_chunk = chunk_idx

    if (first_chunk is not None) and (last_chunk is not None):
        first_chunk = max(0, first_chunk - keep_chunks_before)
        last_chunk = min(num_chunks - 1, last_chunk + keep_chunks_after)

        # Compute offset/duration
        offset_sec = first_chunk * seconds_per_chunk
        last_sec = (last_chunk + 1) * seconds_per_chunk
        duration_sec = last_sec - offset_sec

    return offset_sec, duration_sec
