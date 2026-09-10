from typing import Optional, Tuple

import numpy as np

from .vad import SileroVoiceActivityDetector

# Shortest input the Silero model scores. Below it the ONNX Pad node raises
# rather than returning a probability.
DETECTOR_MIN_SAMPLES = 129


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

    # Determine main block of speech. Every chunk is scored, including the
    # final (possibly short) tail chunk, which is zero-padded only up to the
    # detector's own floor: below that it raises and the exception aborts the
    # whole utterance rather than the chunk. Padding no further leaves every
    # tail the detector already accepted scoring on exactly the samples it
    # scored before, so this cannot move a boundary on a clip that worked.
    # It also does not make a short tail audible -- a speech onset inside one
    # can still fall under the threshold, and does.
    for chunk_idx in range(num_chunks):
        start = chunk_idx * samples_per_chunk
        chunk = audio_array[start:start + samples_per_chunk]
        if len(chunk) < DETECTOR_MIN_SAMPLES:
            chunk = np.pad(chunk, (0, DETECTOR_MIN_SAMPLES - len(chunk)))
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
