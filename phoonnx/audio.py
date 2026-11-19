"""Write synthesized audio chunks to a WAV file."""
import wave
from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from phoonnx.voice import AudioChunk


def write_wav(chunks: Iterable["AudioChunk"], wav_file: wave.Wave_write, *,
              sample_rate: Optional[int] = None, set_wav_format: bool = True) -> None:
    """Write a stream of :class:`~phoonnx.voice.AudioChunk` to an open WAV writer.

    The first chunk sets the format of the file: its sample rate, sample width
    and channel count. Voice conversion and super-resolution can change the
    rate, so the voice's own rate is not always correct.

    :param chunks: The chunks to write, in order, such as the output of
        ``TTSVoice.synthesize`` or ``TTSVoice.synthesize_ssml``.
    :param wav_file: A writer from ``wave.open(path, "wb")``.
    :param sample_rate: The rate to set before the first chunk, so that a
        stream with no chunks still gives a valid, empty WAV file.
    :param set_wav_format: False leaves the format of ``wav_file`` as the
        caller set it.
    """
    if set_wav_format and sample_rate is not None:
        wav_file.setframerate(sample_rate)
        wav_file.setsampwidth(2)
        wav_file.setnchannels(1)
    first = True
    for chunk in chunks:
        if first and set_wav_format:
            wav_file.setframerate(chunk.sample_rate)
            wav_file.setsampwidth(chunk.sample_width)
            wav_file.setnchannels(chunk.sample_channels)
        first = False
        wav_file.writeframes(chunk.audio_int16_bytes)
