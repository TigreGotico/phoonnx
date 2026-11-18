"""``TTSVoice.stream``: audio as the engine makes it, joining to the ``synthesize`` audio."""
import numpy as np
import pytest

from phoonnx.config import SynthesisConfig, VoiceConfig
from phoonnx.voice import TTSVoice
from tests.test_pockettts_stream import DECODE_CHUNK_FRAMES, _RecordingFlowLM, _voice as _pocket_adapter
from tests.test_prepare_render import CONFIG, _text_voice

TEXT = "Hi there, my friend. Who is it at the door?"
FRAMES = 2 * DECODE_CHUNK_FRAMES + 3
RAW = SynthesisConfig(normalize_audio=False)


def _pocket_voice(events=None):
    """A Pocket TTS voice on stub graphs that makes FRAMES latent frames per sentence."""
    events = [] if events is None else events
    config = VoiceConfig.from_dict(dict(CONFIG, phoneme_type="graphemes", alphabet="graphemes"))
    adapter = _pocket_adapter(FRAMES, events)
    adapter.seed = 5
    return TTSVoice(config=config, session=_RecordingFlowLM(events), adapter=adapter)


def _joined(chunks):
    return np.concatenate([c.audio_float_array for c in chunks])


@pytest.mark.parametrize("config", [RAW, SynthesisConfig(), SynthesisConfig(volume=0.5, normalize_audio=False)])
def test_pocket_stream_joins_to_the_synthesized_audio(config):
    whole = list(_pocket_voice().synthesize(TEXT, config))
    parts = list(_pocket_voice().stream(TEXT, config))
    assert len(whole) == 2
    assert np.array_equal(_joined(parts), _joined(whole))
    assert _joined(parts).dtype == _joined(whole).dtype
    assert [i for c in parts for i in c.phoneme_ids] == [i for c in whole for i in c.phoneme_ids]


def test_pocket_stream_yields_slices_without_whole_sentence_steps():
    parts = list(_pocket_voice().stream(TEXT, RAW))
    # 33 frames per sentence decode as 15 + 15 + 3, four samples per frame
    assert [len(c.audio_float_array) for c in parts] == [60, 60, 12] * 2


def test_pocket_stream_gives_its_first_part_before_the_last_flow_step():
    events = []
    stream = _pocket_voice(events).stream(TEXT, RAW)
    next(stream)
    steps_before_first_part = events.count("step")
    list(stream)
    assert steps_before_first_part < events.count("step")


class _PassThroughSR:
    def upscale(self, audio, rate):
        return audio, rate


@pytest.mark.parametrize("config", [
    SynthesisConfig(),
    SynthesisConfig(normalize_audio=False, target_loudness=-20.0),
    SynthesisConfig(normalize_audio=False, super_resolution=True),
    SynthesisConfig(normalize_audio=False, vc_reference="target.wav"),
])
def test_a_whole_sentence_step_gives_one_chunk_per_sentence(config):
    voices = [_pocket_voice(), _pocket_voice()]
    for voice in voices:
        voice._sr_engine, voice._sr_loaded = _PassThroughSR(), True
        # a clip shorter than 0.3 s passes voice conversion untouched
        voice._load_voice_conversion = lambda syn_config: object() if syn_config.vc_reference else None
    whole = list(voices[0].synthesize(TEXT, config))
    parts = list(voices[1].stream(TEXT, config))
    assert len(parts) == len(whole) == 2
    assert np.array_equal(_joined(parts), _joined(whole))


def test_an_engine_without_the_hook_gives_the_synthesize_chunks():
    whole = list(_text_voice().synthesize("Hi there. Who is it?", RAW))
    parts = list(_text_voice().stream("Hi there. Who is it?", RAW))
    assert len(parts) == len(whole) == 2
    for part, chunk in zip(parts, whole):
        assert np.array_equal(part.audio_float_array, chunk.audio_float_array)
        assert part.audio_float_array.dtype == chunk.audio_float_array.dtype
        assert (part.phonemes, part.phoneme_ids, part.sample_rate) == (
            chunk.phonemes, chunk.phoneme_ids, chunk.sample_rate)
