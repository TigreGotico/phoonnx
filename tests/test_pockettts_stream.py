"""Pocket TTS streaming: ``synthesize_stream`` yields the decoded audio of each slice of
latent frames as soon as the flow LM has made it, and the parts joined are the samples
that ``synthesize`` returns."""
import numpy as np
import pytest

from phoonnx.engines.base import AdapterSynthesisRequest, AdapterSynthesisResult, BaseOnnxAdapter
from tests.test_pockettts import _FlowLMSession, _adapter

TOKENS = tuple(range(1, 31))
# Latent frames per decoder call, as Kyutai's reference decodes them.
DECODE_CHUNK_FRAMES = 15


class _StatefulDecoder:
    """Stand-in for mimi_decoder whose audio depends on the latents and on the state the
    previous call returned, so a slice decoded with the wrong state gives other samples."""

    def __init__(self, events):
        self.events = events

    def run(self, output_names, feed):
        self.events.append("decode")
        latent, previous = feed["latent"], feed["state_1"]
        audio = np.repeat(latent.mean(axis=2), 4, axis=1) + previous.sum()
        return [audio.reshape(1, 1, -1).astype(np.float32),
                np.zeros_like(feed["state_0"]), previous + latent.sum()]


class _RecordingFlowLM(_FlowLMSession):
    def __init__(self, events, eos_at=0):
        super().__init__(eos_at=eos_at)
        self.events = events

    def run(self, output_names, feed):
        self.events.append("step")
        return super().run(output_names, feed)


def _req(**params):
    ids = np.array([TOKENS], np.int64)
    return AdapterSynthesisRequest(phoneme_ids=ids, phoneme_lengths=np.array([ids.shape[1]], np.int64),
                                   params={"temperature": 0.7, "seed": 5, **params})


def _voice(frames, events):
    """An adapter whose flow LM makes exactly *frames* latent frames."""
    adapter = _adapter()
    adapter.model_recommended_frames_after_eos = frames
    adapter.mimi_decoder = _StatefulDecoder(events)
    return adapter


@pytest.mark.parametrize("frames", [0, 1, 7, DECODE_CHUNK_FRAMES, DECODE_CHUNK_FRAMES + 1,
                                    2 * DECODE_CHUNK_FRAMES, 3 * DECODE_CHUNK_FRAMES - 1])
def test_joined_stream_is_the_synthesized_audio(frames):
    whole = _voice(frames, []).synthesize(_req(), _RecordingFlowLM([]))
    parts = list(_voice(frames, []).synthesize_stream(_req(), _RecordingFlowLM([])))
    assert whole.extras["frames"] == frames
    assert np.array_equal(np.concatenate([p.audio for p in parts]), whole.audio)
    assert all(p.audio.dtype == np.float32 for p in parts)
    assert [p.extras["frames"] for p in parts] == (
        [DECODE_CHUNK_FRAMES] * (frames // DECODE_CHUNK_FRAMES)
        + ([frames % DECODE_CHUNK_FRAMES] if frames % DECODE_CHUNK_FRAMES or not frames else []))


def test_first_part_comes_before_the_last_latent_step():
    events = []
    frames = 2 * DECODE_CHUNK_FRAMES + 3
    stream = _voice(frames, events).synthesize_stream(_req(), _RecordingFlowLM(events))
    next(stream)
    steps_before_first_part = events.count("step")
    list(stream)
    assert steps_before_first_part < events.count("step")
    # events[0] primes the flow LM; each decode follows the step that made the last frame
    # of its slice, and the tail is decoded after the step that ends the loop
    assert [i for i, e in enumerate(events) if e == "decode"] == [
        1 + DECODE_CHUNK_FRAMES, 2 + 2 * DECODE_CHUNK_FRAMES, len(events) - 1]


class _OneGraphAdapter(BaseOnnxAdapter):
    def build_feed_dict(self, request, session):
        return {"ids": request.phoneme_ids}

    def parse_outputs(self, outputs, request, output_names=None):
        return AdapterSynthesisResult(audio=outputs[0], extras={"n": len(outputs[0])})

    def default_params(self):
        return {}


class _OneGraphSession:
    def run(self, output_names, feed):
        return [np.linspace(-1, 1, feed["ids"].shape[1] * 3, dtype=np.float32)]


def test_default_stream_yields_the_one_synthesized_result():
    adapter = _OneGraphAdapter()
    whole = adapter.synthesize(_req(), _OneGraphSession())
    parts = list(adapter.synthesize_stream(_req(), _OneGraphSession()))
    assert len(parts) == 1
    assert np.array_equal(parts[0].audio, whole.audio)
    assert parts[0].extras == whole.extras
