"""``TTSVoice.prepare`` and ``TTSVoice.render``: the text step and the audio step.

``synthesize`` is ``render`` over ``prepare``. The voices here run a small real
ONNX graph, so the engine call is real. The expected values come from outside
the voice: onnxruntime run on hand-built feeds, phoonnx's own normaliser, and
the ``wave`` module.
"""
import io
import types
import unittest
import warnings
import wave
from dataclasses import FrozenInstanceError

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from phoonnx.audio import write_wav
from phoonnx.config import SynthesisConfig, VoiceConfig
from phoonnx.engines.vits import VitsAdapter
from phoonnx.util import normalize
from phoonnx.version import VERSION_MAJOR
from phoonnx.voice import AudioChunk, TTSVoice, Utterance

LETTERS = " abcdefghijklmnopqrstuvwxyz.,?!"
CONFIG = {"phoonnx_version": "1.0", "engine": "phoonnx", "phoneme_type": "unicode",
          "alphabet": "unicode", "lang_code": "en-US", "audio": {"sample_rate": 16000},
          "num_symbols": 64, "hop_length": 4,
          "inference": {"noise_scale": 0.5, "length_scale": 1.0, "noise_w": 0.5},
          "phoneme_id_map": {"_": [0], "^": [1], "$": [2], **{c: [i + 3] for i, c in enumerate(LETTERS)}}}
SCALES = np.array([0.5, 1.0, 0.5], dtype=np.float32)
TEXT = "Hi there. Who is it?"
RAW = SynthesisConfig(normalize_audio=False)


def _session():
    """``input`` [1, T] -> ``output`` [1, 1, T] = id * sum(scales) / 64, and
    ``durations`` [1, T] = id + 1. The audio of each id is distinct."""
    ids = helper.make_tensor_value_info("input", TensorProto.INT64, [1, "T"])
    lengths = helper.make_tensor_value_info("input_lengths", TensorProto.INT64, [1])
    scales = helper.make_tensor_value_info("scales", TensorProto.FLOAT, [3])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, "T"])
    durations = helper.make_tensor_value_info("durations", TensorProto.FLOAT, [1, "T"])
    nodes = [
        helper.make_node("Cast", ["input"], ["ids_f"], to=TensorProto.FLOAT),
        helper.make_node("ReduceSum", ["scales"], ["scale_sum"], keepdims=0),
        helper.make_node("Mul", ["ids_f", "scale_sum"], ["scaled"]),
        helper.make_node("Div", ["scaled", "denominator"], ["audio"]),
        helper.make_node("Unsqueeze", ["audio", "axis"], ["output"]),
        helper.make_node("Add", ["ids_f", "one"], ["durations"]),
        helper.make_node("Identity", ["input_lengths"], ["unused"]),
    ]
    constants = [numpy_helper.from_array(np.array(64.0, dtype=np.float32), name="denominator"),
                 numpy_helper.from_array(np.array([1], dtype=np.int64), name="axis"),
                 numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="one")]
    graph = helper.make_graph(nodes, "tiny_vits", [ids, lengths, scales], [output, durations], constants)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])


def _engine_audio(session, ids):
    """The graph's audio for *ids*, from onnxruntime with no phoonnx code."""
    feed = {"input": np.array([ids], dtype=np.int64),
            "input_lengths": np.array([len(ids)], dtype=np.int64), "scales": SCALES}
    return session.run(None, feed)[0].squeeze()


class _TextTokenAdapter(VitsAdapter):
    """An engine that reads text: one id per character, one sequence per sentence."""

    def __init__(self):
        super().__init__()
        self.texts = []

    def encode_text(self, text, voice, syn_config):
        self.texts.append(text)
        return [[ord(c) % 60 + 3 for c in sentence] for sentence in text.split(". ")]


def _symbol_voice():
    return TTSVoice(config=VoiceConfig.from_dict(dict(CONFIG)), session=_session())


def _text_voice():
    config = VoiceConfig.from_dict(dict(CONFIG, phoneme_type="graphemes", alphabet="graphemes"))
    return TTSVoice(config=config, session=_session(), adapter=_TextTokenAdapter())


EMPTY_ID_MAP = {c: i for i, c in enumerate(" hitrewos")}
"""A phoneme_id_map with no blank/BOS/EOS: a sentence made only of other
letters tokenizes to an empty id list."""


class _TextSkipAdapter(VitsAdapter):
    """A text engine that gives the second sentence no ids, one sequence per sentence."""

    def __init__(self):
        super().__init__()
        self.texts = []

    def encode_text(self, text, voice, syn_config):
        self.texts.append(text)
        ids = []
        for i, sentence in enumerate(text.split(". ")):
            ids.append([] if i == 1 else [ord(c) % 60 + 3 for c in sentence])
        return ids


def _empty_ids_symbol_voice():
    config = dict(CONFIG)
    config["phoneme_id_map"] = EMPTY_ID_MAP
    return TTSVoice(config=VoiceConfig.from_dict(config), session=_session())


def _text_skip_voice():
    config = VoiceConfig.from_dict(dict(CONFIG, phoneme_type="graphemes", alphabet="graphemes"))
    return TTSVoice(config=config, session=_session(), adapter=_TextSkipAdapter())


def _wav_bytes(write):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        write(wav_file)
    return buffer.getvalue()


class TestPrepare(unittest.TestCase):
    def test_a_symbol_voice_gives_one_utterance_per_sentence(self):
        voice = _symbol_voice()
        utterances = list(voice.prepare(TEXT))
        self.assertEqual([u.symbols for u in utterances],
                         [tuple("Hi there"), tuple("Who is it")])
        # each utterance carries its own chunk's text, not the whole input
        self.assertEqual([u.text for u in utterances], ["Hi there.", "Who is it?"])
        # blank between symbols, BOS and EOS around them, upper case read as lower case
        for utterance in utterances:
            letters = [CONFIG["phoneme_id_map"][s.lower()][0] for s in utterance.symbols]
            expected = [1] + [i for letter in letters for i in (0, letter)] + [0, 2]
            self.assertEqual(utterance.ids, tuple(expected))
            self.assertEqual(dict(utterance.streams), {})

    def test_a_whole_text_utterance_carries_the_joined_text(self):
        voice = _symbol_voice()
        utterances = list(voice.prepare(TEXT, SynthesisConfig(chunk="none")))
        self.assertEqual([u.text for u in utterances], ["Hi there. Who is it?"])

    def test_a_text_voice_gives_the_adapter_ids_and_no_symbols(self):
        voice = _text_voice()
        utterances = list(voice.prepare("Hi there. Who is it 2?"))
        spoken = normalize("Hi there. Who is it 2?", "en-US")
        self.assertEqual(voice.adapter.texts, [spoken])
        self.assertNotIn("2", spoken)
        self.assertEqual([u.symbols for u in utterances], [None, None])
        self.assertEqual([u.ids for u in utterances],
                         [tuple(ord(c) % 60 + 3 for c in s) for s in spoken.split(". ")])

    def test_the_phonetic_spellings_apply_before_the_frontend(self):
        voice = _symbol_voice()
        voice.phonetic_spellings = types.SimpleNamespace(apply=lambda text: text.replace("Hi", "Ha"))
        self.assertEqual(next(voice.prepare(TEXT)).symbols[:2], ("H", "a"))
        off = SynthesisConfig(enable_phonetic_spellings=False)
        self.assertEqual(next(voice.prepare(TEXT, off)).symbols[:2], ("H", "i"))

    def test_an_utterance_is_frozen(self):
        utterance = next(_symbol_voice().prepare(TEXT))
        with self.assertRaises(FrozenInstanceError):
            utterance.ids = (1,)

    def test_the_second_sentence_is_read_after_the_first_is_taken(self):
        voice = _symbol_voice()
        read = []
        tokenize = voice.tokenizer.tokenize
        voice.config.tokenizer = types.SimpleNamespace(
            tokenize=lambda symbols: read.append(symbols) or tokenize(symbols))
        utterances = voice.prepare(TEXT)
        self.assertEqual(read, [])
        next(utterances)
        self.assertEqual(len(read), 1)


class TestPrepareSkipsEmptyIds(unittest.TestCase):
    """A sentence whose ids are empty yields no utterance and no chunk."""

    TEXT = "Hi there. Buzz. Who is it"

    def test_a_symbol_voice_skips_a_sentence_with_no_ids(self):
        voice = _empty_ids_symbol_voice()
        utterances = list(voice.prepare(self.TEXT))
        # "Buzz" is dropped: every symbol is out of vocabulary, and the map
        # holds no blank/BOS/EOS to leave ids behind.
        self.assertEqual([u.symbols for u in utterances],
                         [tuple("Hi there"), tuple("Who is it")])
        for utterance in utterances:
            expected = tuple(EMPTY_ID_MAP[s.lower()] for s in utterance.symbols)
            self.assertEqual(utterance.ids, expected)
        self.assertEqual(len(list(voice.synthesize(self.TEXT))), 2)

    def test_a_text_voice_skips_a_sentence_with_no_ids(self):
        voice = _text_skip_voice()
        utterances = list(voice.prepare(self.TEXT))
        spoken = normalize(self.TEXT, "en-US")
        expected = [tuple(ord(c) % 60 + 3 for c in sentence)
                    for i, sentence in enumerate(spoken.split(". ")) if i != 1]
        self.assertEqual([u.symbols for u in utterances], [None, None])
        self.assertEqual([u.ids for u in utterances], expected)
        self.assertEqual(len(list(voice.synthesize(self.TEXT))), 2)


class TestRender(unittest.TestCase):
    def test_render_runs_the_engine_on_the_utterance_ids(self):
        voice = _symbol_voice()
        utterance = Utterance(text="", symbols=None, ids=(5, 9, 14))
        chunk = voice.render(utterance, RAW)
        self.assertIsInstance(chunk, AudioChunk)
        np.testing.assert_array_equal(chunk.audio_float_array,
                                      _engine_audio(voice.session, [5, 9, 14]).astype(np.float32))
        self.assertEqual(chunk.phoneme_ids, [5, 9, 14])
        self.assertEqual(chunk.phonemes, [])
        self.assertEqual(chunk.sample_rate, 16000)
        self.assertIsNone(chunk.phoneme_alignments)

    def test_render_levels_and_scales_the_audio(self):
        voice = _symbol_voice()
        utterance = Utterance(text="", symbols=None, ids=(5, 9, 14))
        chunk = voice.render(utterance, SynthesisConfig(volume=0.5))
        engine = _engine_audio(voice.session, [5, 9, 14])
        np.testing.assert_allclose(chunk.audio_float_array, engine / engine.max() * 0.5, rtol=1e-6)

    def test_render_with_alignments_returns_a_chunk_with_durations(self):
        voice = _symbol_voice()
        utterance = next(voice.prepare("ab"))
        chunk = voice.render(utterance, alignments=True)
        self.assertIsInstance(chunk, AudioChunk)
        np.testing.assert_array_equal(chunk.phoneme_id_samples, (np.array(utterance.ids) + 1) * 4)
        self.assertEqual([a.phoneme for a in chunk.phoneme_alignments], ["a", "b"])
        self.assertEqual(sum(a.num_samples for a in chunk.phoneme_alignments),
                         int(chunk.phoneme_id_samples.sum()))

    def test_render_passes_the_streams_to_the_engine(self):
        voice = _symbol_voice()
        seen = []
        synthesize = voice.adapter.synthesize
        voice.adapter.synthesize = lambda request, session: seen.append(request) or synthesize(request, session)
        voice.render(Utterance(text="", symbols=("a", "b"), ids=(5, 6), streams={"language_ids": (3, 4)}))
        np.testing.assert_array_equal(seen[0].language_ids, [[3, 4]])

    def test_render_passes_the_utterance_text_to_the_adapter(self):
        voice = _symbol_voice()
        seen = []
        synthesize = voice.adapter.synthesize
        voice.adapter.synthesize = lambda request, session: seen.append(request) or synthesize(request, session)
        voice.render(Utterance(text="Hi there.", symbols=("H", "i"), ids=(5, 6)))
        self.assertEqual(seen[0].text, "Hi there.")


class TestSynthesizeIsRenderOverPrepare(unittest.TestCase):
    def assert_same_chunks(self, voice, text, config=None, alignments=False):
        expected = [voice.render(u, config, alignments=alignments) for u in voice.prepare(text, config)]
        chunks = list(voice.synthesize(text, config, include_alignments=alignments))
        self.assertEqual(len(chunks), len(expected))
        self.assertGreater(len(chunks), 1)
        for chunk, want in zip(chunks, expected):
            self.assertEqual(chunk.audio_int16_bytes, want.audio_int16_bytes)
            self.assertEqual(chunk.phoneme_ids, want.phoneme_ids)
            self.assertEqual(chunk.phonemes, want.phonemes)
            self.assertEqual(chunk.sample_rate, want.sample_rate)
            self.assertEqual(chunk.phoneme_alignments, want.phoneme_alignments)
        return chunks

    def test_a_symbol_voice(self):
        voice = _symbol_voice()
        chunks = self.assert_same_chunks(voice, TEXT)
        self.assert_same_chunks(voice, TEXT, SynthesisConfig(length_scale=1.5, volume=0.7), alignments=True)
        # the engine audio of the first sentence, peak-normalised
        engine = _engine_audio(voice.session, chunks[0].phoneme_ids)
        np.testing.assert_allclose(chunks[0].audio_float_array, engine / engine.max(), rtol=1e-6)

    def test_a_text_voice(self):
        voice = _text_voice()
        chunks = self.assert_same_chunks(voice, "Hi there. Who is it 2?")
        self.assertEqual([c.phonemes for c in chunks], [[], []])


class TestWriteWav(unittest.TestCase):
    def test_synthesize_wav_writes_the_chunks_of_synthesize(self):
        voice = _symbol_voice()
        chunks = list(voice.synthesize(TEXT))
        written = _wav_bytes(lambda wav_file: voice.synthesize_wav(TEXT, wav_file))
        with wave.open(io.BytesIO(written)) as wav_file:
            self.assertEqual(wav_file.getframerate(), 16000)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getnchannels(), 1)
            frames = wav_file.readframes(wav_file.getnframes())
        self.assertEqual(frames, b"".join(c.audio_int16_bytes for c in chunks))
        self.assertEqual(written, _wav_bytes(lambda wav_file: write_wav(chunks, wav_file)))

    def test_the_first_chunk_sets_the_format(self):
        chunk = AudioChunk(sample_rate=48000, sample_width=2, sample_channels=1,
                           audio_float_array=np.zeros(3, dtype=np.float32))
        written = _wav_bytes(lambda wav_file: write_wav([chunk], wav_file, sample_rate=16000))
        with wave.open(io.BytesIO(written)) as wav_file:
            self.assertEqual(wav_file.getframerate(), 48000)
            self.assertEqual(wav_file.getnframes(), 3)

    def test_no_chunks_give_an_empty_wav_at_the_given_rate(self):
        written = _wav_bytes(lambda wav_file: write_wav([], wav_file, sample_rate=22050))
        with wave.open(io.BytesIO(written)) as wav_file:
            self.assertEqual(wav_file.getframerate(), 22050)
            self.assertEqual(wav_file.getnframes(), 0)

    def test_synthesize_ssml_wav_writes_the_chunks_of_synthesize_ssml(self):
        voice = _symbol_voice()
        ssml = '<speak>Hi there.<break time="100ms"/><mark name="m"/>Who is it?</speak>'
        chunks = list(voice.synthesize_ssml(ssml))
        written = _wav_bytes(lambda wav_file: voice.synthesize_ssml_wav(ssml, wav_file))
        self.assertEqual(written, _wav_bytes(lambda wav_file: write_wav(chunks, wav_file)))


class TestDeprecatedShims(unittest.TestCase):
    removal = f"removed in phoonnx {VERSION_MAJOR + 1}.0.0"

    def test_phonemes_to_ids_warns_and_tokenizes(self):
        voice = _symbol_voice()
        with self.assertWarns(DeprecationWarning) as caught:
            ids = voice.phonemes_to_ids(["a", "b"])
        self.assertIn(self.removal, str(caught.warning))
        self.assertIn("tokenizer.tokenize", str(caught.warning))
        self.assertEqual(ids, [1, 0, 4, 0, 5, 0, 2])

    def test_phoneme_ids_to_audio_warns_and_returns_the_engine_audio(self):
        voice = _symbol_voice()
        with self.assertWarns(DeprecationWarning) as caught:
            audio = voice.phoneme_ids_to_audio([5, 9, 14])
        self.assertIn(self.removal, str(caught.warning))
        self.assertIn("TTSVoice.render", str(caught.warning))
        np.testing.assert_array_equal(audio, _engine_audio(voice.session, [5, 9, 14]))
        with self.assertWarns(DeprecationWarning):
            audio, samples = voice.phoneme_ids_to_audio([5, 9], include_alignments=True)
        np.testing.assert_array_equal(samples, [24, 40])

    def test_synthesis_uses_no_deprecated_name(self):
        for voice in (_symbol_voice(), _text_voice()):
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                list(voice.synthesize(TEXT, include_alignments=True))
                list(voice.synthesize_ssml(f"<speak>{TEXT}</speak>"))
                _wav_bytes(lambda wav_file: voice.synthesize_wav(TEXT, wav_file))


if __name__ == "__main__":
    unittest.main()
