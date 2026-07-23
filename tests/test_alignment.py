"""Tests for phoneme alignment (PhonemeAlignment, AudioChunk, TTSVoice alignment logic).

Unit tests are hermetic — no real models, no network.
Integration tests require the downloaded eu-ES dii espeak voice and a patched
alignment ONNX model at ALIGNED_MODEL (set up by setUpClass).

Run unit tests only:
    pytest tests/test_alignment.py -k "not Integration"

Run everything (requires network on first run to download the voice):
    pytest tests/test_alignment.py
"""
import os
import shutil
import tempfile
import unittest
from dataclasses import fields, is_dataclass
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from phoonnx.config import VoiceConfig, DEFAULT_HOP_LENGTH, PhonemeType, Alphabet
from phoonnx.engines.vits import VitsAdapter
from phoonnx.voice import AudioChunk, PhonemeAlignment, TTSVoice

from tests.conftest import retry_download


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

class _Input:
    """Minimal stand-in for an onnxruntime model input."""
    def __init__(self, name):
        self.name = name


class _FakePhonemizer:
    """A minimal phonemizer honouring the parts of the BasePhonemizer contract the
    conversion route uses: ``phonemize`` plus the per-sentence ``phonemize_lazy``
    generator (a MagicMock would return non-iterable sentinels instead).
    """
    def __init__(self, result):
        self._result = result

    def phonemize(self, text, lang=None):
        return self._result

    def phonemize_lazy(self, text, lang=None):
        yield from self._result

    def add_diacritics(self, text, lang, model=None):
        return text


def _fake_audio(n=200):
    return np.linspace(-0.5, 0.5, n, dtype=np.float32)


def _make_vocab(char2idx=None):
    """Build a minimal Vocabulary mock with the standard special tokens."""
    char2idx = char2idx or {"_": 0, "^": 1, "$": 2, "h": 3, "e": 4, "l": 5, "o": 6}
    idx2char = {v: k for k, v in char2idx.items()}
    vocab = MagicMock()
    vocab.char2idx = char2idx
    vocab.idx2char = idx2char
    vocab.blank_id = 0
    vocab.bos_id = 1
    vocab.eos_id = 2
    vocab.pad_id = 0
    vocab.blank = "_"
    vocab.bos = "^"
    vocab.eos = "$"
    vocab.pad = "_"
    return vocab


def _make_tokenizer(char2idx=None):
    """Build a minimal TTSTokenizer mock."""
    vocab = _make_vocab(char2idx)
    tok = MagicMock()
    tok.vocabulary = vocab
    tok.blank_id = 0
    tok.pad_id = 0
    tok.add_blank_char = True
    tok.use_eos_bos = True
    tok.blank_at_start = True
    tok.blank_at_end = True
    return tok


def _make_voice(session_outputs, phonemize_result=None, token_ids=None, char2idx=None):
    """Build a TTSVoice with a mocked ONNX session.

    ``voice.phonemize`` and ``voice.phonemes_to_ids`` are mocked at the
    instance level to bypass all text-processing paths.

    session_outputs: list of numpy arrays the fake session.run() returns.
    token_ids: the full sequence including bos/blank/eos that the tokenizer
               would normally produce (matches what phoneme_id_samples covers).
    """
    if phonemize_result is None:
        phonemize_result = [["h", "e", "l", "l", "o"]]
    if token_ids is None:
        # Simple sequence: [bos, blank, h, blank, e, blank, l, blank, l, blank, o, blank, eos]
        token_ids = [1, 0, 3, 0, 4, 0, 5, 0, 5, 0, 6, 0, 2]

    session = MagicMock()
    session.get_inputs.return_value = [_Input("input"), _Input("input_lengths")]
    session.run.return_value = session_outputs

    cfg = MagicMock(spec=VoiceConfig)
    cfg.lang_code = "en-US"
    cfg.sample_rate = 22050
    cfg.num_speakers = 1
    cfg.phoneme_type = PhonemeType.UNICODE
    # A phoneme model (tgt != GRAPHEMES) so synthesize's alphabet dispatch takes
    # the phonemize hot path (src GRAPHEMES -> tgt IPA), exercising the lazy
    # per-sentence stream that carries phonemes + ids into each AudioChunk.
    cfg.alphabet = Alphabet.IPA
    cfg.noise_scale = 0.667
    cfg.length_scale = 1.0
    cfg.noise_w_scale = 0.8
    cfg.hop_length = DEFAULT_HOP_LENGTH
    cfg.enable_phonetic_spellings = False
    cfg.add_diacritics = False
    cfg.diacritizer_model = None
    cfg.engine_params = {}
    cfg.tokenizer = _make_tokenizer(char2idx)

    voice = TTSVoice.__new__(TTSVoice)
    voice.session = session
    voice.config = cfg
    voice.phonemizer = _FakePhonemizer(phonemize_result)
    voice.phonetic_spellings = None
    voice.adapter = VitsAdapter()
    voice.phonemes_to_ids = MagicMock(return_value=token_ids)
    return voice


# ---------------------------------------------------------------------------
# PhonemeAlignment dataclass
# ---------------------------------------------------------------------------

class TestPhonemeAlignment(unittest.TestCase):
    def test_is_dataclass(self):
        self.assertTrue(is_dataclass(PhonemeAlignment))

    def test_fields(self):
        pa = PhonemeAlignment(phoneme="a", num_samples=512)
        self.assertEqual(pa.phoneme, "a")
        self.assertEqual(pa.num_samples, 512)

    def test_equality(self):
        self.assertEqual(PhonemeAlignment("b", 100), PhonemeAlignment("b", 100))
        self.assertNotEqual(PhonemeAlignment("b", 100), PhonemeAlignment("b", 200))


# ---------------------------------------------------------------------------
# AudioChunk new fields
# ---------------------------------------------------------------------------

class TestAudioChunkFields(unittest.TestCase):
    def _chunk(self, **kw):
        defaults = dict(
            sample_rate=22050, sample_width=2, sample_channels=1,
            audio_float_array=_fake_audio(), phonemes=[], phoneme_ids=[],
        )
        defaults.update(kw)
        return AudioChunk(**defaults)

    def test_phonemes_and_ids_stored(self):
        c = self._chunk(phonemes=["h", "i"], phoneme_ids=[3, 4])
        self.assertEqual(c.phonemes, ["h", "i"])
        self.assertEqual(c.phoneme_ids, [3, 4])

    def test_optional_fields_default_none(self):
        c = self._chunk()
        self.assertIsNone(c.phoneme_id_samples)
        self.assertIsNone(c.phoneme_alignments)

    def test_alignment_objects_stored(self):
        aligns = [PhonemeAlignment("h", 256), PhonemeAlignment("i", 512)]
        c = self._chunk(phoneme_alignments=aligns)
        self.assertEqual(c.phoneme_alignments[0].phoneme, "h")

    def test_id_samples_array_stored(self):
        arr = np.array([100, 200, 300], dtype=np.int64)
        c = self._chunk(phoneme_id_samples=arr)
        np.testing.assert_array_equal(c.phoneme_id_samples, arr)


# ---------------------------------------------------------------------------
# VoiceConfig hop_length
# ---------------------------------------------------------------------------

class TestHopLength(unittest.TestCase):
    def test_default_hop_length_constant(self):
        self.assertEqual(DEFAULT_HOP_LENGTH, 256)

    def test_hop_length_field_default(self):
        hop_field = next(f for f in fields(VoiceConfig) if f.name == "hop_length")
        self.assertEqual(hop_field.default, DEFAULT_HOP_LENGTH)

    def _phoonnx_cfg(self, **extra):
        cfg = {
            "phoonnx_version": "0.0.1",
            "phoneme_type": "espeak",
            "alphabet": "ipa",
            "audio": {"sample_rate": 22050},
            "inference": {},
            "phoneme_id_map": {"_": 0, "^": 1, "$": 2},
        }
        cfg.update(extra)
        return cfg

    def test_from_dict_parses_hop_length(self):
        cfg = VoiceConfig.from_dict(self._phoonnx_cfg(hop_length=512))
        self.assertEqual(cfg.hop_length, 512)

    def test_from_dict_hop_length_default(self):
        cfg = VoiceConfig.from_dict(self._phoonnx_cfg())
        self.assertEqual(cfg.hop_length, DEFAULT_HOP_LENGTH)


# ---------------------------------------------------------------------------
# phoneme_ids_to_audio — return-value shape
# ---------------------------------------------------------------------------

class TestPhonemeIdsToAudio(unittest.TestCase):
    def test_without_alignments_returns_ndarray(self):
        voice = _make_voice([_fake_audio()[np.newaxis, np.newaxis, :]])
        result = voice.phoneme_ids_to_audio([3, 4, 5], include_alignments=False)
        self.assertIsInstance(result, np.ndarray)

    def test_with_alignments_single_output_returns_tuple_none(self):
        voice = _make_voice([_fake_audio()[np.newaxis, np.newaxis, :]])
        audio, samples = voice.phoneme_ids_to_audio([1, 0, 3, 0, 2], include_alignments=True)
        self.assertIsInstance(audio, np.ndarray)
        self.assertIsNone(samples)

    def test_with_alignments_two_outputs_returns_tuple_array(self):
        ids = [1, 0, 3, 0, 4, 0, 2]
        audio_arr = _fake_audio()[np.newaxis, np.newaxis, :]
        dur_arr = np.ones((1, len(ids)), dtype=np.float32)
        voice = _make_voice([audio_arr, dur_arr])
        audio, samples = voice.phoneme_ids_to_audio(ids, include_alignments=True)
        self.assertIsInstance(samples, np.ndarray)
        self.assertEqual(samples.dtype, np.int64)
        self.assertEqual(len(samples), len(ids))

    def test_hop_length_scales_duration(self):
        """Model durations (in frames) are multiplied by hop_length to get samples."""
        ids = [1, 0, 3, 0, 4, 0, 2]
        raw = np.array([[2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 2.0]], dtype=np.float32)
        voice = _make_voice([_fake_audio()[np.newaxis, np.newaxis, :], raw])
        voice.config.hop_length = 100
        _, samples = voice.phoneme_ids_to_audio(ids, include_alignments=True)
        expected = np.array([200, 100, 300, 100, 400, 100, 200], dtype=np.int64)
        np.testing.assert_array_equal(samples, expected)


# ---------------------------------------------------------------------------
# synthesize — alignment reconstruction
# ---------------------------------------------------------------------------

class TestAlignmentReconstruction(unittest.TestCase):
    """Tests for the id→phoneme walk in synthesize(include_alignments=True).

    The new implementation uses tokenizer.vocabulary.idx2char to look up each
    id and folds blank/bos/eos durations into adjacent real phonemes.
    """

    def _synth(self, token_ids, durations, phonemes=None, char2idx=None):
        """Run synthesize and return the single AudioChunk."""
        if phonemes is None:
            phonemes = ["h", "e", "l"]
        audio = _fake_audio()
        dur_arr = np.array([durations], dtype=np.float32)
        voice = _make_voice(
            [audio[np.newaxis, np.newaxis, :], dur_arr],
            phonemize_result=[phonemes],
            token_ids=token_ids,
            char2idx=char2idx,
        )
        voice.config.hop_length = 1  # frame == sample for easy arithmetic
        chunks = list(voice.synthesize("x", include_alignments=True))
        self.assertEqual(len(chunks), 1)
        return chunks[0]

    def test_real_phonemes_extracted(self):
        # ids: [bos=1, blank=0, h=3, blank=0, e=4, blank=0, l=5, blank=0, eos=2]
        ids = [1, 0, 3, 0, 4, 0, 5, 0, 2]
        durs = [10, 5, 20, 5, 30, 5, 40, 5, 10]
        chunk = self._synth(ids, durs, phonemes=["h", "e", "l"])
        self.assertIsNotNone(chunk.phoneme_alignments)
        names = [a.phoneme for a in chunk.phoneme_alignments]
        self.assertEqual(names, ["h", "e", "l"])

    def test_blank_before_phoneme_absorbed_forward(self):
        # ids: bos(10), blank(5), h(20), trailing-blank(5), eos(10)
        # bos + blank_before_h → pending=15; h appended with 15+20=35
        # trailing blank → pending=5; eos → absorbed into h: 35+10=45
        # after loop: trailing pending 5 → absorbed into h: 45+5=50
        ids = [1, 0, 3, 0, 2]
        durs = [10, 5, 20, 5, 10]
        chunk = self._synth(ids, durs, phonemes=["h"])
        self.assertEqual(chunk.phoneme_alignments[0].num_samples, 50)

    def test_eos_absorbed_into_last_phoneme(self):
        # ids: bos, blank, h, blank, eos(10)
        ids = [1, 0, 3, 0, 2]
        durs = [0, 0, 20, 5, 10]
        chunk = self._synth(ids, durs, phonemes=["h"])
        # h = 0+0 (bos+blank_bos) + 20 + 5 (trailing blank) + 10 (eos) = 35
        # bos=0, blank=0 → pending=0; h=20 → PhonemeAlignment("h",0+20)=20; blank=5 → pending=5; eos=10 → absorbed into last → 20+10=30; then trailing pending 5 absorbed → 35
        # Actually: bos=0 pending=0, blank_bos=0 pending=0, h=20 → align(h,20), blank=5 → pending=5, eos=10 → absorbed into last: h.num_samples=20+10=30, then pending=5 absorbed into last: 30+5=35
        self.assertEqual(chunk.phoneme_alignments[0].num_samples, 35)

    def test_total_samples_equal_sum_of_durations(self):
        ids = [1, 0, 3, 0, 4, 0, 5, 0, 2]
        durs = [10, 5, 20, 5, 30, 5, 40, 5, 10]
        chunk = self._synth(ids, durs, phonemes=["h", "e", "l"])
        total_aligned = sum(a.num_samples for a in chunk.phoneme_alignments)
        self.assertEqual(total_aligned, sum(durs))

    def test_length_mismatch_alignments_none(self):
        """phoneme_id_samples length != phoneme_ids length → alignments stay None."""
        # 3 duration values but 9 ids
        audio = _fake_audio()
        ids = [1, 0, 3, 0, 4, 0, 5, 0, 2]
        voice = _make_voice(
            [audio[np.newaxis, np.newaxis, :],
             np.array([[10.0, 20.0, 30.0]], dtype=np.float32)],
            phonemize_result=[["h", "e", "l"]],
            token_ids=ids,
        )
        voice.config.hop_length = 1
        chunk = list(voice.synthesize("x", include_alignments=True))[0]
        self.assertIsNone(chunk.phoneme_alignments)

    def test_unknown_id_becomes_question_mark(self):
        """An id not in idx2char maps to '?' but doesn't crash or fail."""
        char2idx = {"_": 0, "^": 1, "$": 2, "h": 3}
        ids = [1, 0, 3, 0, 99, 0, 2]  # id 99 unknown
        durs = [5, 5, 20, 5, 20, 5, 5]
        chunk = self._synth(ids, durs, phonemes=["h", "?"], char2idx=char2idx)
        # reconstruction should succeed; unknown id maps to "?"
        self.assertIsNotNone(chunk.phoneme_alignments)
        names = [a.phoneme for a in chunk.phoneme_alignments]
        self.assertIn("?", names)

    def test_no_alignments_requested_fields_none(self):
        ids = [1, 0, 3, 0, 4, 0, 2]
        audio = _fake_audio()
        dur = np.ones((1, len(ids)), dtype=np.float32)
        voice = _make_voice([audio[np.newaxis, np.newaxis, :], dur],
                            phonemize_result=[["h", "e"]], token_ids=ids)
        chunk = list(voice.synthesize("hi", include_alignments=False))[0]
        self.assertIsNone(chunk.phoneme_id_samples)
        self.assertIsNone(chunk.phoneme_alignments)

    def test_phonemes_and_ids_always_populated(self):
        ids = [1, 0, 3, 0, 4, 0, 2]
        audio = _fake_audio()
        voice = _make_voice([audio[np.newaxis, np.newaxis, :]],
                            phonemize_result=[["h", "e"]], token_ids=ids)
        chunk = list(voice.synthesize("hi", include_alignments=False))[0]
        self.assertEqual(chunk.phonemes, ["h", "e"])
        self.assertEqual(chunk.phoneme_ids, ids)

    def test_empty_phonemes_yields_no_chunk(self):
        audio = _fake_audio()
        voice = _make_voice([audio[np.newaxis, np.newaxis, :]],
                            phonemize_result=[[]], token_ids=[])
        chunks = list(voice.synthesize("", include_alignments=True))
        self.assertEqual(chunks, [])

    def test_model_without_alignment_output_gives_none(self):
        ids = [1, 0, 3, 0, 4, 0, 2]
        audio = _fake_audio()
        voice = _make_voice([audio[np.newaxis, np.newaxis, :]],
                            phonemize_result=[["h", "e"]], token_ids=ids)
        chunk = list(voice.synthesize("hi", include_alignments=True))[0]
        self.assertIsNone(chunk.phoneme_id_samples)
        self.assertIsNone(chunk.phoneme_alignments)


# ---------------------------------------------------------------------------
# include_alignments=False is a strict no-op on the audio
# ---------------------------------------------------------------------------

class _OrderPhonemizer:
    """A lazy phonemizer that records the index of each sentence as it is
    actually produced, so a test can assert nothing downstream forced later
    sentences to be phonemized early."""
    def __init__(self, sentences, log):
        self._sentences = sentences
        self._log = log

    def phonemize_lazy(self, text, lang=None):
        for i, sentence in enumerate(self._sentences):
            self._log.append(i)
            yield sentence

    def add_diacritics(self, text, lang, model=None):
        return text


class TestAlignmentNoOp(unittest.TestCase):
    def test_off_produces_byte_identical_audio_to_on(self):
        """A two-output (alignment-capable) VITS voice must produce the exact
        same waveform whether alignments are requested or not — the same
        inference runs either way, only post-processing differs."""
        ids = [1, 0, 3, 0, 4, 0, 2]
        audio = _fake_audio()
        dur = np.ones((1, len(ids)), dtype=np.float32)

        v_off = _make_voice([audio[np.newaxis, np.newaxis, :], dur.copy()],
                            phonemize_result=[["h", "e"]], token_ids=ids)
        v_on = _make_voice([audio[np.newaxis, np.newaxis, :], dur.copy()],
                           phonemize_result=[["h", "e"]], token_ids=ids)

        a_off = list(v_off.synthesize("hi", include_alignments=False))[0]
        a_on = list(v_on.synthesize("hi", include_alignments=True))[0]

        np.testing.assert_array_equal(a_off.audio_float_array, a_on.audio_float_array)
        self.assertEqual(a_off.audio_float_array.tobytes(),
                         a_on.audio_float_array.tobytes())
        # off is the no-op: alignment fields stay None, on is populated.
        self.assertIsNone(a_off.phoneme_id_samples)
        self.assertIsNone(a_off.phoneme_alignments)
        self.assertIsNotNone(a_on.phoneme_alignments)

    def test_lazy_second_sentence_not_phonemized_before_first_yield(self):
        """The lazy stream must stay lazy: pulling the first AudioChunk must not
        phonemize the second sentence."""
        log = []
        voice = _make_voice([_fake_audio()[np.newaxis, np.newaxis, :]],
                            phonemize_result=None, token_ids=[1, 0, 3, 0, 2])
        voice.phonemizer = _OrderPhonemizer([["h"], ["e"]], log)

        gen = voice.synthesize("s1. s2", include_alignments=True)
        next(gen)
        self.assertEqual(log, [0], "second sentence phonemized before first yield")
        next(gen)
        self.assertEqual(log, [0, 1])


# ---------------------------------------------------------------------------
# Two-stage engines (mel frames -> samples via hop_length) — voice.py level
# ---------------------------------------------------------------------------

class TestTwoStageEngineAlignment(unittest.TestCase):
    """TTSVoice.phoneme_ids_to_audio scales a two-stage engine's mel-frame
    durations to samples the same way it does VITS's frame-domain durations:
    uniformly, via VoiceConfig.hop_length, regardless of which adapter
    produced them (see phoonnx/engines/mixertts.py / matcha.py / glowtts.py).
    """

    def _make_two_stage_voice(self, mel_frames, durations, hop_length):
        from phoonnx.engines.mixertts import MixerTTSAdapter
        from phoonnx.engines.vocoders.base import BaseVocoder

        n_mel_frames = int(sum(mel_frames))

        class _FakeVocoder(BaseVocoder):
            def mel_to_audio(self, mel, denoise=False):
                return np.zeros(mel.shape[-1], dtype=np.float32)

        mel = np.zeros((1, 80, n_mel_frames), dtype=np.float32)
        durs = np.array([durations], dtype=np.float32)

        session = MagicMock()
        session.get_inputs.return_value = [_Input("token_ids")]
        session.run.return_value = [mel, durs]
        out_mel, out_durs = _Input("mel_spec"), _Input("durations")
        session.get_outputs.return_value = [out_mel, out_durs]

        cfg = MagicMock(spec=VoiceConfig)
        cfg.hop_length = hop_length
        cfg.noise_scale = None
        cfg.length_scale = None
        cfg.noise_w_scale = None
        cfg.engine_params = {}

        voice = TTSVoice.__new__(TTSVoice)
        voice.session = session
        voice.config = cfg
        voice.adapter = MixerTTSAdapter(vocoder=_FakeVocoder())
        return voice

    def test_mel_frames_scaled_to_samples_by_hop_length(self):
        """hop=256, 3 phonemes with mel-frame durations [2, 1, 4] ->
        per-phoneme sample spans [0, 512), [512, 768), [768, 1792)."""
        voice = self._make_two_stage_voice(mel_frames=[2, 1, 4], durations=[2, 1, 4], hop_length=256)
        audio, samples = voice.phoneme_ids_to_audio([10, 11, 12], include_alignments=True)
        np.testing.assert_array_equal(samples, [512, 256, 1024])
        # spans derived from cumulative sums
        starts = np.concatenate([[0], np.cumsum(samples)[:-1]])
        ends = np.cumsum(samples)
        np.testing.assert_array_equal(starts, [0, 512, 768])
        np.testing.assert_array_equal(ends, [512, 768, 1792])

    def test_no_duration_output_gives_none(self):
        from phoonnx.engines.mixertts import MixerTTSAdapter
        from phoonnx.engines.vocoders.base import BaseVocoder

        class _FakeVocoder(BaseVocoder):
            def mel_to_audio(self, mel, denoise=False):
                return np.zeros(mel.shape[-1], dtype=np.float32)

        mel = np.zeros((1, 80, 9), dtype=np.float32)
        session = MagicMock()
        session.get_inputs.return_value = [_Input("token_ids")]
        session.run.return_value = [mel]
        session.get_outputs.return_value = [_Input("mel_spec")]

        cfg = MagicMock(spec=VoiceConfig)
        cfg.hop_length = 256
        cfg.noise_scale = None
        cfg.length_scale = None
        cfg.noise_w_scale = None
        cfg.engine_params = {}

        voice = TTSVoice.__new__(TTSVoice)
        voice.session = session
        voice.config = cfg
        voice.adapter = MixerTTSAdapter(vocoder=_FakeVocoder())

        audio, samples = voice.phoneme_ids_to_audio([10, 11], include_alignments=True)
        self.assertIsNone(samples)


# ---------------------------------------------------------------------------
# Integration tests — real model, real synthesis
# ---------------------------------------------------------------------------

VOICE_ID = "OpenVoiceOS/phoonnx_eu-ES_dii_espeak"
ALIGNED_MODEL = "/tmp/phoonnx_test_aligned_eu-ES.onnx"
VOICE_CONFIG = None  # set in setUpClass


def _ensure_model():
    """Download voice + patch alignment output; return (onnx_path, config_path)."""
    from phoonnx.model_manager import TTSModelManager
    m = TTSModelManager()
    m.merge_default_voices()
    v = m.voices.get(VOICE_ID)
    if v is None:
        raise unittest.SkipTest(f"Voice {VOICE_ID} not in registry")
    retry_download(v.download_config)
    retry_download(v.download_model)
    import glob
    voice_path = v.voice_path
    onnx_files = glob.glob(os.path.join(voice_path, "*.onnx"))
    json_files = glob.glob(os.path.join(voice_path, "*.json"))
    if not onnx_files or not json_files:
        raise unittest.SkipTest("Voice files not downloaded")
    return onnx_files[0], json_files[0]


def _patch_alignment(src_onnx, dst_onnx):
    """Add the Ceil alignment output to an ONNX model."""
    import onnx
    model = onnx.load(src_onnx)
    ceil_names = {o for node in model.graph.node
                  if node.op_type == "Ceil" for o in node.output}
    if not ceil_names:
        raise unittest.SkipTest("No Ceil node found in model — alignment not patchable")
    if len(ceil_names) > 1:
        raise unittest.SkipTest("Multiple Ceil nodes — cannot autodetect")
    # Check not already an output
    existing = {o.name for o in model.graph.output}
    name = next(iter(ceil_names))
    if name not in existing:
        vinfo = onnx.helper.ValueInfoProto()
        vinfo.name = name
        model.graph.output.append(vinfo)
        onnx.save(model, dst_onnx)
    else:
        shutil.copy(src_onnx, dst_onnx)


class TestAlignmentIntegration(unittest.TestCase):
    """End-to-end tests with a real ONNX model.

    Skipped automatically when onnx is not installed or the voice cannot be
    downloaded (no network, CI without model cache, etc.).
    """

    @classmethod
    def setUpClass(cls):
        try:
            import onnx  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("onnx not installed")

        try:
            src_onnx, config_path = _ensure_model()
        except unittest.SkipTest:
            raise
        except Exception as e:
            raise unittest.SkipTest(f"Could not set up model: {e}")

        _patch_alignment(src_onnx, ALIGNED_MODEL)
        cls.voice = TTSVoice.load(ALIGNED_MODEL, config_path)
        cls.voice_no_align = TTSVoice.load(src_onnx, config_path)
        cls._src_onnx = src_onnx

    @classmethod
    def tearDownClass(cls):
        # ``voice_no_align``'s model does have a locatable duration tensor
        # (see test_model_without_alignment_output_triggers_runtime_surgery
        # below), so an ``include_alignments=True`` call against it writes an
        # on-demand ``<model>.alignment.onnx`` sibling next to the
        # downloaded/cached model file. Clean it up rather than leaving a
        # derived artifact behind in the voice cache.
        src_onnx = getattr(cls, "_src_onnx", None)
        if src_onnx:
            p = Path(src_onnx)
            stem = p.name[:-len(".onnx")] if p.name.endswith(".onnx") else p.stem
            derived = p.with_name(f"{stem}.alignment.onnx")
            if derived.is_file():
                derived.unlink()

    def test_synthesize_produces_audio(self):
        chunks = list(self.voice.synthesize("kaixo"))
        self.assertGreater(len(chunks), 0)
        self.assertGreater(len(chunks[0].audio_float_array), 0)

    def test_without_alignments_fields_are_none(self):
        chunk = list(self.voice.synthesize("kaixo", include_alignments=False))[0]
        self.assertIsNone(chunk.phoneme_id_samples)
        self.assertIsNone(chunk.phoneme_alignments)

    def test_phonemes_and_ids_always_present(self):
        chunk = list(self.voice.synthesize("kaixo", include_alignments=False))[0]
        self.assertGreater(len(chunk.phonemes), 0)
        self.assertGreater(len(chunk.phoneme_ids), 0)

    def test_with_alignments_populated(self):
        chunk = list(self.voice.synthesize("kaixo", include_alignments=True))[0]
        self.assertIsNotNone(chunk.phoneme_id_samples)
        self.assertIsNotNone(chunk.phoneme_alignments)
        self.assertGreater(len(chunk.phoneme_alignments), 0)

    def test_alignment_total_equals_audio_length(self):
        """Sum of per-phoneme sample counts must equal the audio frame count."""
        chunk = list(self.voice.synthesize("kaixo", include_alignments=True))[0]
        total = sum(a.num_samples for a in chunk.phoneme_alignments)
        self.assertEqual(total, len(chunk.audio_float_array))

    def test_alignment_covers_all_phonemes(self):
        """Every real phoneme from phonemization appears in the alignment list."""
        chunk = list(self.voice.synthesize("kaixo mundua", include_alignments=True))[0]
        aligned_phonemes = [a.phoneme for a in chunk.phoneme_alignments]
        for p in chunk.phonemes:
            self.assertIn(p, aligned_phonemes)

    def test_all_durations_positive(self):
        chunk = list(self.voice.synthesize("kaixo mundua", include_alignments=True))[0]
        for a in chunk.phoneme_alignments:
            self.assertGreater(a.num_samples, 0,
                               f"phoneme {a.phoneme!r} has zero duration")

    def test_model_without_alignment_output_triggers_runtime_surgery(self):
        """An unpatched model (single output) that does carry a locatable
        duration tensor gets one retrofitted on demand — see
        ``phoonnx.onnx_surgery`` / ``TTSVoice._ensure_alignment_session`` —
        instead of degrading to ``None``. This is the on-demand-alignment
        feature working exactly as intended for a real VITS export."""
        chunk = list(self.voice_no_align.synthesize("kaixo", include_alignments=True))[0]
        self.assertIsNotNone(chunk.phoneme_id_samples)
        self.assertIsNotNone(chunk.phoneme_alignments)
        total = sum(a.num_samples for a in chunk.phoneme_alignments)
        self.assertEqual(total, len(chunk.audio_float_array))

    def test_multiple_sentences(self):
        """Multi-sentence input: each chunk is independently aligned."""
        chunks = list(self.voice.synthesize(
            "kaixo mundua. zer moduz zaude?", include_alignments=True
        ))
        for chunk in chunks:
            if chunk.phoneme_alignments:
                total = sum(a.num_samples for a in chunk.phoneme_alignments)
                self.assertEqual(total, len(chunk.audio_float_array))

    def test_alignment_phoneme_strings_are_real(self):
        """Alignment phoneme strings are non-empty and not all '?'."""
        chunk = list(self.voice.synthesize("kaixo", include_alignments=True))[0]
        phonemes = [a.phoneme for a in chunk.phoneme_alignments]
        self.assertTrue(all(p for p in phonemes), "empty phoneme string in alignments")
        unknown = [p for p in phonemes if p == "?"]
        self.assertEqual(unknown, [], f"unknown ids in alignment: {phonemes}")


if __name__ == "__main__":
    unittest.main()
