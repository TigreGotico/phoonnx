"""Tests for on-demand runtime alignment output (load-time graph surgery).

A voice model exported *without* ``--add-phoneme-alignment`` has no duration
output on its ONNX session. ``TTSVoice.phoneme_ids_to_audio(include_alignments
=True)`` retrofits one on demand: locate the duration (``Ceil``-op) tensor with
``phoonnx.onnx_surgery``, write a ``<model>.alignment.onnx`` sibling with it
promoted to a graph output, rebuild the session, and retry -- at most once per
voice, with the outcome (including a negative one) cached.

These tests build a tiny, real, *runnable* ONNX graph (no torch, no network)
shaped like a VITS duration predictor: ``input`` (phoneme ids) -> per-id
duration via ``Ceil`` (not exposed as an output) -> an audio tensor whose
length is exactly the summed duration. This lets the "sample counts sum to
audio length" and "byte-identical audio pre/post surgery" properties be
checked exactly, not approximately.
"""
import os
import shutil
import unittest
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from unittest.mock import MagicMock

from phoonnx.config import VoiceConfig
from phoonnx.engines.vits import VitsAdapter
from phoonnx.voice import TTSVoice, _UNSET


def _tiny_model_without_alignment():
    """``input`` (int64 [1, T]) -> Ceil-derived per-id durations (hidden,
    not a graph output) -> ``output`` (float [1, sum(durations)]).

    Durations are ``ceil(id + 1.5)`` -- deterministic, always positive, and
    with no reliance on any tensor named like a known duration output, so the
    *only* way to recover them is locating the ``Ceil`` node (exactly what
    on-demand surgery must do).
    """
    inp = helper.make_tensor_value_info("input", TensorProto.INT64, [1, "T"])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, None])

    one_const = numpy_helper.from_array(np.array([1], dtype=np.int64), name="one_const")
    axes_const = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes_const")
    add_const = numpy_helper.from_array(np.array(1.5, dtype=np.float32), name="add_const")
    fill_value = numpy_helper.from_array(np.array([1.0], dtype=np.float32))

    nodes = [
        helper.make_node("Cast", ["input"], ["input_f"], to=TensorProto.FLOAT),
        helper.make_node("Add", ["input_f", "add_const"], ["raw_dur"]),
        helper.make_node("Ceil", ["raw_dur"], ["durations"]),
        helper.make_node("ReduceSum", ["durations", "axes_const"], ["dur_sum"], keepdims=0),
        helper.make_node("Cast", ["dur_sum"], ["dur_sum_i"], to=TensorProto.INT64),
        helper.make_node("Concat", ["one_const", "dur_sum_i"], ["out_shape"], axis=0),
        helper.make_node("ConstantOfShape", ["out_shape"], ["output"], value=fill_value),
    ]
    graph = helper.make_graph(
        nodes, "tiny_align_no_output", [inp], [out],
        [one_const, axes_const, add_const],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def _tiny_model_no_ceil():
    """Same shape, but with no ``Ceil`` node at all -- nothing to locate."""
    inp = helper.make_tensor_value_info("input", TensorProto.INT64, [1, "T"])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, "T"])
    node = helper.make_node("Cast", ["input"], ["output"], to=TensorProto.FLOAT)
    graph = helper.make_graph([node], "tiny_no_ceil", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def _make_voice(model_path):
    cfg = MagicMock(spec=VoiceConfig)
    cfg.hop_length = 1  # native frames == samples, so sums check out exactly
    cfg.noise_scale = None
    cfg.length_scale = None
    cfg.noise_w_scale = None
    cfg.engine_params = {}

    voice = TTSVoice.__new__(TTSVoice)
    voice.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    voice.config = cfg
    voice.adapter = VitsAdapter()
    voice.model_path = str(model_path)
    voice._alignment_session = _UNSET
    return voice


class TestRuntimeAlignmentSurgery(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp(prefix="phoonnx_runtime_align_")
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.model_path = os.path.join(self.tmpdir, "model.onnx")
        onnx.save(_tiny_model_without_alignment(), self.model_path)

    def _alignment_path(self):
        return os.path.join(self.tmpdir, "model.alignment.onnx")

    def test_include_alignments_triggers_surgery_and_populates_alignments(self):
        voice = _make_voice(self.model_path)
        ids = [0, 1, 2, 3, 4]
        audio, samples = voice.phoneme_ids_to_audio(ids, include_alignments=True)

        self.assertTrue(os.path.isfile(self._alignment_path()))
        self.assertIsNotNone(samples)
        np.testing.assert_array_equal(samples, [2, 3, 4, 5, 6])
        self.assertEqual(int(np.sum(samples)), len(audio))

    def test_second_call_reuses_cached_alignment_file(self):
        voice = _make_voice(self.model_path)
        ids = [0, 1, 2]

        with patch(
            "phoonnx.onnx_surgery.add_phoneme_alignment_output",
            wraps=__import__("phoonnx.onnx_surgery", fromlist=["add_phoneme_alignment_output"]).add_phoneme_alignment_output,
        ) as mock_surgery:
            voice.phoneme_ids_to_audio(ids, include_alignments=True)
            voice.phoneme_ids_to_audio(ids, include_alignments=True)
            self.assertEqual(mock_surgery.call_count, 1)

    def test_second_voice_reuses_cached_alignment_file_on_disk(self):
        """A *fresh* voice instance pointed at the same model path reuses the
        already-written ``.alignment.onnx`` sibling without redoing surgery."""
        voice1 = _make_voice(self.model_path)
        voice1.phoneme_ids_to_audio([0, 1], include_alignments=True)
        self.assertTrue(os.path.isfile(self._alignment_path()))

        voice2 = _make_voice(self.model_path)
        with patch(
            "phoonnx.onnx_surgery.add_phoneme_alignment_output"
        ) as mock_surgery:
            audio, samples = voice2.phoneme_ids_to_audio([0, 1, 2], include_alignments=True)
            mock_surgery.assert_not_called()
        self.assertIsNotNone(samples)

    def test_no_locatable_duration_tensor_gives_none_and_caches_negative(self):
        no_ceil_path = os.path.join(self.tmpdir, "no_ceil.onnx")
        onnx.save(_tiny_model_no_ceil(), no_ceil_path)
        voice = _make_voice(no_ceil_path)

        with patch(
            "phoonnx.onnx_surgery.find_duration_tensor",
            wraps=__import__("phoonnx.onnx_surgery", fromlist=["find_duration_tensor"]).find_duration_tensor,
        ) as mock_locate:
            audio1, samples1 = voice.phoneme_ids_to_audio([0, 1], include_alignments=True)
            audio2, samples2 = voice.phoneme_ids_to_audio([0, 1], include_alignments=True)

        self.assertIsNone(samples1)
        self.assertIsNone(samples2)
        self.assertGreater(len(audio1), 0)
        # find_duration_tensor is invoked once inside add_phoneme_alignment_output
        # during the single surgery attempt; the second call short-circuits on
        # the cached negative result and never calls it again.
        self.assertEqual(mock_locate.call_count, 1)
        self.assertFalse(os.path.isfile(os.path.join(self.tmpdir, "no_ceil.alignment.onnx")))

    def test_include_alignments_false_never_triggers_surgery(self):
        voice = _make_voice(self.model_path)
        with patch.object(voice, "_ensure_alignment_session") as mock_ensure:
            audio = voice.phoneme_ids_to_audio([0, 1, 2], include_alignments=False)
            mock_ensure.assert_not_called()
        self.assertGreater(len(audio), 0)
        self.assertFalse(os.path.isfile(self._alignment_path()))

    def test_audio_byte_identical_before_and_after_surgery(self):
        ids = [0, 1, 2, 3]

        voice_before = _make_voice(self.model_path)
        audio_before = voice_before.phoneme_ids_to_audio(ids, include_alignments=False)

        voice_after = _make_voice(self.model_path)
        audio_after, samples = voice_after.phoneme_ids_to_audio(ids, include_alignments=True)
        self.assertIsNotNone(samples)

        np.testing.assert_array_equal(audio_before, audio_after)

    def test_ensure_alignment_session_cached_on_instance(self):
        voice = _make_voice(self.model_path)
        self.assertIs(voice._alignment_session, _UNSET)
        voice.phoneme_ids_to_audio([0, 1], include_alignments=True)
        self.assertIsNotNone(voice._alignment_session)
        cached = voice._alignment_session
        # A further request reuses the exact same cached session object.
        voice.phoneme_ids_to_audio([0, 1], include_alignments=True)
        self.assertIs(voice._alignment_session, cached)


class TestAlignmentModelPath(unittest.TestCase):
    """``_alignment_model_path`` respects ``PHOONNX_ORT_CACHE_DIR``."""

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp(prefix="phoonnx_runtime_align_path_")
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.model_path = os.path.join(self.tmpdir, "model.onnx")
        onnx.save(_tiny_model_without_alignment(), self.model_path)

    def test_default_writes_next_to_model(self):
        voice = _make_voice(self.model_path)
        dest = voice._alignment_model_path()
        self.assertEqual(str(dest), os.path.join(self.tmpdir, "model.alignment.onnx"))

    def test_cache_dir_env_var_redirects_destination(self):
        cache_dir = os.path.join(self.tmpdir, "ort_cache")
        os.makedirs(cache_dir, exist_ok=True)
        voice = _make_voice(self.model_path)
        with patch.dict(os.environ, {"PHOONNX_ORT_CACHE_DIR": cache_dir}):
            dest = voice._alignment_model_path()
            self.assertEqual(str(dest), os.path.join(cache_dir, "model.alignment.onnx"))

            audio, samples = voice.phoneme_ids_to_audio([0, 1], include_alignments=True)
            self.assertIsNotNone(samples)
            self.assertTrue(os.path.isfile(os.path.join(cache_dir, "model.alignment.onnx")))
            self.assertFalse(os.path.isfile(os.path.join(self.tmpdir, "model.alignment.onnx")))


if __name__ == "__main__":
    unittest.main()
