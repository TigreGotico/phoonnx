"""Two threads asking one voice for alignments must not race the surgery.

The patched model is written to a path derived from the model's own name, so
every caller of a shared voice targets the same file. An unguarded
check-then-act let both threads decide the file was missing and both save over
it at once, and because ``onnx.save`` writes incrementally, whichever thread
loaded first could get a truncated graph.

Both threads are made to overlap deterministically by making the save itself
slow, rather than by hoping the scheduler interleaves them.
"""
import os
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from phoonnx.alignment import ensure_alignment_session


def _model_with_hidden_durations():
    """``input`` -> Ceil-derived durations (hidden) -> ``output``.

    The durations are only reachable through the ``Ceil`` node, which is what
    the surgery has to locate.
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
    return model


class TestConcurrentAlignmentSurgery(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="phoonnx_align_race_")
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.model_path = os.path.join(self.tmpdir, "model.onnx")
        onnx.save(_model_with_hidden_durations(), self.model_path)
        self.dest = os.path.join(self.tmpdir, "model.alignment.onnx")

        self.saved_paths = []
        self.overlapped = False
        self._in_save = 0
        self._guard = threading.Lock()
        real_save = onnx.save

        def slow_save(model, path, *args, **kwargs):
            with self._guard:
                self._in_save += 1
                if self._in_save > 1:
                    self.overlapped = True
                self.saved_paths.append(str(path))
            try:
                # Wide enough that an unserialized second thread is certain to
                # be inside the same call, and short enough to stay a unit test.
                threading.Event().wait(0.3)
                return real_save(model, path, *args, **kwargs)
            finally:
                with self._guard:
                    self._in_save -= 1

        patcher = patch("onnx.save", side_effect=slow_save)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _race(self, threads=4):
        start = threading.Barrier(threads)
        results = [None] * threads
        errors = []

        def run(i):
            try:
                start.wait()
                results[i] = ensure_alignment_session(
                    self.model_path, ["CPUExecutionProvider"])
            except BaseException as e:  # pragma: no cover - reported below
                errors.append(e)

        workers = [threading.Thread(target=run, args=(i,)) for i in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join(timeout=60)
        self.assertEqual(errors, [], "the surgery must not raise at all")
        return results

    def test_only_one_thread_performs_the_surgery(self):
        results = self._race()
        self.assertFalse(self.overlapped,
                         "two threads wrote the same alignment model at once")
        self.assertEqual(len(self.saved_paths), 1,
                         "the losing threads must reuse the file the winner "
                         f"wrote, not save again (saved {self.saved_paths})")
        for session in results:
            self.assertIsNotNone(session, "every caller gets a usable session")

    def test_the_patched_model_is_never_written_under_its_final_name(self):
        """A reader that opens the destination while it is being written must
        find either the old file or the finished one, never a partial graph."""
        self._race()
        self.assertEqual(len(self.saved_paths), 1)
        written = self.saved_paths[0]
        self.assertNotEqual(written, self.dest,
                            "the save must go to a temporary file and be "
                            "moved into place")
        self.assertEqual(os.path.dirname(written), self.tmpdir,
                         "the temporary file must share the destination's "
                         "filesystem, or the move is not atomic")
        self.assertTrue(os.path.isfile(self.dest))
        self.assertEqual(
            [f for f in os.listdir(self.tmpdir) if f.endswith(".tmp")], [],
            "no temporary file may be left behind")

    def test_every_racing_thread_sees_a_complete_graph(self):
        for session in self._race():
            names = [o.name for o in session.get_outputs()]
            self.assertIn("durations", names,
                          "a session loaded from a truncated save would be "
                          "missing the promoted duration output")


if __name__ == "__main__":
    unittest.main()
