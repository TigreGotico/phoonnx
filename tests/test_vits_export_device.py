"""The VITS ONNX export must build its dummy inputs on the model's own device.

A checkpoint trained on a GPU restores to that GPU. Dummy inputs built on CPU
regardless meet those weights mid-trace and abort the export:

    RuntimeError: Expected all tensors to be on the same device, but got index
    is on cpu, different from other tensors on cuda:0

which fails on exactly the machines that trained the model.

CI has no accelerator, and a test that simply exported and checked for success
would pass against the unfixed code. The meta device stands in: it is available
everywhere, it is not CPU, and inputs built on it can only have come from
reading the model rather than from a hardcoded default.
"""
import unittest
import unittest.mock
from pathlib import Path

import torch

from phoonnx_train.engines import get_engine
from phoonnx_train.vits.lightning import VitsModel

_TINY = dict(
    inter_channels=32, hidden_channels=32, filter_channels=64, n_heads=1,
    n_layers=1, n_layers_q=1, resblock="2", resblock_kernel_sizes=(3,),
    resblock_dilation_sizes=((1, 2),), upsample_rates=(8,),
    upsample_initial_channel=16, upsample_kernel_sizes=(16,),
    num_speakers=1, num_symbols=32,
)


class _Captured(Exception):
    """Raised once the dummy inputs have been seen, so nothing is written."""


class TestExportBuildsInputsOnTheModelDevice(unittest.TestCase):
    def test_dummy_inputs_follow_the_model_rather_than_defaulting_to_cpu(self):
        model = VitsModel(dataset=None, **_TINY).to("meta")
        captured = {}

        def fake_export(*args, **kwargs):
            captured["args"] = kwargs.get("args", args[1] if len(args) > 1 else None)
            raise _Captured()

        with unittest.mock.patch.object(
            VitsModel, "load_from_checkpoint", return_value=model
        ), unittest.mock.patch(
            "builtins.open", unittest.mock.mock_open(read_data="{}")
        ), unittest.mock.patch("torch.onnx.export", side_effect=fake_export):
            with self.assertRaises(_Captured):
                get_engine("vits").export_onnx(
                    checkpoint_path=Path("unused.ckpt"),
                    config_path=Path("unused.json"),
                    output_dir=Path("unused"),
                )

        tensors = [t for t in captured["args"] if isinstance(t, torch.Tensor)]
        self.assertTrue(tensors, "no dummy input tensors reached torch.onnx.export")
        for t in tensors:
            self.assertEqual(
                t.device.type, "meta",
                f"dummy input on {t.device}, not the model's device")


if __name__ == "__main__":
    unittest.main()
