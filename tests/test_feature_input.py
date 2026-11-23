"""The VITS text-encoder input built from a features.json table."""
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import click.testing
import numpy as np
import torch

from phoonnx_train.vits.input_features import FeatureInput, load_input_features, split_marks
from phoonnx_train.vits.lightning import VitsModel
from phoonnx_train.vits.models import TextEncoder

MAP = {"_": 0, "^": 1, "$": 2, " ": 3, "a": 4, "ɐ": 5, "ɐ̃": 6, "tʲ": 7, "ʎ": 8, "ˈ": 9}
VECTORS = {4: [1, 0, -1], 5: [1, -1, 0], 6: [1, -1, 1], 7: [-1, 1, 0], 8: [-1, 1, 1]}
NUM_SYMBOLS = 12  # two ids past the map, as a fine-tune config can have
TINY = dict(inter_channels=32, hidden_channels=32, filter_channels=64, n_heads=1, n_layers=1,
            n_layers_q=1, resblock="2", resblock_kernel_sizes=(3,), resblock_dilation_sizes=((1, 2),),
            upsample_rates=(8,), upsample_initial_channel=16, upsample_kernel_sizes=(16,))


def _sha(phoneme_id_map) -> str:
    text = json.dumps(phoneme_id_map, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_table(path: Path, phoneme_id_map=MAP, **override) -> Path:
    table = {
        "format": "phoonnx-input-features/1",
        "source": {"name": "test", "version": "0"},
        "feature_names": ["f1", "f2", "f3"],
        "phoneme_id_map_sha256": _sha(phoneme_id_map),
        "vectors": {str(i): {"symbol": s, "vector": VECTORS.get(i)} for s, i in MAP.items()},
        **override,
    }
    path.write_text(json.dumps(table, ensure_ascii=False), encoding="utf-8")
    return path


def _features():
    with TemporaryDirectory() as tmp:
        return load_input_features(_write_table(Path(tmp) / "f.json"), MAP, NUM_SYMBOLS)


class TestLoad(unittest.TestCase):
    def test_rows_follow_the_ids_and_null_rows_are_zero(self):
        feats = _features()
        self.assertEqual(len(feats["features"]), NUM_SYMBOLS)
        self.assertEqual(feats["features"][6], [1.0, -1.0, 1.0])
        self.assertEqual(feats["features"][9], [0.0, 0.0, 0.0])
        self.assertEqual([i for i, v in enumerate(feats["has_vector"]) if v], [4, 5, 6, 7, 8])
        self.assertEqual(feats["symbols"][6], "ɐ̃")

    def test_a_table_made_from_another_map_is_refused(self):
        with TemporaryDirectory() as tmp:
            path = _write_table(Path(tmp) / "f.json", phoneme_id_map={**MAP, "x": 10})
            with self.assertRaisesRegex(ValueError, "another phoneme_id_map"):
                load_input_features(path, MAP, NUM_SYMBOLS)

    def test_another_format_is_refused(self):
        with TemporaryDirectory() as tmp:
            path = _write_table(Path(tmp) / "f.json", format="phoonnx-input-features/2")
            with self.assertRaisesRegex(ValueError, "format"):
                load_input_features(path, MAP, NUM_SYMBOLS)

    def test_a_map_id_without_an_entry_is_refused(self):
        with TemporaryDirectory() as tmp:
            vectors = {str(i): {"symbol": s, "vector": VECTORS.get(i)} for s, i in MAP.items() if i != 7}
            path = _write_table(Path(tmp) / "f.json", vectors=vectors)
            with self.assertRaisesRegex(ValueError, r"no entry for ids \[7\]"):
                load_input_features(path, MAP, NUM_SYMBOLS)


class TestSplitMarks(unittest.TestCase):
    def test_bases_and_marks(self):
        self.assertEqual(split_marks("ɐ̃"), ("ɐ", ["̃"]))
        self.assertEqual(split_marks("ẽ"), ("e", ["̃"]))  # precomposed ẽ
        self.assertEqual(split_marks("tʲ"), ("t", ["ʲ"]))
        self.assertEqual(split_marks("aː"), ("a", ["ː"]))
        self.assertEqual(split_marks("d͡ʒ"), ("d͡ʒ", []))
        self.assertEqual(split_marks("ˈ"), ("ˈ", []))
        self.assertEqual(split_marks("̃"), ("̃", []))


class TestFeatureInput(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.feats = _features()
        self.F = torch.tensor(self.feats["features"])

    def _projection(self, layer):
        return self.F @ layer.proj.weight.T + layer.proj.bias

    def test_a_new_layer_is_the_feature_projection(self):
        for mode in ("residual", "composition"):
            layer = FeatureInput(self.feats, 8, mode)
            self.assertTrue(torch.allclose(layer.table(), self._projection(layer)), mode)

    def test_a_phone_missing_from_training_keeps_its_projection(self):
        for mode in ("residual", "composition"):
            layer = FeatureInput(self.feats, 8, mode)
            opt = torch.optim.SGD(layer.parameters(), lr=0.1)
            trained = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 9]])  # every id but ʎ (8)
            for _ in range(5):
                opt.zero_grad()
                (layer(trained) - 1.0).pow(2).mean().backward()
                opt.step()
            out = layer(torch.tensor([8]))[0]
            self.assertTrue(torch.allclose(out, self._projection(layer)[8], atol=1e-6), mode)
            self.assertFalse(torch.allclose(layer(torch.tensor([5]))[0], self._projection(layer)[5]), mode)

    def test_composition_shares_the_base_and_adds_the_mark(self):
        layer = FeatureInput(self.feats, 8, "composition")
        torch.nn.init.normal_(layer.base.weight)
        torch.nn.init.normal_(layer.marks.weight[1:])
        table, proj = layer.table(), self._projection(layer)
        self.assertEqual(layer.base_ids[5], layer.base_ids[6])  # ɐ and ɐ̃
        tilde = layer.marks.weight[layer.mark_ids[6][0]]
        expected = table[5] - proj[5] + proj[6] + tilde
        self.assertTrue(torch.allclose(table[6], expected, atol=1e-6))
        self.assertTrue(torch.equal(layer.marks.weight[0], torch.zeros(8)))


class TestBake(unittest.TestCase):
    def test_baked_encoder_gives_the_same_output(self):
        feats = _features()
        for mode in ("residual", "composition"):
            torch.manual_seed(1)
            enc = TextEncoder(NUM_SYMBOLS, 16, 32, 64, 1, 1, 3, 0.1,
                              input_layer=FeatureInput(feats, 32, mode)).eval()
            for p in enc.input_layer.parameters():
                torch.nn.init.normal_(p)
            x = torch.tensor([[1, 9, 7, 6, 3, 8, 5, 4, 2]])
            lengths = torch.tensor([9])
            with torch.no_grad():
                before = enc(x, lengths)
                enc.bake_input()
                after = enc(x, lengths)
            self.assertIsNone(enc.input_layer)
            self.assertEqual(set(k for k in enc.state_dict() if "input_layer" in k), set())
            for b, a in zip(before, after):
                self.assertTrue(torch.allclose(b, a, atol=1e-5), mode)


class TestVitsModel(unittest.TestCase):
    def test_checkpoint_round_trip_keeps_the_feature_layer(self):
        model = VitsModel(dataset=None, num_symbols=NUM_SYMBOLS, num_speakers=1,
                          input_layer="residual", input_features=_features(), **TINY)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.ckpt"
            torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                        "pytorch-lightning_version": "0.0.0", "global_step": 0, "epoch": 0}, path)
            restored = VitsModel.load_from_checkpoint(path, dataset=None)
        self.assertEqual(restored.model_g.enc_p.input_layer.mode, "residual")
        self.assertNotIn("model_g.enc_p.emb.weight", restored.state_dict())
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, restored.state_dict()[k]), k)

    def test_an_id_checkpoint_warm_starts_a_feature_model(self):
        from phoonnx_train.engines import get_engine
        base = VitsModel(dataset=None, num_symbols=NUM_SYMBOLS, num_speakers=1, **TINY)
        model = VitsModel(dataset=None, num_symbols=NUM_SYMBOLS, num_speakers=1,
                          input_layer="composition", input_features=_features(), **TINY)
        fresh = {k: v.clone() for k, v in model.state_dict().items() if "input_layer" in k}
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "base.ckpt"
            torch.save({"state_dict": base.state_dict(), "hyper_parameters": dict(base.hparams),
                        "pytorch-lightning_version": "0.0.0", "global_step": 0, "epoch": 0}, path)
            get_engine("vits").load_checkpoint(model, path)
        state = model.state_dict()
        for k, v in base.state_dict().items():
            if k != "model_g.enc_p.emb.weight":
                self.assertTrue(torch.equal(v, state[k]), k)
        for k, v in fresh.items():
            self.assertTrue(torch.equal(v, state[k]), k)

    def test_export_bakes_the_layer_and_the_graph_matches_the_trained_model(self):
        import onnxruntime
        from phoonnx_train.engines import get_engine
        torch.manual_seed(2)
        model = VitsModel(dataset=None, num_symbols=NUM_SYMBOLS, num_speakers=1,
                          input_layer="residual", input_features=_features(), **TINY).eval()
        for p in model.model_g.enc_p.input_layer.parameters():
            torch.nn.init.normal_(p)
        ids = torch.tensor([[1, 9, 7, 6, 3, 8, 5, 4, 2]])
        lengths = torch.tensor([9])
        with torch.no_grad():
            expected = model.model_g.infer(ids, lengths, noise_scale=0.0, length_scale=1.0,
                                           noise_scale_w=0.0)[0].squeeze().numpy()
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = tmp / "config.json"
            config.write_text(json.dumps({"audio": {"sample_rate": 22050}, "phoneme_id_map": MAP,
                                          "num_symbols": NUM_SYMBOLS}), encoding="utf-8")
            with patch.object(VitsModel, "load_from_checkpoint", return_value=model):
                onnx_path = get_engine("vits").export_onnx(tmp / "m.ckpt", config, tmp)
            sess = onnxruntime.InferenceSession(str(onnx_path))
            got = sess.run(None, {"input": ids.numpy(), "input_lengths": lengths.numpy(),
                                  "scales": np.array([0.0, 1.0, 0.0], dtype=np.float32)})[0].squeeze()
        self.assertIsNone(model.model_g.enc_p.input_layer)
        self.assertEqual(got.shape, expected.shape)
        self.assertTrue(np.allclose(got, expected, atol=1e-4), float(np.abs(got - expected).max()))


class TestTrainCli(unittest.TestCase):
    def _invoke(self, tmp: Path, *args):
        from phoonnx_train import train
        (tmp / "config.json").write_text(json.dumps({"phoneme_id_map": MAP, "num_symbols": NUM_SYMBOLS}),
                                         encoding="utf-8")
        with patch("phoonnx_train.train.get_engine") as get_engine:
            get_engine.return_value.quality_presets.return_value = {"medium": {}}
            result = click.testing.CliRunner().invoke(train.main, ["--dataset-dir", str(tmp), *args])
        return result, get_engine

    def test_a_feature_layer_needs_a_table(self):
        with TemporaryDirectory() as tmp:
            result, _ = self._invoke(Path(tmp), "--input-layer", "residual")
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("need --input-features", result.output)

    def test_features_apply_to_vits_only(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            table = _write_table(tmp / "f.json")
            result, _ = self._invoke(tmp, "--engine", "glowtts", "--input-layer", "residual",
                                     "--input-features", str(table))
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("vits engine only", result.output)

    def test_the_table_reaches_the_model(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            table = _write_table(tmp / "f.json")
            _, get_engine = self._invoke(tmp, "--input-layer", "composition", "--input-features", str(table))
            extra = get_engine.return_value.create_model.call_args.kwargs["config"].extra
            self.assertEqual(extra["input_layer"], "composition")
            self.assertEqual(extra["input_features"]["features"][8], [-1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
