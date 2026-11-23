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

MAP = {"_": 0, "^": 1, "$": 2, " ": 3, "a": 4, "ɐ": 5, "ɐ̃": 6, "tʲ": 7, "ʎ": 8, "ˈ": 9, "ɐ̃ː": 10}
VECTORS = {4: [1, 0, -1], 5: [1, -1, 0], 6: [1, -1, 1], 7: [-1, 1, 0], 8: [-1, 1, 1], 10: [1, -1, 1]}
NULL_IDS = [0, 1, 2, 3, 9, 11]
NUM_SYMBOLS = 12  # one id past the map, as a fine-tune config can have
HIDDEN = 8
TINY = dict(inter_channels=32, hidden_channels=32, filter_channels=64, n_heads=1, n_layers=1,
            n_layers_q=1, resblock="2", resblock_kernel_sizes=(3,), resblock_dilation_sizes=((1, 2),),
            upsample_rates=(8,), upsample_initial_channel=16, upsample_kernel_sizes=(16,))


def _sha(phoneme_id_map) -> str:
    """The hash that plans/feature-input-tts.md states, written out."""
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


def _features(**override):
    with TemporaryDirectory() as tmp:
        return load_input_features(_write_table(Path(tmp) / "f.json", **override), MAP, NUM_SYMBOLS)


class TestLoad(unittest.TestCase):
    def test_rows_follow_the_ids_and_null_rows_are_zero(self):
        feats = _features()
        self.assertEqual(len(feats["features"]), NUM_SYMBOLS)
        self.assertEqual(feats["features"][6], [1.0, -1.0, 1.0])
        self.assertEqual(feats["features"][9], [0.0, 0.0, 0.0])
        self.assertEqual([i for i, v in enumerate(feats["has_vector"]) if not v], NULL_IDS)
        self.assertEqual(feats["symbols"][6], "ɐ̃")

    def test_a_list_valued_map_loads(self):
        listed = {s: [i] for s, i in MAP.items()}
        with TemporaryDirectory() as tmp:
            feats = load_input_features(_write_table(Path(tmp) / "f.json", phoneme_id_map=listed), listed, NUM_SYMBOLS)
        self.assertEqual(feats["features"][8], [-1.0, 1.0, 1.0])

    def test_a_table_made_from_another_map_is_refused(self):
        with TemporaryDirectory() as tmp:
            path = _write_table(Path(tmp) / "f.json", phoneme_id_map={**MAP, "x": 11})
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

    def test_an_id_past_the_model_symbols_is_refused(self):
        with TemporaryDirectory() as tmp:
            vectors = {str(i): {"symbol": s, "vector": VECTORS.get(i)} for s, i in MAP.items()}
            vectors["12"] = {"symbol": "x", "vector": None}
            path = _write_table(Path(tmp) / "f.json", vectors=vectors)
            with self.assertRaisesRegex(ValueError, r"ids \[12\] outside"):
                load_input_features(path, MAP, NUM_SYMBOLS)

    def test_a_vector_of_another_length_is_refused(self):
        with TemporaryDirectory() as tmp:
            vectors = {str(i): {"symbol": s, "vector": VECTORS.get(i)} for s, i in MAP.items()}
            vectors["4"]["vector"] = [1, 0]
            path = _write_table(Path(tmp) / "f.json", vectors=vectors)
            with self.assertRaisesRegex(ValueError, "id 4 has 2 values, expected 3"):
                load_input_features(path, MAP, NUM_SYMBOLS)


class TestSplitMarks(unittest.TestCase):
    def test_bases_and_marks(self):
        self.assertEqual(split_marks("ɐ̃"), ("ɐ", ["̃"]))
        self.assertEqual(split_marks("ẽ"), ("e", ["̃"]))  # precomposed ẽ
        self.assertEqual(split_marks("tʲ"), ("t", ["ʲ"]))
        self.assertEqual(split_marks("ɐ̃ː"), ("ɐ", ["̃", "ː"]))
        self.assertEqual(split_marks("d͡ʒ"), ("d͡ʒ", []))
        self.assertEqual(split_marks("ˈ"), ("ˈ", []))
        self.assertEqual(split_marks("̃"), ("̃", []))

    def test_the_cedilla_makes_a_letter(self):
        self.assertEqual(split_marks("ç"), ("ç", []))
        self.assertEqual(split_marks("çʰ"), ("ç", ["ʰ"]))


class TestFeatureInput(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.feats = _features()
        self.F = torch.tensor(self.feats["features"])
        self.has = torch.tensor(self.feats["has_vector"], dtype=torch.float32)[:, None]
        self.scale = HIDDEN ** 0.5

    def _projection(self, layer):
        """The projection of each id's features, and nothing for a null id."""
        return (self.F @ layer.proj.weight.T + layer.proj.bias) * self.has

    def test_a_new_layer_is_the_feature_projection_and_null_ids_are_zero(self):
        for mode in ("residual", "composition"):
            layer = FeatureInput(self.feats, HIDDEN, mode)
            table = layer.table()
            self.assertTrue(torch.allclose(table, self._projection(layer)), mode)
            self.assertTrue(torch.equal(table[NULL_IDS], torch.zeros(len(NULL_IDS), HIDDEN)), mode)

    def test_a_null_id_gets_only_its_learned_part(self):
        layer = FeatureInput(self.feats, HIDDEN, "residual")
        torch.nn.init.normal_(layer.residual.weight)
        out = layer(torch.tensor(NULL_IDS))
        self.assertTrue(torch.allclose(out, layer.residual.weight[NULL_IDS] * self.scale, atol=1e-6))

    def test_a_phone_missing_from_training_keeps_its_projection(self):
        for mode in ("residual", "composition"):
            layer = FeatureInput(self.feats, HIDDEN, mode)
            opt = torch.optim.SGD(layer.parameters(), lr=0.1)
            trained = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 9]])  # every id but ʎ (8) and ɐ̃ː (10)
            for _ in range(5):
                opt.zero_grad()
                (layer(trained) - 1.0).pow(2).mean().backward()
                opt.step()
            out = layer(torch.tensor([8]))[0]
            self.assertTrue(torch.allclose(out, self._projection(layer)[8], atol=1e-6), mode)
            self.assertFalse(torch.allclose(layer(torch.tensor([5]))[0], self._projection(layer)[5]), mode)

    def test_composition_adds_the_base_and_each_mark(self):
        layer = FeatureInput(self.feats, HIDDEN, "composition")
        torch.nn.init.normal_(layer.base.weight)
        torch.nn.init.normal_(layer.marks.weight[1:])
        table, proj = layer.table(), self._projection(layer)
        self.assertEqual(layer.base_ids[5], layer.base_ids[6])  # ɐ and ɐ̃
        self.assertEqual(layer.base_ids[5], layer.base_ids[10])  # ɐ and ɐ̃ː
        base = layer.base.weight[layer.base_ids[5]]
        tilde, long = (layer.marks.weight[i] for i in layer.mark_ids[10])
        self.assertTrue(torch.allclose(table[5], proj[5] + self.scale * base, atol=1e-5))
        self.assertTrue(torch.allclose(table[6], proj[6] + self.scale * (base + tilde), atol=1e-5))
        self.assertTrue(torch.allclose(table[10], proj[10] + self.scale * (base + tilde + long), atol=1e-5))
        self.assertTrue(torch.equal(layer.marks.weight[0], torch.zeros(HIDDEN)))



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


def _save(model, path: Path) -> Path:
    torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": "0.0.0", "global_step": 0, "epoch": 0}, path)
    return path


def _feature_model(mode="residual", seed=0, **feature_override):
    torch.manual_seed(seed)
    model = VitsModel(dataset=None, num_symbols=NUM_SYMBOLS, num_speakers=1, input_layer=mode,
                      input_features=_features(**feature_override), **TINY)
    for p in model.model_g.enc_p.input_layer.parameters():
        torch.nn.init.normal_(p)
    return model


class TestCheckpoints(unittest.TestCase):
    def test_bake_refuses_a_table_that_does_not_match(self):
        enc = _feature_model().model_g.enc_p
        real = torch.nn.Embedding.from_pretrained
        with patch.object(torch.nn.Embedding, "from_pretrained",
                          side_effect=lambda table, freeze: real(table + 1e-3, freeze=freeze)):
            with self.assertRaisesRegex(RuntimeError, "baked id table differs"):
                enc.bake_input()

    def test_a_resume_with_another_table_is_refused(self):
        trained = _feature_model()
        negated = {str(i): {"symbol": s, "vector": [-v for v in VECTORS[i]] if i in VECTORS else None}
                   for s, i in MAP.items()}
        changed = _feature_model(vectors=negated)
        checkpoint = {"hyper_parameters": dict(trained.hparams)}
        with self.assertRaisesRegex(ValueError, "another text-encoder input layer or feature table"):
            changed.on_load_checkpoint(checkpoint)
        trained.on_load_checkpoint(checkpoint)  # the same table resumes

    def test_an_id_model_fine_tunes_from_a_feature_checkpoint_through_its_baked_table(self):
        from phoonnx_train.engines import get_engine
        source = _feature_model("composition").eval()
        with torch.no_grad():
            table = source.model_g.enc_p.input_layer.table()
        model = VitsModel(dataset=None, num_symbols=NUM_SYMBOLS, num_speakers=1, **TINY)
        with TemporaryDirectory() as tmp:
            get_engine("vits").load_checkpoint(model, _save(source, Path(tmp) / "f.ckpt"))
        loaded = model.model_g.enc_p.emb.weight * (TINY["hidden_channels"] ** 0.5)
        self.assertTrue(torch.allclose(loaded, table, atol=1e-5))

    def test_discard_encoder_leaves_the_feature_layer_as_built(self):
        from phoonnx_train.engines import get_engine
        source = _feature_model(seed=1)
        model = _feature_model(seed=2)
        built = {k: v.clone() for k, v in model.state_dict().items() if "input_layer" in k}
        with TemporaryDirectory() as tmp:
            get_engine("vits").load_checkpoint(model, _save(source, Path(tmp) / "f.ckpt"), discard_encoder=True)
        for k, v in built.items():
            self.assertTrue(torch.equal(model.state_dict()[k], v), k)

    def test_a_changed_symbol_list_keeps_the_projection_and_drops_the_tables(self):
        from phoonnx_train.engines import get_engine
        source = _feature_model("composition", seed=1)
        renamed = {str(i): {"symbol": "z" if s == "a" else s, "vector": VECTORS.get(i)} for s, i in MAP.items()}
        model = _feature_model("composition", seed=2, vectors=renamed)
        built = {k: v.clone() for k, v in model.state_dict().items() if "input_layer" in k}
        with TemporaryDirectory() as tmp:
            get_engine("vits").load_checkpoint(model, _save(source, Path(tmp) / "f.ckpt"))
        state, saved = model.state_dict(), source.state_dict()
        for key in ("model_g.enc_p.input_layer.proj.weight", "model_g.enc_p.input_layer.proj.bias"):
            self.assertTrue(torch.equal(state[key], saved[key]), key)
        for key in ("model_g.enc_p.input_layer.base.weight", "model_g.enc_p.input_layer.marks.weight"):
            self.assertTrue(torch.equal(state[key], built[key]), key)

    def test_other_feature_names_drop_the_projection(self):
        from phoonnx_train.engines import get_engine
        source = _feature_model(seed=1)
        model = _feature_model(seed=2, feature_names=["f3", "f2", "f1"])
        built = {k: v.clone() for k, v in model.state_dict().items() if "input_layer" in k}
        with TemporaryDirectory() as tmp:
            get_engine("vits").load_checkpoint(model, _save(source, Path(tmp) / "f.ckpt"))
        state, saved = model.state_dict(), source.state_dict()
        for key in ("model_g.enc_p.input_layer.proj.weight", "model_g.enc_p.input_layer.proj.bias"):
            self.assertTrue(torch.equal(state[key], built[key]), key)
        key = "model_g.enc_p.input_layer.residual.weight"
        self.assertTrue(torch.equal(state[key], saved[key]))

    def test_a_single_speaker_conversion_drops_tables_of_other_symbols(self):
        from phoonnx_train.engines import get_engine
        source = _feature_model("composition", seed=1)
        renamed = {str(i): {"symbol": "z" if s == "a" else s, "vector": VECTORS.get(i)} for s, i in MAP.items()}
        torch.manual_seed(2)
        model = VitsModel(dataset=None, num_symbols=NUM_SYMBOLS, num_speakers=2, input_layer="composition",
                          input_features=_features(vectors=renamed), **TINY)
        for p in model.model_g.enc_p.input_layer.parameters():
            torch.nn.init.normal_(p)
        built = {k: v.clone() for k, v in model.state_dict().items() if "input_layer" in k}
        with TemporaryDirectory() as tmp:
            get_engine("vits").load_checkpoint(model, _save(source, Path(tmp) / "f.ckpt"),
                                               resume_from_single_speaker_checkpoint=True)
        state, saved = model.state_dict(), source.state_dict()
        for key in ("model_g.enc_p.input_layer.base.weight", "model_g.enc_p.input_layer.marks.weight"):
            self.assertTrue(torch.equal(state[key], built[key]), key)
        key = "model_g.enc_p.input_layer.proj.weight"
        self.assertTrue(torch.equal(state[key], saved[key]))

    def test_a_single_speaker_conversion_bakes_a_feature_checkpoint_for_an_id_model(self):
        from phoonnx_train.engines import get_engine
        source = _feature_model("composition").eval()
        with torch.no_grad():
            table = source.model_g.enc_p.input_layer.table()
        model = VitsModel(dataset=None, num_symbols=NUM_SYMBOLS, num_speakers=2, **TINY)
        with TemporaryDirectory() as tmp:
            get_engine("vits").load_checkpoint(model, _save(source, Path(tmp) / "f.ckpt"),
                                               resume_from_single_speaker_checkpoint=True)
        loaded = model.model_g.enc_p.emb.weight * (TINY["hidden_channels"] ** 0.5)
        self.assertTrue(torch.allclose(loaded, table, atol=1e-5))

    def test_a_config_without_a_map_is_refused(self):
        from phoonnx_train import train
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "config.json").write_text(json.dumps({"num_symbols": NUM_SYMBOLS}), encoding="utf-8")
            table = _write_table(tmp / "f.json")
            with patch("phoonnx_train.train.get_engine") as get_engine:
                get_engine.return_value.quality_presets.return_value = {"medium": {}}
                result = click.testing.CliRunner().invoke(train.main, [
                    "--dataset-dir", str(tmp), "--input-layer", "residual", "--input-features", str(table)])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("needs a phoneme_id_map", result.output)


if __name__ == "__main__":
    unittest.main()
