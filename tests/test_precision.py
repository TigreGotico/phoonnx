"""The ``precision`` setting: which files of a voice load, int8 or full.

An index entry names its full files; an optional ``variants["int8"]`` names the
int8 files that differ. An entry with only int8 files carries
``"precision": "int8"``. The setting is ``int8``, ``full`` or ``auto``, from the
load kwarg, the plugin key, the ``PHOONNX_PRECISION`` env var, or the default.
"""
import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import phoonnx.opm as opm
from phoonnx import model_manager
from phoonnx.model_manager import TTSModelInfo, TTSModelManager
from phoonnx.version import VERSION_MAJOR

BASE = "https://huggingface.co/Phoonnx/demo/resolve/main"
CPU = ["CPUExecutionProvider"]
GPUS = ["CUDAExecutionProvider", "ROCMExecutionProvider", "MIGraphXExecutionProvider",
        "DmlExecutionProvider", "CoreMLExecutionProvider"]


def _entry(variant=None, **fields):
    entry = dict(voice_id="demo/voice", lang="pt-BR", engine="rapido",
                 model_url=f"{BASE}/model.onnx", config_url=f"{BASE}/config.json",
                 aux_model_urls={"text_encoder_path": f"{BASE}/text_encoder.onnx",
                                 "decoder_path": f"{BASE}/decoder.onnx"})
    if variant is not None:
        entry["variants"] = {"int8": variant}
    entry.update(fields)
    return TTSModelInfo(**entry)


INT8 = {"model_url": f"{BASE}/model_int8.onnx",
        "aux_model_urls": {"decoder_path": f"{BASE}/decoder_int8.onnx"}}


def _env(value=None):
    env = {k: v for k, v in os.environ.items() if k != "PHOONNX_PRECISION"}
    if value is not None:
        env["PHOONNX_PRECISION"] = value
    return patch.dict(os.environ, env, clear=True)


def _gpu_runtime(*gpus):
    """An onnxruntime build that offers ``gpus`` beside the CPU provider."""
    return patch("phoonnx.providers.available_providers", return_value=[*gpus, *CPU])


def _url_fields(info):
    return {k: v for k, v in info.to_dict().items()
            if k.endswith("_url") or k == "aux_model_urls"}


class TestThePrecisionSetting(unittest.TestCase):

    def test_the_default_is_auto(self):
        with _env():
            self.assertEqual(model_manager.resolve_precision(), "auto")

    def test_the_env_var_beats_the_default(self):
        with _env("int8"):
            self.assertEqual(model_manager.resolve_precision(), "int8")

    def test_the_kwarg_beats_the_env_var(self):
        with _env("int8"):
            self.assertEqual(model_manager.resolve_precision("full"), "full")
            self.assertEqual(_entry(INT8).with_precision("full").model_url, f"{BASE}/model.onnx")

    def test_the_plugin_key_beats_the_env_var(self):
        info = _entry(INT8)
        with _env("full"):
            plugin = _plugin(info, precision="int8")
            self.assertEqual(plugin._as_loaded(info).model_url, f"{BASE}/model_int8.onnx")
        with _env("int8"):
            plugin = _plugin(info, precision="full")
            self.assertIs(plugin._as_loaded(info), info)

    def test_the_plugin_falls_back_to_the_env_var(self):
        info = _entry(INT8)
        with _env("int8"):
            self.assertEqual(_plugin(info)._as_loaded(info).model_url, f"{BASE}/model_int8.onnx")

    def test_an_invalid_value_raises(self):
        with _env():
            with self.assertRaises(ValueError):
                model_manager.resolve_precision("fp16")
            with self.assertRaises(ValueError):
                _entry(INT8).with_precision("half")
        with _env("fp16"), self.assertRaises(ValueError):
            model_manager.resolve_precision()
        with _env(), self.assertRaises(ValueError):
            _plugin(_entry(INT8), precision="fp16")


class TestInt8(unittest.TestCase):

    def test_int8_picks_the_variant_over_the_base_keys(self):
        info = _entry(INT8).with_precision("int8")
        self.assertEqual(info.model_url, f"{BASE}/model_int8.onnx")
        self.assertEqual(info.config_url, f"{BASE}/config.json")
        self.assertEqual(info.aux_model_urls, {"text_encoder_path": f"{BASE}/text_encoder.onnx",
                                               "decoder_path": f"{BASE}/decoder_int8.onnx"})
        self.assertEqual((info.voice_id, info.precision, info.variants), ("demo/voice", "int8", None))

    def test_int8_leaves_the_base_entry_alone(self):
        base = _entry(INT8)
        base.with_precision("int8")
        self.assertEqual(base.model_url, f"{BASE}/model.onnx")
        self.assertEqual(base.aux_model_urls["decoder_path"], f"{BASE}/decoder.onnx")

    def test_int8_without_int8_files_raises(self):
        with self.assertRaises(model_manager.PrecisionUnavailable) as caught:
            _entry().with_precision("int8")
        self.assertIsInstance(caught.exception, ValueError)
        message = str(caught.exception)
        self.assertIn("demo/voice", message)
        self.assertIn("'auto'", message)
        self.assertIn("'full'", message)

    def test_full_on_an_int8_only_entry_raises(self):
        with self.assertRaises(model_manager.PrecisionUnavailable) as caught:
            _entry(precision="int8").with_precision("full")
        self.assertIn("'int8'", str(caught.exception))

    def test_an_int8_only_entry_loads_its_files_under_int8_and_auto(self):
        info = _entry(precision="int8")
        self.assertIs(info.with_precision("int8"), info)
        self.assertIs(info.with_precision("auto", CPU), info)


class TestAuto(unittest.TestCase):

    def test_auto_on_cpu_picks_an_int8_variant_marked_auto(self):
        info = _entry({**INT8, "auto": True}).with_precision("auto", CPU)
        self.assertEqual(info.model_url, f"{BASE}/model_int8.onnx")
        self.assertEqual(info.precision, "int8")

    def test_auto_on_a_gpu_provider_picks_full(self):
        info = _entry({**INT8, "auto": True})
        for gpu in GPUS:
            with self.subTest(gpu), _gpu_runtime(gpu):
                self.assertIs(info.with_precision("auto", [gpu, *CPU]), info)

    def test_auto_skips_a_variant_not_marked_auto(self):
        info = _entry(INT8)
        self.assertIs(info.with_precision("auto", CPU), info)

    def test_auto_without_a_variant_picks_full_and_logs_nothing(self):
        info = _entry()
        with patch.object(model_manager, "LOG") as log, \
                patch.object(model_manager, "resolve_providers") as providers:
            self.assertIs(info.with_precision("auto", CPU), info)
        self.assertEqual(log.method_calls, [])
        providers.assert_not_called()


class TestTheRapidoAlias(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.manager = TTSModelManager(cache_path=str(Path(self.tmp.name) / "voices.json"))
        self.manager.merge_default_voices()
        model_manager._LOGGED_ALIASES.clear()

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_old_id_gives_the_int8_files_and_warns(self):
        with self.assertWarns(DeprecationWarning) as caught, \
                patch.object(model_manager, "LOG") as log:
            info = self.manager.get_voice("edwixx/pt-BR-rapido-int8")
        self.assertEqual(info.voice_id, "edwixx/pt-BR-rapido")
        self.assertEqual(info.precision, "int8")
        self.assertTrue(info.model_url.endswith("/edwixx__rapido-tts-int8/model.onnx"))
        removal = f"{VERSION_MAJOR + 1}.0.0"
        self.assertIn(removal, str(caught.warning))
        self.assertIn(removal, log.warning.call_args.args[0])

    def test_the_old_id_is_not_listed(self):
        self.assertNotIn("edwixx/pt-BR-rapido-int8", self.manager.voices)

    def test_the_old_id_in_a_saved_cache_is_not_listed(self):
        self.manager.cache["edwixx/pt-BR-rapido-int8"] = {
            "voice_id": "edwixx/pt-BR-rapido-int8", "lang": "pt-BR",
            "model_url": f"{BASE}/model.onnx"}
        self.manager.cache.store()
        self.manager.load()
        self.assertIn("edwixx/pt-BR-rapido", self.manager.voices)
        self.assertNotIn("edwixx/pt-BR-rapido-int8", self.manager.voices)

    def test_the_old_id_loads_the_int8_files(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            info = self.manager.get_voice("edwixx/pt-BR-rapido-int8")
        fetched = _load_and_record(info)
        self.assertTrue(fetched)
        self.assertTrue(all("/edwixx__rapido-tts-int8/" in url for url in fetched), fetched)


def _load_and_record(info, **load_kwargs):
    """The ONNX URLs ``info.load`` fetches, with the engine and network stubbed."""
    fetched = []
    with patch.object(TTSModelInfo, "config", new_callable=PropertyMock), \
            patch.object(model_manager, "check_lang_supported"), \
            patch.object(model_manager, "_voice_phonemizer"), \
            patch.object(TTSModelInfo, "hub_path", return_value=None), \
            patch.object(TTSModelInfo, "_fetch_onnx",
                         side_effect=lambda url: fetched.append(url) or Path("model.onnx")), \
            patch.object(model_manager, "TTSVoice"):
        info.load(**load_kwargs)
    return fetched


class TestLoadPicksTheFiles(unittest.TestCase):

    def test_load_with_int8_fetches_the_variant(self):
        fetched = _load_and_record(_entry(INT8), precision="int8")
        self.assertEqual(sorted(fetched), sorted([f"{BASE}/model_int8.onnx",
                                                  f"{BASE}/text_encoder.onnx",
                                                  f"{BASE}/decoder_int8.onnx"]))

    def test_load_with_full_fetches_the_base(self):
        fetched = _load_and_record(_entry(INT8), precision="full")
        self.assertEqual(sorted(fetched), sorted([f"{BASE}/model.onnx",
                                                  f"{BASE}/text_encoder.onnx",
                                                  f"{BASE}/decoder.onnx"]))


class TestCacheKeys(unittest.TestCase):

    def test_the_two_precisions_have_different_artifact_keys(self):
        info = _entry(INT8)
        self.assertNotEqual(info.with_precision("int8").artifact_key(),
                            info.with_precision("full").artifact_key())

    def test_the_two_precisions_have_different_plugin_cache_ids(self):
        info = _entry(INT8)
        voices = {}
        for precision in ("int8", "full"):
            ctxt = _plugin(info, precision=precision)._get_ctxt(
                {"voice": info.voice_id, "lang": "pt-BR"})
            voices[precision] = ctxt.voice
        self.assertEqual(voices, {"int8": "demo/voice#int8", "full": "demo/voice"})

    def test_an_int8_only_voice_keeps_its_plugin_cache_id(self):
        info = _entry(precision="int8")
        ctxt = _plugin(info)._get_ctxt({"voice": info.voice_id, "lang": "pt-BR"})
        self.assertEqual(ctxt.voice, "demo/voice")


class TestTheBundledIndex(unittest.TestCase):

    def test_auto_on_cpu_loads_the_files_each_entry_names(self):
        # No variant is marked "auto" yet, so the default loads, for every
        # voice, exactly the files its entry names.
        seen = 0
        changed = []
        for path in TTSModelManager.voice_index_files():
            for voice_id, entry in json.loads(path.read_text(encoding="utf-8")).items():
                seen += 1
                info = TTSModelInfo(**entry)
                if _url_fields(info.with_precision("auto", CPU)) != _url_fields(info):
                    changed.append(voice_id)
        self.assertGreater(seen, 3000)
        self.assertEqual(changed, [])

    def test_every_int8_only_entry_is_pocket_tts(self):
        int8_only = [voice_id for path in TTSModelManager.voice_index_files()
                     for voice_id, entry in json.loads(path.read_text(encoding="utf-8")).items()
                     if entry.get("precision") == "int8"]
        self.assertEqual(len(int8_only), 156)
        self.assertTrue(all(v.startswith("pockettts/") for v in int8_only))


def _plugin(*infos, **config):
    """A plugin whose catalog holds ``infos``, with no network."""
    manager = MagicMock()
    manager.get_voice.side_effect = {i.voice_id: i for i in infos}.get
    manager.get_lang_voices.return_value = list(infos)
    config = {"lang": "pt-BR", "voice": infos[0].voice_id, "onnx_providers": CPU, **config}
    with patch.object(opm, "TTSModelManager", return_value=manager):
        plugin = opm.PhoonnxTTSPlugin(config=config)
    plugin.config = config
    return plugin


if __name__ == "__main__":
    unittest.main()
