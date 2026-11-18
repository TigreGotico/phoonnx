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
import wave
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

from click.testing import CliRunner

import phoonnx.opm as opm
from phoonnx import model_manager
from phoonnx.cli import cli
from phoonnx.index_schema import validate_entry
from phoonnx.model_manager import TTSModelInfo, TTSModelManager
from phoonnx.version import VERSION_MAJOR

BASE = "https://huggingface.co/Phoonnx/demo/resolve/0123456789abcdef0123456789abcdef01234567"
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

    def test_the_env_var_ignores_case_and_spaces(self):
        with _env(" Int8\n"):
            self.assertEqual(model_manager.resolve_precision(), "int8")
        with _env():
            self.assertEqual(model_manager.resolve_precision(" FULL "), "full")

    def test_an_empty_kwarg_falls_through_to_the_env_var(self):
        with _env("int8"):
            self.assertEqual(model_manager.resolve_precision(""), "int8")
            self.assertEqual(model_manager.resolve_precision(None), "int8")
        with _env():
            self.assertEqual(model_manager.resolve_precision(""), "auto")

    def test_an_empty_plugin_key_falls_through_to_the_env_var(self):
        info = _entry(INT8)
        with _env("int8"):
            self.assertEqual(_plugin(info, precision="")._as_loaded(info).model_url,
                             f"{BASE}/model_int8.onnx")

    def test_a_value_that_is_not_a_string_raises_value_error(self):
        with _env():
            for value in (8, False, ["int8"]):
                with self.subTest(value=value), self.assertRaises(ValueError) as caught:
                    model_manager.resolve_precision(value)
                self.assertIn("int8, full, auto", str(caught.exception))
            with self.assertRaises(ValueError):
                _plugin(_entry(INT8), precision=8)

    def test_the_plugin_key_beats_the_env_var_at_load(self):
        info = _entry(INT8)
        with _env("int8"):
            plugin = _plugin(info, precision="full")
            with _recording_fetches() as fetched:
                plugin.get_model(info.voice_id)
        self.assertEqual(sorted(fetched), sorted([f"{BASE}/model.onnx", f"{BASE}/text_encoder.onnx",
                                                  f"{BASE}/decoder.onnx"]))

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
        self.assertEqual(info.voice_id, "edwixx/pt-BR-rapido-int8")
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

    def test_the_old_id_with_full_names_the_old_id(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            info = self.manager.get_voice("edwixx/pt-BR-rapido-int8")
        with self.assertRaises(model_manager.PrecisionUnavailable) as caught:
            info.with_precision("full")
        self.assertIn("'edwixx/pt-BR-rapido-int8' has no full files", str(caught.exception))

    def test_download_by_id_resolves_the_old_id_from_the_bundled_index(self):
        empty = TTSModelManager(cache_path=str(Path(self.tmp.name) / "empty.json"))
        with _recording_downloads() as downloaded, warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.assertTrue(empty.download_voice_by_id("edwixx/pt-BR-rapido-int8"))
        self.assertEqual(len(downloaded), 1)
        self.assertTrue(downloaded[0].endswith("/edwixx__rapido-tts-int8/model.onnx"), downloaded)


#: The Rápido entry as releases before the variants field cached it.
_OLD_RAPIDO = {
    "voice_id": "edwixx/pt-BR-rapido", "lang": "pt-BR", "engine": "rapido",
    "phoneme_type": "graphemes", "alphabet": "unicode",
    "model_url": "https://huggingface.co/Phoonnx/phoonnx-rapido/resolve/main/edwixx__rapido-tts/model.onnx",
    "config_url": "https://huggingface.co/Phoonnx/phoonnx-rapido/resolve/main/edwixx__rapido-tts/config.json"}
_OLD_RAPIDO_INT8 = {
    **_OLD_RAPIDO, "voice_id": "edwixx/pt-BR-rapido-int8",
    "model_url": "https://huggingface.co/Phoonnx/phoonnx-rapido/resolve/main/edwixx__rapido-tts-int8/model.onnx"}


class TestACacheFromAnEarlierRelease(unittest.TestCase):
    """A voices.json written before the Rápido int8 id became an alias."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "voices.json"
        self.path.write_text(json.dumps({"edwixx/pt-BR-rapido": _OLD_RAPIDO,
                                         "edwixx/pt-BR-rapido-int8": _OLD_RAPIDO_INT8}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_old_id_finds_the_int8_files(self):
        manager = TTSModelManager(cache_path=str(self.path))
        manager.load()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            info = manager.get_voice("edwixx/pt-BR-rapido-int8")
        self.assertTrue(info.model_url.endswith("/edwixx__rapido-tts-int8/model.onnx"))

    def test_an_alias_lookup_keeps_the_entry_the_registry_holds(self):
        manager = TTSModelManager(cache_path=str(self.path))
        manager.load()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            manager.get_voice("edwixx/pt-BR-rapido-int8")
        kept = manager.voices["edwixx/pt-BR-rapido"]
        self.assertEqual(kept.model_url, _OLD_RAPIDO["model_url"])
        self.assertFalse(kept.variants)

    def test_the_cli_precision_option_ignores_case(self):
        empty = str(Path(self.tmp.name) / "empty.json")
        with patch("phoonnx.cli.TTSModelManager",
                   side_effect=lambda: TTSModelManager(cache_path=empty)), \
                _recording_downloads() as downloaded:
            result = CliRunner().invoke(cli, ["download", "edwixx/pt-BR-rapido", "--precision", "INT8"])
        self.assertIsNone(result.exception, result.output)
        self.assertEqual(len(downloaded), 1, result.output)
        self.assertTrue(downloaded[0].endswith("/edwixx__rapido-tts-int8/model.onnx"), downloaded)

    def test_the_cli_downloads_the_old_id(self):
        with patch("phoonnx.cli.TTSModelManager",
                   side_effect=lambda: TTSModelManager(cache_path=str(self.path))), \
                _recording_downloads() as downloaded:
            result = CliRunner().invoke(cli, ["download", "edwixx/pt-BR-rapido-int8"])
        self.assertIsNone(result.exception, result.output)
        self.assertEqual(len(downloaded), 1, result.output)
        self.assertTrue(downloaded[0].endswith("/edwixx__rapido-tts-int8/model.onnx"), downloaded)


class TestTheOldIdInEachPluginSetting(unittest.TestCase):
    """The old Rápido id loads the int8 files however the plugin names it,
    and keeps the audio-cache id it had before it became an alias."""

    OLD = "edwixx/pt-BR-rapido-int8"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        model_manager._LOGGED_ALIASES.clear()

    def tearDown(self):
        self.tmp.cleanup()

    def _synthesize(self, config, env=None):
        cache = str(Path(self.tmp.name) / "voices.json")
        with _env(), patch.dict(os.environ, env or {}), \
                patch.object(opm, "TTSModelManager", side_effect=lambda: TTSModelManager(cache_path=cache)), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            plugin = opm.PhoonnxTTSPlugin(config={"lang": "pt-BR", "onnx_providers": CPU, **config})
            plugin.config = {"lang": "pt-BR", "onnx_providers": CPU, **config}
            ctxt = plugin._get_ctxt({"lang": "pt-BR"})
            with _recording_fetches() as fetched:
                plugin.get_tts("Olá.", str(Path(self.tmp.name) / "out.wav"), lang="pt-BR",
                               voice=None if ctxt.voice == "default" else ctxt.voice)
        return ctxt.voice, fetched

    def _check(self, config, env=None, cache_voice="default"):
        cache_id, fetched = self._synthesize(config, env)
        self.assertEqual(cache_id, cache_voice)
        self.assertTrue(fetched)
        self.assertTrue(all("/edwixx__rapido-tts-int8/" in url for url in fetched), fetched)

    def test_as_the_voice(self):
        self._check({"voice": self.OLD}, cache_voice=self.OLD)

    def test_as_the_lang2voice_default(self):
        self._check({"lang2voice": {"pt-BR": self.OLD}})

    def test_as_the_env_default(self):
        self._check({}, env={"PHOONNX_DEFAULT_VOICE_PT_BR": self.OLD})


def _write_silence(text, wav_file, *args, **kwargs):
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(16000)
    wav_file.writeframes(b"\0\0")


@contextmanager
def _recording_fetches():
    """Record the ONNX URLs that loads and downloads fetch, with the engine and network stubbed."""
    fetched = []
    voice = MagicMock()
    voice.synthesize_wav.side_effect = _write_silence
    with patch.object(TTSModelInfo, "config", new_callable=PropertyMock), \
            patch.object(model_manager, "check_lang_supported"), \
            patch.object(model_manager, "_voice_phonemizer"), \
            patch.object(TTSModelInfo, "hub_path", return_value=None), \
            patch.object(TTSModelInfo, "_fetch_onnx",
                         side_effect=lambda url: fetched.append(url) or Path("model.onnx")), \
            patch.object(model_manager, "TTSVoice") as tts_voice:
        tts_voice.load.return_value = voice
        yield fetched


@contextmanager
def _recording_downloads():
    """Record the model URL of each ``download_all`` call, which downloads nothing."""
    downloaded = []

    def download_all(info, precision=None, providers=None):
        chosen = info.with_precision(precision, providers)
        downloaded.append(chosen.model_url)
        return Path("model.onnx")

    with patch.object(TTSModelInfo, "download_all", download_all):
        yield downloaded


def _load_and_record(info, **load_kwargs):
    """The ONNX URLs ``info.load`` fetches, with the engine and network stubbed."""
    with _recording_fetches() as fetched:
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


class TestDownloadPicksTheFiles(unittest.TestCase):

    def test_download_all_with_int8_fetches_the_variant(self):
        with _recording_fetches() as fetched:
            _entry(INT8).download_all(precision="int8")
        self.assertIn(f"{BASE}/model_int8.onnx", fetched)
        self.assertIn(f"{BASE}/decoder_int8.onnx", fetched)
        self.assertNotIn(f"{BASE}/model.onnx", fetched)

    def test_download_all_with_full_fetches_the_base(self):
        with _recording_fetches() as fetched:
            _entry(INT8).download_all(precision="full")
        self.assertIn(f"{BASE}/model.onnx", fetched)
        self.assertNotIn(f"{BASE}/model_int8.onnx", fetched)


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

    def test_the_default_voice_gets_int8_in_its_plugin_cache_id(self):
        info = _entry(INT8)
        voices = {precision: _plugin(info, voice=None, precision=precision)._get_ctxt({"lang": "pt-BR"}).voice
                  for precision in ("int8", "full")}
        self.assertEqual(voices, {"int8": "default#int8", "full": "default"})

    def test_an_int8_only_voice_keeps_its_plugin_cache_id(self):
        info = _entry(precision="int8")
        ctxt = _plugin(info)._get_ctxt({"voice": info.voice_id, "lang": "pt-BR"})
        self.assertEqual(ctxt.voice, "demo/voice")


class TestUnknownIndexKeys(unittest.TestCase):

    def test_an_unknown_key_is_ignored_with_one_warning(self):
        model_manager._LOGGED_UNKNOWN_KEYS.clear()
        entry = {"voice_id": "demo/voice", "lang": "pt-BR", "model_url": f"{BASE}/model.onnx",
                 "a_future_field": {"x": 1}}
        with patch.object(model_manager, "LOG") as log:
            first = TTSModelInfo(**model_manager._known_fields(entry))
            TTSModelInfo(**model_manager._known_fields(entry))
        self.assertEqual(first.model_url, f"{BASE}/model.onnx")
        self.assertEqual(len(log.warning.call_args_list), 1)
        self.assertIn("a_future_field", log.warning.call_args.args[0])
        self.assertIn("demo/voice", log.warning.call_args.args[0])

    def test_a_cached_entry_with_an_unknown_key_is_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voices.json"
            path.write_text(json.dumps({"demo/voice": {
                "voice_id": "demo/voice", "lang": "pt-BR", "model_url": f"{BASE}/model.onnx",
                "a_future_field": 1}}))
            manager = TTSModelManager(cache_path=str(path))
            manager.load()
        self.assertEqual(manager.get_voice("demo/voice").model_url, f"{BASE}/model.onnx")

    def test_a_variant_replaces_only_the_files(self):
        info = _entry({**INT8, "lang": "en-US", "precision": "full"}).with_precision("int8")
        self.assertEqual(info.model_url, f"{BASE}/model_int8.onnx")
        self.assertEqual(info.lang, "pt-BR")
        self.assertEqual(info.precision, "int8")

    def test_an_unknown_variant_key_is_ignored(self):
        info = _entry({**INT8, "rtf": 0.2}).with_precision("int8")
        self.assertEqual(info.model_url, f"{BASE}/model_int8.onnx")


class TestPrecisionSchema(unittest.TestCase):

    def _problems(self, **fields):
        entry = {"voice_id": "demo/voice", "lang": "pt-BR", "model_url": f"{BASE}/model.onnx", **fields}
        return validate_entry("demo/voice", entry)

    def test_a_valid_variant_passes(self):
        self.assertEqual(self._problems(variants={"int8": {**INT8, "auto": True}}), [])
        self.assertEqual(self._problems(precision="int8"), [])

    def test_each_defect_is_named(self):
        cases = {
            "precision is 'fp16'": dict(precision="fp16"),
            "unknown variant 'fp16'": dict(variants={"fp16": INT8}),
            "int8-only entry cannot carry": dict(precision="int8", variants={"int8": INT8}),
            "names no file": dict(variants={"int8": {"auto": True}}),
            "'lang' is not a file field": dict(variants={"int8": {**INT8, "lang": "en"}}),
            "'auto' is 'yes'": dict(variants={"int8": {**INT8, "auto": "yes"}}),
            "'aux_model_urls' is null": dict(variants={"int8": {**INT8, "aux_model_urls": None}}),
            "'model_url' has 5": dict(variants={"int8": {"model_url": 5}}),
        }
        for expected, fields in cases.items():
            with self.subTest(expected):
                problems = self._problems(**fields)
                self.assertEqual(len(problems), 1, problems)
                self.assertIn(expected, problems[0])


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
        self.assertEqual(len(int8_only), 130)
        self.assertTrue(all(v.startswith("pockettts/") for v in int8_only))

    def test_italian_pocket_tts_defaults_to_full_and_other_languages_to_int8(self):
        # The int8 Italian graphs repeat words on long sentences, so the
        # Italian voices load the full graphs of the same folder and commit
        # unless int8 is asked for. Every other language has only int8 files.
        path = next(p for p in TTSModelManager.voice_index_files() if p.name == "pockettts.json")
        entries = json.loads(path.read_text(encoding="utf-8"))
        langs = {entry["lang"].split("-")[0] for entry in entries.values()}
        self.assertEqual(langs, {"en", "fr", "de", "it", "pt", "es"})
        for voice_id, entry in entries.items():
            with self.subTest(voice_id):
                info = TTSModelInfo(**entry)
                graphs = [info.model_url, info.aux_model_urls["flow_lm_flow_path"],
                          info.aux_model_urls["mimi_decoder_path"]]
                if info.lang != "it":
                    self.assertEqual(info.precision, "int8")
                    self.assertIsNone(info.variants)
                    self.assertTrue(all(url.endswith("_int8.onnx") for url in graphs))
                    continue
                self.assertIsNone(info.precision)
                self.assertEqual([url.rsplit("/", 1)[1] for url in graphs],
                                 ["flow_lm_main.onnx", "flow_lm_flow.onnx", "mimi_decoder.onnx"])
                self.assertEqual(_url_fields(info.with_precision("auto", CPU)), _url_fields(info))
                int8 = info.with_precision("int8")
                self.assertEqual(int8.precision, "int8")
                self.assertEqual(int8.model_url, info.model_url.replace(".onnx", "_int8.onnx"))
                for key in ("flow_lm_flow_path", "mimi_decoder_path"):
                    self.assertEqual(int8.aux_model_urls[key],
                                     info.aux_model_urls[key].replace(".onnx", "_int8.onnx"))
                for key in set(info.aux_model_urls) - {"flow_lm_flow_path", "mimi_decoder_path"}:
                    self.assertEqual(int8.aux_model_urls[key], info.aux_model_urls[key])


def _plugin(*infos, **config):
    """A plugin whose catalog holds ``infos``, with no network. ``voice=None`` leaves the voice unset."""
    manager = MagicMock()
    manager.get_voice.side_effect = {i.voice_id: i for i in infos}.get
    manager.get_lang_voices.return_value = list(infos)
    config = {"lang": "pt-BR", "voice": infos[0].voice_id, "onnx_providers": CPU, **config}
    if config["voice"] is None:
        del config["voice"]
    with patch.object(opm, "TTSModelManager", return_value=manager):
        plugin = opm.PhoonnxTTSPlugin(config=config)
    plugin.config = config
    return plugin


if __name__ == "__main__":
    unittest.main()
