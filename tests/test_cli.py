import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import requests
from click.testing import CliRunner

from phoonnx.cli import cli, print_voice_info
from phoonnx.model_manager import TTSModelInfo, TTSModelManager


def make_voice(voice_id="test/voice", lang="en-US", display_name=None):
    voice = MagicMock()
    voice.voice_id = voice_id
    voice.lang = lang
    voice.display_name = display_name
    voice.engine.value = "piper"
    voice.phoneme_type = "espeak"
    voice.model_url = "https://example.com/model.onnx"
    voice.config_url = "https://example.com/config.json"
    voice.voice_path = "/tmp/voice"
    return voice


class TestPrintVoiceInfo(unittest.TestCase):
    def test_prints_all_fields(self):
        runner = CliRunner()
        voice = make_voice(display_name="Test Voice")
        with runner.isolation() as (out, _, _):
            print_voice_info(voice)
            result = out.getvalue().decode()
        self.assertIn("ID:          test/voice", result)
        self.assertIn("Name:        Test Voice", result)
        self.assertIn("Language:    en-US", result)
        self.assertIn("Engine:      piper", result)

    def test_omits_name_when_absent(self):
        runner = CliRunner()
        voice = make_voice(display_name=None)
        with runner.isolation() as (out, _, _):
            print_voice_info(voice)
            result = out.getvalue().decode()
        self.assertNotIn("Name:", result)


class TestUpdateCache(unittest.TestCase):
    @patch("phoonnx.cli.TTSModelManager")
    def test_clears_cache_by_default(self, mock_manager_cls):
        manager = mock_manager_cls.return_value
        manager.all_voices = [make_voice()]
        manager.supported_langs = ["en-US"]
        runner = CliRunner()
        result = runner.invoke(cli, ["update-cache"])
        self.assertEqual(result.exit_code, 0)
        manager.clear.assert_called_once()
        manager.merge_default_voices.assert_called_once_with(store=True)
        self.assertIn("Cache updated successfully", result.output)

    @patch("phoonnx.cli.TTSModelManager")
    def test_no_clear_loads_existing(self, mock_manager_cls):
        manager = mock_manager_cls.return_value
        manager.all_voices = []
        manager.supported_langs = []
        runner = CliRunner()
        result = runner.invoke(cli, ["update-cache", "--no-clear"])
        self.assertEqual(result.exit_code, 0)
        manager.clear.assert_not_called()
        manager.load.assert_called_once()

    @patch("phoonnx.cli.TTSModelManager")
    def test_merge_failure_reports_error_no_traceback(self, mock_manager_cls):
        manager = mock_manager_cls.return_value
        manager.merge_default_voices.side_effect = RuntimeError("index corrupt")
        runner = CliRunner()
        result = runner.invoke(cli, ["update-cache"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("An unexpected error occurred while loading voice lists: index corrupt", result.output)
        self.assertNotIn("Traceback", result.output)

    @patch("phoonnx.cli.TTSModelManager")
    def test_unwritable_cache_dir_reports_error(self, mock_manager_cls):
        manager = mock_manager_cls.return_value
        manager.clear.side_effect = PermissionError("Permission denied: /readonly/cache")
        runner = CliRunner()
        result = runner.invoke(cli, ["update-cache"])
        self.assertEqual(result.exit_code, 1)
        self.assertNotIn("Traceback", result.output)


class TestListLangs(unittest.TestCase):
    @patch("phoonnx.cli.TTSModelManager")
    def test_empty_cache_prompts_update(self, mock_manager_cls):
        manager = mock_manager_cls.return_value
        manager.supported_langs = []
        runner = CliRunner()
        result = runner.invoke(cli, ["list-langs"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Run 'update-cache' first", result.output)

    @patch("phoonnx.cli.TTSModelManager")
    def test_lists_langs(self, mock_manager_cls):
        manager = mock_manager_cls.return_value
        manager.supported_langs = ["en-US", "pt-PT"]
        runner = CliRunner()
        result = runner.invoke(cli, ["list-langs"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("en-US", result.output)
        self.assertIn("pt-PT", result.output)


class TestListVoices(unittest.TestCase):
    @patch("phoonnx.cli.TTSModelManager")
    def test_no_voices_in_cache(self, mock_manager_cls):
        manager = mock_manager_cls.return_value
        manager.all_voices = []
        runner = CliRunner()
        result = runner.invoke(cli, ["list-voices"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Run 'update-cache' first", result.output)

    @patch("phoonnx.cli.TTSModelManager")
    def test_unknown_lang_filter_returns_empty_not_crash(self, mock_manager_cls):
        manager = mock_manager_cls.return_value
        manager.all_voices = [make_voice()]
        manager.get_lang_voices.return_value = []
        runner = CliRunner()
        result = runner.invoke(cli, ["list-voices", "--lang", "zz-ZZ"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Found 0 voices for language 'zz-ZZ'", result.output)
        self.assertIn("No voices found for 'zz-ZZ'", result.output)

    @patch("phoonnx.cli.TTSModelManager")
    def test_lists_all_voices_default(self, mock_manager_cls):
        manager = mock_manager_cls.return_value
        manager.all_voices = [make_voice(display_name="Voice A")]
        runner = CliRunner()
        result = runner.invoke(cli, ["list-voices"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Voice A (en-US)", result.output)

    @patch("phoonnx.cli.TTSModelManager")
    def test_lists_voice_id_when_no_display_name(self, mock_manager_cls):
        manager = mock_manager_cls.return_value
        manager.all_voices = [make_voice(voice_id="piper/foo", display_name=None)]
        runner = CliRunner()
        result = runner.invoke(cli, ["list-voices"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("piper/foo (en-US)", result.output)

    @patch("phoonnx.cli.TTSModelManager")
    def test_verbose_prints_details(self, mock_manager_cls):
        manager = mock_manager_cls.return_value
        manager.all_voices = [make_voice()]
        runner = CliRunner()
        result = runner.invoke(cli, ["list-voices", "-v"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Model URL:", result.output)


class TestListAvailable(unittest.TestCase):
    @patch("phoonnx.cli.TTSModelManager")
    def test_groups_by_source(self, mock_manager_cls):
        manager = mock_manager_cls.return_value
        manager.get_available_voice_ids_by_source.return_value = {
            "piper": ["piper/a", "piper/b"],
            "mms": ["mms/x"],
        }
        runner = CliRunner()
        result = runner.invoke(cli, ["list-available"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Total available voices found (bundled indexes): 3", result.output)
        self.assertIn("PIPER Voices (2)", result.output)
        self.assertIn("piper/a", result.output)
        self.assertIn("MMS Voices (1)", result.output)

    @patch("phoonnx.cli.TTSModelManager")
    def test_empty_result(self, mock_manager_cls):
        manager = mock_manager_cls.return_value
        manager.get_available_voice_ids_by_source.return_value = {}
        runner = CliRunner()
        result = runner.invoke(cli, ["list-available"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Total available voices found (bundled indexes): 0", result.output)


class TestListVoicesLangFilter(unittest.TestCase):
    LANGS = ("pt-PT", "pt-BR", "pt", "pt-PT-x-lisbon", "en-US", "zh", "zh-CN", "zh-TW")  # zh tags as the index has them

    def _ranked(self, lang):
        """What the manager itself ranks for ``lang``, without the CLI filter."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with patch("phoonnx.model_manager.os.path.expanduser", return_value=tmp.name):
            manager = TTSModelManager(cache_path=os.path.join(tmp.name, "cache.json"))
        manager.voices = {f"v/{l}": TTSModelInfo(voice_id=f"v/{l}", lang=l,
                                                  model_url=f"https://example.com/{l}.onnx")
                          for l in self.LANGS}
        return sorted(v.lang for v in manager.get_lang_voices(lang))

    def _list(self, lang):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with patch("phoonnx.model_manager.os.path.expanduser", return_value=tmp.name):
            manager = TTSModelManager(cache_path=os.path.join(tmp.name, "cache.json"))
        manager.load = lambda: None
        manager.voices = {f"v/{l}": TTSModelInfo(voice_id=f"v/{l}", lang=l,
                                                  model_url=f"https://example.com/{l}.onnx")
                          for l in self.LANGS}
        with patch("phoonnx.cli.TTSModelManager", return_value=manager):
            result = CliRunner().invoke(cli, ["list-voices", "--lang", lang])
        self.assertEqual(result.exit_code, 0)
        return sorted(line.split("(")[-1].rstrip(")") for line in result.output.splitlines()
                      if line.startswith("* "))

    def test_region_tag_lists_only_that_region(self):
        self.assertEqual(self._list("pt-PT"), ["pt", "pt-PT", "pt-PT-x-lisbon"])

    def test_tag_spelling_does_not_matter(self):
        self.assertEqual(self._list("PT_pt"), self._list("pt-PT"))

    def test_bare_tag_lists_every_region(self):
        self.assertEqual(self._list("pt"), ["pt", "pt-BR", "pt-PT", "pt-PT-x-lisbon"])

    def test_script_tag_without_region_keeps_every_voice(self):
        self.assertTrue(self._ranked("zh-Hans"))
        self.assertEqual(self._list("zh-Hans"), self._ranked("zh-Hans"))

    def test_region_with_no_voice_lists_the_nearest_voices(self):
        self.assertTrue(self._ranked("en-AU"))
        self.assertEqual(self._list("en-AU"), self._ranked("en-AU"))


class TestDownloadVoice(unittest.TestCase):
    @patch("phoonnx.cli.TTSModelManager")
    def test_downloads_cached_voice(self, mock_manager_cls):
        manager = mock_manager_cls.return_value
        voice = make_voice(voice_id="piper/foo")
        manager.get_voice.side_effect = {"piper/foo": voice}.get
        runner = CliRunner()
        result = runner.invoke(cli, ["download", "piper/foo"])
        self.assertEqual(result.exit_code, 0)
        voice.download_all.assert_called_once()
        self.assertIn("Download complete", result.output)

    @patch("phoonnx.cli.TTSModelManager")
    def test_unknown_voice_id_reports_error_nonzero_exit(self, mock_manager_cls):
        manager = mock_manager_cls.return_value
        manager.get_voice.return_value = None
        manager.download_voice_by_id.return_value = False
        runner = CliRunner()
        result = runner.invoke(cli, ["download", "no/such-voice"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Error: Voice ID 'no/such-voice' not found.", result.output)
        self.assertIn("list-available", result.output)
        self.assertNotIn("Traceback", result.output)

    @patch("phoonnx.cli.TTSModelManager")
    def test_fallback_download_succeeds(self, mock_manager_cls):
        manager = mock_manager_cls.return_value
        manager.get_voice.return_value = None
        manager.download_voice_by_id.return_value = True
        runner = CliRunner()
        result = runner.invoke(cli, ["download", "piper/other"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Download complete", result.output)

    @patch("phoonnx.cli.TTSModelManager")
    def test_network_failure_reports_error_no_traceback(self, mock_manager_cls):
        manager = mock_manager_cls.return_value
        voice = make_voice(voice_id="piper/foo")
        voice.download_all.side_effect = requests.exceptions.ConnectionError("network down")
        manager.get_voice.side_effect = {"piper/foo": voice}.get
        runner = CliRunner()
        result = runner.invoke(cli, ["download", "piper/foo"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Download failed due to network error", result.output)
        self.assertNotIn("Traceback", result.output)

    @patch("phoonnx.cli.TTSModelManager")
    def test_unexpected_exception_reports_error_no_traceback(self, mock_manager_cls):
        manager = mock_manager_cls.return_value
        voice = make_voice(voice_id="piper/foo")
        voice.download_all.side_effect = ValueError("bad data")
        manager.get_voice.side_effect = {"piper/foo": voice}.get
        runner = CliRunner()
        result = runner.invoke(cli, ["download", "piper/foo"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("An unexpected error occurred during download: bad data", result.output)
        self.assertNotIn("Traceback", result.output)


class TestDownloadFailureExitsNonZero(unittest.TestCase):
    """Each failure leaves a non-zero exit status and a message on stderr."""

    def _run(self, voice_id, **manager_attrs):
        with patch("phoonnx.cli.TTSModelManager") as manager_cls:
            manager = manager_cls.return_value
            manager.get_voice.return_value = None
            for name, value in manager_attrs.items():
                setattr(manager, name, value)
            return CliRunner().invoke(cli, ["download", voice_id])

    def test_unknown_id(self):
        result = self._run("no/such-voice", download_voice_by_id=MagicMock(return_value=False))
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(len(result.stderr.strip().splitlines()), 1)
        self.assertIn("no/such-voice", result.stderr)

    def test_network_failure_on_bundled_voice(self):
        down = MagicMock(side_effect=requests.exceptions.ConnectionError("network down"))
        result = self._run("piper/foo", download_voice_by_id=down)
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.stderr.strip().splitlines(),
                         ["Download failed due to network error: network down"])

    def test_offline_voice_not_cached(self):
        from huggingface_hub.errors import LocalEntryNotFoundError
        down = MagicMock(side_effect=LocalEntryNotFoundError("offline and not cached"))
        result = self._run("piper/foo", download_voice_by_id=down)
        self.assertEqual(result.exit_code, 1)
        self.assertIn("offline and not cached", result.stderr)


class TestAddAlignment(unittest.TestCase):
    """The offline build step for phoneme alignments."""

    def test_writes_the_patched_model_beside_the_original(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("model.onnx", "wb") as f:
                f.write(b"not really onnx")
            with patch("phoonnx.alignment.export_alignment_model",
                       return_value="model.alignment.onnx") as export:
                result = runner.invoke(cli, ["add-alignment", "model.onnx"])
            export.assert_called_once_with("model.onnx", None)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("model.alignment.onnx", result.output)

    def test_a_model_without_durations_fails_loudly(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("model.onnx", "wb") as f:
                f.write(b"not really onnx")
            with patch("phoonnx.alignment.export_alignment_model",
                       side_effect=ValueError("no unique duration tensor")):
                result = runner.invoke(cli, ["add-alignment", "model.onnx"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("no unique duration tensor", result.output)

    def test_a_missing_onnx_package_names_the_extra(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("model.onnx", "wb") as f:
                f.write(b"not really onnx")
            with patch("phoonnx.alignment.export_alignment_model",
                       side_effect=ImportError("no module named onnx")):
                result = runner.invoke(cli, ["add-alignment", "model.onnx"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("phoonnx[streaming]", result.output)


if __name__ == "__main__":
    unittest.main()
