import unittest
from unittest.mock import patch, MagicMock

import requests
from click.testing import CliRunner

from phoonnx.cli import cli, print_voice_info


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


class TestDownloadVoice(unittest.TestCase):
    @patch("phoonnx.cli.TTSModelManager")
    def test_downloads_cached_voice(self, mock_manager_cls):
        manager = mock_manager_cls.return_value
        voice = make_voice(voice_id="piper/foo")
        manager.voices = {"piper/foo": voice}
        runner = CliRunner()
        result = runner.invoke(cli, ["download", "piper/foo"])
        self.assertEqual(result.exit_code, 0)
        voice.download_model.assert_called_once()
        self.assertIn("Download complete", result.output)

    @patch("phoonnx.cli.TTSModelManager")
    def test_unknown_voice_id_reports_error_nonzero_exit(self, mock_manager_cls):
        manager = mock_manager_cls.return_value
        manager.voices = {}
        manager.download_voice_by_id.return_value = False
        runner = CliRunner()
        result = runner.invoke(cli, ["download", "no/such-voice"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Error: Voice ID 'no/such-voice' not found.", result.output)
        self.assertIn("list-available", result.output)
        self.assertNotIn("Traceback", result.output)

    @patch("phoonnx.cli.TTSModelManager")
    def test_fallback_download_succeeds(self, mock_manager_cls):
        manager = mock_manager_cls.return_value
        manager.voices = {}
        manager.download_voice_by_id.return_value = True
        runner = CliRunner()
        result = runner.invoke(cli, ["download", "piper/other"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Download complete", result.output)

    @patch("phoonnx.cli.TTSModelManager")
    def test_network_failure_reports_error_no_traceback(self, mock_manager_cls):
        manager = mock_manager_cls.return_value
        voice = make_voice(voice_id="piper/foo")
        voice.download_model.side_effect = requests.exceptions.ConnectionError("network down")
        manager.voices = {"piper/foo": voice}
        runner = CliRunner()
        result = runner.invoke(cli, ["download", "piper/foo"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Download failed due to network error", result.output)
        self.assertNotIn("Traceback", result.output)

    @patch("phoonnx.cli.TTSModelManager")
    def test_unexpected_exception_reports_error_no_traceback(self, mock_manager_cls):
        manager = mock_manager_cls.return_value
        voice = make_voice(voice_id="piper/foo")
        voice.download_model.side_effect = ValueError("bad data")
        manager.voices = {"piper/foo": voice}
        runner = CliRunner()
        result = runner.invoke(cli, ["download", "piper/foo"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("An unexpected error occurred during download: bad data", result.output)
        self.assertNotIn("Traceback", result.output)


if __name__ == "__main__":
    unittest.main()
