import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import requests

from phoonnx.model_manager import TTSModelInfo, TTSModelManager
from phoonnx.config import Engine, PhonemeType, Alphabet


def _mock_response(status_code=200, json_data=None, content=b"", text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.text = text
    resp.iter_content.return_value = [content] if content else []
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError("No JSON")
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{status_code} error")
    else:
        resp.raise_for_status.return_value = None
    return resp


class VoicePathTestCase(unittest.TestCase):
    """Base class that redirects TTSModelInfo.voice_path into a tmp dir."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        patcher = patch("phoonnx.model_manager.os.path.expanduser", return_value=self._tmpdir.name)
        self.addCleanup(patcher.stop)
        patcher.start()

    def make_info(self, **kwargs):
        defaults = dict(
            voice_id="test/voice",
            lang="en-US",
            model_url="https://example.com/model.onnx",
        )
        defaults.update(kwargs)
        return TTSModelInfo(**defaults)


class TestFetchOnnx(VoicePathTestCase):
    @patch("phoonnx.model_manager.requests.get")
    def test_downloads_model_and_sidecar_missing_is_404(self, mock_get):
        info = self.make_info()
        model_resp = _mock_response(200, content=b"onnxdata")
        sidecar_resp = _mock_response(404)

        def side_effect(url, timeout=None, stream=None):
            if url == info.model_url:
                cm = MagicMock()
                cm.__enter__.return_value = model_resp
                return cm
            cm = MagicMock()
            cm.__enter__.return_value = sidecar_resp
            return cm

        mock_get.side_effect = side_effect
        dest = info.voice_path / "model.onnx"
        result = info._fetch_onnx(info.model_url, dest)
        self.assertTrue(dest.is_file())
        self.assertEqual(dest.read_bytes(), b"onnxdata")
        self.assertEqual(result, dest)
        self.assertFalse((info.voice_path / "model.onnx_data").is_file())

    @patch("phoonnx.model_manager.requests.get")
    def test_http_404_on_primary_download_raises(self, mock_get):
        info = self.make_info()
        resp = _mock_response(404)
        cm = MagicMock()
        cm.__enter__.return_value = resp
        mock_get.return_value = cm
        dest = info.voice_path / "model.onnx"
        with self.assertRaises(requests.exceptions.HTTPError):
            info._fetch_onnx(info.model_url, dest)

    @patch("phoonnx.model_manager.requests.get")
    def test_http_500_on_primary_download_raises(self, mock_get):
        info = self.make_info()
        resp = _mock_response(500)
        cm = MagicMock()
        cm.__enter__.return_value = resp
        mock_get.return_value = cm
        dest = info.voice_path / "model.onnx"
        with self.assertRaises(requests.exceptions.HTTPError):
            info._fetch_onnx(info.model_url, dest)

    @patch("phoonnx.model_manager.requests.get")
    def test_sidecar_network_failure_treated_as_absent(self, mock_get):
        info = self.make_info()
        model_resp = _mock_response(200, content=b"data")

        def side_effect(url, timeout=None, stream=None):
            if url == info.model_url:
                cm = MagicMock()
                cm.__enter__.return_value = model_resp
                return cm
            raise requests.exceptions.ConnectionError("offline")

        mock_get.side_effect = side_effect
        dest = info.voice_path / "model.onnx"
        result = info._fetch_onnx(info.model_url, dest)
        self.assertEqual(result, dest)
        self.assertTrue(dest.is_file())

    @patch("phoonnx.model_manager.requests.get")
    def test_already_cached_skips_download(self, mock_get):
        info = self.make_info()
        dest = info.voice_path / "model.onnx"
        dest.write_bytes(b"cached")
        data_dest = info.voice_path / (info.model_url.split("/")[-1] + "_data")
        data_dest.write_bytes(b"cached-data")
        result = info._fetch_onnx(info.model_url, dest)
        mock_get.assert_not_called()
        self.assertEqual(result, dest)


class TestDownloadHelpers(VoicePathTestCase):
    @patch("phoonnx.model_manager.requests.get")
    def test_download_model_delegates_to_fetch_onnx(self, mock_get):
        info = self.make_info()
        resp = _mock_response(200, content=b"x")
        cm = MagicMock()
        cm.__enter__.return_value = resp
        mock_get.return_value = cm
        with patch.object(info, "_fetch_onnx", wraps=info._fetch_onnx) as fetch:
            path = info.download_model()
            fetch.assert_called_once_with(info.model_url, info.voice_path / "model.onnx")
        self.assertTrue(path.is_file())

    def test_download_vocoder_none_when_no_url(self):
        info = self.make_info()
        self.assertIsNone(info.download_vocoder())

    @patch("phoonnx.model_manager.requests.get")
    def test_download_vocoder_with_config(self, mock_get):
        info = self.make_info(vocoder_url="https://example.com/vocoder.onnx",
                               vocoder_config_url="https://example.com/vocoder.json")
        vocoder_resp = _mock_response(200, content=b"vdata")
        config_resp = _mock_response(200, json_data={"a": 1})

        def side_effect(url, timeout=None, stream=None):
            if stream:
                cm = MagicMock()
                cm.__enter__.return_value = vocoder_resp
                return cm
            return config_resp

        mock_get.side_effect = side_effect
        path = info.download_vocoder()
        self.assertTrue(path.is_file())
        self.assertTrue((info.voice_path / "vocoder.json").is_file())

    def test_download_style_none_when_no_url(self):
        info = self.make_info()
        self.assertIsNone(info.download_style())

    @patch("phoonnx.model_manager.requests.get")
    def test_download_style_downloads(self, mock_get):
        info = self.make_info(style_url="https://example.com/style.bin")
        resp = _mock_response(200, content=b"styledata")
        cm = MagicMock()
        cm.__enter__.return_value = resp
        mock_get.return_value = cm
        path = info.download_style()
        self.assertTrue(path.is_file())
        self.assertEqual(path.read_bytes(), b"styledata")

    def test_download_speaker_encoder_none_when_no_url(self):
        info = self.make_info()
        self.assertIsNone(info.download_speaker_encoder())

    @patch("phoonnx.model_manager.requests.get")
    def test_download_speaker_encoder_downloads(self, mock_get):
        info = self.make_info(speaker_encoder_url="https://example.com/enc.onnx")
        resp = _mock_response(200, content=b"encdata")
        cm = MagicMock()
        cm.__enter__.return_value = resp
        mock_get.return_value = cm
        path = info.download_speaker_encoder()
        self.assertTrue(path.is_file())

    @patch("phoonnx.model_manager.requests.get")
    def test_download_speaker_encoder_http_500(self, mock_get):
        info = self.make_info(speaker_encoder_url="https://example.com/enc.onnx")
        resp = _mock_response(500)
        cm = MagicMock()
        cm.__enter__.return_value = resp
        mock_get.return_value = cm
        with self.assertRaises(requests.exceptions.HTTPError):
            info.download_speaker_encoder()

    def test_download_aux_models_empty(self):
        info = self.make_info()
        self.assertEqual(info.download_aux_models(), {})

    @patch("phoonnx.model_manager.requests.get")
    def test_download_aux_models_downloads_each(self, mock_get):
        info = self.make_info(aux_model_urls={
            "preprocess_path": "https://example.com/preprocess.onnx",
            "decode_path": "https://example.com/decode.onnx",
        })
        resp = _mock_response(200, content=b"aux")
        cm = MagicMock()
        cm.__enter__.return_value = resp
        mock_get.return_value = cm
        paths = info.download_aux_models()
        self.assertEqual(set(paths.keys()), {"preprocess_path", "decode_path"})
        for p in paths.values():
            self.assertTrue(p.is_file())

    def test_download_bpe_tokenizer_none_without_url(self):
        info = self.make_info()
        self.assertIsNone(info.download_bpe_tokenizer())

    @patch("phoonnx.model_manager.requests.get")
    def test_download_bpe_tokenizer_downloads(self, mock_get):
        info = self.make_info(tokenizer_config_url="https://example.com/tokenizer.json")
        resp = _mock_response(200, content=b'{"vocab": {}}')
        mock_get.return_value = resp
        path = info.download_bpe_tokenizer()
        self.assertTrue(path.is_file())
        self.assertEqual(path.read_bytes(), b'{"vocab": {}}')

    @patch("phoonnx.model_manager.requests.get")
    def test_download_bpe_tokenizer_http_404(self, mock_get):
        info = self.make_info(tokenizer_config_url="https://example.com/tokenizer.json")
        resp = _mock_response(404)
        mock_get.return_value = resp
        with self.assertRaises(requests.exceptions.HTTPError):
            info.download_bpe_tokenizer()


class TestDownloadConfig(VoicePathTestCase):
    @patch("phoonnx.model_manager.requests.get")
    def test_truncated_json_response_raises(self, mock_get):
        info = self.make_info(config_url="https://example.com/config.json")
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_get.return_value = resp
        with self.assertRaises(json.JSONDecodeError):
            info.download_config()

    @patch("phoonnx.model_manager.requests.get")
    def test_http_404_raises(self, mock_get):
        info = self.make_info(config_url="https://example.com/config.json")
        resp = _mock_response(404)
        mock_get.return_value = resp
        with self.assertRaises(requests.exceptions.HTTPError):
            info.download_config()

    def test_corrupt_cached_file_raises_on_load(self):
        info = self.make_info(config_url="https://example.com/config.json")
        config_path = info.voice_path / "model.json"
        config_path.write_text("{not valid json", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            info.download_config()

    @patch("phoonnx.model_manager.requests.get")
    def test_uses_cache_when_present(self, mock_get):
        info = self.make_info(config_url="https://example.com/config.json")
        config_path = info.voice_path / "model.json"
        config_path.write_text(json.dumps({"cached": True}), encoding="utf-8")
        result = info.download_config()
        mock_get.assert_not_called()
        self.assertEqual(result, {"cached": True})


class TestConfigProperty(VoicePathTestCase):
    @patch("phoonnx.model_manager.requests.get")
    def test_config_with_vocab_override(self, mock_get):
        info = self.make_info(
            config_url="https://example.com/config.json",
            vocab_override={"a": 1, "b": 2},
            phoneme_type="graphemes",
            alphabet="unicode",
        )
        mock_get.return_value = _mock_response(200, json_data={"blank": "a"})
        config = info.config
        self.assertIsNotNone(config)
        self.assertIs(info.config, config)

    @patch("phoonnx.model_manager.requests.get")
    def test_config_with_vocab_url_and_tokenizer_config(self, mock_get):
        info = self.make_info(
            config_url="https://example.com/config.json",
            vocab_url="https://example.com/vocab.json",
            tokenizer_config_url="https://example.com/tokenizer_config.json",
            phoneme_type="graphemes",
            alphabet="unicode",
        )

        def side_effect(url, timeout=None):
            if url == info.vocab_url:
                return _mock_response(200, json_data={"a": 0, "b": 1})
            if url == info.tokenizer_config_url:
                return _mock_response(200, json_data={"add_blank": True, "language": "en", "pad_token": "a"})
            return _mock_response(200, json_data={"blank": "a"})

        mock_get.side_effect = side_effect
        config = info.config
        self.assertIsNotNone(config)

    @patch("phoonnx.model_manager.requests.get")
    def test_config_with_tokens_url(self, mock_get):
        info = self.make_info(tokens_url="https://example.com/tokens.txt", phoneme_type="graphemes", alphabet="unicode")
        mock_get.return_value = _mock_response(200, text="a\nb\nc\n")
        config = info.config
        self.assertIsNotNone(config)
        self.assertTrue((info.voice_path / "tokens.txt").is_file())

    @patch("phoonnx.model_manager.requests.get")
    def test_config_espeak_phoneme_type_hack(self, mock_get):
        info = self.make_info(config_url="https://example.com/config.json")
        mock_get.return_value = _mock_response(200, json_data={"phoneme_type": "PhonemeType.ESPEAK", "alphabet": "ipa"})
        config = info.config
        self.assertIsNotNone(config)


class TestVocabAndTokenizerDownloads(VoicePathTestCase):
    @patch("phoonnx.model_manager.requests.get")
    def test_download_tokenizer_config_downloads_and_caches(self, mock_get):
        info = self.make_info(tokenizer_config_url="https://example.com/tokenizer_config.json")
        mock_get.return_value = _mock_response(200, json_data={"model_max_length": 64})
        result = info.download_tokenizer_config()
        self.assertEqual(result, {"model_max_length": 64})
        mock_get.reset_mock()
        result2 = info.download_tokenizer_config()
        mock_get.assert_not_called()
        self.assertEqual(result2, {"model_max_length": 64})

    @patch("phoonnx.model_manager.requests.get")
    def test_download_tokenizer_config_http_500(self, mock_get):
        info = self.make_info(tokenizer_config_url="https://example.com/tokenizer_config.json")
        mock_get.return_value = _mock_response(500)
        with self.assertRaises(requests.exceptions.HTTPError):
            info.download_tokenizer_config()

    @patch("phoonnx.model_manager.requests.get")
    def test_download_vocab_downloads_and_caches(self, mock_get):
        info = self.make_info(vocab_url="https://example.com/vocab.json")
        mock_get.return_value = _mock_response(200, json_data={"a": 0})
        result = info.download_vocab()
        self.assertEqual(result, {"a": 0})

    @patch("phoonnx.model_manager.requests.get")
    def test_download_vocab_http_404(self, mock_get):
        info = self.make_info(vocab_url="https://example.com/vocab.json")
        mock_get.return_value = _mock_response(404)
        with self.assertRaises(requests.exceptions.HTTPError):
            info.download_vocab()

    @patch("phoonnx.model_manager.requests.get")
    def test_download_tokens_txt_downloads_and_caches(self, mock_get):
        info = self.make_info(tokens_url="https://example.com/tokens.txt")
        mock_get.return_value = _mock_response(200, text="a\nb\n")
        result = info.download_tokens_txt()
        self.assertEqual(result, "a\nb\n")
        mock_get.reset_mock()
        result2 = info.download_tokens_txt()
        mock_get.assert_not_called()
        self.assertEqual(result2, "a\nb\n")

    @patch("phoonnx.model_manager.requests.get")
    def test_download_tokens_txt_http_500(self, mock_get):
        info = self.make_info(tokens_url="https://example.com/tokens.txt")
        mock_get.return_value = _mock_response(500)
        with self.assertRaises(requests.exceptions.HTTPError):
            info.download_tokens_txt()


class TestEngineParams(VoicePathTestCase):
    def test_empty_for_single_stage_engine(self):
        info = self.make_info()
        self.assertEqual(info.engine_params(), {})

    @patch("phoonnx.model_manager.requests.get")
    def test_combines_vocoder_style_speaker_encoder(self, mock_get):
        info = self.make_info(
            vocoder_url="https://example.com/vocoder.onnx",
            vocoder_type="hifigan",
            style_url="https://example.com/style.bin",
            speaker_encoder_url="https://example.com/enc.onnx",
            speaker_encoder_type="dvector",
        )
        resp = _mock_response(200, content=b"data")
        cm = MagicMock()
        cm.__enter__.return_value = resp
        mock_get.return_value = cm
        params = info.engine_params()
        self.assertEqual(params["vocoder_type"], "hifigan")
        self.assertEqual(params["speaker_encoder_type"], "dvector")
        self.assertIn("style_path", params)
        self.assertIn("vocoder_path", params)
        self.assertIn("speaker_encoder_path", params)

    @patch("phoonnx.model_manager.requests.get")
    def test_parametric_vocoder_no_model_file(self, mock_get):
        info = self.make_info(vocoder_type="griffinlim", vocoder_config_url="https://example.com/vocoder.json")
        mock_get.return_value = _mock_response(200, content=b'{"n_fft": 1024}')
        params = info.engine_params()
        self.assertEqual(params["vocoder_type"], "griffinlim")
        self.assertIn("vocoder_config", params)
        self.assertEqual(params["vocoder_config"], {"n_fft": 1024})

    @patch("phoonnx.model_manager.requests.get")
    def test_aux_model_url_without_filename_uses_key(self, mock_get):
        info = self.make_info(aux_model_urls={"weird_path": "https://example.com/graphs/"})
        resp = _mock_response(200, content=b"data")
        cm = MagicMock()
        cm.__enter__.return_value = resp
        mock_get.return_value = cm
        paths = info.download_aux_models()
        self.assertTrue(str(paths["weird_path"]).endswith("weird_path.onnx"))

    @patch("phoonnx.model_manager.requests.get")
    def test_chatterbox_aux_graphs(self, mock_get):
        info = self.make_info(
            speech_encoder_url="https://example.com/speech_encoder.onnx",
            embed_tokens_url="https://example.com/embed_tokens.onnx",
            conditional_decoder_url="https://example.com/conditional_decoder.onnx",
        )
        resp = _mock_response(200, content=b"data")
        cm = MagicMock()
        cm.__enter__.return_value = resp
        mock_get.return_value = cm
        params = info.engine_params()
        self.assertIn("speech_encoder_path", params)
        self.assertIn("embed_tokens_path", params)
        self.assertIn("conditional_decoder_path", params)


class TestLoad(VoicePathTestCase):
    @patch("phoonnx.model_manager.TTSVoice")
    @patch("phoonnx.model_manager.requests.get")
    def test_load_downloads_model_then_builds_voice(self, mock_get, mock_ttsvoice_cls):
        info = self.make_info(config_url="https://example.com/config.json")
        resp = _mock_response(200, content=b"data")
        cm = MagicMock()
        cm.__enter__.return_value = resp
        mock_get.return_value = cm
        info.voice_path.joinpath("model.json").write_text(
            json.dumps({"phoneme_type": "graphemes", "alphabet": "unicode"}), encoding="utf-8"
        )
        fake_voice = MagicMock()
        fake_voice.config.phoneme_type = info.config.phoneme_type
        fake_voice.config.alphabet = info.config.alphabet
        mock_ttsvoice_cls.load.return_value = fake_voice
        result = info.load()
        mock_ttsvoice_cls.load.assert_called_once()
        self.assertEqual(result, fake_voice)

    @patch("phoonnx.model_manager.VoiceConfig")
    @patch("phoonnx.model_manager.onnxruntime")
    @patch("phoonnx.model_manager.TTSVoice")
    @patch("phoonnx.model_manager.requests.get")
    def test_load_engine_without_config_url_uses_session(self, mock_get, mock_ttsvoice_cls, mock_ort, mock_voiceconfig_cls):
        info = self.make_info(engine=Engine.CHATTERBOX, tokenizer_config_url="https://example.com/tokenizer.json")

        def side_effect(url, timeout=None, stream=None):
            if stream:
                resp = _mock_response(200, content=b"data")
                cm = MagicMock()
                cm.__enter__.return_value = resp
                return cm
            return _mock_response(200, content=b'{"vocab": {}}')

        mock_get.side_effect = side_effect
        fake_config = MagicMock()
        fake_config.engine_params = {}
        mock_voiceconfig_cls.from_dict.return_value = fake_config
        session = MagicMock()
        mock_ort.InferenceSession.return_value = session
        fake_voice = MagicMock()
        mock_ttsvoice_cls.return_value = fake_voice
        result = info.load()
        mock_ort.InferenceSession.assert_called_once()
        self.assertEqual(result, fake_voice)


class TestTTSModelInfoStringEnumCoercion(VoicePathTestCase):
    def test_string_engine_alphabet_phoneme_type_coerced(self):
        info = self.make_info(engine="piper", alphabet="ipa", phoneme_type="espeak")
        self.assertEqual(info.engine, Engine.PIPER)
        self.assertEqual(info.alphabet, Alphabet.IPA)
        self.assertEqual(info.phoneme_type, PhonemeType.ESPEAK)

    def test_invalid_enum_value_raises(self):
        with self.assertRaises(ValueError):
            self.make_info(engine="not-a-real-engine")

    def test_display_name_placeholder_resolution(self):
        info = self.make_info(engine="piper", phoneme_type="espeak", display_name="{engine}/{phoneme_type}")
        self.assertEqual(info.display_name, "piper/espeak")


class ManagerTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.cache_path = os.path.join(self._tmpdir.name, "cache.json")
        expand_patcher = patch("phoonnx.model_manager.os.path.expanduser", return_value=self._tmpdir.name)
        self.addCleanup(expand_patcher.stop)
        expand_patcher.start()


class TestManagerLoadMalformed(ManagerTestCase):
    def test_load_skips_malformed_entries(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        manager.cache["good/voice"] = {
            "voice_id": "good/voice",
            "lang": "en-US",
            "model_url": "https://example.com/model.onnx",
        }
        manager.cache["missing_required_field"] = {"voice_id": "bad/voice"}
        manager.cache["bad_engine_value"] = {
            "voice_id": "bad2/voice",
            "lang": "en-US",
            "model_url": "https://example.com/model.onnx",
            "engine": "not-a-real-engine",
        }
        manager.cache.store()
        manager.load()
        self.assertIn("good/voice", manager.voices)
        self.assertNotIn("missing_required_field", manager.voices)
        self.assertNotIn("bad_engine_value", manager.voices)

    def test_merge_default_voices_skips_bad_entries_and_stores(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        with patch.object(manager.cache, "update") as mock_update:
            manager.merge_default_voices(store=True)
        self.assertTrue(mock_update.called)


class TestManagerRegistry(ManagerTestCase):
    def test_all_voices_and_supported_langs(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        v1 = TTSModelInfo(voice_id="a/1", lang="en-US", model_url="https://example.com/a.onnx")
        v2 = TTSModelInfo(voice_id="a/2", lang="pt-PT", model_url="https://example.com/b.onnx")
        manager.voices = {"a/1": v1, "a/2": v2}
        self.assertEqual(sorted(manager.all_voices, key=lambda v: v.voice_id), [v1, v2])
        self.assertEqual(manager.supported_langs, ["en-US", "pt-PT"])

    def test_clear_resets_voices_and_cache(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        manager.voices = {"x": MagicMock()}
        manager.cache["x"] = {"voice_id": "x"}
        manager.clear()
        self.assertEqual(manager.voices, {})
        self.assertEqual(len(manager.cache), 0)

    def test_add_voice_persists_to_cache(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        voice = TTSModelInfo(voice_id="a/1", lang="en-US", model_url="https://example.com/a.onnx")
        manager.add_voice(voice)
        self.assertIn("a/1", manager.voices)
        self.assertEqual(manager.cache["a/1"]["voice_id"], "a/1")

    def test_save_persists_all_voices(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        voice = TTSModelInfo(voice_id="a/1", lang="en-US", model_url="https://example.com/a.onnx")
        manager.voices = {"a/1": voice}
        manager.save()
        manager.cache.reload()
        self.assertIn("a/1", manager.cache)
        self.assertEqual(manager.cache["a/1"]["model_url"], "https://example.com/a.onnx")

    def test_get_lang_voices_filters_by_match(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        v_en = TTSModelInfo(voice_id="a/1", lang="en-US", model_url="https://example.com/a.onnx")
        v_pt = TTSModelInfo(voice_id="a/2", lang="pt-PT", model_url="https://example.com/b.onnx")
        manager.voices = {"a/1": v_en, "a/2": v_pt}
        result = manager.get_lang_voices("en-US")
        self.assertEqual(result, [v_en])

    def test_get_lang_voices_unknown_lang_returns_empty(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        v_en = TTSModelInfo(voice_id="a/1", lang="en-US", model_url="https://example.com/a.onnx")
        manager.voices = {"a/1": v_en}
        result = manager.get_lang_voices("zz-ZZ")
        self.assertEqual(result, [])


class TestGetAvailableVoiceIdsBySource(ManagerTestCase):
    def test_groups_by_source(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        result = manager.get_available_voice_ids_by_source()
        self.assertIn("piper", result)
        self.assertIsInstance(result["piper"], list)
        self.assertEqual(result["piper"], sorted(result["piper"]))


class TestDownloadVoiceById(ManagerTestCase):
    def test_voice_found_in_registry(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        voice = MagicMock()
        manager.voices = {"registry/voice": voice}
        result = manager.download_voice_by_id("registry/voice")
        self.assertTrue(result)
        voice.download_model.assert_called_once()

    def test_voice_found_only_in_on_disk_index(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        manager.voices = {}
        with patch.object(TTSModelInfo, "download_model") as mock_download:
            result = manager.download_voice_by_id("piper/ar_JO-kareem-low")
        self.assertTrue(result)
        mock_download.assert_called_once()

    def test_voice_not_found_anywhere_returns_false(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        manager.voices = {}
        result = manager.download_voice_by_id("totally/unknown-voice-id-xyz")
        self.assertFalse(result)


class TestUnwritableCachePath(unittest.TestCase):
    def test_unwritable_cache_dir_raises_clear_error(self):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda: (os.chmod(tmpdir, 0o700), __import__("shutil").rmtree(tmpdir)))
        os.chmod(tmpdir, 0o500)
        bad_path = os.path.join(tmpdir, "sub", "cache.json")
        with self.assertRaises(PermissionError):
            TTSModelManager(cache_path=bad_path)


if __name__ == "__main__":
    unittest.main()
