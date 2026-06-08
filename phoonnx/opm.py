# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import wave
from typing import Dict
from ovos_utils.log import LOG
from ovos_plugin_manager.templates.tts import TTS

from phoonnx.model_manager import TTSModelManager, TTSModelInfo
from phoonnx.voice import TTSVoice, SynthesisConfig


class PhoonnxTTSPlugin(TTS):
    """Interface to Phoonnx TTS."""
    engines = {}

    def __init__(self, config=None):
        """
        Initialize the PhoonnxTTSPlugin and prepare its model manager and initial voice cache.
        
        Creates a TTSModelManager, loads models, merges default voices into the manager, and caches either the configured non-default voice or the default voice for the plugin language.
        
        Parameters:
        	config (dict | None): Optional configuration passed to the base TTS initializer.
        """
        super().__init__(config=config)
        self.model_manager = TTSModelManager()
        self.model_manager.load()
        self.model_manager.merge_default_voices()

        self.voices: Dict[str, TTSVoice] = {}
        if self.voice and self.voice != "default":
            self.voices[self.voice] = self.get_model(self.voice)
        else:
            default = self.get_default_voice(self.lang)
            self.voices[default.voice_id] = self.get_model(default.voice_id)

    def _cfg_opt(self, default, *keys):
        """
        Read a synthesis option from the plugin config, accepting any of the
        given key aliases.

        Historically some option keys drifted from what the docs advertise
        (e.g. ``noise-scale`` vs ``noise_scale``). The documented underscore
        names are listed first; legacy hyphenated names are kept as fallbacks
        so existing configs keep working.

        Parameters:
            default: Value returned when none of the keys are present.
            *keys (str): Config keys to try, in priority order.

        Returns:
            The first present config value, otherwise ``default``.
        """
        for key in keys:
            if key in self.config:
                return self.config[key]
        return default

    def refresh_voices(self, force=False):
        """
        Refresh available voices from the model manager when none are loaded or when forcing an update.
        
        Parameters:
        	force (bool): If True, force a refresh even if voices are already present.
        """
        if not self.model_manager.voices or force:
            try:
                self.model_manager.merge_default_voices()
            except Exception as exc:
                LOG.warning(f"Voice refresh failed: {exc}")

    def get_default_voice(self, lang: str) -> TTSModelInfo:
        """
        Selects the default TTS model for the given language.
        
        Parameters:
        	lang (str): Language tag used to look up available voices (e.g., "en-US", "pt-PT").
        
        Returns:
        	TTSModelInfo: The first/default voice model info for the specified language.
        
        Raises:
        	ValueError: If no voices are available for the given language.
        """
        voices = self.model_manager.get_lang_voices(lang)
        if not voices:
            LOG.info(f"{lang} voices not found - refreshing voice list")
            self.refresh_voices(force=True)
            voices = self.model_manager.get_lang_voices(lang)
            if not voices:
                raise ValueError(f"No voices available for language: {lang}")
        return voices[0]

    def get_model(self, voice_id: str) -> TTSVoice:
        """
        Retrieve and cache the TTSVoice instance for a given voice identifier.
        
        Parameters:
            voice_id (str): Identifier of the voice to load.
        
        Returns:
            TTSVoice: The loaded voice model corresponding to `voice_id`.
        
        Raises:
            Exception: If `voice_id` is not found after refreshing available voices.
        """
        if voice_id in self.voices:
            return self.voices[voice_id]
        if voice_id not in self.model_manager.voices:
            LOG.info(f"{voice_id} not found - refreshing voice list")
            self.refresh_voices(force=True)
            if voice_id not in self.model_manager.voices:
                raise Exception(f"Unknown voice: {voice_id}")
        LOG.debug(f"Using voice: {voice_id}")
        self.voices[voice_id] = self.model_manager.voices[voice_id].load()
        return self.voices[voice_id]

    def get_tts(self, sentence, wav_file, lang=None, voice=None):
        """
        Synthesize the given text into speech and write the result to the specified WAV file.
        
        Parameters:
            sentence (str): Text to synthesize.
            wav_file (str): Path where the WAV audio will be written.
            lang (str, optional): Language hint used to select a default voice when `voice` is not provided.
            voice (str, optional): Specific voice identifier to use; treat `None` or `"default"` as no explicit selection.
        
        Returns:
            tuple: (`wav_file`, `phonemes`) where `wav_file` is the path to the written WAV file and `phonemes` is `None` when no phoneme output is produced.
        """
        if voice and voice != "default":
            # load first so the model manager is refreshed if the voice
            # isn't cached yet, then read its info (avoids a KeyError when a
            # configured voice hasn't been fetched before)
            model = self.get_model(voice)
            voice_info = self.model_manager.voices[voice]
        else:
            voice_info = self.get_default_voice(lang or self.lang)
            model = self.get_model(voice_info.voice_id)

        synth_params = SynthesisConfig(
            enable_phonetic_spellings=self._cfg_opt(
                voice_info.config.enable_phonetic_spellings
                if hasattr(voice_info.config, "enable_phonetic_spellings") else True,
                "enable_phonetic_spellings", "enable_phonetic_spelling"),
            add_diacritics=self._cfg_opt(
                voice_info.config.add_diacritics,  # arabic and hebrew only
                "add_diacritics"),
            noise_scale=self._cfg_opt(
                voice_info.config.noise_scale,  # generator noise
                "noise_scale", "noise-scale"),
            length_scale=self._cfg_opt(
                voice_info.config.length_scale,  # phoneme length
                "length_scale", "length-scale"),
            noise_w_scale=self._cfg_opt(
                voice_info.config.noise_w_scale,  # phoneme width noise
                "noise_w_scale", "noise_w", "noise-w"),
            # zero-shot voice cloning (cloning engines: YourTTS, StyleTTS2) — a path
            # to a reference wav; the voice's speaker encoder turns it into the
            # conditioning d-vector/style. e.g. {"ref_wav": "/home/user/me.wav"}
            speaker_reference=self._cfg_opt(
                None, "speaker_reference", "ref_wav", "clone_voice"),
        )
        with wave.open(wav_file, "wb") as wav_out:
            model.synthesize_wav(sentence, wav_out, synth_params)

        return wav_file, None


if __name__ == "__main__":
    utterance = "Guimarães é uma das mais importantes cidades históricas do país, estando o seu centro histórico inscrito na lista de Património Mundial da UNESCO desde 2001, o que a torna definitivamente num dos maiores centros turísticos da região. As suas ruas e monumentos respiram história e encantam quem a visita."
    #utterance = "Um arco-íris, também popularmente denominado arco-da-velha, é um fenômeno óptico e meteorológico que separa a luz do sol em seu espectro contínuo quando o sol brilha sobre gotículas de água suspensas no ar."

    tts = PhoonnxTTSPlugin()
    tts.get_tts(utterance, "tmiro-pt-PT.wav",
                voice="OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone")
    tts.get_tts(utterance, "tdii-pt-PT.wav",
                voice="OpenVoiceOS/phoonnx_pt-PT_dii_tugaphone")
    tts.get_tts(utterance, "miro-pt-PT.wav",
                voice="OpenVoiceOS/pipertts_pt-PT_miro")
    tts.get_tts(utterance, "dii-pt-PT.wav",
                voice="OpenVoiceOS/pipertts_pt-PT_dii")