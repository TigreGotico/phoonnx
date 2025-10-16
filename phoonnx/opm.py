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
        Initialize the PhoonnxTTSPlugin, prepare the model manager, and load an initial voice.
        
        Creates a TTSModelManager and attempts to refresh available voices; if refresh fails the manager falls back to loading voices. If a non-default voice is configured on the instance, that voice is loaded and cached in self.voices. Otherwise the language-specific default voice is determined, loaded, and stored in self.voices.
        
        Parameters:
            config (dict|None): Configuration passed to the base TTS initializer; may influence plugin behavior (e.g., selected voice or language).
        """
        super().__init__(config=config)
        self.model_manager = TTSModelManager()
        try:
            self.model_manager.refresh_voices()
        except Exception as exc:
            LOG.warning(f"Voice refresh failed; retrying with cached voices: {exc}")
            self.model_manager.load()

        self.voices: Dict[str, TTSVoice] = {}
        if self.voice and self.voice != "default":
            self.voices[self.voice] = self.get_model(self.voice)
        else:
            default = self.get_default_voice(self.lang)
            self.voices[default.voice_id] = self.get_model(default.voice_id)

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
            raise ValueError(f"No voices available for language: {lang}")
        return voices[0]

    def get_model(self, voice_id: str) -> TTSVoice:
        if voice_id in self.voices:
            return self.voices[voice_id]
        if voice_id not in self.model_manager.voices:
            raise Exception(f"Unknown voice: {voice_id}")
        LOG.debug(f"Using voice: {voice_id}")
        self.voices[voice_id] = self.model_manager.voices[voice_id].load()
        return self.voices[voice_id]

    def get_tts(self, sentence, wav_file, lang=None, voice=None):
        """
        Synthesize speech for a sentence and write the audio to the specified WAV file.
        
        Parameters:
            sentence (str): Text to synthesize.
            wav_file (str): Path to the output WAV file that will be written.
            lang (str, optional): Language override used to select the default voice when no `voice` is provided.
            voice (str, optional): Voice identifier override to select a specific model.
        
        Returns:
            tuple: (wav_file, phonemes) where `wav_file` is the path to the written WAV file and `phonemes` is `None` when phoneme output is not produced.
        """
        if voice:
            voice_info = self.model_manager.voices[voice]
            model = self.get_model(voice)
        else:
            voice_info = self.get_default_voice(lang or self.lang)
            model = self.get_model(voice_info.voice_id)

        synth_params = SynthesisConfig(
            enable_phonetic_spellings=self.config.get("enable_phonetic_spelling", True),
            add_diacritics=self.config.get("add_diacritics", voice_info.config.add_diacritics), # arabic and hebrew only
            noise_scale=self.config.get("noise-scale", voice_info.config.noise_scale),  # generator noise
            length_scale=self.config.get("length-scale", voice_info.config.length_scale),  # Phoneme length
            noise_w_scale=self.config.get("noise-w", voice_info.config.noise_w_scale)  # Phoneme width noise
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