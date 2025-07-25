import json
import os.path
import re
import string
import unicodedata
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union, Dict
from langcodes import closest_match
import numpy as np
import onnxruntime

from phoonnx.config import PhonemeType, VoiceConfig, SynthesisConfig
from phoonnx.graphemes import Graphemes, CotoviaGraphemes
from phoonnx.phoneme_ids import phonemes_to_ids, DEFAULT_PHONEME_ID_MAP
from phoonnx.phonemizers import ByT5Phonemizer, EspeakPhonemizer, RawPhonemizedChunks
from phoonnx.tashkeel import TashkeelDiacritizer

_PHONEME_BLOCK_PATTERN = re.compile(r"(\[\[.*?\]\])")

try:
    from ovos_utils.log import LOG
except ImportError:
    import logging
    LOG = logging.getLogger(__name__)
    LOG.setLevel("DEBUG")

@dataclass
class PhoneticSpellings:
    replacements: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_lang(lang: str, locale_path: str = f"{os.path.dirname(__file__)}/locale"):
        langs = os.listdir(locale_path)
        lang2, distance = closest_match(lang, langs)
        if distance <= 10:
            spellings_file = f"{locale_path}/{lang2}/phonetic_spellings.txt"
            return PhoneticSpellings.from_path(spellings_file)
        raise FileNotFoundError(f"Spellings file for '{lang}' not found")

    @staticmethod
    def from_path(spellings_file: str):
        replacements = {}
        with open(spellings_file) as f:
            lines = f.read().split("\n")
            for l in lines:
                word, spelling = l.split(":", 1)
                replacements[word.strip()] = spelling.strip()
        return PhoneticSpellings(replacements)

    def apply(self, text: str) -> str:
        for k, v in self.replacements.items():
            # Use regex to ensure word boundaries
            pattern = r'\b' + re.escape(k) + r'\b'
            # Replace using regex with case insensitivity
            text = re.sub(pattern, v, text, flags=re.IGNORECASE)
        return text

@dataclass
class AudioChunk:
    """Chunk of raw audio."""

    sample_rate: int
    """Rate of chunk samples in Hertz."""

    sample_width: int
    """Width of chunk samples in bytes."""

    sample_channels: int
    """Number of channels in chunk samples."""

    audio_float_array: np.ndarray
    """Audio data as float numpy array in [-1, 1]."""

    _audio_int16_array: Optional[np.ndarray] = None
    _audio_int16_bytes: Optional[bytes] = None
    _MAX_WAV_VALUE: float = 32767.0

    @property
    def audio_int16_array(self) -> np.ndarray:
        """
        Get audio as an int16 numpy array.

        :return: Audio data as int16 numpy array.
        """
        if self._audio_int16_array is None:
            self._audio_int16_array = np.clip(
                self.audio_float_array * self._MAX_WAV_VALUE, -self._MAX_WAV_VALUE, self._MAX_WAV_VALUE
            ).astype(np.int16)

        return self._audio_int16_array

    @property
    def audio_int16_bytes(self) -> bytes:
        """
        Get audio as 16-bit PCM bytes.

        :return: Audio data as signed 16-bit sample bytes.
        """
        return self.audio_int16_array.tobytes()


PhonemizedChunks = list[list[str]]


@dataclass
class TTSVoice:
    session: onnxruntime.InferenceSession

    config: VoiceConfig

    phonetic_spellings: Optional[PhoneticSpellings] = None

    espeak_phonemizer: Optional[EspeakPhonemizer] = None
    byt5_phonemizer: Optional[ByT5Phonemizer] = None

    cotovia_phonemizer: Optional[CotoviaGraphemes] = None
    graphemes_tokenizer: Optional[Graphemes] = None

    # For Arabic text only
    use_tashkeel: bool = True
    tashkeel_diacritizier: Optional[TashkeelDiacritizer] = None  # For Arabic text only
    taskeen_threshold: Optional[float] = 0.8

    def __post_init__(self):
        try:
            self.phonetic_spellings = PhoneticSpellings.from_lang(self.config.lang_code)
        except FileNotFoundError:
            pass

        if self.config.phoneme_type == PhonemeType.ESPEAK and self.espeak_phonemizer is None:
            self.espeak_phonemizer = EspeakPhonemizer()
        if self.config.phoneme_type == PhonemeType.BYT5 and self.byt5_phonemizer is None:
            self.byt5_phonemizer = ByT5Phonemizer()
        if self.config.phoneme_type == PhonemeType.COTOVIA and self.cotovia_phonemizer is None:
            # cotovia_bin_path can be passed via config if needed, for now use default search
            self.cotovia_phonemizer = CotoviaGraphemes()
        if self.config.phoneme_type in [PhonemeType.GRAPHEMES, PhonemeType.COTOVIA] and self.graphemes_tokenizer is None:
            # Use characters_config from VoiceConfig to initialize Graphemes
            characters = self.config.characters_config.get("characters")
            punctuations = self.config.characters_config.get("punctuations")
            pad = self.config.characters_config.get("pad")
            eos = self.config.characters_config.get("eos")
            bos = self.config.characters_config.get("bos")
            blank = self.config.characters_config.get("blank", "<BLNK>") # Default blank token

            self.graphemes_tokenizer = Graphemes(
                characters=characters,
                punctuations=punctuations,
                pad=pad,
                eos=eos,
                bos=bos,
                blank=blank
            )


        # compat with piper arabic models
        if self.config.lang_code.split("-")[0] == "ar" and self.use_tashkeel and self.tashkeel_diacritizier is None:
            self.tashkeel_diacritizier = TashkeelDiacritizer()

    @staticmethod
    def load(
            model_path: Union[str, Path],
            config_path: Optional[Union[str, Path]] = None,
            use_cuda: bool = False
    ) -> "TTSVoice":
        """
        Load an ONNX model and config.

        :param model_path: Path to ONNX voice model.
        :param config_path: Path to JSON voice config (defaults to model_path + ".json").
        :param use_cuda: True if CUDA (GPU) should be used instead of CPU.
        :return: Voice object.
        """
        if config_path is None:
            config_path = f"{model_path}.json"
            LOG.debug("Guessing voice config path: %s", config_path)

        with open(config_path, "r", encoding="utf-8") as config_file:
            config_dict = json.load(config_file)

        providers: list[Union[str, tuple[str, dict[str, Any]]]]
        if use_cuda:
            providers = [
                (
                    "CUDAExecutionProvider",
                    {"cudnn_conv_algo_search": "HEURISTIC"},
                )
            ]
            LOG.debug("Using CUDA")
        else:
            providers = ["CPUExecutionProvider"]

        return TTSVoice(
            config=VoiceConfig.from_dict(config_dict),
            session=onnxruntime.InferenceSession(
                str(model_path),
                sess_options=onnxruntime.SessionOptions(),
                providers=providers,
            )
        )

    def _process_phones(self, raw_phones: RawPhonemizedChunks) -> PhonemizedChunks:
        """Text to phonemes grouped by sentence."""

        all_phonemes: list[list[str]] = []
        sentence_phonemes: list[str] = []

        for phonemes_str, terminator_str, end_of_sentence in raw_phones:
            # Filter out (lang) switch (flags).
            # These surround words from languages other than the current voice.
            phonemes_str = re.sub(r"\([^)]+\)", "", phonemes_str)

            # Keep punctuation even though it's not technically a phoneme
            phonemes_str += terminator_str
            if terminator_str in (",", ":", ";"):
                # Not a sentence boundary
                phonemes_str += " "

            # Decompose phonemes into UTF-8 codepoints.
            # This separates accent characters into separate "phonemes".
            sentence_phonemes.extend(list(unicodedata.normalize("NFD", phonemes_str)))

            if end_of_sentence:
                all_phonemes.append(sentence_phonemes)
                sentence_phonemes = []

        if sentence_phonemes:
            all_phonemes.append(sentence_phonemes)

        return all_phonemes

    def phonemize(self, text: str) -> PhonemizedChunks:
        """
        Text to phonemes grouped by sentence.

        :param text: Text to phonemize.
        :return: List of phonemes for each sentence.
        """
        # This method's behavior depends heavily on the phoneme_type.
        # For GRAPHEMES, it will return a list of characters as "phonemes"
        # For others, it will return actual phonemes.

        phonemes: list[list[str]] = []

        if self.config.phoneme_type == PhonemeType.UNICODE:
            # Phonemes = codepoints
            return [list(unicodedata.normalize("NFD", text))]

        text_parts = _PHONEME_BLOCK_PATTERN.split(text)

        for i, text_part in enumerate(text_parts):
            if text_part.startswith("[["):
                # Phonemes
                if not phonemes:
                    # Start new sentence
                    phonemes.append([])

                if (i > 0) and (text_parts[i - 1].endswith(" ")):
                    phonemes[-1].append(" ")

                phonemes[-1].extend(list(text_part[2:-2].strip())) # Ensure characters are split

                if (i < (len(text_parts)) - 1) and (text_parts[i + 1].startswith(" ")):
                    phonemes[-1].append(" ")

                continue

            # Arabic diacritization
            if self.config.lang_code.split("-")[0] == "ar" and self.use_tashkeel:
                text_part = self.tashkeel_diacritizier(
                    text_part, taskeen_threshold=self.taskeen_threshold
                )

            # Phonemization
            if self.config.phoneme_type == PhonemeType.BYT5:
                raw_phonemes = self.byt5_phonemizer.phonemize(
                    text_part, self.config.lang_code
                )
                text_part_phonemes = self._process_phones(raw_phonemes)
            elif self.config.phoneme_type == PhonemeType.ESPEAK:
                raw_phonemes = self.espeak_phonemizer.phonemize(
                    text_part, self.config.lang_code
                )
                text_part_phonemes = self._process_phones(raw_phonemes)
            else:
                raise ValueError(f"Invalid phonemizer config: {self.config.phoneme_type}")

            phonemes.extend(text_part_phonemes)

        if phonemes and (not phonemes[-1]):
            # Remove empty phonemes
            phonemes.pop()

        return phonemes

    def phonemes_to_ids(self, phonemes: list[str]) -> list[int]:
        """
        Phonemes to ids.

        :param phonemes: List of phonemes (or characters for grapheme models).
        :return: List of phoneme ids.
        """
        # If the model is grapheme-based, the "phonemes" here are actually characters
        # and the IDs are already generated by Graphemes.text_to_ids.
        if self.config.phoneme_type in [PhonemeType.GRAPHEMES, PhonemeType.COTOVIA]:
            # This case should ideally not be reached if synthesize handles it correctly
            # by getting IDs directly from graphemes_tokenizer.
            # However, if it does, we need to map characters to IDs using the grapheme vocab.
            if self.graphemes_tokenizer is None:
                raise RuntimeError("Graphemes tokenizer not initialized for GRAPHEMES phoneme type.")
            return [self.graphemes_tokenizer.char_to_id(p) for p in phonemes]

        # For phoneme-based models, use the phoneme_id_map
        if self.config.phoneme_id_map is None:
            LOG.warning("phoneme_id_map is None. Using DEFAULT_PHONEME_ID_MAP.")
            return phonemes_to_ids(phonemes, DEFAULT_PHONEME_ID_MAP)

        return phonemes_to_ids(phonemes, self.config.phoneme_id_map)

    def synthesize(
            self,
            text: str,
            syn_config: Optional[SynthesisConfig] = None,
    ) -> Iterable[AudioChunk]:
        """
        Synthesize one audio chunk per sentence from from text.

        :param text: Text to synthesize.
        :param syn_config: Synthesis configuration.
        """
        if syn_config is None:
            syn_config = SynthesisConfig()

        LOG.debug("text=%s", text)

        # user defined word-level replacements to force correct pronounciation
        if self.phonetic_spellings and syn_config.enable_phonetic_spellings:
            text = self.phonetic_spellings.apply(text)

        # Determine phoneme_ids based on phoneme_type
        if self.config.phoneme_type in [PhonemeType.GRAPHEMES, PhonemeType.COTOVIA]:
            if self.graphemes_tokenizer is None:
                raise RuntimeError("Graphemes tokenizer not initialized for GRAPHEMES phoneme type.")

            if self.config.phoneme_type == PhonemeType.COTOVIA:
                # NOTE: cotovia phonemizer is used as a pre-processing
                # step for text since it does not output IPA
                text = self.cotovia_phonemizer.phonemize(text)

            # For grapheme models, we get IDs directly
            # Graphemes.text_to_ids already applies add_blank and use_eos_bos based on config
            all_phoneme_ids_for_synthesis = [
                self.graphemes_tokenizer.text_to_ids(
                    text,
                    add_blank=self.config.add_blank,
                    use_eos_bos=self.config.use_eos_bos
                )
            ]
        else:
            # For phoneme-based models (ESPEAK, BYT5), use the existing phonemize -> phonemes_to_ids flow
            sentence_phonemes = self.phonemize(text)
            LOG.debug("phonemes=%s", sentence_phonemes)
            all_phoneme_ids_for_synthesis = [
                self.phonemes_to_ids(phonemes) for phonemes in sentence_phonemes if phonemes
            ]

        for phoneme_ids in all_phoneme_ids_for_synthesis:
            if not phoneme_ids:
                continue

            audio = self.phoneme_ids_to_audio(phoneme_ids, syn_config)

            if syn_config.normalize_audio:
                max_val = np.max(np.abs(audio))
                if max_val < 1e-8:
                    # Prevent division by zero
                    audio = np.zeros_like(audio)
                else:
                    audio = audio / max_val

            if syn_config.volume != 1.0:
                audio = audio * syn_config.volume

            audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

            yield AudioChunk(
                sample_rate=self.config.sample_rate,
                sample_width=2,
                sample_channels=1,
                audio_float_array=audio,
            )

    def synthesize_wav(
            self,
            text: str,
            wav_file: wave.Wave_write,
            syn_config: Optional[SynthesisConfig] = None,
            set_wav_format: bool = True,
    ) -> None:
        """
        Synthesize and write WAV audio from text.

        :param text: Text to synthesize.
        :param wav_file: WAV file writer.
        :param syn_config: Synthesis configuration.
        :param set_wav_format: True if the WAV format should be set automatically.
        """

        # 16-bit samples for silence
        sentence_silence = 0.0  # Seconds of silence after each sentence
        silence_int16_bytes = bytes(
            int(self.config.sample_rate * sentence_silence * 2)
        )
        first_chunk = True
        for audio_chunk in self.synthesize(text, syn_config=syn_config):
            if first_chunk:
                if set_wav_format:
                    # Set audio format on first chunk
                    wav_file.setframerate(audio_chunk.sample_rate)
                    wav_file.setsampwidth(audio_chunk.sample_width)
                    wav_file.setnchannels(audio_chunk.sample_channels)

                first_chunk = False

            if not first_chunk:
                wav_file.writeframes(silence_int16_bytes)

            wav_file.writeframes(audio_chunk.audio_int16_bytes)

    def phoneme_ids_to_audio(
            self, phoneme_ids: list[int], syn_config: Optional[SynthesisConfig] = None
    ) -> np.ndarray:
        """
        Synthesize raw audio from phoneme ids.

        :param phoneme_ids: List of phoneme ids.
        :param syn_config: Synthesis configuration.
        :return: Audio float numpy array from voice model (unnormalized, in range [-1, 1]).
        """
        if syn_config is None:
            syn_config = SynthesisConfig()

        speaker_id = syn_config.speaker_id
        length_scale = syn_config.length_scale
        noise_scale = syn_config.noise_scale
        noise_w_scale = syn_config.noise_w_scale

        if length_scale is None:
            length_scale = self.config.length_scale

        if noise_scale is None:
            noise_scale = self.config.noise_scale

        if noise_w_scale is None:
            noise_w_scale = self.config.noise_w_scale

        phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)
        scales = np.array(
            [noise_scale, length_scale, noise_w_scale],
            dtype=np.float32,
        )

        args = {
            "input": phoneme_ids_array,
            "input_lengths": phoneme_ids_lengths,
            "scales": scales,
        }

        if self.config.num_speakers <= 1:
            speaker_id = None

        if (self.config.num_speakers > 1) and (speaker_id is None):
            # Default speaker
            speaker_id = 0

        if speaker_id is not None:
            sid = np.array([speaker_id], dtype=np.int64)
            args["sid"] = sid

        # Synthesize through onnx
        audio = self.session.run(
            None,
            args,
        )[0].squeeze()

        return audio



if __name__ == "__main__":

    syn_config = SynthesisConfig(enable_phonetic_spellings=True)

    model = "/home/miro/PycharmProjects/piper/src/miro_pt-PT.onnx"
    config = model + ".json"

    voice = TTSVoice.load(model_path=model, config_path=config, use_cuda=False)
    # manually load byt5 phonemizer (model requests espeak)
    voice.byt5_phonemizer = ByT5Phonemizer()

    sentence = "A rainbow is a meteorological phenomenon that is caused by reflection, refraction and dispersion of light in water droplets resulting in a spectrum of light appearing in the sky."
    sentence = "hey mycroft"
   # sentence = "OpenVoiceOS"

    for phonemizer_type in [PhonemeType.ESPEAK, PhonemeType.BYT5]:
        voice.config.phoneme_type = phonemizer_type

        slug = "".join([c for c in sentence if c not in string.punctuation]).replace(" ", "_")
        slug += f"_{phonemizer_type.value}_{voice.config.lang_code}"
        with wave.open(f"{slug}.wav", "wb") as wav_file:
            voice.synthesize_wav(sentence, wav_file, syn_config)



    model = "/home/miro/PycharmProjects/ovos-tts-plugin-nos/celtia/model.onnx"
    config = "/home/miro/PycharmProjects/ovos-tts-plugin-nos/celtia/config.json"

    voice = TTSVoice.load(model_path=model, config_path=config, use_cuda=False)

    sentence = "Este é un sistema de conversión de texto a voz en lingua galega baseado en redes neuronais artificiais. Ten en conta que as funcionalidades incluídas nesta páxina ofrécense unicamente con fins de demostración. Se tes algún comentario, suxestión ou detectas algún problema durante a demostración, ponte en contacto connosco."
    sentence = "hey mycroft"
    #sentence = "OpenVoiceOS"

    slug = "".join([c for c in sentence if c not in string.punctuation]).replace(" ", "_")
    slug += f"_{voice.config.phoneme_type.value}_{voice.config.lang_code}"
    with wave.open(f"{slug[-25:]}.wav", "wb") as wav_file:
        voice.synthesize_wav(sentence, wav_file, syn_config)


    model = "/home/miro/PycharmProjects/ovos-tts-plugin-nos/sabela/model.onnx"
    config = "/home/miro/PycharmProjects/ovos-tts-plugin-nos/sabela/config.json"

    voice = TTSVoice.load(model_path=model, config_path=config, use_cuda=False)

    slug = "".join([c for c in sentence if c not in string.punctuation]).replace(" ", "_")
    slug += f"_{voice.config.phoneme_type.value}_{voice.config.lang_code}"
    with wave.open(f"{slug[-25:]}.wav", "wb") as wav_file:
        voice.synthesize_wav(sentence, wav_file, syn_config)
