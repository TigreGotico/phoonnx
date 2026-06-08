"""
TTSVoice — architecture-agnostic TTS voice.

This module is the main user-facing synthesis interface.  It delegates
all ONNX-specific I/O to an engine adapter (``phoonnx.engines``),
which means the same TTSVoice class works for VITS or
any future architecture without code changes here.
"""
import json
import os.path
import re
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union, Dict

import numpy as np
import onnxruntime
from langcodes import closest_match

from phoonnx.alphabet_convert import convert
from phoonnx.config import PhonemeType, VoiceConfig, SynthesisConfig, get_phonemizer
from phoonnx.engines import detect_engine, get_adapter
from phoonnx.engines.base import (
    AdapterSynthesisRequest,
    AdapterSynthesisResult,
    BaseOnnxAdapter,
)
from phoonnx.phonemizers import Phonemizer
from phoonnx.phonemizers.base import PhonemizedChunks
from phoonnx.tokenizer import TTSTokenizer
from phoonnx.util import LOG


_PHONEME_BLOCK_PATTERN = re.compile(r"(\[\[.*?\]\])")


def _load_reference_audio(ref: Any) -> tuple:
    """Normalise a cloning reference to ``(mono float32 audio, sample_rate)``.

    Accepts an ``(audio, sample_rate)`` tuple or a path to an audio file.
    """
    if isinstance(ref, tuple) and len(ref) == 2:
        audio, sr = ref
        return np.asarray(audio, dtype=np.float32).reshape(-1), int(sr)
    try:
        import soundfile as sf
        audio, sr = sf.read(str(ref), dtype="float32")
    except ImportError:
        with wave.open(str(ref), "rb") as w:
            sr, ch = w.getframerate(), w.getnchannels()
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
            if ch > 1:
                audio = audio.reshape(-1, ch).mean(axis=1)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32).reshape(-1), int(sr)


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


@dataclass
class TTSVoice:
    session: onnxruntime.InferenceSession
    config: VoiceConfig
    phonetic_spellings: Optional[PhoneticSpellings] = None
    phonemizer: Optional[Phonemizer] = None
    adapter: Optional[BaseOnnxAdapter] = None

    def __post_init__(self):
        """
        Initialize optional phonetic resources after dataclass construction.
        
        Attempts to load phonetic spellings for the voice's language and, if a phonemizer was not provided, selects and assigns one based on the voice configuration.
        
        Notes:
        - If no phonetic spellings file is found for the configured language, the absence is ignored.
        """
        try:
            self.phonetic_spellings = PhoneticSpellings.from_lang(self.config.lang_code)
        except FileNotFoundError:
            pass

        # Phonemizer
        if self.phonemizer is None:
            self.phonemizer = get_phonemizer(
                self.config.phoneme_type,
                self.config.alphabet,
                self.config.phonemizer_model,
            )

        # Engine adapter — auto-detect if not explicitly provided
        if self.adapter is None:
            engine_name = self.config.engine.value if self.config.engine else None
            try:
                # Try by engine name first
                if engine_name and engine_name not in ("piper", "mimic3", "coqui"):
                    self.adapter = get_adapter(engine_name)
                else:
                    # For piper/mimic3/coqui all use the vits adapter
                    self.adapter = get_adapter("vits")
            except KeyError:
                # Fall back to auto-detection
                self.adapter = detect_engine(session=self.session)

        # Let the adapter set up engine-specific runtime state (e.g. Matcha
        # building its vocoder from config.engine_params).
        self.adapter.configure(self.config)

    @property
    def tokenizer(self) -> TTSTokenizer:
        """
        Return the tokenizer configured for this voice.
        
        Returns:
            TTSTokenizer: The TTSTokenizer instance from the voice configuration.
        """
        return self.config.tokenizer

    @staticmethod
    def load(
            model_path: Union[str, Path],
            config_path: Optional[Union[str, Path]] = None,
            vocab_path: Optional[Union[str, Path]] = None,
            tokenizer_config_path: Optional[Union[str, Path]] = None,
            phonemes_txt: Optional[str] = None,
            phoneme_map: Optional[str] = None,
            lang_code: Optional[str] = None,
            phoneme_type_str: Optional[str] = None,
            alphabet_str: Optional[str] = None,
            engine_params: Optional[Dict[str, Any]] = None,
            use_cuda: bool = False
    ) -> "TTSVoice":
        """
        Load a TTS voice ONNX model and its configuration into a TTSVoice instance.
        
        Parameters:
            model_path (str | Path): Path to the ONNX voice model file.
            config_path (str | Path, optional): Path to the JSON voice configuration. If omitted, defaults to model_path + ".json".
            phonemes_txt (str, optional): Optional phonemes definition or file content to override the config's phoneme list.
            phoneme_map (str, optional): Optional phoneme mapping specification or file path used to map phonemes (may be None).
            lang_code (str, optional): Language code to override or set in the loaded voice configuration.
            phoneme_type_str (str, optional): Phoneme type identifier to override the configuration (for example, "arpabet" or "ipa").
            alphabet_str (str, optional): Alphabet override to pass into the VoiceConfig during load.
            use_cuda (bool): If true, prefer CUDA execution provider for ONNX Runtime; otherwise use the CPU provider.
        
        Returns:
            TTSVoice: A TTSVoice instance prepared with the loaded ONNX session and merged configuration.
        """
        if config_path is None:
            config_path = f"{model_path}.json"

        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as config_file:
                config_dict = json.load(config_file)
        else:
            config_dict = {"phoneme_type": "unicode", "alphabet": "unicode"}

        vocab_dict = {}
        tokenizer_dict = {}
        if vocab_path and os.path.isfile(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as vocab_file:
                vocab_dict = json.load(vocab_file)
            if tokenizer_config_path and os.path.isfile(tokenizer_config_path):
                with open(tokenizer_config_path, "r", encoding="utf-8") as tokenizer_file:
                    tokenizer_dict = json.load(tokenizer_file)
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

        session = onnxruntime.InferenceSession(
            str(model_path),
            sess_options=onnxruntime.SessionOptions(),
            providers=providers,
        )

        # Auto-detect engine adapter from config + session
        adapter = detect_engine(config=config_dict, session=session)

        # Engine-specific params (e.g. Matcha's vocoder_path) come either from
        # the caller (the voice manager injects locally-downloaded paths) or
        # from the model's own config JSON.
        engine_params = engine_params or config_dict.get("engine_params") or {}

        voice_config = VoiceConfig.from_dict(
            config_dict,
            vocab=vocab_dict,
            tokenizer_config=tokenizer_dict,
            alphabet=alphabet_str,
            tokens_txt=phonemes_txt,
            lang_code=lang_code,
            phoneme_type=phoneme_type_str,
            engine_params=engine_params,
        )

        return TTSVoice(
            config=voice_config,
            session=session,
            adapter=adapter,
        )

    def phonemize(self, text: str, lang: Optional[str] = None) -> PhonemizedChunks:
        """
        Text to phonemes grouped by sentence.

        :param text: Text to phonemize.
        :param lang: Language code to phonemize in; defaults to the voice's own
            ``lang_code``. Used to phonemize a cloning reference transcription in
            *its* language (which may differ from the target's).
        :return: List of phonemes for each sentence.
        """
        lang = lang or self.config.lang_code
        phonemes: list[list[str]] = []

        text_parts = _PHONEME_BLOCK_PATTERN.split(text)

        for i, text_part in enumerate(text_parts):
            if text_part.startswith("[["):
                # Phonemes
                if not phonemes:
                    # Start new sentence
                    phonemes.append([])

                if (i > 0) and (text_parts[i - 1].endswith(" ")):
                    phonemes[-1].append(" ")

                phonemes[-1].extend(list(text_part[2:-2].strip()))  # Ensure characters are split

                if (i < (len(text_parts)) - 1) and (text_parts[i + 1].startswith(" ")):
                    phonemes[-1].append(" ")

                continue

            phonemes.extend(self.phonemizer.phonemize(text_part, lang))

        if phonemes and (not phonemes[-1]):
            # Remove empty phonemes
            phonemes.pop()

        return phonemes

    def phonemes_to_ids(self, phonemes: list[str]) -> list[int]:
        """
        Convert a sequence of phoneme tokens (or characters for grapheme models) into token IDs.
        
        Parameters:
            phonemes (list[str]): Sequence of phoneme strings or individual characters to be tokenized.
        
        Returns:
            list[int]: Token IDs corresponding to each input phoneme or character.
        """
        token_ids =  self.tokenizer.tokenize(phonemes)
        return token_ids

    def synthesize(
            self,
            text: str,
            syn_config: Optional[SynthesisConfig] = None,
    ) -> Iterable[AudioChunk]:
        """
        Synthesize speech from input text, yielding one AudioChunk per sentence.
        
        Generates sentence-level audio by phonemizing the input text and synthesizing each sentence into a float32 PCM audio array in the range [-1.0, 1.0]. If enabled in the synthesis configuration, user-provided phonetic spellings and diacritic augmentation are applied before phonemization. Output audio may be normalized and volume-scaled according to the configuration; samples are clipped to [-1.0, 1.0].
        
        Parameters:
            text (str): The input text to synthesize.
            syn_config (Optional[SynthesisConfig]): Optional synthesis options (e.g., enable_phonetic_spellings, add_diacritics, normalize_audio, volume). If omitted, a default SynthesisConfig is used.
        
        Returns:
            Iterable[AudioChunk]: An iterable that yields one AudioChunk per synthesized sentence. Each AudioChunk contains a float32 audio array, sample rate taken from the voice config, 2-byte sample width, and 1 channel.
        """
        if syn_config is None:
            syn_config = SynthesisConfig()

        LOG.debug("text=%s", text)

        # user defined word-level replacements to force correct pronunciation
        if self.phonetic_spellings and syn_config.enable_phonetic_spellings:
            text = self.phonetic_spellings.apply(text)

        if syn_config.add_diacritics:
            text = self.phonemizer.add_diacritics(text, self.config.lang_code)
            LOG.debug("text+diacritics=%s", text)

        # Script-conversion: bridge the caller's alphabet to the model's alphabet.
        # syn_config.alphabet is the user-input alphabet; None means "assume model's".
        # self.config.alphabet is the model-expected alphabet.
        # No-op when src == dst or when the pair has no registered converter.
        if self.config.alphabet is not None:
            src_alphabet = syn_config.alphabet or self.config.alphabet
            text = convert(text, src=src_alphabet, dst=self.config.alphabet)
            LOG.debug("text+alphabet_convert=%s", text)

        # All phonemization goes through the unified self.phonemize method
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


    def _prompt_token_ids(self, text: str, lang: Optional[str] = None) -> list[int]:
        """Tokenize a cloning reference transcription into prompt token ids, reusing
        the voice's own phonemizer + tokenizer (for in-context engines like ZipVoice).

        ``lang`` phonemizes the transcription in the reference's own language (which
        may differ from the target's); defaults to the voice's ``lang_code``.
        """
        ids: list[int] = []
        for phonemes in self.phonemize(text, lang):
            if phonemes:
                ids.extend(self.phonemes_to_ids(phonemes))
        return ids

    def phoneme_ids_to_audio(
            self, phoneme_ids: list[int], syn_config: Optional[SynthesisConfig] = None
    ) -> np.ndarray:
        """
        Synthesize raw audio from phoneme ids.

        Delegates all ONNX I/O to ``self.adapter`` so the same code
        path works for VITS or any future engine.

        :param phoneme_ids: List of phoneme ids.
        :param syn_config: Synthesis configuration.
        :return: Audio float numpy array (unnormalized, in range [-1, 1]).
        """
        syn_config = syn_config or SynthesisConfig()

        # Build the architecture-agnostic request
        phoneme_ids_array = np.expand_dims(
            np.array(phoneme_ids, dtype=np.int64), 0
        )
        phoneme_ids_lengths = np.array(
            [phoneme_ids_array.shape[1]], dtype=np.int64
        )

        # Merge defaults from adapter → VoiceConfig → SynthesisConfig
        params = dict(self.adapter.default_params())
        # Override with voice-config level defaults
        params.update({
            k: v for k, v in {
                "noise_scale": self.config.noise_scale,
                "length_scale": self.config.length_scale,
                "noise_w_scale": self.config.noise_w_scale,
            }.items() if v is not None
        })
        # Override with any engine-specific extras from voice config
        if hasattr(self.config, 'engine_params'):
            params.update(self.config.engine_params)
        # Override with per-call SynthesisConfig
        if syn_config.noise_scale is not None:
            params["noise_scale"] = syn_config.noise_scale
        if syn_config.length_scale is not None:
            params["length_scale"] = syn_config.length_scale
        if syn_config.noise_w_scale is not None:
            params["noise_w_scale"] = syn_config.noise_w_scale
        # Engine-specific extras from SynthesisConfig
        params.update(syn_config.extra_params)
        # Voice cloning: hand the cloning adapter the reference clip and — for
        # in-context engines (ZipVoice) — the prompt tokens of its transcription.
        if syn_config.speaker_reference is not None:
            params["reference_audio"] = _load_reference_audio(syn_config.speaker_reference)
        if syn_config.speaker_reference_text:
            params["prompt_tokens"] = self._prompt_token_ids(
                syn_config.speaker_reference_text, syn_config.speaker_reference_lang)

        request = AdapterSynthesisRequest(
            phoneme_ids=phoneme_ids_array,
            phoneme_lengths=phoneme_ids_lengths,
            speaker_id=syn_config.speaker_id or 0,
            language_id=syn_config.lang_id or 0,
            params=params,
        )

        # Delegate to the adapter (single-graph engines use the default
        # build_feed_dict → run → parse_outputs; iterative engines override).
        result = self.adapter.synthesize(request, self.session)

        return result.audio

if __name__ == "__main__":
    from phoonnx.phonemizers.gl import CotoviaPhonemizer
    from phoonnx.phonemizers.he import PhonikudPhonemizer
    from phoonnx.phonemizers.mul import (EspeakPhonemizer, EpitranPhonemizer, GruutPhonemizer, ByT5Phonemizer)

    syn_config = SynthesisConfig(enable_phonetic_spellings=True)

    # test hebrew piper
    model = "/home/miro/PycharmProjects/phoonnx_tts/phonikud/model.onnx"
    config = "/home/miro/PycharmProjects/phoonnx_tts/phonikud/model.config.json"

    voice = TTSVoice.load(model_path=model, config_path=config, use_cuda=False)

    print("\n################")
    # hebrew phonemes (raw input model)
    pho = PhonikudPhonemizer(diacritics=True)
    sentence = "הכוח לשנות מתחיל ברגע שבו אתה מאמין שזה אפשרי!"
    sentence = pho.phonemize_string(sentence, "he")

    print("## piper hebrew (raw)")
    print("-", voice.config.phoneme_type)
    slug = f"piper_{voice.config.phoneme_type.value}_{voice.config.lang_code}"
    with wave.open(f"{slug}.wav", "wb") as wav_file:
        voice.synthesize_wav(sentence, wav_file, syn_config)

    print("\n################")
    sentence = "הכוח לשנות מתחיל ברגע שבו אתה מאמין שזה אפשרי!"
    voice.config.phoneme_type = PhonemeType.PHONIKUD
    voice.phonemizer = pho

    print("## piper hebrew (phonikud)")
    print("-", voice.config.phoneme_type)
    slug = f"piper_{voice.config.phoneme_type.value}_{voice.config.lang_code}"
    with wave.open(f"{slug}.wav", "wb") as wav_file:
        voice.synthesize_wav(sentence, wav_file, syn_config)

    exit()
    # test piper
    model = "/home/miro/PycharmProjects/phoonnx_tts/miro_en-GB.onnx"
    config = "/home/miro/PycharmProjects/phoonnx_tts/piper_espeak.json"

    voice = TTSVoice.load(model_path=model, config_path=config, use_cuda=False)
    byt5_phonemizer = ByT5Phonemizer()
    gruut_phonemizer = GruutPhonemizer()
    espeak_phonemizer = EspeakPhonemizer()
    epitran_phonemizer = EpitranPhonemizer()
    cotovia_phonemizer = CotoviaPhonemizer()

    sentence = "A rainbow is a meteorological phenomenon that is caused by reflection, refraction and dispersion of light in water droplets resulting in a spectrum of light appearing in the sky. It takes the form of a multi-colored circular arc. Rainbows caused by sunlight always appear in the section of sky directly opposite the Sun."

    print("\n################")
    print("## piper")
    for phonemizer_type, phonemizer in [
        (PhonemeType.ESPEAK, espeak_phonemizer),
        (PhonemeType.BYT5, byt5_phonemizer),
        (PhonemeType.GRUUT, gruut_phonemizer),
        (PhonemeType.EPITRAN, epitran_phonemizer)
    ]:
        voice.config.phoneme_type = phonemizer_type
        voice.phonemizer = phonemizer
        print("-", phonemizer_type)

        slug = f"piper_{phonemizer_type.value}_{voice.config.lang_code}"
        with wave.open(f"{slug}.wav", "wb") as wav_file:
            voice.synthesize_wav(sentence, wav_file, syn_config)

    print("\n################")
    print("## mimic3")
    model = "/home/miro/PycharmProjects/phoonnx_tts/mimic3_ap/generator.onnx"
    config = "/home/miro/PycharmProjects/phoonnx_tts/mimic3_ap/config.json"
    phonemes_txt = "/home/miro/PycharmProjects/phoonnx_tts/mimic3_ap/phonemes.txt"
    phoneme_map = "/home/miro/PycharmProjects/phoonnx_tts/mimic3_ap/phoneme_map.txt"

    voice = TTSVoice.load(model_path=model, config_path=config,
                          phonemes_txt=phonemes_txt, phoneme_map=phoneme_map,
                          use_cuda=False)
    for phonemizer_type, phonemizer in [
        (PhonemeType.ESPEAK, espeak_phonemizer),
        (PhonemeType.BYT5, byt5_phonemizer),
        (PhonemeType.GRUUT, gruut_phonemizer),
        (PhonemeType.EPITRAN, epitran_phonemizer)
    ]:
        voice.config.phoneme_type = phonemizer_type
        voice.phonemizer = phonemizer
        print("-", phonemizer_type)
        slug = f"mimic3_{voice.config.phoneme_type.value}_{voice.config.lang_code}"
        with wave.open(f"{slug}.wav", "wb") as wav_file:
            voice.synthesize_wav(sentence, wav_file, syn_config)

    # Test grapheme model directly
    print("\n################")
    print("## coqui vits")
    model = "/home/miro/PycharmProjects/phoonnx_tts/celtia_vits/model.onnx"
    config = "/home/miro/PycharmProjects/phoonnx_tts/celtia_vits/config.json"

    sentence = "Este é un sistema de conversión de texto a voz en lingua galega baseado en redes neuronais artificiais. Ten en conta que as funcionalidades incluídas nesta páxina ofrécense unicamente con fins de demostración. Se tes algún comentario, suxestión ou detectas algún problema durante a demostración, ponte en contacto connosco."

    voice = TTSVoice.load(model_path=model, config_path=config,
                          use_cuda=False, lang_code="gl-ES")
    print("-", voice.config.phoneme_type)
    print(voice.config)
    phones = voice.phonemize(sentence)
    print(phones)
    print(voice.phonemes_to_ids(phones[0]))

    slug = f"vits_{voice.config.phoneme_type.value}_{voice.config.lang_code}"
    with wave.open(f"{slug}.wav", "wb") as wav_file:
        voice.synthesize_wav(sentence, wav_file, syn_config)

    # Test cotovia phonemizer
    print("\n################")
    print("## cotovia coqui vits")
    model = "/home/miro/PycharmProjects/phoonnx_tts/sabela_cotovia/model.onnx"
    config = "/home/miro/PycharmProjects/phoonnx_tts/sabela_cotovia/config.json"

    sentence = "Este é un sistema de conversión de texto a voz en lingua galega baseado en redes neuronais artificiais. Ten en conta que as funcionalidades incluídas nesta páxina ofrécense unicamente con fins de demostración. Se tes algún comentario, suxestión ou detectas algún problema durante a demostración, ponte en contacto connosco."

    voice = TTSVoice.load(model_path=model, config_path=config,
                          use_cuda=False, lang_code="gl-ES")
    print("-", voice.config.phoneme_type)
    print(voice.config)
    phones = voice.phonemize(sentence)
    print(phones)
    print(voice.phonemes_to_ids(phones[0]))

    slug = f"vits_{voice.config.phoneme_type.value}_{voice.config.lang_code}"
    with wave.open(f"{slug}.wav", "wb") as wav_file:
        voice.synthesize_wav(sentence, wav_file, syn_config)
