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
import hashlib
import os
import tempfile
import wave
from contextlib import suppress
from collections import OrderedDict
from threading import RLock
from typing import List, Optional
from ovos_utils.log import LOG
from ovos_plugin_manager.templates.tts import TTS, TTSContext

from phoonnx.model_manager import TTSModelManager, TTSModelInfo
from phoonnx.voice import TTSVoice, SynthesisConfig
from phoonnx.voice_cache import VoiceCache, VoiceExceedsMemoryBudget

__all__ = ["PhoonnxTTSPlugin", "VoiceExceedsMemoryBudget"]


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

        self.voice_cache = VoiceCache(
            resolve=lambda voice_id: self.get_voice_info(voice_id),
            load=lambda info: info.load(providers=self._providers()),
            max_loaded_voices=self.config.get("max_loaded_voices"),
            max_loaded_bytes=self.config.get("max_loaded_bytes"),
            pinned_voices=self.config.get("pinned_voices"),
            load_wait_timeout=self.config.get("load_wait_timeout"))
        # Cloned cache identities, most recent last. Bounded because the
        # reference that creates one comes from the request.
        self._cloned_caches: "OrderedDict[str, bool]" = OrderedDict()
        self._cloned_lock = RLock()
        # Resolve the configured voice now, but do not load it. A voice that was
        # named explicitly and does not exist is a configuration error and still
        # raises here; an unset voice (or "default") resolves to the language's
        # default. Loading is deferred to the first synthesis so that fetching a
        # model is never what decides whether the TTS service can start.
        if self.voice and self.voice != "default":
            self.voice_info = self.get_voice_info(self.voice)
        else:
            self.voice_info = self.get_default_voice(self.lang)

        self.voice_cache.preload_pinned()

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

    def _providers(self) -> Optional[List[str]]:
        """
        Read the ONNX Runtime execution providers from the plugin config.

        ``onnx_providers`` (or ``providers``) takes an ordered list, e.g.
        ``["ROCMExecutionProvider", "CPUExecutionProvider"]``; a bare string is
        accepted for a single provider. Unset, providers come from
        ``PHOONNX_ONNX_PROVIDERS`` or auto-detection.
        """
        providers = self._cfg_opt(None, "onnx_providers", "providers")
        if isinstance(providers, str):
            return [providers]
        return list(providers) if providers else None

    def _resolve_speaker(self, voice_info) -> Optional[int]:
        """
        Resolve the speaker index for multi-speaker voices from the plugin config.

        Accepts either ``speaker_id`` (an integer index, or its digit string) or
        ``speaker`` (a name resolved against the voice's ``speaker_id_map``). The
        name may be given bare (``"elia"``) or accent-qualified
        (``"central/elia"``); the trailing segment is tried when the full string
        is not a map key.

        Returns:
            int | None: The speaker index, or None for single-speaker voices /
            when nothing is configured (the engine then defaults to speaker 0).
        """
        spk = self._cfg_opt(None, "speaker_id", "speaker")
        if spk is None or isinstance(spk, bool):  # bools are ints in python
            return None
        if isinstance(spk, int):
            return spk
        if isinstance(spk, str):
            if spk.lstrip("-").isdigit():
                return int(spk)
            smap = getattr(voice_info.config, "speaker_id_map", None) or {}
            if spk in smap:
                return smap[spk]
            bare = spk.split("/")[-1]
            if bare in smap:
                return smap[bare]
            LOG.warning(f"Unknown speaker '{spk}' for voice "
                        f"{voice_info.voice_id}; known speakers: "
                        f"{sorted(smap.keys())}")
        return None

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
            VoiceExceedsMemoryBudget: The voice cannot fit `max_loaded_bytes`
                even on its own.
            Exception: If `voice_id` is not found after refreshing available
                voices.
        """
        return self.voice_cache.get(voice_id)

    def get_voice_info(self, voice_id: str) -> TTSModelInfo:
        """
        Look up a voice in the catalog without loading it, refreshing the
        catalog once if the id is not already known.

        Parameters:
            voice_id (str): Identifier of the voice to look up.

        Returns:
            TTSModelInfo: Catalog entry for the voice.

        Raises:
            Exception: If `voice_id` is still unknown after a refresh.
        """
        info = self.model_manager.get_voice(voice_id)
        if info is None:
            LOG.info(f"{voice_id} not found - refreshing voice list")
            self.refresh_voices(force=True)
            info = self.model_manager.get_voice(voice_id)
            if info is None:
                raise Exception(f"Unknown voice: {voice_id}")
        return info

    def _get_ctxt(self, kwargs=None):
        """Keep cloned audio out of the shared voice's cache entry.

        The cache identifies a voice by ``plugin_id/voice/lang`` and then keys
        audio by the sentence alone, so two requests that clone different
        people saying the same sentence with the same base voice collide: the
        second caller is served the first one's cloned audio. On a shared
        server that is both the wrong voice and a leak of someone else's
        cloned speech, and it is silent — the audio plays perfectly.

        The reference is therefore folded into the cache identity. Only the
        identity changes; ``synth_kwargs`` still carries the real voice id, so
        the model that gets loaded is unaffected.
        """
        ctxt = super()._get_ctxt(kwargs)
        kwargs = kwargs or {}
        reference = tuple(
            kwargs.get(key) for key in
            ("speaker_reference", "ref_wav",
             "speaker_reference_text", "ref_text",
             "speaker_reference_lang", "ref_lang"))
        if any(reference):
            digest = hashlib.sha1(
                repr(reference).encode("utf-8")).hexdigest()[:12]
            ctxt.voice = f"{ctxt.voice}#{digest}"
            self._remember_cloned_cache(ctxt.tts_id)
        return ctxt

    #: How many cloned identities keep a cache object in memory. Each distinct
    #: reference makes one, and the reference comes from the request, so the
    #: count is whatever a caller decides it is; the files on disk are already
    #: bounded by the cache's own free-space curation, but this dictionary is
    #: not bounded by anything.
    MAX_CLONED_CACHES = 32

    def _remember_cloned_cache(self, tts_id: str) -> None:
        """Keep only the most recent cloned caches in memory."""
        with self._cloned_lock:
            self._cloned_caches.pop(tts_id, None)
            self._cloned_caches[tts_id] = True
            while len(self._cloned_caches) > self.MAX_CLONED_CACHES:
                oldest, _ = self._cloned_caches.popitem(last=False)
                # Only the in-memory handle is dropped. The audio stays on
                # disk for the cache's own curation to reclaim; deleting a
                # directory another request may be reading from would trade a
                # bounded dictionary for a much worse bug.
                TTSContext._caches.pop(oldest, None)

    def get_tts(self, sentence, wav_file, lang=None, voice=None,
                speaker_reference=None, speaker_reference_text=None,
                speaker_reference_lang=None,
                ref_wav=None, ref_text=None, ref_lang=None):
        """
        Synthesize the given text into speech and write the result to the specified WAV file.

        Parameters:
            sentence (str): Text to synthesize.
            wav_file (str): Path where the WAV audio will be written.
            lang (str, optional): Language hint used to select a default voice when `voice` is not provided.
            voice (str, optional): Specific voice identifier to use; treat `None` or `"default"` as no explicit selection.
            speaker_reference (str, optional): Reference clip to clone, as a URL
                or a path readable by the server. Cloning engines (chatterbox,
                f5tts, zipvoice, styletts2, ...) cannot synthesize without one.
            speaker_reference_text (str, optional): What the reference clip says.
                In-context engines such as ZipVoice need it.
            speaker_reference_lang (str, optional): Language of that transcription.
            ref_wav, ref_text, ref_lang: short aliases for the three above,
                matching the config key aliases.

        Every caller-supplied value overrides the configured default, so one
        server can clone a different voice per request. These are named
        explicitly rather than collected with ``**kwargs`` because the plugin
        manager forwards only parameters it can see in this signature.

        Returns:
            tuple: (`wav_file`, `phonemes`) where `wav_file` is the path to the written WAV file and `phonemes` is `None` when no phoneme output is produced.
        """
        # The caller wins over the config; the aliases are equal citizens, so
        # whichever of the pair was sent is the one that counts.
        speaker_reference = speaker_reference or ref_wav
        speaker_reference_text = speaker_reference_text or ref_text
        speaker_reference_lang = speaker_reference_lang or ref_lang
        # The voice info is read from the manager only after the load, which
        # refreshes it: a configured voice that has never been fetched is not
        # in the catalog until then.
        voice_info = (None if voice and voice != "default"
                      else self.get_default_voice(lang or self.lang))
        voice_id = voice if voice_info is None else voice_info.voice_id
        # Leased for the whole synthesis: the cache may evict the entry
        # meanwhile, but the weights this call is using stay charged to the
        # memory budget, so no concurrent load is admitted against them.
        with self.voice_cache.lease(voice_id) as model:
            voice_info = voice_info or self.get_voice_info(voice_id)

            synth_params = SynthesisConfig(
                # speaker selection for multi-speaker voices (e.g. the Catalan
                # multiaccent matxa model). ``speaker_id`` (int) or ``speaker``
                # (name via the voice's speaker_id_map); ignored by single-speaker
                # voices, which only have speaker 0.
                speaker_id=self._resolve_speaker(voice_info),
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
                # zero-shot voice cloning. A path to a reference wav; cloning engines turn
                # it into the conditioning signal. In-context engines (ZipVoice) also need
                # the reference's transcription + its language, e.g.:
                #   {"ref_wav": "/home/user/me.wav", "ref_text": "olá tudo bem",
                #    "ref_lang": "pt"}   ->   clone a Portuguese voice speaking English
                speaker_reference=speaker_reference or self._cfg_opt(
                    None, "speaker_reference", "ref_wav", "clone_voice"),
                speaker_reference_text=speaker_reference_text or self._cfg_opt(
                    None, "speaker_reference_text", "ref_text"),
                speaker_reference_lang=speaker_reference_lang or self._cfg_opt(
                    None, "speaker_reference_lang", "ref_lang"),
                # optional post-synthesis audio super-resolution (audiosronnx), off unless
                # switched on in mycroft.conf. The core TTSVoice loads the engine lazily and
                # upscales each chunk; synthesize_wav takes the sample rate from the first
                # chunk, so the WAV header follows.
                super_resolution=bool(self._cfg_opt(False, "super_resolution")),
                super_resolution_model=self._cfg_opt(None, "super_resolution_model"),
            )
            # Written beside the target and moved into place only once the
            # synthesis has finished. Opening the target directly creates it
            # before the audio exists, so a failure part-way leaves a valid but
            # empty WAV — 44 bytes of header — where the cache expects the
            # finished file. The cache decides by existence, so every later
            # request for that sentence is then served silence with a 200, and
            # nothing ever retries it.
            # Written to a private temporary file and moved into place only once
            # the synthesis has finished. Opening the destination directly creates
            # it before the audio exists, so a failure part-way left a valid but
            # empty WAV — 44 bytes of header — exactly where the cache looks. The
            # cache decides by existence, so every later request for that sentence
            # was answered with that silence and an HTTP 200, and nothing retried.
            #
            # The name is unique per call rather than "<target>.part": two requests
            # for the same sentence and voice would otherwise share one temporary
            # file, and whichever lost the race found it already renamed away.
            directory = os.path.dirname(wav_file) or "."
            handle, tmp_file = tempfile.mkstemp(dir=directory, suffix=".part")
            os.close(handle)
            try:
                wav_out = wave.open(tmp_file, "wb")
                try:
                    model.synthesize_wav(sentence, wav_out, synth_params)
                    # Closed inside the try: the final flush is where a full disk
                    # surfaces, and swallowing that would publish a truncated file
                    # as a finished one — the very bug this avoids.
                    wav_out.close()
                except BaseException:
                    # Now the failure is already on its way out. Closing a wave
                    # file that never got its parameters raises "# channels not
                    # specified", which would replace the engine's own exception
                    # and hide why the synthesis actually failed.
                    with suppress(Exception):
                        wav_out.close()
                    raise
                os.replace(tmp_file, wav_file)
            except BaseException:
                with suppress(OSError):
                    os.remove(tmp_file)
                raise

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