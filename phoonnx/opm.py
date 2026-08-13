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
import threading
from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, List, Optional
from ovos_utils.log import LOG
from ovos_plugin_manager.templates.tts import TTS, TTSContext

from phoonnx.model_manager import TTSModelManager, TTSModelInfo
from phoonnx.voice import TTSVoice, SynthesisConfig


@dataclass
class _Loading:
    """A load in progress, and how it ended.

    Callers that arrive while a voice is loading wait on ``done`` and then read
    ``error``: either the load succeeded and the voice is in the cache, or it
    failed and they get the same exception rather than repeating the attempt.
    """
    done: threading.Event = field(default_factory=threading.Event)
    error: Optional[BaseException] = None


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

        # Ordered by least-recently-used, so eviction has an obvious victim.
        self.voices: "OrderedDict[str, TTSVoice]" = OrderedDict()
        self._voice_lock = RLock()
        # Cloned cache identities, most recent last. Bounded because the
        # reference that creates one comes from the request.
        self._cloned_caches: "OrderedDict[str, bool]" = OrderedDict()
        self._cloned_lock = RLock()
        # One gate per voice currently being loaded, so simultaneous callers
        # wait for the load already running instead of starting their own.
        self._loading: Dict[str, threading.Event] = {}
        # Voices that must never be evicted. A cold load costs seconds to
        # minutes depending on the engine, while a resident voice answers in
        # milliseconds, so the voices a deployment actually serves are worth
        # keeping loaded.
        pinned = self.config.get("pinned_voices") or []
        # A single voice written as a bare string is the obvious way to get
        # this wrong, and iterating it would pin one voice per character.
        if isinstance(pinned, str):
            pinned = [pinned]
        # Order is kept so the first pin stays the first loaded; duplicates
        # would otherwise inflate the ceiling that is raised to fit the pins.
        self.pinned_voices: List[str] = list(dict.fromkeys(v for v in pinned if v))
        # How many voices may be resident at once, pinned ones included.
        # Unset keeps the previous behaviour: never evict anything.
        self.max_loaded_voices = self._parse_max_loaded(
            self.config.get("max_loaded_voices"))
        if (self.max_loaded_voices is not None
                and len(self.pinned_voices) > self.max_loaded_voices):
            LOG.error(
                f"{len(self.pinned_voices)} voices are pinned but "
                f"max_loaded_voices is {self.max_loaded_voices}; raising the "
                f"limit to fit them, because a pinned voice is a promise")
            self.max_loaded_voices = len(self.pinned_voices)
        # Resolve the configured voice now, but do not load it. A voice that was
        # named explicitly and does not exist is a configuration error and still
        # raises here; an unset voice (or "default") resolves to the language's
        # default. Loading is deferred to the first synthesis so that fetching a
        # model is never what decides whether the TTS service can start.
        if self.voice and self.voice != "default":
            self.voice_info = self.get_voice_info(self.voice)
        else:
            self.voice_info = self.get_default_voice(self.lang)

        for voice_id in self.pinned_voices:
            try:
                self.get_model(voice_id)
                LOG.info(f"pinned voice loaded: {voice_id}")
            except Exception as e:
                # A pinned voice that cannot load must not stop the service
                # from starting; it simply is not resident.
                LOG.error(f"pinned voice '{voice_id}' failed to load: {e}")

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
            Exception: If `voice_id` is not found after refreshing available voices.
        """
        while True:
            with self._voice_lock:
                if voice_id in self.voices:
                    # Touch it so the least recently used voice is evicted.
                    self.voices.move_to_end(voice_id)
                    return self.voices[voice_id]
                waiting = self._loading.get(voice_id)
                if waiting is None:
                    # This caller owns the load; everyone else waits on it.
                    self._loading[voice_id] = _Loading()
                    break
            # Another caller is loading this voice. Waiting costs nothing and
            # saves a duplicate multi-gigabyte load.
            waiting.done.wait()
            if waiting.error is not None:
                # Share the failure rather than each waiter retrying it in
                # turn: the retries are serial, so N callers behind a slow
                # failure would each wait for the one before. A later request
                # still gets a fresh attempt, because the gate is gone by then.
                raise waiting.error

        # Everything from here is inside the try, so no failure can leave the
        # gate installed. An unknown voice id raises out of get_voice_info, and
        # that is a request parameter, so this path is reachable from outside.
        try:
            with self._voice_lock:
                info = self.get_voice_info(voice_id)

            LOG.debug(f"Using voice: {voice_id}")
            # Loaded outside the lock: a cold load takes seconds to minutes,
            # and holding the lock would stall every other request meanwhile.
            voice = info.load(providers=self._providers())

        except BaseException as exc:
            with self._voice_lock:
                gate = self._loading.pop(voice_id, None)
            if gate is not None:
                gate.error = exc
                gate.done.set()
            raise

        else:
            # Cached and released under one hold of the lock. Releasing the
            # gate first would open a window in which the voice is neither
            # cached nor being loaded, and a waiter that woke inside it would
            # elect itself and start a second full cold load — the duplicate
            # this gate exists to prevent.
            with self._voice_lock:
                if voice_id not in self.voices:
                    self._evict_for(voice_id)
                    self.voices[voice_id] = voice
                self.voices.move_to_end(voice_id)
                cached = self.voices[voice_id]
                gate = self._loading.pop(voice_id, None)
            if gate is not None:
                gate.done.set()
            return cached

    @staticmethod
    def _parse_max_loaded(value) -> Optional[int]:
        """Normalize ``max_loaded_voices``; None means no limit."""
        if value in (None, "", 0):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            LOG.error(f"ignoring invalid max_loaded_voices: {value!r}")
            return None
        if parsed < 1:
            LOG.error(f"ignoring max_loaded_voices={parsed}, it must be at "
                      f"least 1")
            return None
        return parsed

    def _evict_for(self, incoming: str) -> None:
        """Make room for ``incoming``, never dropping a pinned voice."""
        if self.max_loaded_voices is None:
            return
        while len(self.voices) >= self.max_loaded_voices:
            victim = next((v for v in self.voices if v not in self.pinned_voices),
                          None)
            if victim is None:
                # Everything resident is pinned: serve the request anyway
                # rather than evict a voice an operator asked us to keep.
                LOG.warning(
                    f"all {len(self.voices)} loaded voices are pinned; "
                    f"loading '{incoming}' above max_loaded_voices")
                return
            self.voices.pop(victim, None)
            LOG.debug(f"evicted least recently used voice: {victim}")

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
        if voice_id not in self.model_manager.voices:
            LOG.info(f"{voice_id} not found - refreshing voice list")
            self.refresh_voices(force=True)
            if voice_id not in self.model_manager.voices:
                raise Exception(f"Unknown voice: {voice_id}")
        return self.model_manager.voices[voice_id]

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