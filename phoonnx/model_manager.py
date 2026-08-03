import onnxruntime
import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any, Sequence
import requests
from json_database import JsonStorageXDG, JsonStorage

from phoonnx.config import PhonemeType, get_phonemizer, VoiceConfig, Engine, Alphabet
from phoonnx.util import match_lang, normalize_lang, LOG
from phoonnx.providers import ProviderSpec
from phoonnx.voice import TTSVoice


def _tmp_path(dest: Path) -> Path:
    """Sibling scratch path used while a download is in flight.

    Every write goes through here, so this is also where the voice directory is
    created — a voice's directory should exist because something was written
    into it, not because its catalog entry was constructed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest.with_suffix(dest.suffix + ".part")


def _is_cached(path: Path) -> bool:
    """A cached artifact counts only if it exists and is non-empty."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _expected_size(response) -> Optional[int]:
    """Content-Length of a response, or None when absent/unusable."""
    try:
        raw = response.headers.get("Content-Length")
    except AttributeError:
        return None
    if not isinstance(raw, (str, int)) or isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _stream_to_file(url: str, dest: Path, timeout: int = 120) -> Path:
    """Stream ``url`` into ``dest`` atomically.

    The body is written to a sibling ``.part`` file and only renamed onto
    ``dest`` once it has been fully received (and matches ``Content-Length``,
    when the server sends one). An interrupted download therefore leaves at
    most a stale ``.part`` file, never a truncated artifact that later runs
    would mistake for a complete one.
    """
    tmp = _tmp_path(dest)
    written = 0
    try:
        with requests.get(url, timeout=timeout, stream=True) as r:
            r.raise_for_status()
            expected = _expected_size(r)
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        written += len(chunk)
        if expected is not None and written != expected:
            raise IOError(f"incomplete download for '{url}': "
                          f"got {written} bytes, expected {expected}")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dest)
    return dest


def _write_bytes_atomic(dest: Path, data: bytes) -> Path:
    """Write ``data`` to ``dest`` via a sibling ``.part`` file + rename."""
    tmp = _tmp_path(dest)
    try:
        tmp.write_bytes(data)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dest)
    return dest


def _write_json_atomic(dest: Path, data: Any) -> Path:
    """Serialize ``data`` to ``dest`` via a sibling ``.part`` file + rename."""
    tmp = _tmp_path(dest)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dest)
    return dest


def _load_cached_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a cached JSON file, self-healing a corrupt one.

    Returns ``None`` when the file is missing, empty or undecodable; in the
    corrupt case the file is unlinked so the caller re-downloads it once
    instead of failing forever on a bad cache entry.
    """
    if not _is_cached(path):
        path.unlink(missing_ok=True)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
        LOG.warning(f"discarding corrupt cached json '{path}': {e}")
        path.unlink(missing_ok=True)
        return None


@dataclass
class TTSModelInfo:
    voice_id: str
    lang: str  # not always present in config.json and often wrong if present
    model_url: str
    config_url: Optional[str] = None  # some models only provide tokens.txt
    vocab_url: Optional[str] = None  # transformers provides vocab.json with tokens
    tokenizer_config_url: Optional[str] = None  # transformers provides tokenizer_config.json with metadata
    tokens_url: Optional[str] = None  # mimic3/sherpa provide phoneme_map in this format
    phoneme_map_url: Optional[str] = None  # json lookup table for phoneme replacement
    phoneme_type: Optional[PhonemeType] = None
    phonemizer_model: Optional[str] = None  # per-phonemizer variant (e.g. AhoTTS classic/modern/northern)
    alphabet: Optional[Alphabet] = None
    engine: Optional[Engine] = None
    vocab_override: Optional[Dict[str, int]] = field(default_factory=dict)

    # Two-stage engines (Matcha-TTS) pair the acoustic model with a *separate*
    # vocoder published as its own artifact.  These fields link the tested
    # vocoder for this voice; the manager downloads it alongside the model.
    vocoder_url: Optional[str] = None
    vocoder_config_url: Optional[str] = None
    vocoder_type: Optional[str] = None  # "vocos" | "wavenext" | "hifigan"

    # StyleTTS2/Kokoro carry a per-voice *style* embedding (Kokoro: a [510, 256]
    # pack indexed by token length). The manager downloads it alongside the model
    # and the StyleTTS2 adapter loads it from engine_params["style_path"].
    style_url: Optional[str] = None

    # Cloning models (YourTTS) carry a separate speaker-encoder ONNX (reference audio
    # -> d-vector); the manager downloads it and the adapter loads it from
    # engine_params["speaker_encoder_path"].
    speaker_encoder_url: Optional[str] = None
    speaker_encoder_type: Optional[str] = None

    # Multi-graph iterative engines (F5-TTS) split inference across several
    # ONNX files beyond the primary model.onnx. Each entry maps an
    # engine_params key (e.g. "preprocess_path", "decode_path") to the URL of
    # that graph; the manager downloads each one alongside the model and
    # injects the local path under the same key.
    aux_model_urls: Optional[Dict[str, str]] = field(default_factory=dict)
    # Chatterbox (autoregressive codec-LM) splits inference across four graphs: the
    # language_model is `model_url`; these three aux graphs load from engine_params
    # (speech_encoder_path / embed_tokens_path / conditional_decoder_path).
    speech_encoder_url: Optional[str] = None
    embed_tokens_url: Optional[str] = None
    conditional_decoder_url: Optional[str] = None

    # Free-form engine settings that are plain values rather than downloadable files
    # (``aux_model_urls`` covers the files). Merged into ``engine_params()`` as-is, so an
    # index entry can pin a per-voice knob the adapter reads — e.g. NeuTTS voices, which
    # all share one checkpoint and differ only by which ``voices.json`` preset they use.
    engine_options: Optional[Dict[str, Any]] = field(default_factory=dict)

    # Optional BCP47 -> language-token map: how this voice's lang_code becomes the model's
    # language token. Dialect models (e.g. lahgtna) repurpose the base tokens, so a literal
    # token like "eg" is mapped here rather than derived from the (normalized) lang code.
    lang_tokens: Optional[Dict[str, str]] = None

    # user-friendly name for UIs/CLIs; may contain "{engine}"/"{phoneme_type}"
    # placeholders that get resolved once those fields are known
    display_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize every field to a JSON-safe dict, matching the format used by the
        bundled voice-index JSON files (enum fields become their plain string value).
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
        return d

    @property
    def config(self) -> VoiceConfig:
        # lazy loaded
        """
        Return the model's VoiceConfig, loading and synchronizing it on first access.
        
        On first access, load or construct a VoiceConfig from cached files or remote URLs and apply any provided vocab, tokenizer, or tokens overrides; normalize and propagate language, alphabet, phoneme type, and engine values between the dataclass and the loaded config. Subsequent accesses return the cached VoiceConfig.
        
        Returns:
            VoiceConfig: The loaded and normalized voice configuration instance.
        """
        if not self._config:
            if self.config_url:
                config = self.download_config()
                # HACK: seen in some published piper voices
                # "es_MX-ald-medium"
                if config.get('phoneme_type', "") == "PhonemeType.ESPEAK":
                    config["phoneme_type"] = "espeak"
            else:
                config = {"phoneme_type": "graphemes", "alphabet": "unicode"}
            if self.phoneme_type:
                config["phoneme_type"] = self.phoneme_type
            if self.phonemizer_model:
                config["phonemizer_model"] = self.phonemizer_model

            lang_code = normalize_lang(self.lang) if self.lang else None
            alphabet = self.alphabet if self.alphabet else None
            phoneme_type = self.phoneme_type if self.phoneme_type else None
            engine = self.engine if self.engine else None

            if self.engine == Engine.CHATTERBOX or config.get("engine") == "chatterbox":
                # Chatterbox carries its own subword BPE (tokenizer.json); no vocab/tokens.
                tok_json = self.download_bpe_tokenizer()
                self._config = VoiceConfig.from_dict(config,
                                                     alphabet=alphabet,
                                                     phoneme_type=phoneme_type,
                                                     engine=engine,
                                                     lang_code=lang_code,
                                                     bpe_tokenizer_json=str(tok_json) if tok_json else None,
                                                     lang_tokens=self.lang_tokens)
            elif self.vocab_override:
                self._config = VoiceConfig.from_dict(config,
                                                     vocab=self.vocab_override,
                                                     alphabet=alphabet,
                                                     phoneme_type=phoneme_type,
                                                     engine=engine,
                                                     lang_code=lang_code)
            elif self.vocab_url:
                vocab = self.download_vocab()
                if self.tokenizer_config_url:
                    tokenizer_config = self.download_tokenizer_config()
                else:
                    tokenizer_config = {}
                self._config = VoiceConfig.from_dict(config,
                                                     vocab=vocab,
                                                     alphabet=alphabet,
                                                     phoneme_type=phoneme_type,
                                                     engine=engine,
                                                     tokenizer_config=tokenizer_config,
                                                     lang_code=lang_code)
            elif self.tokens_url:
                self.download_tokens_txt()
                self._config = VoiceConfig.from_dict(config,
                                                     alphabet=alphabet,
                                                     phoneme_type=phoneme_type,
                                                     engine=engine,
                                                     tokens_txt=str(self.voice_path / "tokens.txt"),
                                                     lang_code=lang_code)
            else:
                self._config = VoiceConfig.from_dict(config,
                                                     alphabet=alphabet,
                                                     phoneme_type=phoneme_type,
                                                     engine=engine,
                                                     lang_code=lang_code)

            # populate any missing properties
            self.lang = self.lang or normalize_lang(self._config.lang_code)
            self.alphabet = self.alphabet or self._config.alphabet
            self.phoneme_type = self.phoneme_type or self._config.phoneme_type
            self.engine = self.engine or self._config.engine

        return self._config

    def __post_init__(self):
        """
        Initialize internal state for TTSModelInfo and normalize persisted values.
        
        Sets up the private config storage, ensures the voice cache directory exists, and converts string representations of engine, alphabet, and phoneme_type into their corresponding Enum values so the instance fields are normalized.
        """
        self._config: Optional[VoiceConfig] = None

        # cast strings to enum for consistency
        if not isinstance(self.engine, Engine) and isinstance(self.engine, str):
            self.engine = Engine(self.engine)
        if not isinstance(self.alphabet, Alphabet) and isinstance(self.alphabet, str):
            self.alphabet = Alphabet(self.alphabet)
        if not isinstance(self.phoneme_type, PhonemeType) and isinstance(self.phoneme_type, str):
            self.phoneme_type = PhonemeType(self.phoneme_type)

        # resolve "{engine}"/"{phoneme_type}" placeholders in display_name,
        # if those fields are already known (avoids triggering a config
        # download just to format a label)
        if self.display_name and "{" in self.display_name:
            try:
                self.display_name = self.display_name.format(
                    engine=self.engine.value if self.engine else "",
                    phoneme_type=self.phoneme_type.value if self.phoneme_type else "",
                )
            except (AttributeError, KeyError):
                LOG.warning(f"Could not format display_name for {self.voice_id}.")

    @property
    def voice_path(self) -> Path:
        return Path(os.path.expanduser("~")) / ".cache" / "phoonnx" / "voices" / self.voice_id

    def download_config(self) -> Dict[str, Any]:
        """
        Ensure the model configuration file exists locally and return its parsed contents.
        
        If the configuration file is not present in the voice cache directory, download it from the instance's configured URL and save it as model.json; otherwise load the existing file.
        
        Returns:
            dict: Parsed JSON configuration for the TTS model.
        """
        config_path = self.voice_path / "model.json"
        cached = _load_cached_json(config_path)
        if cached is not None:
            return cached
        r = requests.get(self.config_url, timeout=30)
        r.raise_for_status()
        cfg = r.json()  # validate received json
        _write_json_atomic(config_path, cfg)
        return cfg

    def download_tokenizer_config(self) -> Dict[str, Any]:
        """
        Download and cache the tokenizer configuration for this voice, returning it as a parsed dictionary.
        
        If a local cached file exists, it is loaded and returned; otherwise the configuration is fetched from the configured URL, saved to the voice cache, and returned.
        
        Returns:
            dict: The tokenizer configuration parsed from JSON.
        
        Raises:
            requests.HTTPError: If the HTTP request for the tokenizer configuration returns an error status.
            json.JSONDecodeError: If a retrieved or cached file contains invalid JSON.
        """
        config_path = self.voice_path / "tokenizer_config.json"
        cached = _load_cached_json(config_path)
        if cached is not None:
            return cached
        r = requests.get(self.tokenizer_config_url, timeout=30)
        r.raise_for_status()
        cfg = r.json()  # validate received json
        _write_json_atomic(config_path, cfg)
        return cfg

    def download_vocab(self) -> Dict[str, Any]:
        """
        Load the voice vocabulary from the local cache or download it from the configured URL.
        
        If a cached vocab.json exists in the voice's cache directory, it is read and returned.
        If no cached file exists and `vocab_url` is set, the vocabulary JSON is fetched from that URL,
        saved to the cache as vocab.json (UTF-8), and the parsed dictionary is returned.
        
        Returns:
            dict: The vocabulary mapping loaded from vocab.json.
        
        Raises:
            requests.RequestException: On network errors or non-success HTTP responses.
            OSError: On file read/write errors.
        """
        vocab_path = self.voice_path / "vocab.json"
        cached = _load_cached_json(vocab_path)
        if cached is not None:
            return cached
        if not self.vocab_url:
            # preserve the historical error for a missing, un-downloadable vocab
            with open(vocab_path, "r", encoding="utf-8") as f:
                return json.load(f)
        r = requests.get(self.vocab_url, timeout=30)
        r.raise_for_status()
        cfg = r.json()
        _write_json_atomic(vocab_path, cfg)
        return cfg

    def download_tokens_txt(self) -> str:
        """
        Ensure a local tokens.txt exists for this voice and return its contents.
        
        If `tokens_url` is set and the file does not exist in the voice cache directory, download the tokens file, save it to the cache using UTF-8 encoding, and return its text. If the file already exists, read and return its contents.
        
        Returns:
            str: Contents of the tokens file.
        
        Raises:
            requests.exceptions.RequestException: If the HTTP request to `tokens_url` fails or the response status is not successful.
        """
        tokens_path = self.voice_path / "tokens.txt"
        if self.tokens_url and not _is_cached(tokens_path):
            tokens_path.unlink(missing_ok=True)  # discard an empty/partial cache
            r = requests.get(self.tokens_url, timeout=30)
            r.raise_for_status()
            tokens = r.text
            _write_bytes_atomic(tokens_path, tokens.encode("utf-8"))
            return tokens
        with open(tokens_path, "r", encoding="utf-8") as f:
            return f.read()

    def _fetch_onnx(self, url: str, dest: Path) -> Path:
        """Download an ONNX graph (and its external-data sidecar, if present) into the
        voice cache.

        A graph with external weights references them by the original
        ``<name>.onnx_data`` filename, so that sidecar is saved under exactly that name
        next to the graph for the reference to resolve. Chatterbox's four graphs all use
        external data; single-file graphs (piper/vits exports) simply have no sidecar.

        A missing sidecar may surface as an HTTP 404 (online) or as a connection
        error (offline); both mean "no sidecar here", so a fully-cached voice loads
        without the probe crashing when there is no network.
        """
        if not _is_cached(dest):
            _stream_to_file(url, dest)
        data_dest = dest.parent / (url.split("/")[-1] + "_data")   # e.g. model_q4.onnx_data
        if not _is_cached(data_dest):
            try:
                with requests.get(url + "_data", timeout=600, stream=True) as r:
                    if r.status_code != 404:
                        r.raise_for_status()
                        tmp = _tmp_path(data_dest)
                        try:
                            with open(tmp, "wb") as f:
                                for chunk in r.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)
                        except BaseException:
                            tmp.unlink(missing_ok=True)
                            raise
                        os.replace(tmp, data_dest)
            except requests.exceptions.RequestException as e:
                # Offline/DNS/timeout: treat like a 404 (sidecar simply absent) so a
                # fully-cached voice still loads; a genuinely needed sidecar surfaces
                # later when the graph is loaded.
                LOG.debug(f"external-data sidecar unavailable for '{url}': {e}")
        return dest

    def download_model(self) -> Path:
        """Download the primary ONNX graph (+ external-data sidecar, if any)."""
        return self._fetch_onnx(self.model_url, self.voice_path / "model.onnx")

    def download_vocoder(self) -> Optional[Path]:
        """
        Download the separate vocoder ONNX (and optional config) for two-stage
        engines such as Matcha-TTS into the voice cache directory.

        Returns the local vocoder path, or ``None`` if this voice has no
        ``vocoder_url``.
        """
        if not self.vocoder_url:
            return None
        vocoder_path = self.voice_path / "vocoder.onnx"
        if not _is_cached(vocoder_path):
            _stream_to_file(self.vocoder_url, vocoder_path)
        if self.vocoder_config_url:
            vocoder_cfg_path = self.voice_path / "vocoder.json"
            if _load_cached_json(vocoder_cfg_path) is None:
                r = requests.get(self.vocoder_config_url, timeout=30)
                r.raise_for_status()
                _write_json_atomic(vocoder_cfg_path, r.json())
        return vocoder_path

    def download_style(self) -> Optional[Path]:
        """Download the per-voice StyleTTS2/Kokoro style embedding, if any."""
        if not self.style_url:
            return None
        style_path = self.voice_path / "style.bin"
        if not _is_cached(style_path):
            _stream_to_file(self.style_url, style_path)
        return style_path

    def download_speaker_encoder(self) -> Optional[Path]:
        """Download the speaker-encoder ONNX (cloning models), if any."""
        if not self.speaker_encoder_url:
            return None
        enc_path = self.voice_path / "speaker_encoder.onnx"
        if not _is_cached(enc_path):
            _stream_to_file(self.speaker_encoder_url, enc_path)
        return enc_path

    def download_aux_models(self) -> Dict[str, Path]:
        """Download the auxiliary ONNX graphs of multi-graph engines (F5-TTS),
        if any. Returns {engine_params_key: local_path}."""
        paths: Dict[str, Path] = {}
        for key, url in (self.aux_model_urls or {}).items():
            fname = url.rsplit("/", 1)[-1] or f"{key}.onnx"
            aux_path = self.voice_path / fname
            if not _is_cached(aux_path):
                _stream_to_file(url, aux_path)
            paths[key] = aux_path
        return paths

    def download_bpe_tokenizer(self) -> Optional[Path]:
        """Download the HF ``tokenizer.json`` (Chatterbox subword BPE) from
        ``tokenizer_config_url``, if any."""
        if not self.tokenizer_config_url:
            return None
        path = self.voice_path / "tokenizer.json"
        if _load_cached_json(path) is None:
            r = requests.get(self.tokenizer_config_url, timeout=60)
            r.raise_for_status()
            _write_bytes_atomic(path, r.content)
        return path

    def engine_params(self) -> Dict[str, Any]:
        """
        Build the engine_params dict for synthesis, resolving the vocoder to
        its locally-downloaded path.  Empty for single-stage engines.
        """
        params: Dict[str, Any] = dict(self.engine_options or {})
        for key, aux_path in self.download_aux_models().items():
            params[key] = str(aux_path)
        style_path = self.download_style()
        if style_path:
            params["style_path"] = str(style_path)
        enc_path = self.download_speaker_encoder()
        if enc_path:
            params["speaker_encoder_path"] = str(enc_path)
            if self.speaker_encoder_type:
                params["speaker_encoder_type"] = self.speaker_encoder_type
        vocoder_path = self.download_vocoder()
        if vocoder_path:
            params["vocoder_path"] = str(vocoder_path)
            if self.vocoder_type:
                params["vocoder_type"] = self.vocoder_type
            vocoder_cfg_path = self.voice_path / "vocoder.json"
            if vocoder_cfg_path.is_file():
                with open(vocoder_cfg_path, "r", encoding="utf-8") as f:
                    params["vocoder_config"] = json.load(f)
        elif self.vocoder_type:
            # Parametric vocoder (e.g. Griffin-Lim) — no model file, just config.
            params["vocoder_type"] = self.vocoder_type
            if self.vocoder_config_url:
                cfg_path = self.voice_path / "vocoder.json"
                cached = _load_cached_json(cfg_path)
                if cached is None:
                    r = requests.get(self.vocoder_config_url, timeout=30)
                    r.raise_for_status()
                    _write_bytes_atomic(cfg_path, r.content)
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                params["vocoder_config"] = cached

        # Chatterbox's three auxiliary graphs (the language_model is the primary model).
        for url, key, fname in (
            (self.speech_encoder_url, "speech_encoder_path", "speech_encoder.onnx"),
            (self.embed_tokens_url, "embed_tokens_path", "embed_tokens.onnx"),
            (self.conditional_decoder_url, "conditional_decoder_path", "conditional_decoder.onnx"),
        ):
            if url:
                params[key] = str(self._fetch_onnx(url, self.voice_path / fname))
        return params

    def download_all(self) -> Path:
        """Download everything this voice needs to load offline.

        The primary graph, the config JSON, whichever tokenizer artifact the
        voice uses (BPE tokenizer / vocab + tokenizer config / tokens.txt) and
        every auxiliary artifact resolved by :meth:`engine_params` (vocoder,
        style embedding, speaker encoder, extra ONNX graphs).

        This is the single definition of "fully downloaded" shared by the CLI
        and :meth:`TTSModelManager.download_voice_by_id`.

        Returns:
            Path: local path of the primary ONNX model.
        """
        model_path = self.download_model()

        cfg: Dict[str, Any] = self.download_config() if self.config_url else {}

        if self.engine == Engine.CHATTERBOX or cfg.get("engine") == "chatterbox":
            self.download_bpe_tokenizer()
        elif self.vocab_override:
            pass  # vocab ships inside the index entry
        elif self.vocab_url:
            self.download_vocab()
            if self.tokenizer_config_url:
                self.download_tokenizer_config()
        elif self.tokens_url:
            self.download_tokens_txt()

        # vocoder / style / speaker-encoder / aux graphs
        self.engine_params()
        return model_path

    def load(self, providers: Optional[Sequence[ProviderSpec]] = None) -> TTSVoice:
        """
        Load and return a TTSVoice for this model, ensuring the ONNX model is downloaded and the voice configuration is applied.

        Parameters:
            providers (Sequence, optional): Ordered ONNX Runtime execution providers,
                e.g. ``["ROCMExecutionProvider", "CPUExecutionProvider"]``. When omitted the
                providers are resolved from the environment / auto-detection.
        
        Loads a TTSVoice from the cached model and config files (and tokens file if available). If this TTSModelInfo specifies a different phoneme type or alphabet than the loaded voice, updates the loaded voice's phoneme_type and alphabet and rebuilds its phonemizer accordingly.
        
        Returns:
            TTSVoice: The configured TTSVoice instance ready for synthesis.
        """
        model_path = self.voice_path / "model.onnx"
        config_path = self.voice_path / "model.json"
        vocab_path = self.voice_path / "vocab.json"
        tokenizer_config_path = self.voice_path / "tokenizer_config.json"
        tokens_path = self.voice_path / "tokens.txt"
        self.download_model()

        # Voices without a published config.json (Chatterbox) can't be engine-detected
        # from files on disk — build the voice from the index-derived VoiceConfig, which
        # already carries the engine, BPE tokenizer and lang_tokens.
        if self.engine and not self.config_url:
            config = self.config
            config.engine_params = {**(config.engine_params or {}), **self.engine_params()}
            session = onnxruntime.InferenceSession(
                str(model_path),
                sess_options=onnxruntime.SessionOptions(),
                providers=["CPUExecutionProvider"],
            )
            return TTSVoice(session=session, config=config)

        voice = TTSVoice.load(model_path=model_path,
                              config_path=config_path,
                              vocab_path=vocab_path,
                              tokenizer_config_path=tokenizer_config_path,
                              lang_code=self.config.lang_code,
                              phoneme_type_str=self.config.phoneme_type,
                              alphabet_str=self.config.alphabet,
                              engine_params=self.engine_params() or None,
                              providers=providers,
                              phonemes_txt=str(tokens_path) if self.tokens_url else None)
        # override phoneme_type, if config.json is wrong
        if self.phoneme_type != voice.config.phoneme_type or self.alphabet != voice.config.alphabet:
            voice.config.phoneme_type = self.phoneme_type
            voice.config.alphabet = self.alphabet
            voice.phonemizer = get_phonemizer(self.phoneme_type,
                                              alphabet=self.alphabet,
                                              model=voice.config.phonemizer_model)
        return voice


class TTSModelManager:
    # Explicit merge order for the bundled indexes: later files win when two
    # sources publish the same voice_id. Any index file not listed here is
    # still picked up (appended, alphabetically) so a newly added index can
    # never silently go missing from either the catalog or the listing.
    _VOICE_INDEX_ORDER = (
        "OVOS.json", "MMS.json", "proxectonos.json", "piper.json",
        "phonikud.json", "neurlang.json", "mimic3.json",
        "transformers_community.json", "piper_community.json",
        "optispeech.json", "glowtts.json", "mixertts.json", "fastpitch.json",
        "coqui_community.json", "vits2.json", "styletts2.json", "f5tts.json",
        "coqui_vits.json", "BSC.json", "shami.json", "chatterbox.json",
        "supertonic.json", "neutts.json", "pockettts.json", "sparktts.json",
        "qwen3tts.json",
    )

    @classmethod
    def voice_index_path(cls) -> Path:
        """Directory holding the bundled voice-index JSON files."""
        return Path(os.path.dirname(__file__)) / "voice_index"

    @classmethod
    def voice_index_files(cls) -> List[Path]:
        """Every bundled voice-index file, in merge order.

        The single source of truth for both ``merge_default_voices`` and
        ``get_available_voice_ids_by_source`` - they cannot drift apart.
        """
        base_path = cls.voice_index_path()
        known = [base_path / f for f in cls._VOICE_INDEX_ORDER
                 if (base_path / f).is_file()]
        extra = sorted(p for p in base_path.glob("*.json") if p not in known)
        return known + extra

    def __init__(self, cache_path: Optional[str] = None):
        """
        Initialize the TTSModelManager and prepare persistent cache storage.
        
        Creates an empty in-memory registry for voices and sets up JSON-backed cache storage using the provided path or the XDG cache location; ensures the cache file exists.
        
        Parameters:
            cache_path (Optional[str]): Filesystem path for the cache file. If omitted, uses the user's XDG cache directory under "phoonnx/voices".
        """
        self.voices: Dict[str, TTSModelInfo] = {}
        if cache_path:
            self.cache = JsonStorage(cache_path)
        else:
            self.cache = JsonStorageXDG("voices", subfolder="phoonnx")
        self.cache.store() # ensure file exists

    @property
    def all_voices(self) -> List[TTSModelInfo]:
        """
        List all registered TTSModelInfo instances.
        
        Returns:
            List[TTSModelInfo]: A list containing the current TTSModelInfo objects stored in the manager.
        """
        return list(self.voices.values())

    @property
    def supported_langs(self) -> List[str]:
        return sorted(set(l.lang for l in self.all_voices))

    def clear(self):
        self.cache.clear()
        self.voices = {}

    def load(self):
        """
        Reload the on-disk voice cache and rebuild the in-memory voice registry.
        
        Clears the current in-memory voices, calls the cache reload, and attempts to instantiate
        TTSModelInfo for each cached entry. Entries that fail to construct are skipped and an
        error is logged; successful entries are stored in self.voices keyed by voice_id.
        """
        self.cache.reload()
        self.voices = {}
        for voice_id, voice_dict in self.cache.items():
            try:
                self.voices[voice_id] =  TTSModelInfo(**voice_dict)
            except Exception as e:
                LOG.error(f"Failed to load '{voice_id}': ({e})")
                continue

    def save(self):
        """
        Persist in-memory voice metadata to the configured cache storage.

        Writes each managed voice's full metadata (every TTSModelInfo field) to the
        cache and then persists the cache to disk.
        """
        self.cache.clear()
        for voice_id, voice_info in self.voices.items():
            self.cache[voice_id] = voice_info.to_dict()
        self.cache.store()

    def add_voice(self, voice_info: TTSModelInfo):
        """
        Add or update a TTS voice in the manager's in-memory registry and persist its full metadata to the cache.

        This stores the given TTSModelInfo under its voice_id in memory and writes every
        field into the persistent cache, overwriting any existing entry for the same voice_id.

        Parameters:
            voice_info (TTSModelInfo): The voice metadata to add or update.
        """
        self.voices[voice_info.voice_id] = voice_info
        self.cache[voice_info.voice_id] = voice_info.to_dict()

    def get_lang_voices(self, lang: str) -> List[TTSModelInfo]:
        voices = sorted(
            [
                (voice_info, match_lang(voice_info.lang, lang)[-1])
                for voice_info in self.voices.values()
            ], key=lambda k: k[1])
        return [v[0] for v in voices if v[1] < 10]

    def merge_default_voices(self, store=False):
        """
        Merge a set of predefined voice indexes into the manager's cache and rebuild the in-memory voice registry.
        
        Loads multiple JSON voice index files bundled with the package into the manager's cache, constructs TTSModelInfo objects for each entry, and logs (but skips) any entries that fail to load. If store is True, the updated cache is persisted to disk.
        
        Parameters:
            store (bool): If True, persist the updated cache to disk after merging.
        """
        for index_file in self.voice_index_files():
            self.cache.update(JsonStorage(str(index_file)))
        self.voices = {}
        for voice_id, voice_dict in self.cache.items():
            try:
                self.voices[voice_id] =  TTSModelInfo(**voice_dict)
            except Exception as e:
                LOG.error(f"Failed to load '{voice_id}': ({e})")
                continue

        if store:
            self.cache.store()

    def get_available_voice_ids_by_source(self) -> Dict[str, List[str]]:
        """
        List all voice IDs bundled with phoonnx, grouped by source.

        Reads the ``voice_index`` JSON files that ship with the package
        directly (plain ``json.load``, no ``TTSModelInfo`` construction),
        so this performs no network access and downloads no model/config
        files - suitable for a quick "what's available" listing before
        committing to downloading a specific voice via ``download_voice_by_id``.

        Returns:
            Dict[str, List[str]]: mapping of source name to sorted voice IDs.
        """
        result: Dict[str, List[str]] = {}
        for path in self.voice_index_files():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            result[path.stem.lower()] = sorted(data.keys())
        return result

    def download_voice_by_id(self, voice_id: str) -> bool:
        """
        Download everything a voice needs to load offline, by its ID.

        Looks the voice up in the in-memory registry first (populated via
        ``load()``/``merge_default_voices()``); if not found there, falls
        back to the bundled voice indexes so a voice can be downloaded
        on-demand without first loading the full catalog into memory.

        Fetches the full set of artifacts via ``TTSModelInfo.download_all``:
        the ONNX graph, the config JSON, the tokenizer artifacts and any
        vocoder / style / speaker-encoder / auxiliary graphs.

        Returns:
            bool: True if the voice was found and its download was attempted,
            False if the voice ID could not be resolved.
        """
        voice_info = self.voices.get(voice_id)
        if not voice_info:
            for path in self.voice_index_files():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if voice_id in data:
                    voice_info = TTSModelInfo(**data[voice_id])
                    break

        if not voice_info:
            LOG.error(f"Cannot find voice information for ID: {voice_id}")
            return False

        voice_info.download_all()
        return True



if __name__ == "__main__":
    manager = TTSModelManager()
    manager.cache.clear()
    manager.merge_default_voices(store=True)  # load and cache known voice models

    print(f"Total voices: {len(manager.all_voices)}")
    print(f"Total langs: {len(manager.supported_langs)}")

    # Total voices: 1462
    # Total langs: 1205

    for voice in manager.get_lang_voices("gl"):
        print(voice)
        # TTSModelInfo(voice_id='proxectonos/brais', lang='gl-ES', model_url='https://huggingface.co/OpenVoiceOS/proxectonos-brais-vits-graphemes-onnx/resolve/main/model.onnx', config_url='https://huggingface.co/OpenVoiceOS/proxectonos-brais-vits-graphemes-onnx/resolve/main/config.json', vocab_url=None, tokenizer_config_url=None, tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=0, num_langs=1, sample_rate=16000, lang_code='gl-ES', phoneme_type='graphemes', alphabet='unicode', phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine='coqui', length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8, add_diacritics=False, tokenizer=TTSTokenizer(vocabulary=Vocabulary(char2idx={'<PAD>': 0, '!': 1, '¡': 2, "'": 3, '(': 4, ')': 5, ',': 6, '-': 7, '.': 8, ':': 9, ';': 10, '¿': 11, '?': 12, ' ': 13, '"': 14, '\n': 15, 'A': 16, 'B': 17, 'C': 18, 'D': 19, 'E': 20, 'F': 21, 'G': 22, 'H': 23, 'I': 24, 'J': 25, 'K': 26, 'L': 27, 'M': 28, 'N': 29, 'O': 30, 'P': 31, 'Q': 32, 'R': 33, 'S': 34, 'T': 35, 'U': 36, 'V': 37, 'W': 38, 'X': 39, 'Y': 40, 'Z': 41, 'Ç': 42, 'Á': 43, 'É': 44, 'Í': 45, 'Ï': 46, 'Ó': 47, 'Ú': 48, 'Ü': 49, 'a': 50, 'b': 51, 'c': 52, 'd': 53, 'e': 54, 'f': 55, 'g': 56, 'h': 57, 'i': 58, 'j': 59, 'k': 60, 'l': 61, 'm': 62, 'n': 63, 'o': 64, 'p': 65, 'q': 66, 'r': 67, 's': 68, 't': 69, 'u': 70, 'v': 71, 'w': 72, 'x': 73, 'y': 74, 'z': 75, 'ñ': 76, 'á': 77, 'é': 78, 'í': 79, 'ï': 80, 'ó': 81, 'ú': 82, 'ü': 83, '<BLNK>': 84}, pad='<PAD>', eos='<EOS>', bos='<BOS>', blank='<BLNK>', blank_word=None), add_blank_char=True, add_blank_word=False, use_eos_bos=False, blank_at_end=True, blank_at_start=True), blank_at_start=True, blank_at_end=True, pad_token=None, blank_token=None, bos_token=None, eos_token=None, word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.GRAPHEMES: 'graphemes'>, alphabet=<Alphabet.UNICODE: 'unicode'>, engine=<Engine.COQUI: 'coqui'>, vocab_override={})
        # TTSModelInfo(voice_id='proxectonos/celtia', lang='gl-ES', model_url='https://huggingface.co/OpenVoiceOS/proxectonos-celtia-vits-graphemes-onnx/resolve/main/model.onnx', config_url='https://huggingface.co/OpenVoiceOS/proxectonos-celtia-vits-graphemes-onnx/resolve/main/config.json', vocab_url=None, tokenizer_config_url=None, tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=0, num_langs=1, sample_rate=16000, lang_code='gl-ES', phoneme_type='graphemes', alphabet='unicode', phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine='coqui', length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8, add_diacritics=False, tokenizer=TTSTokenizer(vocabulary=Vocabulary(char2idx={'_': 0, '!': 1, '"': 2, '(': 3, ')': 4, ',': 5, '-': 6, '.': 7, ':': 8, ';': 9, '?': 10, '¡': 11, '¿': 12, ' ': 13, 'A': 14, 'B': 15, 'C': 16, 'D': 17, 'E': 18, 'F': 19, 'G': 20, 'H': 21, 'I': 22, 'J': 23, 'K': 24, 'L': 25, 'M': 26, 'N': 27, 'O': 28, 'P': 29, 'Q': 30, 'R': 31, 'S': 32, 'T': 33, 'U': 34, 'V': 35, 'X': 36, 'Y': 37, 'Z': 38, 'a': 39, 'b': 40, 'c': 41, 'd': 42, 'e': 43, 'f': 44, 'g': 45, 'h': 46, 'i': 47, 'j': 48, 'k': 49, 'l': 50, 'm': 51, 'n': 52, 'o': 53, 'p': 54, 'q': 55, 'r': 56, 's': 57, 't': 58, 'u': 59, 'v': 60, 'w': 61, 'x': 62, 'y': 63, 'z': 64, 'Á': 65, 'É': 66, 'Í': 67, 'Ó': 68, 'Ú': 69, 'á': 70, 'é': 71, 'í': 72, 'ñ': 73, 'ó': 74, 'ú': 75, 'ü': 76, '<BLNK>': 77}, pad='_', eos='*', bos='^', blank='<BLNK>', blank_word=None), add_blank_char=True, add_blank_word=False, use_eos_bos=False, blank_at_end=True, blank_at_start=True), blank_at_start=True, blank_at_end=True, pad_token=None, blank_token=None, bos_token=None, eos_token=None, word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.GRAPHEMES: 'graphemes'>, alphabet=<Alphabet.UNICODE: 'unicode'>, engine=<Engine.COQUI: 'coqui'>, vocab_override={})
        # TTSModelInfo(voice_id='proxectonos/brais-cotovia', lang='gl-ES', model_url='https://huggingface.co/OpenVoiceOS/proxectonos-brais-vits-phonemes-onnx/resolve/main/model.onnx', config_url='https://huggingface.co/OpenVoiceOS/proxectonos-brais-vits-phonemes-onnx/resolve/main/config.json', vocab_url=None, tokenizer_config_url=None, tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=0, num_langs=1, sample_rate=16000, lang_code='gl-ES', phoneme_type='cotovia', alphabet='cotovia', phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine='coqui', length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8, add_diacritics=False, tokenizer=TTSTokenizer(vocabulary=Vocabulary(char2idx={'<PAD>': 0, '!': 1, '¡': 2, "'": 3, '(': 4, ')': 5, ',': 6, '-': 7, '.': 8, ':': 9, ';': 10, '¿': 11, '?': 12, ' ': 13, '"': 14, '\n': 15, 'A': 16, 'B': 17, 'C': 18, 'D': 19, 'E': 20, 'F': 21, 'G': 22, 'H': 23, 'I': 24, 'J': 25, 'K': 26, 'L': 27, 'M': 28, 'N': 29, 'O': 30, 'P': 31, 'Q': 32, 'R': 33, 'S': 34, 'T': 35, 'U': 36, 'V': 37, 'W': 38, 'X': 39, 'Y': 40, 'Z': 41, 'Ç': 42, 'Á': 43, 'É': 44, 'Í': 45, 'Ó': 46, 'Ú': 47, 'Ü': 48, 'a': 49, 'b': 50, 'c': 51, 'd': 52, 'e': 53, 'f': 54, 'g': 55, 'h': 56, 'i': 57, 'j': 58, 'k': 59, 'l': 60, 'm': 61, 'n': 62, 'o': 63, 'p': 64, 'q': 65, 'r': 66, 's': 67, 't': 68, 'u': 69, 'v': 70, 'w': 71, 'x': 72, 'y': 73, 'z': 74, 'ñ': 75, 'á': 76, 'é': 77, 'í': 78, 'ó': 79, 'ú': 80, 'ü': 81, '<BLNK>': 82}, pad='<PAD>', eos='<EOS>', bos='<BOS>', blank='<BLNK>', blank_word=None), add_blank_char=True, add_blank_word=False, use_eos_bos=False, blank_at_end=True, blank_at_start=True), blank_at_start=True, blank_at_end=True, pad_token=None, blank_token=None, bos_token=None, eos_token=None, word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.COTOVIA: 'cotovia'>, alphabet=<Alphabet.COTOVIA: 'cotovia'>, engine=<Engine.COQUI: 'coqui'>, vocab_override={})
        # TTSModelInfo(voice_id='proxectonos/celtia-cotovia', lang='gl-ES', model_url='https://huggingface.co/OpenVoiceOS/proxectonos-celtia-vits-phonemes-onnx/resolve/main/model.onnx', config_url='https://huggingface.co/OpenVoiceOS/proxectonos-celtia-vits-phonemes-onnx/resolve/main/config.json', vocab_url=None, tokenizer_config_url=None, tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=0, num_langs=1, sample_rate=16000, lang_code='gl-ES', phoneme_type='cotovia', alphabet='cotovia', phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine='coqui', length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8, add_diacritics=False, tokenizer=TTSTokenizer(vocabulary=Vocabulary(char2idx={'<PAD>': 0, '!': 1, '¡': 2, "'": 3, '(': 4, ')': 5, ',': 6, '-': 7, '.': 8, ':': 9, ';': 10, '¿': 11, '?': 12, ' ': 13, '"': 14, '\n': 15, 'A': 16, 'B': 17, 'C': 18, 'D': 19, 'E': 20, 'F': 21, 'G': 22, 'H': 23, 'I': 24, 'J': 25, 'K': 26, 'L': 27, 'M': 28, 'N': 29, 'O': 30, 'P': 31, 'Q': 32, 'R': 33, 'S': 34, 'T': 35, 'U': 36, 'V': 37, 'W': 38, 'X': 39, 'Y': 40, 'Z': 41, 'Ç': 42, 'Á': 43, 'É': 44, 'Í': 45, 'Ó': 46, 'Ú': 47, 'Ü': 48, 'a': 49, 'b': 50, 'c': 51, 'd': 52, 'e': 53, 'f': 54, 'g': 55, 'h': 56, 'i': 57, 'j': 58, 'k': 59, 'l': 60, 'm': 61, 'n': 62, 'o': 63, 'p': 64, 'q': 65, 'r': 66, 's': 67, 't': 68, 'u': 69, 'v': 70, 'w': 71, 'x': 72, 'y': 73, 'z': 74, 'ñ': 75, 'á': 76, 'é': 77, 'í': 78, 'ó': 79, 'ú': 80, 'ü': 81, '<BLNK>': 82}, pad='<PAD>', eos='<EOS>', bos='<BOS>', blank='<BLNK>', blank_word=None), add_blank_char=True, add_blank_word=False, use_eos_bos=False, blank_at_end=True, blank_at_start=True), blank_at_start=True, blank_at_end=True, pad_token=None, blank_token=None, bos_token=None, eos_token=None, word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.COTOVIA: 'cotovia'>, alphabet=<Alphabet.COTOVIA: 'cotovia'>, engine=<Engine.COQUI: 'coqui'>, vocab_override={})
        # TTSModelInfo(voice_id='proxectonos/sabela-cotovia', lang='gl-ES', model_url='https://huggingface.co/OpenVoiceOS/proxectonos-sabela-vits-phonemes-onnx/resolve/main/model.onnx', config_url='https://huggingface.co/OpenVoiceOS/proxectonos-sabela-vits-phonemes-onnx/resolve/main/config.json', vocab_url=None, tokenizer_config_url=None, tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=0, num_langs=1, sample_rate=16000, lang_code='gl-ES', phoneme_type='cotovia', alphabet='cotovia', phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine='coqui', length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8, add_diacritics=False, tokenizer=TTSTokenizer(vocabulary=Vocabulary(char2idx={'<PAD>': 0, '!': 1, '¡': 2, "'": 3, '(': 4, ')': 5, ',': 6, '-': 7, '.': 8, ':': 9, ';': 10, '¿': 11, '?': 12, ' ': 13, '"': 14, '\n': 15, 'A': 16, 'B': 17, 'C': 18, 'D': 19, 'E': 20, 'F': 21, 'G': 22, 'H': 23, 'I': 24, 'J': 25, 'K': 26, 'L': 27, 'M': 28, 'N': 29, 'O': 30, 'P': 31, 'Q': 32, 'R': 33, 'S': 34, 'T': 35, 'U': 36, 'V': 37, 'W': 38, 'X': 39, 'Y': 40, 'Z': 41, 'Ç': 42, 'Á': 43, 'É': 44, 'Í': 45, 'Ó': 46, 'Ú': 47, 'Ü': 48, 'a': 49, 'b': 50, 'c': 51, 'd': 52, 'e': 53, 'f': 54, 'g': 55, 'h': 56, 'i': 57, 'j': 58, 'k': 59, 'l': 60, 'm': 61, 'n': 62, 'o': 63, 'p': 64, 'q': 65, 'r': 66, 's': 67, 't': 68, 'u': 69, 'v': 70, 'w': 71, 'x': 72, 'y': 73, 'z': 74, 'ñ': 75, 'á': 76, 'é': 77, 'í': 78, 'ó': 79, 'ú': 80, 'ü': 81, '<BLNK>': 82}, pad='<PAD>', eos='<EOS>', bos='<BOS>', blank='<BLNK>', blank_word=None), add_blank_char=True, add_blank_word=False, use_eos_bos=False, blank_at_end=True, blank_at_start=True), blank_at_start=True, blank_at_end=True, pad_token=None, blank_token=None, bos_token=None, eos_token=None, word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.COTOVIA: 'cotovia'>, alphabet=<Alphabet.COTOVIA: 'cotovia'>, engine=<Engine.COQUI: 'coqui'>, vocab_override={})
        # TTSModelInfo(voice_id='proxectonos/icia-cotovia', lang='gl-ES', model_url='https://huggingface.co/OpenVoiceOS/proxectonos-icia-vits-phonemes-onnx/resolve/main/model.onnx', config_url='https://huggingface.co/OpenVoiceOS/proxectonos-icia-vits-phonemes-onnx/resolve/main/config.json', vocab_url=None, tokenizer_config_url=None, tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=0, num_langs=1, sample_rate=16000, lang_code='gl-ES', phoneme_type='cotovia', alphabet='cotovia', phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine='coqui', length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8, add_diacritics=False, tokenizer=TTSTokenizer(vocabulary=Vocabulary(char2idx={'<PAD>': 0, '<EOS>': 1, '<BOS>': 2, '<BLNK>': 3, '\n': 4, '"': 5, 'A': 6, 'B': 7, 'C': 8, 'D': 9, 'E': 10, 'F': 11, 'G': 12, 'H': 13, 'I': 14, 'J': 15, 'K': 16, 'L': 17, 'M': 18, 'N': 19, 'O': 20, 'P': 21, 'Q': 22, 'R': 23, 'S': 24, 'T': 25, 'U': 26, 'V': 27, 'W': 28, 'X': 29, 'Y': 30, 'Z': 31, 'a': 32, 'b': 33, 'c': 34, 'd': 35, 'e': 36, 'f': 37, 'g': 38, 'h': 39, 'i': 40, 'j': 41, 'k': 42, 'l': 43, 'm': 44, 'n': 45, 'o': 46, 'p': 47, 'q': 48, 'r': 49, 's': 50, 't': 51, 'u': 52, 'v': 53, 'w': 54, 'x': 55, 'y': 56, 'z': 57, 'Á': 58, 'Ç': 59, 'É': 60, 'Í': 61, 'Ó': 62, 'Ú': 63, 'Ü': 64, 'á': 65, 'é': 66, 'í': 67, 'ñ': 68, 'ó': 69, 'ú': 70, 'ü': 71, '!': 72, '¡': 73, "'": 74, '(': 75, ')': 76, ',': 77, '-': 78, '.': 79, ':': 80, ';': 81, '¿': 82, '?': 83, ' ': 84}, pad='<PAD>', eos='<EOS>', bos='<BOS>', blank='<BLNK>', blank_word=None), add_blank_char=True, add_blank_word=False, use_eos_bos=False, blank_at_end=True, blank_at_start=True), blank_at_start=True, blank_at_end=True, pad_token=None, blank_token=None, bos_token=None, eos_token=None, word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.COTOVIA: 'cotovia'>, alphabet=<Alphabet.COTOVIA: 'cotovia'>, engine=<Engine.COQUI: 'coqui'>, vocab_override={})
        # TTSModelInfo(voice_id='proxectonos/paulo-cotovia', lang='gl-ES', model_url='https://huggingface.co/OpenVoiceOS/proxectonos-paulo-vits-phonemes-onnx/resolve/main/model.onnx', config_url='https://huggingface.co/OpenVoiceOS/proxectonos-paulo-vits-phonemes-onnx/resolve/main/config.json', vocab_url=None, tokenizer_config_url=None, tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=0, num_langs=1, sample_rate=16000, lang_code='gl-ES', phoneme_type='cotovia', alphabet='cotovia', phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine='coqui', length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8, add_diacritics=False, tokenizer=TTSTokenizer(vocabulary=Vocabulary(char2idx={'<PAD>': 0, '<EOS>': 1, '<BOS>': 2, '<BLNK>': 3, 'A': 4, 'B': 5, 'C': 6, 'D': 7, 'E': 8, 'F': 9, 'G': 10, 'H': 11, 'I': 12, 'J': 13, 'K': 14, 'L': 15, 'M': 16, 'N': 17, 'O': 18, 'P': 19, 'Q': 20, 'R': 21, 'S': 22, 'T': 23, 'U': 24, 'V': 25, 'W': 26, 'X': 27, 'Y': 28, 'Z': 29, 'a': 30, 'b': 31, 'c': 32, 'd': 33, 'e': 34, 'f': 35, 'g': 36, 'h': 37, 'i': 38, 'j': 39, 'k': 40, 'l': 41, 'm': 42, 'n': 43, 'o': 44, 'p': 45, 'q': 46, 'r': 47, 's': 48, 't': 49, 'u': 50, 'v': 51, 'w': 52, 'x': 53, 'y': 54, 'z': 55, 'Á': 56, 'Ç': 57, 'É': 58, 'Í': 59, 'Ó': 60, 'Ú': 61, 'Ü': 62, 'á': 63, 'é': 64, 'í': 65, 'ñ': 66, 'ó': 67, 'ú': 68, 'ü': 69, '!': 70, '¡': 71, "'": 72, '(': 73, ')': 74, ',': 75, '-': 76, '.': 77, ':': 78, ';': 79, '¿': 80, '?': 81, ' ': 82, '"': 83, '\n': 84}, pad='<PAD>', eos='<EOS>', bos='<BOS>', blank='<BLNK>', blank_word=None), add_blank_char=True, add_blank_word=False, use_eos_bos=False, blank_at_end=True, blank_at_start=True), blank_at_start=True, blank_at_end=True, pad_token=None, blank_token=None, bos_token=None, eos_token=None, word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.COTOVIA: 'cotovia'>, alphabet=<Alphabet.COTOVIA: 'cotovia'>, engine=<Engine.COQUI: 'coqui'>, vocab_override={})
        # TTSModelInfo(voice_id='proxectonos/iago-cotovia', lang='gl-ES', model_url='https://huggingface.co/OpenVoiceOS/proxectonos-iago-vits-phonemes-onnx/resolve/main/model.onnx', config_url='https://huggingface.co/OpenVoiceOS/proxectonos-iago-vits-phonemes-onnx/resolve/main/config.json', vocab_url=None, tokenizer_config_url=None, tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=0, num_langs=1, sample_rate=16000, lang_code='gl-ES', phoneme_type='cotovia', alphabet='cotovia', phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine='coqui', length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8, add_diacritics=False, tokenizer=TTSTokenizer(vocabulary=Vocabulary(char2idx={'<PAD>': 0, '<EOS>': 1, '<BOS>': 2, '<BLNK>': 3, 'A': 4, 'B': 5, 'C': 6, 'D': 7, 'E': 8, 'F': 9, 'G': 10, 'H': 11, 'I': 12, 'J': 13, 'K': 14, 'L': 15, 'M': 16, 'N': 17, 'O': 18, 'P': 19, 'Q': 20, 'R': 21, 'S': 22, 'T': 23, 'U': 24, 'V': 25, 'W': 26, 'X': 27, 'Y': 28, 'Z': 29, 'a': 30, 'b': 31, 'c': 32, 'd': 33, 'e': 34, 'f': 35, 'g': 36, 'h': 37, 'i': 38, 'j': 39, 'k': 40, 'l': 41, 'm': 42, 'n': 43, 'o': 44, 'p': 45, 'q': 46, 'r': 47, 's': 48, 't': 49, 'u': 50, 'v': 51, 'w': 52, 'x': 53, 'y': 54, 'z': 55, 'Á': 56, 'Ç': 57, 'É': 58, 'Í': 59, 'Ó': 60, 'Ú': 61, 'Ü': 62, 'á': 63, 'é': 64, 'í': 65, 'ñ': 66, 'ó': 67, 'ú': 68, 'ü': 69, '!': 70, '¡': 71, "'": 72, '(': 73, ')': 74, ',': 75, '-': 76, '.': 77, ':': 78, ';': 79, '¿': 80, '?': 81, ' ': 82, '"': 83, '\n': 84}, pad='<PAD>', eos='<EOS>', bos='<BOS>', blank='<BLNK>', blank_word=None), add_blank_char=True, add_blank_word=False, use_eos_bos=False, blank_at_end=True, blank_at_start=True), blank_at_start=True, blank_at_end=True, pad_token=None, blank_token=None, bos_token=None, eos_token=None, word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.COTOVIA: 'cotovia'>, alphabet=<Alphabet.COTOVIA: 'cotovia'>, engine=<Engine.COQUI: 'coqui'>, vocab_override={})

    print(manager.supported_langs)
    # ['abi', 'abp', 'aca', 'acd', 'ace', 'acf', 'ach', 'acn', 'acr', 'acu', 'ade', 'adh', 'adj', 'adx', 'aeu',
    # 'af-ZA', 'agd', 'agg', 'agn', 'agr', 'agu', 'agx', 'aha', 'ahk', 'aia', 'aka', 'akb', 'ake', 'akp', 'alj',
    # 'alp', 'alt', 'alz', 'ame', 'amf', 'amh', 'ami', 'amk', 'ann', 'any', 'aoz', 'apb', 'apr', 'ar', 'ar-JO',
    # 'ar-SA', 'ara', 'arl', 'asa', 'asg', 'asm', 'ata', 'atb', 'atg', 'ati', 'atq', 'ava', 'avn', 'avu', 'awa', 'awb',
    # 'ayo', 'ayr', 'ayz', 'azb', 'azg', 'azj-script_cyrillic', 'azj-script_latin', 'azz', 'bak', 'bam', 'ban', 'bao',
    # 'bav', 'bba', 'bbb', 'bbc', 'bbo', 'bcc-script_arabic', 'bcc-script_latin', 'bcl', 'bcw', 'bdg', 'bdh', 'bdq',
    # 'bdu', 'bdv', 'beh', 'bem', 'ben', 'bep', 'bex', 'bfa', 'bfo', 'bfy', 'bfz', 'bg-BG', 'bgc', 'bgq', 'bgr', 'bgt',
    # 'bgw', 'bha', 'bht', 'bhz', 'bib', 'bim', 'bis', 'biv', 'bjr', 'bjv', 'bjw', 'bjz', 'bkd', 'bkv', 'blh', 'blt',
    # 'blx', 'blz', 'bmq', 'bmr', 'bmu', 'bmv', 'bn', 'bng', 'bno', 'bnp', 'boa', 'bod', 'boj', 'bom', 'bor', 'bov',
    # 'box', 'bpr', 'bps', 'bqc', 'bqi', 'bqj', 'bqp', 'bru', 'bsc', 'bsq', 'bss', 'btd', 'bts', 'btt', 'btx', 'bud',
    # 'bul', 'bus', 'bvc', 'bvz', 'bwq', 'bwu', 'byr', 'bzh', 'bzi', 'bzj', 'ca-ES', 'caa', 'cab',
    # 'cak-dialect_central', 'cak-dialect_santodomingoxenacoj', 'cak-dialect_southcentral', 'cak-dialect_western',
    # 'cak-dialect_yepocapa', 'cap', 'car', 'cas', 'cat', 'cax', 'cbc', 'cbi', 'cbr', 'cbs', 'cbt', 'cbu', 'cbv', 'cce',
    # 'cco', 'cdj', 'ceb', 'ceg', 'cek', 'cfm', 'cgc', 'che', 'chf', 'chv', 'chz', 'cjo', 'cjp', 'cjs', 'cko', 'ckt',
    # 'cla', 'cle', 'cly', 'cme', 'cmo-script_khmer', 'cmo-script_latin', 'cmr', 'cnh', 'cni', 'cnl', 'cnt', 'coe',
    # 'cof', 'cok', 'con', 'cot', 'cou', 'cpa', 'cpb', 'cpu', 'crh', 'crk-script_latin', 'crk-script_syllabics', 'crn',
    # 'crq', 'crs', 'crt', 'cs-CZ', 'csk', 'cso', 'ctd', 'ctg', 'cto', 'ctu', 'cuc', 'cui', 'cuk', 'cul', 'cwa', 'cwe',
    # 'cwt', 'cy-GB', 'cya', 'cym', 'da-DK', 'daa', 'dah', 'dar', 'dbj', 'dbq', 'ddn', 'de-DE', 'ded', 'des', 'deu',
    # 'dga', 'dgi', 'dgk', 'dgo', 'dgr', 'dhi', 'did', 'dig', 'dik', 'dip', 'div', 'djk', 'dnj-dialect_blowowest',
    # 'dnj-dialect_gweetaawueast', 'dnt', 'dnw', 'dop', 'dos', 'dsh', 'dso', 'dtp', 'dts', 'dug', 'dwr', 'dyi', 'dyo',
    # 'dyu', 'dzo', 'eip', 'eka', 'el-GR', 'ell', 'emp', 'en', 'en-GB', 'en-IE', 'en-US', 'en-cy', 'enb', 'eng', 'enx',
    # 'es-AR', 'es-CL', 'es-CO', 'es-ES', 'es-MX', 'ese', 'ess', 'eu-ES', 'eus', 'evn', 'ewe', 'eza', 'fa', 'fa-IR',
    # 'fal', 'fao', 'far', 'fas', 'fi-FI', 'fij', 'fin', 'flr', 'fmu', 'fon', 'fr-FR', 'fra', 'frd', 'ful',
    # 'gag-script_cyrillic', 'gag-script_latin', 'gai', 'gam', 'gau', 'gbi', 'gbk', 'gbm', 'gbo', 'gde', 'geb', 'gej',
    # 'gil', 'gjn', 'gkn', 'gl-ES', 'gld', 'glk', 'gmv', 'gna', 'gnd', 'gng', 'gof-script_latin', 'gog', 'gor', 'gqr',
    # 'grc', 'gri', 'grn', 'grt', 'gso', 'gu-IN', 'gub', 'guc', 'gud', 'guh', 'guj', 'guk', 'gum', 'guo', 'guq', 'guu',
    # 'gux', 'gvc', 'gvl', 'gwi', 'gwr', 'gym', 'gyr', 'ha-NE', 'had', 'hag', 'hak', 'hap', 'hat', 'hau', 'hay', 'he',
    # 'heb', 'heh', 'hi-IN', 'hif', 'hig', 'hil', 'hin', 'hlb', 'hlt', 'hne', 'hnn', 'hns', 'hoc', 'hoy', 'hto', 'hu-HU',
    # 'hub', 'hui', 'hun', 'hus-dialect_centralveracruz', 'hus-dialect_westernpotosino', 'huu', 'huv', 'hvn', 'hwc',
    # 'hy-AM', 'hyw', 'iba', 'icr', 'id-ID', 'idd', 'ifa', 'ifb', 'ife', 'ifk', 'ifu', 'ify', 'ign', 'ikk', 'ilb', 'ilo',
    # 'imo', 'inb', 'ind', 'iou', 'ipi', 'iqw', 'iri', 'irk', 'is-IS', 'isl', 'it-IT', 'itl', 'itv',
    # 'ixl-dialect_sangasparchajul', 'ixl-dialect_sanjuancotzal', 'ixl-dialect_santamarianebaj', 'izr', 'izz', 'jac',
    # 'jam', 'jav', 'jbu', 'jen', 'jic', 'jiv', 'jmc', 'jmd', 'jun', 'juy', 'jv-ID', 'jvn', 'ka-GE', 'kaa', 'kab',
    # 'kac', 'kak', 'kan', 'kao', 'kaq', 'kay', 'kaz', 'kbo', 'kbp', 'kbq', 'kbr', 'kby', 'kca', 'kcg', 'kdc', 'kde',
    # 'kdh', 'kdi', 'kdj', 'kdl', 'kdn', 'kdt', 'kek', 'ken', 'keo', 'ker', 'key', 'kez', 'kfb', 'kff-script_telugu',
    # 'kfw', 'kfx', 'khg', 'khm', 'khq', 'kia', 'kij', 'kik', 'kin', 'kir', 'kjb', 'kje', 'kjg', 'kjh', 'kk-KZ', 'kki',
    # 'kkj', 'kle', 'klu', 'klv', 'klw', 'kma', 'kmd', 'kml', 'kmr-script_arabic', 'kmr-script_cyrillic',
    # 'kmr-script_latin', 'kmu', 'knb', 'kne', 'knf', 'knj', 'knk', 'kno', 'ko-KO', 'kog', 'kor', 'kpq', 'kps', 'kpv',
    # 'kpy', 'kpz', 'kqe', 'kqp', 'kqr', 'kqy', 'krc', 'kri', 'krj', 'krl', 'krr', 'krs', 'kru', 'ksb', 'ksr', 'kss',
    # 'ktb', 'ktj', 'kub', 'kue', 'kum', 'kus', 'kvn', 'kvw', 'kwd', 'kwf', 'kwi', 'kxc', 'kxf', 'kxm', 'kxv', 'kyb',
    # 'kyc', 'kyf', 'kyg', 'kyo', 'kyq', 'kyu', 'kyz', 'kzf', 'lac', 'laj', 'lam', 'lao', 'las', 'lat', 'lav', 'law',
    # 'lb-LU', 'lbj', 'lbw', 'lcp', 'lee', 'lef', 'lem', 'lew', 'lex', 'lgg', 'lgl', 'lhu', 'lia', 'lid', 'lif', 'lip',
    # 'lis', 'lje', 'ljp', 'llg', 'lln', 'lme', 'lnd', 'lns', 'lob', 'lok', 'lom', 'lon', 'loq', 'lsi', 'lsm', 'luc',
    # 'lug', 'lv-LV', 'lwo', 'lww', 'lzz', 'maa-dialect_sanantonio', 'mad', 'mag', 'mah', 'mai', 'maj', 'mak', 'mal',
    # 'mam-dialect_central', 'mam-dialect_northern', 'mam-dialect_southern', 'mam-dialect_western', 'maq', 'mar', 'maw',
    # 'maz', 'mbb', 'mbc', 'mbh', 'mbj', 'mbt', 'mbu', 'mbz', 'mca', 'mcb', 'mcd', 'mco', 'mcp', 'mcq', 'mcu', 'mda',
    # 'mdv', 'mdy', 'med', 'mee', 'mej', 'men', 'meq', 'met', 'mev', 'mfe', 'mfh', 'mfi', 'mfk', 'mfq', 'mfy', 'mfz',
    # 'mgd', 'mge', 'mgh', 'mgo', 'mhi', 'mhr', 'mhu', 'mhx', 'mhy', 'mib', 'mie', 'mif', 'mih', 'mil', 'mim', 'min',
    # 'mio', 'mip', 'miq', 'mit', 'miy', 'miz', 'mjl', 'mjv', 'mkl', 'mkn', 'ml-IN', 'mlg', 'mmg', 'mnb', 'mnf', 'mnk',
    # 'mnw', 'mnx', 'moa', 'mog', 'mon', 'mop', 'mor', 'mos', 'mox', 'moz', 'mpg', 'mpm', 'mpp', 'mpx', 'mqb', 'mqf',
    # 'mqj', 'mqn', 'mrw', 'msy', 'mtd', 'mtj', 'mto', 'muh', 'mup', 'mur', 'muv', 'muy', 'mvp', 'mwq', 'mwv', 'mxb',
    # 'mxq', 'mxt', 'mxv', 'mya', 'myb', 'myk', 'myl', 'myv', 'myx', 'myy', 'mza', 'mzi', 'mzj', 'mzk', 'mzm', 'mzw',
    # 'nab', 'nag', 'nan', 'nas', 'naw', 'nca', 'nch', 'ncj', 'ncl', 'ncu', 'ndj', 'ndp', 'ndv', 'ndy', 'ndz', 'ne-NP',
    # 'neb', 'new', 'nfa', 'nfr', 'nga', 'ngl', 'ngp', 'ngu', 'nhe', 'nhi', 'nhu', 'nhw', 'nhx', 'nhy', 'nia', 'nij',
    # 'nim', 'nin', 'nko', 'nl', 'nl-BE', 'nl-NL', 'nlc', 'nld', 'nlg', 'nlk', 'nmz', 'nnb', 'nnq', 'nnw', 'no-NO',
    # 'noa', 'nod', 'nog', 'not', 'npl', 'npy', 'nst', 'nsu', 'ntm', 'ntr', 'nuj', 'nus', 'nuz', 'nwb', 'nxq', 'nya',
    # 'nyf', 'nyn', 'nyo', 'nyy', 'nzi', 'obo', 'ojb-script_latin', 'ojb-script_syllabics', 'oku', 'old', 'omw', 'onb',
    # 'ood', 'orm', 'ory', 'oss', 'ote', 'otq', 'ozm', 'pab', 'pad', 'pag', 'pam', 'pan', 'pao', 'pap', 'pau', 'pbb',
    # 'pbc', 'pbi', 'pce', 'pcm', 'peg', 'pez', 'pib', 'pil', 'pir', 'pis', 'pjt', 'pkb', 'pl-PL', 'pls', 'plw', 'pmf',
    # 'pny', 'poh-dialect_eastern', 'poh-dialect_western', 'poi', 'pol', 'por', 'poy', 'ppk', 'pps', 'prf', 'prk', 'prt',
    # 'pse', 'pss', 'pt-BR', 'pt-PT', 'ptu', 'pui', 'pwg', 'pww', 'pxm', 'qub', 'quc-dialect_central', 'quc-dialect_east',
    # 'quc-dialect_north', 'quf', 'quh', 'qul', 'quw', 'quy', 'quz', 'qvc', 'qve', 'qvh', 'qvm', 'qvn', 'qvo', 'qvs',
    # 'qvw', 'qvz', 'qwh', 'qxh', 'qxl', 'qxn', 'qxo', 'qxr', 'rah', 'rai', 'rap', 'rav', 'raw', 'rej', 'rel', 'rgu',
    # 'rhg', 'rif-script_arabic', 'rif-script_latin', 'ril', 'rim', 'rjs', 'rkt', 'rmc-script_cyrillic',
    # 'rmc-script_latin', 'rmo', 'rmy-script_cyrillic', 'rmy-script_latin', 'rng', 'rnl', 'ro-RO', 'rol', 'ron', 'rop',
    # 'rro', 'ru-RU', 'rub', 'ruf', 'rug', 'run', 'rus', 'sab', 'sag', 'sah', 'saj', 'saq', 'sas', 'sba', 'sbd', 'sbl',
    # 'sbp', 'sch', 'sck', 'sda', 'sea', 'seh', 'ses', 'sey', 'sgb', 'sgj', 'sgw', 'shi', 'shk', 'shn', 'sho', 'shp',
    # 'sid', 'sig', 'sil', 'sja', 'sjm', 'sk-SK', 'sl-SI', 'sld', 'slu', 'sml', 'smo', 'sna', 'sne', 'snn', 'snp',
    # 'snw', 'som', 'soy', 'spa', 'spp', 'spy', 'sqi', 'sr-RS', 'sri', 'srm', 'srn', 'srx', 'stn', 'stp', 'suc', 'suk',
    # 'sun', 'sur', 'sus', 'suv', 'suz', 'sv-SE', 'sw', 'sw-CD', 'swe', 'swh', 'sxb', 'sxn', 'sya', 'syl', 'sza', 'tac',
    # 'taj', 'tam', 'tao', 'tap', 'taq', 'tat', 'tav', 'tbc', 'tbg', 'tbk', 'tbl', 'tby', 'tbz', 'tca', 'tcc', 'tcs',
    # 'tcz', 'tdj', 'tdt-TL', 'te-IN', 'ted', 'tee', 'tel', 'tem', 'teo', 'ter', 'tes', 'tew', 'tex', 'tfr', 'tgj',
    # 'tgk', 'tgl', 'tgo', 'tgp', 'tha', 'thk', 'thl', 'tih', 'tik', 'tir', 'tkr', 'tlb', 'tlj', 'tly', 'tmc', 'tmf',
    # 'tn-ZA', 'tna', 'tng', 'tnk', 'tnn', 'tnp', 'tnr', 'tnt', 'tob', 'toc', 'toh', 'tom', 'tos', 'tpi', 'tpm', 'tpp',
    # 'tpt', 'tr-TR', 'trc', 'tri', 'trn', 'trs', 'tso', 'tsz', 'ttc', 'tte', 'ttq-script_tifinagh', 'tue', 'tuf',
    # 'tuk-script_arabic', 'tuk-script_latin', 'tuo', 'tur', 'tvw', 'twb', 'twe', 'twu', 'txa', 'txq', 'txu', 'tye',
    # 'tzh-dialect_tenejapa', 'tzj-dialect_eastern', 'tzj-dialect_western', 'tzo-dialect_chamula', 'udm', 'udu',
    # 'uig-script_arabic', 'uig-script_cyrillic', 'uk-GB', 'uk-UA', 'ukr', 'unr', 'upv', 'ura', 'urb',
    # 'urd-script_arabic', 'urd-script_devanagari', 'urd-script_latin', 'urk', 'urt', 'ury', 'usp',
    # 'uzb-script_cyrillic', 'vag', 'vi-VN', 'vid', 'vie', 'vif', 'vmw', 'vmy', 'vun', 'vut', 'wal-script_ethiopic',
    # 'wal-script_latin', 'wap', 'war', 'waw', 'way', 'wba', 'wlo', 'wlx', 'wmw', 'wob', 'wsg', 'wwa', 'xal', 'xdy',
    # 'xed', 'xer', 'xmm', 'xnj', 'xnr', 'xog', 'xon', 'xrb', 'xsb', 'xsm', 'xsr', 'xsu', 'xta', 'xtd', 'xte', 'xtm',
    # 'xtn', 'xua', 'xuo', 'yaa', 'yad', 'yal', 'yam', 'yao', 'yas', 'yat', 'yaz', 'yba', 'ybb', 'ycl', 'ycn', 'yea',
    # 'yka', 'yli', 'yo', 'yor', 'yre', 'yua', 'yuz', 'yva', 'zaa', 'zab', 'zac', 'zad', 'zae', 'zai', 'zam', 'zao',
    # 'zaq', 'zar', 'zas', 'zav', 'zaw', 'zca', 'zga', 'zh-CN', 'zh-TW', 'zim', 'ziw', 'zlm', 'zmz', 'zne', 'zos',
    # 'zpc', 'zpg', 'zpi', 'zpl', 'zpm', 'zpo', 'zpt', 'zpu', 'zpz', 'ztq', 'zty', 'zyb', 'zyp', 'zza']


    def generate_voices_markdown(manager: TTSModelManager, output_file: str = "../VOICES.md"):
        """
        Generate a Markdown table listing all registered voices and save it to a file.
        
        Creates a sorted table of all voices (by language code then voice ID) including voice ID, language code, engine, and phoneme type, writes the table to output_file, and prints a success message. If writing the file fails due to an I/O error, prints the error and emits the generated Markdown to stdout.
        
        Parameters:
            manager (TTSModelManager): Manager containing the loaded voices to document.
            output_file (str): Path to the file where the Markdown table will be written.
        """

        # Sort voices by language code (lang) then by voice ID
        sorted_voices = sorted(
            manager.all_voices,
            key=lambda v: (v.lang.lower(), v.voice_id.lower())
        )

        markdown_output = [
            "## Supported Voices",
            "",
            f"**Total Voices:** {len(sorted_voices)}",
            f"**Total Languages:** {len(manager.supported_langs)}",
            "",
            "> ⚠️ some languages are duplicated, either using a different script or less specific language code (eg. Kurdish is available in latin, cyrillic and arabic scripts)",
            "",
            "> ⚠️ some models are duplicated, piper is known to mirror community voices and those will often show up twice (under piper and under original author)",
            "",
            "| Voice ID | Language Code | Engine | Phoneme Type |",
            "| :--- | :--- | :--- | :--- |"
        ]

        for voice in sorted_voices:
            # Get the value from the Enum object, handling None/string if post_init didn't run
            engine = voice.engine.value if hasattr(voice.engine, 'value') else str(voice.engine)
            phoneme_type = voice.phoneme_type.value if hasattr(voice.phoneme_type, 'value') else str(voice.phoneme_type)

            # Format the row for the markdown table
            row = (
                f"| `{voice.voice_id}` | `{voice.lang}` | "
                f"`{engine}` | `{phoneme_type}` |"
            )
            markdown_output.append(row)

        # Write the output to the specified file
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(markdown_output))
            print(f"\n✅ Successfully generated and saved voice table to: **{output_file}**")
        except IOError as e:
            print(f"\n❌ Error writing to file {output_file}: {e}")
            print("\n--- Start of Markdown Output ---")
            print("\n".join(markdown_output))
            print("--- End of Markdown Output ---")


    generate_voices_markdown(manager, output_file="../VOICES.md")