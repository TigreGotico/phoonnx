import contextlib
import contextvars
import functools
import hashlib
import json
import os
import posixpath
import re
import tempfile
import threading
import warnings
from dataclasses import asdict, dataclass, field, fields, replace
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Iterator, List, Any, Callable, Sequence, Set, Tuple
from urllib.parse import quote, unquote
import requests
from huggingface_hub import constants as hub_constants, hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import (EntryNotFoundError, LocalEntryNotFoundError,
                                    OfflineModeIsEnabled)
from json_database import JsonStorageXDG, JsonStorage, load_commented_json

from phoonnx.config import (PhonemeType, _removal_version, _voice_phonemizer, VoiceConfig, Engine,
                            Alphabet, check_lang_supported)
from phoonnx.util import match_lang, normalize_lang, LOG
from phoonnx.providers import ProviderSpec, make_session, resolve_providers, uses_gpu
from phoonnx.voice import TTSVoice
from phoonnx._onnx_external import external_data_locations


def _tmp_path(dest: Path) -> Path:
    """Sibling scratch path used while a download is in flight.

    The name is unique per call rather than "<target>.part": two fetches of the
    same artifact would otherwise share one temporary file, interleaving their
    bytes, and whichever lost the race would find it already renamed away.

    Every write goes through here, so this is also where the voice directory is
    created — a voice's directory should exist because something was written
    into it, not because its catalog entry was constructed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".",
                                   suffix=".part")
    os.close(handle)
    # mkstemp creates 0600 and os.replace carries the mode onto the artifact.
    # Models are read by whoever serves them, which is not always the account
    # that downloaded them: a shared HF_HOME, or a prefetch run as another user.
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(tmp, 0o666 & ~umask)
    return Path(tmp)


def _parses(path: str) -> bool:
    """Whether the index on disk is readable, by reading it.

    An empty registry has two causes that look identical in memory: a file that
    is genuinely empty, and one the storage layer could not parse. Only the
    second is worth telling an operator about, and a size comparison answers
    neither — ``{}\n`` is three bytes and perfectly valid, an empty file is
    zero and is not.
    """
    try:
        load_commented_json(path)
        return True
    except Exception:
        return False


def _is_cached(path: Path) -> bool:
    """A cached artifact counts only if it exists and is non-empty."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


# The hub is asked for a file's metadata before its body. That call is one
# small request, so it should give up long before the body timeout would: a
# blackholed network otherwise stalls the caller for the full body budget
# before the download it is blocking has even started.
_ETAG_TIMEOUT = 10


def _sidecar_url(url: str) -> str:
    """The external-weights URL for ``url``, keeping any query intact.

    The suffix belongs on the path, not on the end of the string: appending it
    to ``model.onnx?download=true`` yields ``download=true_data``, a query the
    hub ignores, so the request comes back as the graph itself. The sidecar
    then looks like it was fetched when it never was, and the graph is handed
    to onnxruntime with its weights missing.
    """
    head, sep, tail = url.partition("?")
    return f"{head}_data{sep}{tail}"


def _sidecar_urls(url: str, graph: Optional[Path]) -> List[str]:
    """The URLs of the external-data files that the graph at ``graph`` names.

    onnxruntime reads each file from the folder of the graph, by the name the
    graph gives it, so the files come from the folder of ``url``. Exporters do
    not agree on the names: ``model.onnx_data``, a series ``model.onnx_data_1``
    to ``_6``, ``model_int8.onnx.data`` from onnxruntime quantization. Only a
    graph that cannot be read falls back to the ``<graph>_data`` guess.

    Raises:
        ValueError: the graph names a file outside its own folder.
    """
    if graph is None:
        return [_sidecar_url(url)]
    try:
        locations = external_data_locations(graph)
    except (OSError, ValueError) as e:
        LOG.debug(f"cannot read the external-data names of '{url}', "
                  f"guessing '<graph>_data': {e}")
        return [_sidecar_url(url)]
    head, sep, tail = url.partition("?")
    folder = head.rsplit("/", 1)[0]
    for location in locations:
        parts = location.replace("\\", "/").split("/")
        if parts[0] == "" or ":" in parts[0] or ".." in parts:
            raise ValueError(f"'{url}' names external data outside its folder: "
                             f"{location!r}")
    if not _HF_URL.match(head):
        # a location is a literal file name, and a URL reads %, # and ? as syntax
        locations = [quote(location, safe="/") for location in locations]
    return [f"{folder}/{location}{sep}{tail}" for location in locations]


def _expected_size(response) -> Optional[int]:
    """Content-Length of a response, or None when absent/unusable.

    A compressed response is a deliberate ``None``: ``Content-Length`` then
    counts the bytes on the wire, while the stream below yields the decoded
    body, so comparing the two condemns a download that arrived intact.
    """
    try:
        headers = response.headers
    except AttributeError:
        return None
    try:
        encoding = (headers.get("Content-Encoding") or "").strip().lower()
    except AttributeError:
        encoding = ""
    if encoding and encoding != "identity":
        return None
    raw = headers.get("Content-Length")
    if not isinstance(raw, (str, int)) or isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None



def _direct_stream(url: str, dest: Path, timeout: int = 120) -> Path:
    """Stream ``url`` into ``dest`` atomically, outside the shared cache.

    The body is written to a sibling ``.part`` file and only renamed onto
    ``dest`` once it has been fully received (and matches ``Content-Length``,
    when the server sends one), so no run ever sees a truncated artifact. A
    download that fails or is interrupted removes its own scratch file; one
    killed outright — SIGKILL, OOM, power loss — leaves it behind, and nothing
    reclaims it, because a sweep on entry cannot tell an abandoned file from
    one another download is still writing.

    This is the path for a self-hosted or mirrored voice, which the hub client
    cannot serve. It gets a private copy.
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


def _direct_dir(url: str) -> Path:
    """Where a file the hub cannot serve is kept.

    Self-hosted and mirrored voices still have to land somewhere, and that
    somewhere has the same requirement as a hub snapshot: a graph and the
    external-data sidecar it names must share one directory, or onnxruntime
    refuses to follow the reference. The directory is keyed by the URL's own
    directory, so files published together stay together.

    It lives inside the HuggingFace cache, not in a phoonnx-owned one. There is
    exactly one place voice files are kept, and ``HF_HOME`` moves all of it.
    """
    base = url.split("?")[0].rsplit("/", 1)[0]
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:32]
    return Path(HF_HUB_CACHE) / "phoonnx-direct" / digest


def _direct_path(url: str, graph_url: Optional[str] = None) -> Path:
    """Where the file at a URL the hub cannot serve is kept.

    The external data of a graph keeps its place relative to the graph, also in a
    subfolder, because onnxruntime reads it there. ``graph_url`` names that graph.
    """
    head = url.split("?")[0]
    if graph_url is None:
        return _direct_dir(url) / (head.rsplit("/", 1)[-1] or "artifact")
    folder = graph_url.split("?")[0].rsplit("/", 1)[0]
    return _direct_dir(graph_url) / unquote(head[len(folder) + 1:])


def _resolve(url: str, timeout: int = 120, graph_url: Optional[str] = None) -> Path:
    """Give the local path of ``url``, downloading it once.

    Hub files are fetched with the hub client and used exactly where it puts
    them: one copy per file and revision, in the cache the rest of the machine
    already shares, so a model named by hundreds of voices is stored once and a
    model another program has already pulled is not pulled again. Nothing is
    copied or linked out of it — a graph that names an external-data sidecar
    only loads from the directory that holds both, and that is the hub's.

    A URL the hub cannot serve is streamed into :func:`_direct_dir` instead,
    with a warning, because breaking self-hosted voices costs more than the
    duplicate copy they keep. ``graph_url`` names the graph whose external data
    ``url`` is, so the file lands beside it (:func:`_direct_path`).

    Raises:
        huggingface_hub.errors.EntryNotFoundError: no such file on the hub.
        requests.exceptions.RequestException: the direct download failed.
    """
    cached = _hf_fetch(url, timeout)
    if cached is not None:
        return cached
    _warn_offhub(url)
    dest = _direct_path(url, graph_url)
    if _is_cached(dest):
        return dest
    return _direct_stream(url, dest, timeout)


_HF_URL = re.compile(
    r"https?://huggingface\.co/(?P<repo>[^/]+/[^/]+)/(?:resolve|raw)/(?P<rev>[^/]+)/(?P<path>.+)")

# Raised by the hub client when the file, not the network, is the problem, and
# when the client is offline with nothing cached. Both mean "no file here".
_NO_SUCH_FILE = (EntryNotFoundError, OfflineModeIsEnabled, LocalEntryNotFoundError)


# The Phoonnx repos lived in OpenVoiceOS under the same names, so a voice fetched before the
# move sits in the cache under the old repo id. While the cache holds the repo under that id,
# every file of the repo is fetched through it: offline it loads from that cache, online the
# Hub redirect serves the current Phoonnx file into the same folder, as before the move. The
# choice is per repo, never per file, so a graph and its weights files share one folder, a
# commit to the moved repo does not make the cached files look missing, and a file another
# tool fetched under the new id does not cut the voice off from its cache. Drop this once
# a stable release has shipped the Phoonnx URLs for a full release cycle.
_MOVED_FROM = {"Phoonnx": "OpenVoiceOS"}
# Repos that also changed name: current id -> the ids it had before, newest first.
_RENAMED_FROM: Dict[str, List[str]] = json.loads(
    (Path(__file__).parent / "moved_repos.json").read_text(encoding="utf-8"))


def _earlier_ids(repo_id: str) -> List[str]:
    """Every id the repo had before, newest first."""
    if repo_id in _RENAMED_FROM:
        return _RENAMED_FROM[repo_id]
    org, name = repo_id.split("/", 1)
    return [f"{_MOVED_FROM[org]}/{name}"] if org in _MOVED_FROM else []


def _repo_cached_before_move(repo_id: str) -> Optional[str]:
    """The newest old repo id that the cache holds the repo under. Reads no network."""
    return next((old for old in _earlier_ids(repo_id)
                 if (Path(HF_HUB_CACHE) / f"models--{old.replace('/', '--')}").is_dir()), None)


# The snapshot an offline load reads in place of a pinned commit the cache does not hold. The
# index pins every file, so a device that cached a voice at an older commit would lose it
# offline after a pin bump; it keeps the voice it has instead. A voice load chooses one
# snapshot per repo and commit that holds every file the voice names there
# (_VOICE_SNAPSHOTS), so a graph and the voice state in another folder of the same repo come
# from the same commit, and two voices cached at different commits each keep their own. A
# file fetched outside a voice load gets one choice per cache folder, pinned commit and
# folder inside the repo (_OFFLINE_SNAPSHOTS), never per file, so a graph and every data
# file it names still come from the same commit.
_VOICE_SNAPSHOTS: Dict[Tuple[Path, str, frozenset], Path] = {}
_OFFLINE_SNAPSHOTS: Dict[Tuple[Path, str, str], Path] = {}


# The hub files of the voice being loaded, by (repo id, pinned commit).
_VOICE_FILES: contextvars.ContextVar[Optional[Dict[Tuple[str, str], Set[str]]]] = \
    contextvars.ContextVar("phoonnx_voice_files", default=None)


@contextlib.contextmanager
def _voice_scope(info: "TTSModelInfo") -> Iterator[None]:
    """Fetch the files of ``info`` so that an offline fallback reads them all from one commit."""
    files: Dict[Tuple[str, str], Set[str]] = {}
    # the vosk dictionary is optional: a load goes on without it, so it does not bind the choice
    urls = [getattr(info, f.name) for f in fields(info)
            if f.name.endswith("_url") and f.name != "dictionary_url"]
    urls += list((info.aux_model_urls or {}).values())
    for url in urls:
        match = _HF_URL.match(url.split("?")[0]) if isinstance(url, str) else None
        if match:
            repo_id = _repo_cached_before_move(match["repo"]) or match["repo"]
            files.setdefault((repo_id, match["rev"]), set()).add(match["path"])
    token = _VOICE_FILES.set(files)
    try:
        yield
    finally:
        _VOICE_FILES.reset(token)


def _in_voice_scope(method: Callable) -> Callable:
    """Run a :class:`TTSModelInfo` method inside :func:`_voice_scope` of its voice."""
    @functools.wraps(method)
    def scoped(self, *args, **kwargs):
        with _voice_scope(self):
            return method(self, *args, **kwargs)
    return scoped


def _voice_key(repo_id: str, revision: str, filename: str) -> Optional[Tuple[Path, str, frozenset]]:
    """The :data:`_VOICE_SNAPSHOTS` key of the voice being loaded, if it names files in ``repo_id``
    at ``revision``.

    A file the voice does not name there, such as the external data of its graph, follows the
    same key, so it comes from the snapshot of the files beside it.
    """
    files = _VOICE_FILES.get()
    wanted = files.get((repo_id, revision)) if files is not None else None
    if not wanted:
        return None
    return Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}", revision, frozenset(wanted)


def _voice_snapshot(repo_id: str, revision: str, filename: str,
                    record: bool) -> Tuple[bool, Optional[Path]]:
    """The fallback for ``filename`` chosen for the voice being loaded.

    Returns ``(False, None)`` when no voice is being loaded or the voice names no file in
    ``repo_id`` at ``revision``, so the folder rule decides. Otherwise the newest snapshot that
    holds ``filename`` and every file the voice names there is chosen, and none when the
    pinned snapshot holds one of them already, as after an interrupted download.
    """
    key = _voice_key(repo_id, revision, filename)
    if key is None:
        return False, None
    repo_dir, _, wanted = key
    needed = wanted | {filename}
    if any((repo_dir / "snapshots" / revision / name).exists() for name in needed):
        return True, None
    chosen = _VOICE_SNAPSHOTS.get(key)
    if chosen is None or not chosen.is_dir():
        holding = [snapshot for snapshot in (repo_dir / "snapshots").glob("*")
                   if snapshot.name != revision
                   and all((snapshot / name).exists() for name in needed)]
        if not holding:
            return True, None
        chosen = max(holding, key=lambda snapshot: snapshot.stat().st_mtime)
        if not record:
            path = chosen / filename
            return True, path if path.exists() else None
        _VOICE_SNAPSHOTS[key] = chosen
        LOG.warning(f"{repo_id} is not cached at the pinned commit {revision} and the hub cannot "
                    f"be reached; loading the cached snapshot {chosen.name} instead")
    path = chosen / filename
    return True, path if path.exists() else None


def _offline_snapshot(repo_id: str, revision: str, filename: str,
                      record: bool = True) -> Optional[Path]:
    """The cached ``filename`` from another snapshot of ``repo_id``, for when ``revision`` is not cached.

    The newest snapshot by modification time that holds ``filename`` is chosen, never the one
    ``refs/main`` names: a fetch at a commit writes no ref, so ``refs/main`` stays at the last
    commit an unpinned install fetched. Later files of the same folder come from the same
    snapshot or not at all. A folder that the pinned snapshot holds in part, as after an
    interrupted download, never falls back, so its files cannot mix two commits. Reads no
    network. Inside a voice load, :func:`_voice_snapshot` chooses instead.

    ``record`` is False when the caller only measures: the chosen snapshot is then not
    remembered, so a size question cannot make a voice look like it loaded from a fallback
    it never took.
    """
    scoped, path = _voice_snapshot(repo_id, revision, filename, record)
    if scoped:
        return path
    repo_dir = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
    folder = posixpath.dirname(filename)
    pinned_folder = repo_dir / "snapshots" / revision / folder
    if pinned_folder.is_dir() and any(entry.is_file() for entry in pinned_folder.iterdir()):
        return None
    key = (repo_dir, revision, folder)
    chosen = _OFFLINE_SNAPSHOTS.get(key)
    if chosen is None or not chosen.is_dir():
        holding = [snapshot for snapshot in (repo_dir / "snapshots").glob("*")
                   if snapshot.name != revision and (snapshot / filename).exists()]
        if not holding:
            return None
        chosen = max(holding, key=lambda snapshot: snapshot.stat().st_mtime)
        if record:
            _OFFLINE_SNAPSHOTS[key] = chosen
            LOG.warning(f"{repo_id} is not cached at the pinned commit {revision} and the hub cannot "
                        f"be reached; loading the cached snapshot {chosen.name} instead")
    path = chosen / filename
    return path if path.exists() else None


def _hf_fetch(url: str, timeout: int = 120) -> Optional[Path]:
    """Give the shared HuggingFace cache path for ``url``, downloading it once.

    Both hub file forms are understood: ``/resolve/<rev>/`` (the form the voice
    index uses) and ``/raw/<rev>/``, which serves the same bytes for the small
    text artifacts.

    Returns ``None`` for a URL the hub cannot serve — a self-hosted or mirrored
    voice. Those are rare but legitimate, so the caller downloads them
    directly; they just do not get the deduplication.
    """
    match = _HF_URL.match(url.split("?")[0])
    if not match:
        return None
    repo_id = _repo_cached_before_move(match["repo"]) or match["repo"]
    try:
        return Path(hf_hub_download(repo_id=repo_id,
                                    filename=match["path"],
                                    revision=match["rev"],
                                    etag_timeout=min(timeout, _ETAG_TIMEOUT)))
    except (LocalEntryNotFoundError, OfflineModeIsEnabled):
        # offline, or the hub did not answer, and the cache lacks this commit
        fallback = _offline_snapshot(repo_id, match["rev"], match["path"])
        if fallback is None:
            raise
        return fallback


def _warn_offhub(url: str) -> None:
    LOG.warning(f"not a HuggingFace file URL, so this download is a private copy "
                f"that no other voice or program can share: {url}")


def _cached_path(url: str, graph_url: Optional[str] = None) -> Optional[Path]:
    """Local path of ``url`` if it is already downloaded, else None.

    Never touches the network: this answers "how big is what we already
    fetched", and a size question must not be able to start a download or
    block on a hub that is slow to answer.
    """
    if not url:
        return None
    match = _HF_URL.match(url.split("?")[0])
    if match:
        repo_id = _repo_cached_before_move(match["repo"]) or match["repo"]
        try:
            return Path(hf_hub_download(repo_id=repo_id,
                                        filename=match["path"],
                                        revision=match["rev"],
                                        local_files_only=True))
        except Exception:
            pass
        # A fallback that _hf_fetch took is what loaded, also when HF_HUB_OFFLINE is
        # unset and the hub was unreachable, so its file is the one to measure.
        repo_dir = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
        key = _voice_key(repo_id, match["rev"], match["path"])
        chosen = _VOICE_SNAPSHOTS.get(key) if key is not None else None
        if chosen is None:
            chosen = _OFFLINE_SNAPSHOTS.get((repo_dir, match["rev"], posixpath.dirname(match["path"])))
        if chosen is not None:
            path = chosen / match["path"]
            if path.exists():
                return path
        if not hub_constants.HF_HUB_OFFLINE:
            return None
        try:
            return _offline_snapshot(repo_id, match["rev"], match["path"], record=False)
        except OSError:
            return None
    dest = _direct_path(url, graph_url)
    return dest if _is_cached(dest) else None


def _file_bytes(*urls: str) -> int:
    """On-disk size of the cached copies of ``urls`` and of their external data.

    A file counts once, also when two URLs name it, or a URL and a graph that
    keeps its weights in it.
    """
    files = set()
    for url in urls:
        graph = _cached_path(url)
        if graph is None:
            continue
        sidecars = []
        if url.split("?")[0].endswith(".onnx"):  # only a graph has external data
            try:
                sidecars = _sidecar_urls(url, graph)
            except ValueError:
                pass
        for path in [graph, *(_cached_path(sidecar, graph_url=url) for sidecar in sidecars)]:
            if path is not None:
                files.add(os.path.realpath(path))
    total = 0
    for path in files:
        try:
            total += os.stat(path).st_size
        except OSError:
            continue
    return total


def _read_json(path: Path) -> Any:
    """Parse the JSON at ``path``."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _hub_json(url: str) -> Any:
    """Give the parsed JSON at ``url``, from the shared cache."""
    return _read_json(_resolve(url))


#: The values of the ``precision`` setting. ``int8`` and ``full`` name the
#: files to load; ``auto`` lets :func:`choose_precision` pick one of them.
PRECISIONS = ("int8", "full", "auto")
DEFAULT_PRECISION = "auto"
#: Environment variable that sets the precision when no argument and no
#: plugin key sets it.
PRECISION_ENV_VAR = "PHOONNX_PRECISION"


class PrecisionUnavailable(ValueError):
    """Raised when a voice has no files for the precision a caller asked for."""

    def __init__(self, voice_id: str, precision: str):
        self.voice_id = voice_id
        self.precision = precision
        other = "full" if precision == "int8" else "int8"
        super().__init__(
            f"voice {voice_id!r} has no {precision} files; precision 'auto' "
            f"or {other!r} loads it")


def resolve_precision(precision: Optional[str] = None) -> str:
    """The precision setting in force: the argument, else the env var, else ``auto``.

    The argument is the load kwarg or the plugin key ``precision``; the plugin
    passes its key as the argument, so a kwarg and a plugin key never meet.
    ``None`` and an empty string count as unset. Case and surrounding spaces
    are ignored.

    Raises:
        ValueError: the value is not a string, or not one of :data:`PRECISIONS`.
    """
    if precision is not None and not isinstance(precision, str):
        raise ValueError(f"precision must be one of {', '.join(PRECISIONS)}, not {precision!r}")
    value = (precision or "").strip() or os.environ.get(PRECISION_ENV_VAR, "").strip()
    value = value.lower() or DEFAULT_PRECISION
    if value not in PRECISIONS:
        raise ValueError(f"precision must be one of {', '.join(PRECISIONS)}, not {value!r}")
    return value


def choose_precision(info: "TTSModelInfo", precision: str,
                     providers: Optional[Sequence[ProviderSpec]] = None) -> str:
    """Which files of ``info`` load under ``precision``: ``int8`` or ``full``.

    ``int8`` and ``full`` raise :class:`PrecisionUnavailable` when the entry has
    no such files. ``auto`` picks int8 only for an int8 variant marked
    ``"auto": true`` while no GPU provider is in the resolved providers; an
    entry that has only int8 files loads them. In every other case ``auto``
    picks full.
    """
    int8_only = info.precision == "int8"
    variant = (info.variants or {}).get("int8")
    if precision == "int8":
        if int8_only or variant:
            return "int8"
        raise PrecisionUnavailable(info.voice_id, "int8")
    if precision == "full":
        if int8_only:
            raise PrecisionUnavailable(info.voice_id, "full")
        return "full"
    if int8_only:
        return "int8"
    if variant and variant.get("auto") is True and not uses_gpu(resolve_providers(providers)):
        return "int8"
    return "full"


#: Voice ids folded into another voice as one of its precisions:
#: old id -> (voice id, precision).
_PRECISION_ALIASES = {
    "edwixx/pt-BR-rapido-int8": ("edwixx/pt-BR-rapido", "int8"),
}
# Aliases already named in the log. A plugin looks a voice up for every
# sentence, and one log line per process is enough.
_LOGGED_ALIASES: set = set()
# Index keys this release does not know, already named in the log.
_LOGGED_UNKNOWN_KEYS: set = set()


def _alias_info(voice_id: str, lookup: Callable[[str], Optional["TTSModelInfo"]]
                ) -> Optional["TTSModelInfo"]:
    """The entry an old precision id names, with a deprecation warning.

    The entry keeps the old id, so a caller that looks the voice up again by
    the id it got back finds the same files.
    """
    target, precision = _PRECISION_ALIASES[voice_id]
    info = lookup(target)
    if info is None:
        return None
    message = (f"voice id {voice_id!r} is deprecated and will be removed in phoonnx "
               f"{_removal_version()}; use {target!r} with precision {precision!r}")
    warnings.warn(message, DeprecationWarning, stacklevel=3)
    if voice_id not in _LOGGED_ALIASES:
        _LOGGED_ALIASES.add(voice_id)
        LOG.warning(message)
    return replace(info.with_precision(precision), voice_id=voice_id)


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
    dictionary_url: Optional[str] = None  # vosk: word -> phonemes pronunciation dictionary
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

    # True when this voice cannot synthesize from a bare (text, lang) request:
    # it needs a caller-supplied reference (e.g. a cloning engine's
    # ``reference_audio``, or a StyleTTS2/Kokoro graph with no baked-in style
    # and no ``engine_params['style_path']``). A catalog sweep can use this to
    # skip these voices instead of rediscovering the same failures every run.
    requires_reference: bool = False

    # Override for the phonemizer's lang_code, when it differs from ``lang``.
    # ``lang`` is catalogue-facing (listings, get_lang_voices) and is not always
    # the language the model was actually trained to phonemize — some index
    # entries carry a dialect/locale code (e.g. "tdt-TL") no phonemizer serves,
    # while the model itself was trained with a different phonemizer voice
    # (e.g. Portuguese espeak). When set, this field — not ``lang`` — becomes
    # the lang_code override passed into VoiceConfig.from_dict.
    phonemizer_lang: Optional[str] = None

    # The precision of the files this entry names: "int8" for an entry that has
    # only int8 files, unset for full files.
    precision: Optional[str] = None
    # Other precisions of the same voice. ``{"int8": {...}}`` holds only the URL
    # fields that differ from this entry, and ``"auto": true`` when the
    # ``auto`` precision may pick it. See :meth:`with_precision`.
    variants: Optional[Dict[str, Dict[str, Any]]] = None

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
            # vosk voices ship a pronunciation dictionary; its local path is the
            # phonemizer_model the VoskPhonemizer loads. A failed fetch leaves
            # phonemizer_model unset rather than pointing at nothing.
            if self.dictionary_url and not self.phonemizer_model:
                dict_path = self.download_dictionary()
                if dict_path:
                    self.phonemizer_model = str(dict_path)
            if self.phoneme_type:
                config["phoneme_type"] = self.phoneme_type
            if self.phonemizer_model:
                config["phonemizer_model"] = self.phonemizer_model

            override_lang = self.phonemizer_lang or self.lang
            lang_code = normalize_lang(override_lang, default_region=False) if override_lang else None
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
                                                     tokens_txt=str(self.hub_path(self.tokens_url) or ""),
                                                     lang_code=lang_code)
            else:
                self._config = VoiceConfig.from_dict(config,
                                                     alphabet=alphabet,
                                                     phoneme_type=phoneme_type,
                                                     engine=engine,
                                                     lang_code=lang_code)

            # populate any missing properties. The catalogue lang takes the
            # default region and the phonemizer lang_code does not, so a voice
            # whose lang comes from its config also records its phonemizer
            # language: a saved voice then reloads with the phonemes it had.
            if not self.lang:
                self.lang = normalize_lang(self._config.lang_code)
                self.phonemizer_lang = self.phonemizer_lang or self._config.lang_code
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

    def with_precision(self, precision: Optional[str] = None,
                       providers: Optional[Sequence[ProviderSpec]] = None) -> "TTSModelInfo":
        """This entry as it loads under ``precision``.

        ``precision`` is ``int8``, ``full`` or ``auto``; unset, it comes from
        ``PHOONNX_PRECISION``, else ``auto``. ``providers`` matter only to
        ``auto``. When the int8 variant is chosen, the result is a new entry
        with the variant's URLs over this entry's, ``aux_model_urls`` merged
        key by key. Otherwise the result is this entry.

        Raises:
            ValueError: ``precision`` is not a known value.
            PrecisionUnavailable: the entry has no files for ``precision``.
        """
        chosen = choose_precision(self, resolve_precision(precision), providers)
        if chosen == "full" or self.precision == "int8":
            return self
        urls = {k: v for k, v in self.variants["int8"].items() if k in VARIANT_FIELDS}
        if "aux_model_urls" in urls:
            urls["aux_model_urls"] = {**(self.aux_model_urls or {}), **urls["aux_model_urls"]}
        return replace(self, **urls, precision="int8", variants=None)

    def hub_path(self, url: Optional[str]) -> Optional[Path]:
        """Give the local path of ``url``, downloading it if needed.

        The path is the one the file already lives at — the hub's, or the
        directory a self-hosted voice was streamed into. Nothing is copied out
        of either, because a graph only loads from the directory that also
        holds the sidecar it names.

        ``None`` only when there is no URL to fetch.
        """
        if not url:
            return None
        return _resolve(url)

    @property
    def voice_path(self) -> Path:
        """The directory this voice's graph lives in.

        The hub's snapshot directory, or the direct-download directory for a
        voice the hub cannot serve. phoonnx owns neither: there is no
        phoonnx-managed voice cache any more, so nothing here is a copy.
        """
        return self.download_model().parent

    def download_config(self) -> Dict[str, Any]:
        """Give the voice's model configuration, from the hub cache."""
        path = self.hub_path(self.config_url)
        return _read_json(path) if path else {}

    def download_tokenizer_config(self) -> Dict[str, Any]:
        """Give the tokenizer configuration, from the hub cache."""
        path = self.hub_path(self.tokenizer_config_url)
        return _read_json(path) if path else {}

    def download_vocab(self) -> Dict[str, Any]:
        """Give the vocabulary, from the cache or from the index entry."""
        if self.vocab_override:
            return self.vocab_override
        path = self.hub_path(self.vocab_url)
        return _read_json(path) if path else {}

    def download_tokens_txt(self) -> str:
        """Give the token list, from the hub cache."""
        path = self.hub_path(self.tokens_url)
        return path.read_text(encoding="utf-8") if path else ""

    def download_dictionary(self) -> Optional[Path]:
        """Give the vosk pronunciation ``dictionary`` (word -> phonemes) path,
        from the hub cache, if any.

        The file is large (tens of MB) but optional: a failed fetch (offline,
        a dropped connection, a pruned hub revision) degrades to rule-based
        g2p for every word rather than failing the whole voice load, the same
        way the phonemizer already degrades per out-of-dictionary word.
        """
        if not self.dictionary_url:
            return None
        try:
            return self.hub_path(self.dictionary_url)
        except Exception as exc:
            LOG.warning(f"Could not fetch vosk pronunciation dictionary for "
                        f"{self.voice_id} ({exc}); falling back to "
                        f"rule-based g2p for every word")
            return None

    def _fetch_onnx(self, url: str) -> Path:
        """Give the local path of an ONNX graph, with its sidecars alongside.

        A graph with external weights names each sidecar by filename and
        onnxruntime resolves that name against the directory the graph itself
        resolves to, so they must share a directory. Fetching every sidecar
        the graph names from the same source puts it there, and the graph
        loads where it lies. Single-file graphs (piper/vits exports) simply
        have no sidecar. A graph that names a file outside its own folder
        raises ``ValueError``, and that file is not fetched.

        "No sidecar" and "the sidecar failed to arrive" are not the same
        answer. The sidecar carries the weights, so a failure that is mistaken
        for absence produces a graph that loads and then synthesizes silence.
        Only a 404, and being offline with nothing cached, count as absence.
        """
        path = _resolve(url)
        for sidecar in _sidecar_urls(url, path):
            try:
                _resolve(sidecar, graph_url=url)
            except _NO_SUCH_FILE as e:
                # There is no sidecar at that URL, or we are offline and have
                # never seen one; a single-file graph loads fine either way.
                LOG.debug(f"external-data sidecar unavailable for '{url}': {e}")
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                # Offline/DNS/timeout on the direct (non-hub) path.
                LOG.debug(f"external-data sidecar unavailable for '{url}': {e}")
            except requests.exceptions.HTTPError as e:
                # Only a 404 means "no sidecar here". Anything else — a 500, a
                # 403, a status we cannot read — is a real failure and must not
                # be swallowed.
                if getattr(getattr(e, "response", None), "status_code", None) != 404:
                    raise
                LOG.debug(f"external-data sidecar unavailable for '{url}': {e}")
        return path

    def download_model(self) -> Path:
        """Give the ONNX graph's path in the hub cache."""
        return self._fetch_onnx(self.model_url)

    def download_vocoder(self) -> Optional[Path]:
        """Give the vocoder graph's path, for two-stage engines."""
        if not self.vocoder_url:
            return None
        return self._fetch_onnx(self.vocoder_url)

    def download_style(self) -> Optional[Path]:
        """Give the style embedding's path, if this voice has one."""
        return self.hub_path(self.style_url)

    def download_speaker_encoder(self) -> Optional[Path]:
        """Give the speaker encoder's path, if this voice has one."""
        if not self.speaker_encoder_url:
            return None
        return self._fetch_onnx(self.speaker_encoder_url)

    def download_aux_models(self) -> Dict[str, Path]:
        """Give the auxiliary graphs' paths, keyed as the engine expects."""
        out: Dict[str, Path] = {}
        for key, url in (self.aux_model_urls or {}).items():
            if not url:
                continue
            out[key] = (self._fetch_onnx(url) if str(url).endswith(".onnx")
                        else _resolve(url))
        return out

    def download_bpe_tokenizer(self) -> Optional[Path]:
        """Give the BPE tokenizer's path in the hub cache."""
        return self.hub_path(self.tokenizer_config_url)

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
            if self.vocoder_config_url:
                params["vocoder_config"] = _hub_json(self.vocoder_config_url)
        elif self.vocoder_type:
            # Parametric vocoder (e.g. Griffin-Lim) — no model file, just config.
            params["vocoder_type"] = self.vocoder_type
            if self.vocoder_config_url:
                params["vocoder_config"] = _hub_json(self.vocoder_config_url)

        # Chatterbox's three auxiliary graphs (the language_model is the primary model).
        for url, key, fname in (
            (self.speech_encoder_url, "speech_encoder_path", "speech_encoder.onnx"),
            (self.embed_tokens_url, "embed_tokens_path", "embed_tokens.onnx"),
            (self.conditional_decoder_url, "conditional_decoder_path", "conditional_decoder.onnx"),
        ):
            if url:
                params[key] = str(self._fetch_onnx(url))
        return params

    @_in_voice_scope
    def download_all(self, precision: Optional[str] = None,
                     providers: Optional[Sequence[ProviderSpec]] = None) -> Path:
        """Download everything this voice needs to load offline.

        The primary graph, the config JSON, whichever tokenizer artifact the
        voice uses (BPE tokenizer / vocab + tokenizer config / tokens.txt) and
        every auxiliary artifact resolved by :meth:`engine_params` (vocoder,
        style embedding, speaker encoder, extra ONNX graphs).

        This is the single definition of "fully downloaded" shared by the CLI
        and :meth:`TTSModelManager.download_voice_by_id`.

        ``precision`` and ``providers`` pick the files as :meth:`load` does.

        Returns:
            Path: local path of the primary ONNX model.
        """
        info = self.with_precision(precision, providers)
        if info is not self:
            return info.download_all(precision="int8")
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

        # vosk pronunciation dictionary (the phonemizer's model)
        self.download_dictionary()

        # vocoder / style / speaker-encoder / aux graphs
        self.engine_params()
        return model_path

    def artifact_urls(self) -> List[str]:
        """Every graph this voice loads, each named once, in a stable order.

        One file is often named by several fields, and — far more importantly —
        by several voices: the bundled omnivoice index has 646 entries over a
        single 3 GB backbone, and qwen3tts 15 over one 4.2 GB talker. Two
        voices with the same list here load exactly the same weights.
        """
        urls = [self.model_url, self.vocoder_url, self.style_url,
                self.speaker_encoder_url, self.speech_encoder_url,
                self.embed_tokens_url, self.conditional_decoder_url,
                self.dictionary_url]
        urls += [u for u in (self.aux_model_urls or {}).values() if u]
        return list(dict.fromkeys(u for u in urls if u))

    def artifact_key(self) -> str:
        """Identity of the *weights* this voice loads, not of the voice.

        Voices that share a key share their memory, so a cache that charges
        memory has to charge it per key rather than per voice id. What makes
        two entries different voices — the language, the display name, the
        engine options applied at synthesis time — is deliberately not in
        here, because none of it costs a second copy of the graph.
        """
        return "|".join(self.artifact_urls())

    @_in_voice_scope
    def disk_size(self) -> int:
        """Total on-disk size, in bytes, of this voice's downloaded artifacts.

        Sums every graph the voice loads — the primary ONNX model, its
        external-weights sidecar when it has one, the vocoder, the style
        embedding, the speaker encoder and any auxiliary graphs — as they lie
        in the shared cache.

        This is a *proxy* for how much memory the loaded voice occupies, not a
        measurement of it. It is the honest one available without an optional
        dependency, and the weights dominate both numbers, so the ordering it
        gives between a 60 MB piper voice and a 2.2 GB omnivoice voice is the
        ordering that matters. What it does NOT account for:

        - onnxruntime arena allocations, activation buffers and per-session
          overhead, which grow with the graph and with sequence length;
        - weights the runtime materializes larger than they are stored (a
          quantized graph dequantized at load, for instance);
        - memory shared between voices that name the same file;
        - anything the voice allocates later, during synthesis;
        - artifacts that are not yet downloaded, which count as zero.

        Returns:
            int: bytes, or 0 when nothing is cached locally.
        """
        # One file may be named by several fields (and several voices); count
        # each distinct file once.
        return _file_bytes(*self.artifact_urls())

    @_in_voice_scope
    def load(self, providers: Optional[Sequence[ProviderSpec]] = None,
             precision: Optional[str] = None) -> TTSVoice:
        """
        Load and return a TTSVoice for this model, ensuring the ONNX model is downloaded and the voice configuration is applied.

        Parameters:
            providers (Sequence, optional): Ordered ONNX Runtime execution providers,
                e.g. ``["ROCMExecutionProvider", "CPUExecutionProvider"]``. When omitted the
                providers are resolved from the environment / auto-detection.
            precision (str, optional): ``int8``, ``full`` or ``auto``. When omitted it comes
                from ``PHOONNX_PRECISION``, else ``auto``. See :meth:`with_precision`.

        Loads a TTSVoice from the cached model and config files (and tokens file if available). If this TTSModelInfo specifies a different phoneme type or alphabet than the loaded voice, updates the loaded voice's phoneme_type and alphabet and rebuilds its phonemizer accordingly.

        Returns:
            TTSVoice: The configured TTSVoice instance ready for synthesis.

        Raises:
            PrecisionUnavailable: the voice has no files for ``precision``.
        """
        info = self.with_precision(precision, providers)
        if info is not self:
            return info.load(providers=providers, precision="int8")
        # self.config resolves the final phoneme_type/lang_code (index override
        # applied) from the small config.json, if any -- cheap next to
        # download_model()'s multi-MB model weights, so the language check
        # runs before that download rather than after it.
        check_lang_supported(self.voice_id, self.config.lang_code, self.config.phoneme_type)

        model_path = self.download_model()
        config_path = self.hub_path(self.config_url)
        vocab_path = self.hub_path(self.vocab_url)
        tokenizer_config_path = self.hub_path(self.tokenizer_config_url)
        tokens_path = self.hub_path(self.tokens_url)

        # Voices without a published config.json can't be engine-detected from
        # files on disk. Build them from the index-derived VoiceConfig, which
        # already carries the engine and its tokenizer/runtime metadata.
        if self.engine and not self.config_url:
            config = self.config
            resolved_providers = resolve_providers(providers)
            engine_params = {
                **(config.engine_params or {}),
                **self.engine_params(),
            }
            engine_params["providers"] = resolved_providers
            config.engine_params = engine_params
            # Through make_session, so this path shares one session between
            # voices that name the same graph like every other path does.
            session = make_session(model_path, providers=resolved_providers)
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
        # A phonemizer_model resolved from the index (the vosk pronunciation
        # dictionary) is an index-level artifact: the published config.json never
        # references it, so the config loaded from disk cannot carry it.
        # TTSVoice.frontend reads the phonemizer assigned here, also after a first synthesis.
        if self.phonemizer_model and not voice.config.phonemizer_model:
            voice.config.phonemizer_model = self.phonemizer_model
            voice.phonemizer = _voice_phonemizer(voice.config.phoneme_type,
                                                 alphabet=voice.config.alphabet,
                                                 model=self.phonemizer_model,
                                                 keep_punctuation=voice.config.keep_punctuation)
        # override phoneme_type, if config.json is wrong
        if self.phoneme_type != voice.config.phoneme_type or self.alphabet != voice.config.alphabet:
            voice.config.phoneme_type = self.phoneme_type
            voice.config.alphabet = self.alphabet
            # voice.config.lang_code is read from the downloaded config.json and
            # can differ from the pre-download self.config.lang_code above, so
            # this re-checks against the phoneme_type actually being installed.
            check_lang_supported(self.voice_id, voice.config.lang_code, self.phoneme_type)
            voice.phonemizer = _voice_phonemizer(self.phoneme_type,
                                                 alphabet=self.alphabet,
                                                 model=voice.config.phonemizer_model,
                                                 keep_punctuation=voice.config.keep_punctuation)
        return voice


_INDEX_FIELDS = frozenset(f.name for f in fields(TTSModelInfo))
#: Fields a precision variant may replace: the files of the voice.
VARIANT_FIELDS = frozenset(name for name in _INDEX_FIELDS
                           if name.endswith("_url") or name == "aux_model_urls")


def _known_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
    """An index or cache record without the keys this release does not know.

    Each unknown key is logged once as a warning, so a typo in a hand-written
    entry is seen, and a field that a newer index adds cannot stop the voice
    from loading.
    """
    for key in sorted(entry.keys() - _INDEX_FIELDS):
        if key not in _LOGGED_UNKNOWN_KEYS:
            _LOGGED_UNKNOWN_KEYS.add(key)
            LOG.warning(f"ignoring key {key!r} of voice {entry.get('voice_id')!r}: "
                        f"this phoonnx release does not know it")
    return {k: v for k, v in entry.items() if k in _INDEX_FIELDS}


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
        # neutts..mosstts: catch-up entries for engines merged since this list was
        # last touched; indic_parler.json, llasa.json, orpheus.json and
        # mosstts.json are new to this change.
        "outetts.json", "arktts.json", "omnivoice.json", "indic_parler.json",
        "parlertts.json", "llasa.json", "orpheus.json", "mosstts.json", "vosk.json",
        "ghananlp.json", "rapido.json",
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
        # The registry is rebuilt wholesale by ``load``/``merge_default_voices``
        # while other threads are looking voices up in it. Rebuilding in place
        # made a known voice look unknown for as long as the repopulation took
        # — thousands of entries — and callers treat "unknown" as a hard error.
        # Every rebuild therefore happens in a local dict and is published by a
        # single assignment, under this lock so two rebuilds cannot interleave.
        self._registry_lock = threading.RLock()
        # bundled entries an alias read, so a lookup per sentence reads the index once
        self._alias_fallbacks: Dict[str, Optional[TTSModelInfo]] = {}
        if cache_path:
            self.cache = JsonStorage(cache_path)
        else:
            self.cache = JsonStorageXDG("voices", subfolder="phoonnx")
        # Create the file, never write over one that is already there. A
        # cache that cannot be parsed loads as an empty registry, and storing
        # that registry back replaces the index with "{}": the catalog is lost
        # and so is the evidence of what broke it.
        if not os.path.isfile(self.cache.path):
            self.cache.store()
        elif not self.cache and not _parses(self.cache.path):
            LOG.error(f"voice index at {self.cache.path} could not be read and "
                      f"is being ignored; the catalog will look empty until it "
                      f"is rebuilt with 'phoonnx-voices update-cache'")

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
        return sorted(set(voice.lang for voice in self.all_voices))

    def clear(self):
        with self._registry_lock:
            self.cache.clear()
            self.voices = {}

    def get_voice(self, voice_id: str) -> Optional[TTSModelInfo]:
        """The catalog entry for ``voice_id``, or ``None`` if it is unknown.

        The lookup readers should use: it sees the registry either as it was
        before a concurrent rebuild or as it is after, never mid-rebuild.

        An old id that now names one precision of another voice gives that
        voice at that precision, with a deprecation warning.
        """
        with self._registry_lock:
            if voice_id in _PRECISION_ALIASES:
                return _alias_info(voice_id, self._alias_target)
            return self.voices.get(voice_id)

    def load(self):
        """
        Reload the on-disk voice cache and rebuild the in-memory voice registry.
        
        Clears the current in-memory voices, calls the cache reload, and attempts to instantiate
        TTSModelInfo for each cached entry. Entries that fail to construct are skipped and an
        error is logged; successful entries are stored in self.voices keyed by voice_id.
        """
        with self._registry_lock:
            self.cache.reload()
            self.voices = self._registry_from_cache()

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
        with self._registry_lock:
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
        indexes = [JsonStorage(str(f)) for f in self.voice_index_files()]
        with self._registry_lock:
            for index in indexes:
                self.cache.update(index)
            self.voices = self._registry_from_cache()
            if store:
                self.cache.store()

    def _registry_from_cache(self) -> Dict[str, TTSModelInfo]:
        """Build the registry from the cache, skipping entries that don't parse.

        Returned rather than assigned, so the caller publishes it in one
        assignment instead of leaving a half-filled registry visible.
        """
        voices: Dict[str, TTSModelInfo] = {}
        for voice_id, voice_dict in self.cache.items():
            if voice_id in _PRECISION_ALIASES:
                # Left in a cache saved before the id became an alias.
                continue
            try:
                voices[voice_id] = TTSModelInfo(**_known_fields(voice_dict))
            except Exception as e:
                LOG.error(f"Failed to load '{voice_id}': ({e})")
        return voices

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

    def _bundled_voice(self, voice_id: str) -> Optional[TTSModelInfo]:
        """The entry for ``voice_id`` read from the bundled indexes, or None."""
        for path in self.voice_index_files():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if voice_id in data:
                return TTSModelInfo(**_known_fields(data[voice_id]))
        return None

    def _alias_target(self, voice_id: str) -> Optional[TTSModelInfo]:
        """The registry entry an alias points at, for :func:`_alias_info`.

        A cache written before the alias existed holds the target without its
        variants. The alias then reads the bundled entry, so it still finds its
        files, and the registry keeps the entry it holds.
        """
        info = self.voices.get(voice_id)
        if info is not None and not info.variants and info.precision is None:
            if voice_id not in self._alias_fallbacks:
                self._alias_fallbacks[voice_id] = self._bundled_voice(voice_id)
            info = self._alias_fallbacks[voice_id] or info
        return info

    def download_voice_by_id(self, voice_id: str, precision: Optional[str] = None) -> bool:
        """
        Download everything a voice needs to load offline, by its ID.

        Looks the voice up in the in-memory registry first (populated via
        ``load()``/``merge_default_voices()``); if not found there, falls
        back to the bundled voice indexes so a voice can be downloaded
        on-demand without first loading the full catalog into memory.

        Fetches the full set of artifacts via ``TTSModelInfo.download_all``:
        the ONNX graph, the config JSON, the tokenizer artifacts and any
        vocoder / style / speaker-encoder / auxiliary graphs. ``precision``
        picks the files as ``TTSModelInfo.load`` does.

        Returns:
            bool: True if the voice was found and its download was attempted,
            False if the voice ID could not be resolved.
        """
        voice_info = self.get_voice(voice_id)
        if not voice_info:
            if voice_id in _PRECISION_ALIASES:
                voice_info = _alias_info(voice_id, self._bundled_voice)
            else:
                voice_info = self._bundled_voice(voice_id)

        if not voice_info:
            LOG.error(f"Cannot find voice information for ID: {voice_id}")
            return False

        voice_info.download_all(precision=precision)
        return True
