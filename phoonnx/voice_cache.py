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
"""A memory-bounded cache of loaded voices.

A count of voices cannot bound memory on a mixed catalog: a piper voice is
~60 MB and an omnivoice voice is ~2.2 GB, so the count that is safe for the big
one wastes the cache for the small ones. The bound that matters is bytes, and it
has to hold at the *peak*, because that is what the OOM killer reads: a cold
load reserves its bytes before it allocates them, and concurrent loads are
admitted only while they fit.

Memory is charged per model, not per voice. A voice id is a catalog entry; the
weights are the artifacts it names, and one bundled index names the same
artifacts from hundreds of entries (646 omnivoice voices over one 3 GB
backbone). Charging each entry separately makes the cache evict a model to make
room for itself.
"""
import math
import re
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from ovos_utils.log import LOG

from phoonnx.model_manager import TTSModelInfo
from phoonnx.voice import TTSVoice


class VoiceExceedsMemoryBudget(RuntimeError):
    """A voice's on-disk size alone is larger than ``max_loaded_bytes``.

    Loading it is not a degraded path, it is a guaranteed OOM kill: a 15.4 GB
    voice against a 5 GB budget in an 8 GiB cgroup killed the process, and the
    container restarted straight back into the same load, over and over. The
    size is known before the load is attempted whenever the voice was already
    on disk from an earlier attempt (its own weights survive the restart even
    though the process does not), which is exactly the case that used to
    loop. Refusing here turns that loop into one failed request.
    """


@dataclass
class _Loading:
    """A load in progress, and how it ended.

    Callers that arrive while a voice is loading wait on ``done`` and then read
    ``error``: either the load succeeded and the voice is in the cache, or it
    failed and they get the same exception rather than repeating the attempt.
    """
    done: threading.Event = field(default_factory=threading.Event)
    error: Optional[BaseException] = None


def parse_max_voices(value) -> Optional[int]:
    """Normalize ``max_loaded_voices``; None means no limit."""
    if value in (None, "", 0):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        LOG.error(f"ignoring invalid max_loaded_voices: {value!r}")
        return None
    if parsed < 1:
        LOG.error(f"ignoring max_loaded_voices={parsed}, it must be at least 1")
        return None
    return parsed


def parse_max_bytes(value) -> Optional[int]:
    """Normalize ``max_loaded_bytes``; None means no budget.

    Accepts a plain number of bytes (``6000000000``) or a human-friendly
    size (``"6GB"``, ``"512 MB"``, ``"1.5GiB"``). ``KB/MB/GB/TB`` are
    powers of 1000, ``KiB/MiB/GiB/TiB`` powers of 1024, as those suffixes
    are defined.

    An unusable value is rejected and the budget stays unset. It must never
    end up as zero: a zero budget evicts every voice on every request, and
    an unbounded cache is a far better answer to a typo than a cache that
    cold-loads a multi-gigabyte model for each call.
    """
    if value in (None, "", 0):
        return None
    units = {"": 1, "B": 1,
             "KB": 10 ** 3, "MB": 10 ** 6, "GB": 10 ** 9, "TB": 10 ** 12,
             "KIB": 2 ** 10, "MIB": 2 ** 20, "GIB": 2 ** 30, "TIB": 2 ** 40}
    if isinstance(value, bool):  # bools are ints in python
        LOG.error(f"ignoring invalid max_loaded_bytes: {value!r}")
        return None
    if isinstance(value, (int, float)):
        number, unit = float(value), ""
    else:
        text = str(value).strip().replace(" ", "")
        match = re.fullmatch(r"([0-9]*\.?[0-9]+)([a-zA-Z]*)", text)
        if not match:
            LOG.error(f"ignoring invalid max_loaded_bytes: {value!r}")
            return None
        number, unit = float(match.group(1)), match.group(2).upper()
    if unit not in units:
        LOG.error(f"ignoring max_loaded_bytes={value!r}: unknown unit "
                  f"'{unit}', expected one of {sorted(units)}")
        return None
    if not math.isfinite(number):
        # YAML writes infinity as ``.inf`` and json.loads accepts
        # ``Infinity``/``NaN``; int() raises on all three, and this is called
        # while the plugin is starting, so an uncaught raise here is a plugin
        # that will not start.
        LOG.error(f"ignoring max_loaded_bytes={value!r}: not a finite size")
        return None
    parsed = int(number * units[unit])
    if parsed < 1:
        LOG.error(f"ignoring max_loaded_bytes={value!r}, it must be at least "
                  f"1 byte")
        return None
    return parsed


def parse_pinned(value) -> List[str]:
    """Normalize ``pinned_voices`` to an ordered list of distinct voice ids.

    A single voice written as a bare string is the obvious way to get this
    wrong, and iterating it would pin one voice per character. Order is kept so
    the first pin stays the first loaded; duplicates would otherwise inflate
    the ceiling that is raised to fit the pins.
    """
    if isinstance(value, str):
        value = [value]
    return list(dict.fromkeys(v for v in (value or []) if v))


class VoiceCache:
    """Loaded voices, bounded by a count of voices and a budget of bytes.

    The cache is keyed by voice id and charged by artifact key. It never loads
    the same voice twice at once, it never lets a load allocate memory the
    budget has not admitted, and it never evicts a pinned voice.

    Residency — the set of weights actually in memory — is *cached entries plus
    leased entries*. Eviction removes a cache entry, it does not free a model a
    request is three minutes into synthesizing with, so a caller that is using
    a voice says so by holding a lease:

        with cache.lease(voice_id) as voice:
            voice.synthesize_wav(...)

    A lease is a refcount, not a garbage-collection hint. That distinction is
    the whole point: ``TTSVoice`` holds a reference cycle (its adapter points
    back at it), so CPython's refcounting alone never frees a released voice
    and its memory only comes back on a cyclic collection — which a budget
    cannot wait on. With leases the cycle is irrelevant to the budget: memory
    is charged from the moment a lease is taken until the moment it is
    released, and releasing one wakes every loader waiting for room.

    Parameters:
        resolve: Maps a voice id to its catalog entry. Raises for an unknown id.
        load: Loads a catalog entry's weights and returns the voice.
        max_loaded_voices: Voices resident at once, pinned ones included.
        max_loaded_bytes: Byte budget for resident weights.
        pinned_voices: Voices that may never be evicted.
        load_wait_timeout: Seconds a load waits for room before proceeding
            anyway. Waiting forever would turn a memory bound into a hang.
    """

    def __init__(self,
                 resolve: Callable[[str], TTSModelInfo],
                 load: Callable[[TTSModelInfo], TTSVoice],
                 max_loaded_voices=None,
                 max_loaded_bytes=None,
                 pinned_voices: Optional[Iterable[str]] = None,
                 load_wait_timeout=None):
        self._resolve = resolve
        self._load = load
        # Ordered by least-recently-used, so eviction has an obvious victim.
        self.voices: "OrderedDict[str, TTSVoice]" = OrderedDict()
        self._lock = RLock()
        # Signalled whenever a reservation or a lease is released, so a waiting
        # loader can re-check whether it fits.
        self._budget_free = threading.Condition(self._lock)
        # One gate per voice currently being loaded, so simultaneous callers
        # wait for the load already running instead of starting their own.
        self._loading: Dict[str, _Loading] = {}
        # Bytes promised to loads that are running right now. A load allocates
        # its memory before the cache ever sees it, so the budget has to be
        # spent when the load starts, not when it finishes.
        self._reserved_bytes = 0
        # Artifact key per cached voice, and measured size per artifact key.
        # Sizes are never pruned: a key whose voices were all evicted may still
        # be leased by a request mid-synthesis, and its size is what says so.
        self._voice_keys: Dict[str, str] = {}
        self._key_bytes: Dict[str, int] = {}
        # Outstanding leases per artifact key: how many callers are using those
        # weights right now, whether or not the cache still lists them.
        self._leases: Dict[str, int] = {}
        # A cold load costs seconds to minutes depending on the engine, while a
        # resident voice answers in milliseconds, so the voices a deployment
        # actually serves are worth keeping loaded.
        self.pinned_voices: List[str] = parse_pinned(pinned_voices)
        self.max_loaded_voices = parse_max_voices(max_loaded_voices)
        if (self.max_loaded_voices is not None
                and len(self.pinned_voices) > self.max_loaded_voices):
            LOG.error(
                f"{len(self.pinned_voices)} voices are pinned but "
                f"max_loaded_voices is {self.max_loaded_voices}; raising the "
                f"limit to fit them, because a pinned voice is a promise")
            self.max_loaded_voices = len(self.pinned_voices)
        self.max_loaded_bytes = parse_max_bytes(max_loaded_bytes)
        self.load_wait_timeout = float(load_wait_timeout or 300)

    def preload_pinned(self) -> None:
        """Load the pinned voices, reporting what does not fit.

        A pinned voice that cannot load must not stop the service from
        starting; it simply is not resident.
        """
        for voice_id in self.pinned_voices:
            try:
                self.get(voice_id)
                LOG.info(f"pinned voice loaded: {voice_id}")
            except Exception as e:
                LOG.error(f"pinned voice '{voice_id}' failed to load: {e}")
        if (self.max_loaded_bytes is not None
                and self.resident_bytes() > self.max_loaded_bytes):
            # Said once, at startup, the way a too-small max_loaded_voices is.
            # The pins win either way, so this is the only warning an operator
            # gets that the budget they set is not the memory they will use.
            LOG.error(
                f"the pinned voices need {self.resident_bytes()} bytes, more "
                f"than max_loaded_bytes ({self.max_loaded_bytes}); the pins "
                f"win, so memory use will exceed the budget")

    def get(self, voice_id: str) -> TTSVoice:
        """Return the loaded voice for ``voice_id``, loading it if needed.

        The returned voice is not charged to the budget once the cache drops
        it; a caller that is going to use the weights should hold a
        :meth:`lease` instead.

        Raises:
            VoiceExceedsMemoryBudget: The voice cannot fit the budget alone.
            Exception: ``voice_id`` is not in the catalog, or the load failed.
        """
        return self._acquire(voice_id, lease=False)[0]

    @contextmanager
    def lease(self, voice_id: str) -> Iterator[TTSVoice]:
        """Hold a voice's weights resident for the duration of the block.

        The lease is taken in the same critical section that finds or caches
        the voice, so the weights are charged to the budget from before the
        caller can see them until the block exits. While a lease is held the
        cache may still evict the entry — that only removes the lookup, the
        charge stays — and no other load is admitted against memory this
        caller is using. Releasing the lease wakes every waiting loader.

        Raises what :meth:`get` raises, before the block is entered.
        """
        voice, key = self._acquire(voice_id, lease=True)
        try:
            yield voice
        finally:
            with self._budget_free:
                remaining = self._leases.get(key, 0) - 1
                if remaining > 0:
                    self._leases[key] = remaining
                else:
                    self._leases.pop(key, None)
                self._budget_free.notify_all()

    def resident_bytes(self) -> int:
        """Total measured size of the weights in memory right now."""
        with self._lock:
            return sum(self._key_bytes.get(k, 0) for k in self._resident_keys())

    def _acquire(self, voice_id: str, lease: bool) -> Tuple[TTSVoice, str]:
        """Find or load ``voice_id``, optionally leasing it, and return its key."""
        while True:
            with self._lock:
                if voice_id in self.voices:
                    # Touch it so the least recently used voice is evicted.
                    self.voices.move_to_end(voice_id)
                    key = self._voice_keys.get(voice_id, voice_id)
                    if lease:
                        self._leases[key] = self._leases.get(key, 0) + 1
                    return self.voices[voice_id], key
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
        # gate installed. An unknown voice id raises out of the catalog lookup,
        # and that is a request parameter, so this path is reachable from
        # outside.
        reserved = 0
        try:
            with self._lock:
                info = self._resolve(voice_id)

            LOG.debug(f"Using voice: {voice_id}")
            # Room is claimed BEFORE the load allocates anything. Evicting
            # after the fact bounds the steady state and nothing else: four
            # concurrent 2.2 GB loads each evicted the one before and the
            # cache looked healthy, after the OOM killer had already taken
            # the process.
            reserved, measured, key = self._reserve(voice_id, info)
            # Loaded outside the lock: a cold load takes seconds to minutes,
            # and holding the lock would stall every other request meanwhile.
            voice = self._load(info)

        except BaseException as exc:
            with self._lock:
                self._release(reserved)
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
            #
            # The try covers the measuring and the caching too. An exception
            # raised here is not caught by the except above it — that is what
            # `else` means — so anything that failed while storing the voice
            # used to leave the gate installed with nothing to ever set it,
            # and every later caller for that voice id waited on it forever.
            try:
                # Measured outside the lock: it only stats files, but it stats
                # one per artifact and there is no reason to hold up every
                # other request for it. A voice that could be measured before
                # the load is not measured twice; one that could not (its
                # files were not downloaded yet) now can be.
                size = measured if measured else self._measure(voice_id, info)

                with self._lock:
                    self._release(reserved)
                    # Released exactly once: the failure path below runs the
                    # same call, and releasing twice would shrink the budget
                    # by a voice that was never holding it.
                    reserved = 0
                    self._key_bytes.setdefault(key, size)
                    if voice_id not in self.voices:
                        # Evicted before this voice is tracked: _evict_for
                        # skips a key that is already resident, and this
                        # voice's own weights would otherwise satisfy that
                        # check against itself, making the post-load
                        # eviction — the only one that sees a size measured
                        # after a cold download — a permanent no-op.
                        self._evict_for(voice_id, key, size)
                        self.voices[voice_id] = voice
                        self._voice_keys[voice_id] = key
                    self.voices.move_to_end(voice_id)
                    key = self._voice_keys.get(voice_id, key)
                    cached = self.voices[voice_id]
                    gate = self._loading.pop(voice_id, None)
                    # Taken last, once nothing that could raise is left: a
                    # lease that is handed out and then thrown away by an
                    # exception on the way out is never released, and its
                    # charge stays on the budget for the life of the process.
                    if lease:
                        self._leases[key] = self._leases.get(key, 0) + 1
                if gate is not None:
                    gate.done.set()
                return cached, key

            except BaseException as exc:
                with self._lock:
                    self._release(reserved)
                    gate = self._loading.pop(voice_id, None)
                if gate is not None:
                    gate.error = exc
                    gate.done.set()
                raise

    def _measure(self, voice_id: str, info: TTSModelInfo) -> int:
        """Size of a loaded voice in bytes, as a proxy: its files on disk.

        See :meth:`TTSModelInfo.disk_size` for what that proxy does and does
        not account for. Nothing is measured unless a budget is set, so a
        deployment that does not use one pays nothing for this.

        A voice that cannot be measured counts as free rather than as
        infinite: refusing to cache a voice because its size is unknown would
        turn a measurement problem into a synthesis problem.
        """
        if self.max_loaded_bytes is None:
            return 0
        try:
            return int(info.disk_size())
        except Exception as exc:
            LOG.error(f"could not measure the size of voice '{voice_id}', "
                      f"counting it as free: {exc}")
            return 0

    def _key_for(self, voice_id: str, info: TTSModelInfo) -> str:
        """Artifact key of a voice: what it would share with another voice.

        Falls back to the voice id when the catalog entry cannot name its
        artifacts, which charges that voice on its own — the safe direction to
        be wrong in.
        """
        try:
            key = info.artifact_key()
        except Exception as exc:
            LOG.debug(f"voice '{voice_id}' has no artifact key ({exc}); "
                      f"charging it on its own")
            return voice_id
        return key if isinstance(key, str) and key else voice_id

    def _resident_keys(self) -> Set[str]:
        """Every artifact key whose weights are in memory right now.

        Cached voices, plus leased voices the cache has already evicted.
        Eviction pops a dict entry; it does not free a model a caller is
        three minutes into synthesizing with.
        """
        keys = {self._voice_keys.get(v, v) for v in self.voices}
        keys.update(k for k, n in self._leases.items() if n)
        return keys

    def _reserve(self, voice_id: str, info: TTSModelInfo) -> Tuple[int, int, str]:
        """Claim budget for a load that has not started yet.

        Returns ``(reserved, measured, key)``: the bytes claimed, the size the
        voice could be measured at before loading (0 when it could not be),
        and its artifact key.

        A voice whose weights are already resident claims nothing and waits
        for nothing: it is going to reuse the session that is already loaded,
        so there is no second allocation to make room for.

        Otherwise this waits until the voice fits beside what is resident and
        what other loads have already claimed, evicting to make room. Three
        rules keep it from ever becoming a hang:

        - a loader waits only while memory could still come back — another
          load in flight, or leased weights a caller will eventually let go of
          — so a voice that fits beside nothing does not wait for room that can
          never appear;
        - a voice whose known on-disk size alone is bigger than the whole
          budget is refused with ``VoiceExceedsMemoryBudget`` instead:
          loading it is not "the memory in use will exceed the budget", it is
          a guaranteed OOM kill, and on a size that is already known (the
          case that matters: files that survived a previous, killed attempt)
          there is no reason to repeat it;
        - the wait is bounded by ``load_wait_timeout`` regardless, after which
          the load proceeds and says so;
        - every reservation is released on both the success and the failure
          path, and each release wakes the waiters, as does every lease that
          ends.

        A voice whose files are not downloaded yet cannot be measured, and an
        unknown size is treated as the whole budget: it loads alone. That
        costs the cache once per voice, the first time it is ever fetched,
        and it is the only honest way to keep an unknown allocation inside a
        known bound.
        """
        key = self._key_for(voice_id, info)
        if self.max_loaded_bytes is None:
            return 0, 0, key
        measured = self._measure(voice_id, info)
        with self._budget_free:
            if key in self._resident_keys():
                LOG.debug(f"voice '{voice_id}' reuses weights that are "
                          f"already loaded; charging nothing for it")
                self._key_bytes.setdefault(key, measured)
                return 0, measured, key
            if measured > self.max_loaded_bytes:
                # Known in advance, not merely estimated: refuse before
                # touching eviction or the loader rather than guarantee an
                # OOM kill. A voice whose size is not yet known (0, not yet
                # downloaded) still falls through below.
                raise VoiceExceedsMemoryBudget(
                    f"voice '{voice_id}' needs {measured} bytes on disk, "
                    f"more than the whole max_loaded_bytes budget of "
                    f"{self.max_loaded_bytes}; refusing to load it rather "
                    f"than guarantee an out-of-memory kill")
            need = measured if measured > 0 else self.max_loaded_bytes
            deadline = time.monotonic() + self.load_wait_timeout
            while (self._over_budget(need)
                   and (self._reserved_bytes or self._releasable_bytes())):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    LOG.warning(
                        f"waited {self.load_wait_timeout:.0f}s for room to "
                        f"load '{voice_id}' and the memory in use did not "
                        f"come back; loading it anyway, so memory use will "
                        f"exceed the budget")
                    break
                self._budget_free.wait(remaining)
            # Make the room now, so the bytes are free when the load takes
            # them rather than after it already has.
            self._evict_for(voice_id, key, need)
            self._reserved_bytes += need
        return need, measured, key

    def _over_budget(self, need: int) -> bool:
        """Whether admitting ``need`` more bytes would break the budget."""
        return (sum(self._key_bytes.get(k, 0) for k in self._resident_keys())
                + self._reserved_bytes + need > self.max_loaded_bytes)

    def _releasable_bytes(self) -> int:
        """Resident bytes that will come back on their own.

        Weights that are no longer cached but are still leased by a request in
        flight. Nothing else can shrink without a caller finishing.
        """
        cached = {self._voice_keys.get(v, v) for v in self.voices}
        return sum(self._key_bytes.get(k, 0)
                   for k in self._resident_keys() if k not in cached)

    def _release(self, reserved: int) -> None:
        """Give back a reservation and wake whoever is waiting for room.

        Called with ``self._lock`` held.
        """
        if not reserved:
            return
        self._reserved_bytes = max(0, self._reserved_bytes - reserved)
        self._budget_free.notify_all()

    def _drop(self, victim: str) -> None:
        self.voices.pop(victim, None)
        self._voice_keys.pop(victim, None)
        LOG.debug(f"evicted least recently used voice: {victim}")

    def _lru_victim(self) -> Optional[str]:
        """The least recently used voice that may be evicted, if any."""
        return next((v for v in self.voices if v not in self.pinned_voices),
                    None)

    def _byte_victim(self) -> Optional[str]:
        """The least recently used voice whose eviction frees memory.

        Evicting one of 646 voices that share a backbone frees nothing while
        the others still reference it, and the eviction loop would otherwise
        empty the whole cache discovering that one entry at a time.

        Neither is a leased voice a candidate: the lease keeps its weights
        charged, so dropping the entry frees nothing and only throws away the
        lookup for a model that is still in memory, which the next request for
        it then loads a second copy of.
        """
        for candidate in self.voices:
            if candidate in self.pinned_voices:
                continue
            key = self._voice_keys.get(candidate, candidate)
            others = [v for v in self.voices
                      if v != candidate
                      and self._voice_keys.get(v, v) == key]
            if others or self._leases.get(key, 0):
                continue
            return candidate
        return None

    def _evict_for(self, incoming: str, key: str, size: int = 0) -> None:
        """Make room for ``incoming``, never dropping a pinned voice.

        Both bounds apply when both are set: a voice is evicted when either the
        count or the byte budget would be exceeded.
        """
        if self.max_loaded_voices is not None:
            while len(self.voices) >= self.max_loaded_voices:
                victim = self._lru_victim()
                if victim is None:
                    # Everything resident is pinned: serve the request anyway
                    # rather than evict a voice an operator asked us to keep.
                    LOG.warning(
                        f"all {len(self.voices)} loaded voices are pinned; "
                        f"loading '{incoming}' above max_loaded_voices")
                    break
                self._drop(victim)

        if self.max_loaded_bytes is None:
            return

        if key in self._resident_keys():
            # Its weights are already in memory; admitting it costs nothing.
            return

        if size > self.max_loaded_bytes:
            # Serving it is still better than not serving it. Everything
            # evictable goes first, so the overshoot is as small as it can be.
            LOG.warning(
                f"voice '{incoming}' needs {size} bytes on disk, more than the "
                f"whole max_loaded_bytes budget of {self.max_loaded_bytes}; "
                f"loading it anyway, so memory use will exceed the budget")

        while self.voices and self._over_budget(size):
            victim = self._byte_victim()
            if victim is None:
                LOG.warning(
                    f"nothing more can be evicted "
                    f"({self.resident_bytes()} bytes resident); loading "
                    f"'{incoming}' above max_loaded_bytes")
                break
            self._drop(victim)
