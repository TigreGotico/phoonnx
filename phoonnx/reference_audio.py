"""Loading the reference clip a cloning voice is conditioned on.

A reference may arrive as ``(audio, sample_rate)``, as a path on disk, or as
an http(s) URL. The URL case exists because a caller talking to a TTS server
over HTTP has no way to put a file on that server's disk — without it the
cloning engines cannot be reached remotely at all — and it is also the case
that needs guarding: the URL is attacker-controlled, so the fetch is bounded
by address, by size, and by wall-clock time.
"""
import ipaddress
import os
import socket
import tempfile
import time
import wave
from contextlib import ExitStack, suppress
from typing import Any, Tuple
from urllib.parse import urlparse

import numpy as np


def is_public_address(host: str) -> bool:
    """Whether ``host`` resolves only to addresses outside this network.

    A reference URL is supplied by whoever is calling, so without this a
    public TTS server will happily fetch from its own loopback, its Docker
    neighbours, or a cloud metadata endpoint on the caller's behalf, and
    report back whether it worked.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    if not infos:
        return False
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (address.is_private or address.is_loopback or address.is_reserved
                or address.is_link_local or address.is_multicast
                or address.is_unspecified):
            return False
    return True


def check_reference_url(url: str) -> None:
    """Refuse a reference URL that points inside this network."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"reference clip URL must be http(s): {url}")
    if not parsed.hostname:
        raise ValueError(f"reference clip URL has no host: {url}")
    if not is_public_address(parsed.hostname):
        raise ValueError(
            f"refusing to fetch a reference clip from {parsed.hostname}: it "
            f"resolves inside this network, and the URL came from the caller")


def fetch_reference_audio(url: str, timeout: int = 10,
                          max_bytes: int = 32 * 1024 * 1024,
                          deadline: float = 30.0) -> str:
    """Download a cloning reference to a temporary file and return its path.

    Bounded three ways, because every one of them is attacker-controlled:
    the host must be outside this network (each redirect hop is checked, not
    just the first), the body must be small enough to be a few seconds of
    speech, and the whole transfer must finish within ``deadline``. The
    timeout ``requests`` takes applies per read, so a sender trickling one
    byte at a time would otherwise hold a synthesis worker open forever.
    """
    import requests

    started = time.monotonic()
    check_reference_url(url)
    with ExitStack() as stack:
        r = stack.enter_context(
            requests.get(url, timeout=timeout, stream=True,
                        allow_redirects=False))
        hops = 0
        while r.is_redirect or r.is_permanent_redirect:
            hops += 1
            if hops > 5:
                raise ValueError(f"too many redirects fetching {url}")
            target = requests.compat.urljoin(r.url, r.headers["Location"])
            # Checked per hop: a redirect is the easy way to turn an
            # innocent-looking URL into a request to a private address.
            check_reference_url(target)
            r.close()
            # Entered on the stack too, so every hop — including this final
            # one, whose body is streamed below — gets closed on the way out.
            r = stack.enter_context(
                requests.get(target, timeout=timeout, stream=True,
                            allow_redirects=False))
        r.raise_for_status()

        suffix = os.path.splitext(urlparse(url).path)[1] or ".wav"
        written = 0
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            for chunk in r.iter_content(chunk_size=8192):
                if time.monotonic() - started > deadline:
                    raise ValueError(
                        f"reference clip at {url} took longer than "
                        f"{deadline}s to arrive")
                if not chunk:
                    continue
                written += len(chunk)
                if written > max_bytes:
                    raise ValueError(
                        f"reference clip at {url} is larger than "
                        f"{max_bytes} bytes; a reference is a few seconds of "
                        f"speech, not a recording session")
                tmp.write(chunk)
        except BaseException:
            tmp.close()
            with suppress(OSError):
                os.unlink(tmp.name)
            raise
        tmp.close()
        return tmp.name


def load_reference_audio(ref: Any) -> Tuple[np.ndarray, int]:
    """Normalise a cloning reference to ``(mono float32 audio, sample_rate)``.

    Accepts an ``(audio, sample_rate)`` tuple, a URL, or a path to an audio
    file.
    """
    if isinstance(ref, tuple) and len(ref) == 2:
        audio, sr = ref
        return np.asarray(audio, dtype=np.float32).reshape(-1), int(sr)
    if isinstance(ref, str) and ref.lower().startswith(("http://", "https://")):
        # Read and then removed. The download exists only to be decoded here,
        # and on a server whose temporary directory is a tmpfs every clip left
        # behind is resident memory that nothing ever reclaims.
        downloaded = fetch_reference_audio(ref)
        try:
            return load_reference_audio(downloaded)
        finally:
            with suppress(OSError):
                os.unlink(downloaded)
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
