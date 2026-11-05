"""Shared test helpers.

``retry_download`` wraps test-time network calls (model/tokenizer downloads
from HF Hub or phoonnx's own voice manager) with a small retry-with-backoff
loop. CI boxes occasionally see the download connection drop mid-transfer
(``requests.exceptions.ChunkedEncodingError`` / ``IncompleteRead``) on
multi-hundred-MB model files; these are transient network conditions, not
test bugs, so we retry a bounded number of times before letting the error
propagate.
"""
import time

import requests
import urllib3

# Exception classes that indicate a transient, retryable network failure
# (as opposed to a real bug in the code under test).
_TRANSIENT_EXCEPTIONS = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    urllib3.exceptions.IncompleteRead,
    urllib3.exceptions.ProtocolError,
    ConnectionError,
)


def retry_download(func, *args, attempts=3, base_delay=1.0, **kwargs):
    """Call ``func(*args, **kwargs)``, retrying on transient network errors.

    Retries up to ``attempts`` times total, with exponential backoff
    (``base_delay * 2**n`` seconds between attempts). Re-raises the last
    exception if every attempt fails. Non-transient exceptions propagate
    immediately without retry.
    """
    last_exc = None
    for attempt in range(attempts):
        try:
            return func(*args, **kwargs)
        except _TRANSIENT_EXCEPTIONS as exc:
            last_exc = exc
            if attempt == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))
    raise last_exc  # pragma: no cover - unreachable, satisfies linters
