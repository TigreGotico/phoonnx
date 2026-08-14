# phoonnx as an OVOS TTS server, batteries included.
#
# Bakes in every optional dependency (all language phonemizers + voice cloning),
# the espeak-ng binary, and a pre-filled voice index so the server starts cleanly
# on an empty cache (see issue #98).
FROM python:3.14-slim

# System deps: espeak-ng (phonemization), libsndfile1 (soundfile, for cloning
# reference audio), git/build tooling for any source wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng \
        libsndfile1 \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# phoonnx + all optional deps + the OVOS TTS server.
# - CPU-only torch first (misaki/spacy pull it transitively) so the multi-GB CUDA
#   wheels never land in this CPU-inference image.
# - setuptools<81 keeps ovos-plugin-manager's pkg_resources usage working.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "setuptools<81" ".[all]" ovos-tts-server \
    && (python -m spacy download en_core_web_sm || true) \
    && (python -m unidic download || true)

RUN useradd -m -u 1000 ovos
USER ovos

# Pre-fill the voice index so the server doesn't choke on a cold cache (issue #98).
RUN phoonnx-voices update-cache || true

EXPOSE 9666

# --cache persists synthesized audio across restarts. The selected voice (and any
# cloning settings) are configured via mycroft.conf — see docs/docker.md.
# The entrypoint optionally prefetches voice weights (PHOONNX_PREFETCH_VOICES)
# before exec'ing the server — see docs/deployment.md.
ENTRYPOINT ["/app/docker-entrypoint.sh"]
