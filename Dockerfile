# phoonnx as an OVOS TTS server, batteries included.
#
# Bakes in every optional dependency (all language phonemizers + voice cloning),
# the espeak-ng binary, and a pre-filled voice index so the server starts cleanly
# on an empty cache (see issue #98).
FROM python:3.11-slim

# System deps: espeak-ng (phonemization), libsndfile1 (soundfile, for cloning
# reference audio), ffmpeg (pydub encodes the non-WAV response formats of the
# vendor-compat routers through it), git/build tooling for any source wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng \
        libsndfile1 \
        ffmpeg \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# phoonnx + all optional deps + the OVOS TTS server.
# - uv does the resolving, not pip. The "all" extra asks for 12 gruut language
#   extras, 4 misaki extras, a scriptconv with 11 extras of its own, spacy and
#   the TTS server in one step, and pip's resolver gives up on it with
#   "error: resolution-too-deep" (issue: the docker job was red on dev from
#   2026-09-19). uv resolves the same set to 1012 pinned packages.
# - --prerelease=allow because the set names alphas on purpose: scriptconv
#   0.0.4a23, stressonnx 0.0.3a2 and ovos-tts-server 1.14.1a2.
# - CPU-only torch first (misaki/spacy pull it transitively) so the multi-GB CUDA
#   wheels never land in this CPU-inference image. uv keeps an installed package
#   that already satisfies a later requirement, so the CPU build stays.
# - setuptools<81 keeps ovos-plugin-manager's pkg_resources usage working.
RUN pip install --no-cache-dir --upgrade pip uv \
    && uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --system --no-cache --prerelease=allow \
        "setuptools<81" ".[all]" "ovos-tts-server[mcp]>=1.14.1a2" \
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
