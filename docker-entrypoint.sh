#!/usr/bin/env sh
# Entrypoint for the phoonnx TTS server image.
#
# When PHOONNX_PREFETCH_VOICES is set, download the listed voices' ONNX
# weights before the server starts accepting traffic. Without this, a
# voice's weights download lazily on its first synth request, so a cold
# public server blocks the first caller on a real download.
#
# PHOONNX_PREFETCH_VOICES is a comma-separated list of voice ids, e.g.:
#   PHOONNX_PREFETCH_VOICES=OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone,OpenVoiceOS/pipertts_en-US_miro
#
# Every listed voice must download. A voice id is an explicit request for a
# specific voice, and a server that starts without it answers requests for
# that voice with an error at synthesis time instead. Failing at startup
# turns a silent capability loss into a visible one.
set -e

# The server is exec'd only after prefetching, so until then this script is
# PID 1 -- and a PID 1 with no installed handler ignores SIGTERM. Without this
# trap, `docker stop` during a long download waits out the whole grace period
# and then SIGKILLs.
prefetch_pid=""
terminate() {
    [ -n "$prefetch_pid" ] && kill -TERM "$prefetch_pid" 2>/dev/null
    exit 143
}
trap terminate TERM INT

if [ -n "${PHOONNX_PREFETCH_VOICES:-}" ]; then
    python - "$PHOONNX_PREFETCH_VOICES" <<'PYEOF' &
import sys

from phoonnx.model_manager import TTSModelManager

# Duplicates are collapsed so a repeated id is downloaded once.
seen = set()
voice_ids = []
for raw in sys.argv[1].split(","):
    voice_id = raw.strip()
    if voice_id and voice_id not in seen:
        seen.add(voice_id)
        voice_ids.append(voice_id)

manager = TTSModelManager()
manager.load()

failed = []
for voice_id in voice_ids:
    print(f"[prefetch] downloading {voice_id} ...", flush=True)
    try:
        voice_info = manager.voices.get(voice_id)
        if voice_info:
            voice_info.download_all()
        elif not manager.download_voice_by_id(voice_id):
            raise RuntimeError(f"voice id '{voice_id}' not found in bundled indexes")
        print(f"[prefetch] done: {voice_id}", flush=True)
    except Exception as exc:
        failed.append(voice_id)
        print(f"[prefetch] FAILED: {voice_id}: {exc}", file=sys.stderr, flush=True)

if failed:
    print(f"[prefetch] {len(failed)}/{len(voice_ids)} voice(s) failed: "
          f"{', '.join(failed)}; aborting startup", file=sys.stderr, flush=True)
    sys.exit(1)
PYEOF
    prefetch_pid=$!
    wait "$prefetch_pid"
    prefetch_pid=""
fi

exec ovos-tts-server --engine ovos-tts-plugin-phoonnx --cache "$@"
