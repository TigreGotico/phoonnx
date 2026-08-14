# Deployment

This page is for operators running the phoonnx TTS server as a public or long-lived
service. It covers persisting the voice cache, prefetching voice weights before serving
traffic, health checks, resource sizing and memory budgeting, diagnosing OOM kills,
prosody enrichment behavior, and verifying that `mycroft.conf` actually applied. For
the image itself and its quick start, see [docker.md](docker.md).

## Persist the cache volume

The container's voice index and downloaded voice weights live under
`/home/ovos/.cache`. Mount that path as a named volume, as the bundled
[`docker-compose.yml`](../docker-compose.yml) does:

```yaml
volumes:
  - phoonnx-cache:/home/ovos/.cache
```

Without a persistent volume, every container restart starts from the image's
pre-filled voice index but with no voice weights downloaded, so the next request
for each voice re-downloads it.

A volume is seeded from the image only when it is first created. An existing
volume keeps its own copy of the voice index, so a newer image's index stays
masked and prefetch resolves ids against the older catalog. Refresh it after
upgrading the image:

```bash
docker compose exec phoonnx-tts phoonnx-voices update-cache
```

## A cold cache is slow, not broken

`phoonnx-voices update-cache` — baked into the image at build time — only fills the
voice **index**: the catalog of known voice ids, languages and download URLs. It does
not fetch any ONNX weights. A voice's actual model file downloads on first use, the
first time a synth request asks for that voice.

On a public server this looks like a hang: the first caller for a given voice waits on
a real network download before hearing anything back. It is not a fault — the fix is to
download the voices you plan to serve before routing traffic to the instance.

## Prefetch voices before serving traffic

Set `PHOONNX_PREFETCH_VOICES` to a comma-separated list of voice ids. The container
entrypoint downloads each one — the same weights `phoonnx-voices download
<voice_id>` fetches — before starting the TTS server:

```yaml
services:
  phoonnx-tts:
    environment:
      PHOONNX_PREFETCH_VOICES: "OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone,OpenVoiceOS/phoonnx_ca_miro_espeak"
```

Every listed voice must download, and startup aborts if any of them fails. A voice
id names one specific voice; a server that starts without it answers requests for
that voice with an error at synthesis time instead, which turns a missing voice into
a runtime surprise. Failing at startup makes it visible immediately. Check the
container logs for `[prefetch] FAILED` lines to see which id is wrong.

With a `restart` policy set, a failing prefetch means the container restarts and
tries again, so a wrong voice id or an unreachable model host produces a restart
loop rather than a running server. That is the intended trade for an explicit
request; remove the id from the list to start without that voice.

Only prefetch the voices `mycroft.conf` actually configures. Weights for a voice not
in that list still download lazily on its first request.

## Health checks

`/status` answers without touching a voice model, so it is the right endpoint for a
container health check or a load balancer — it is what the bundled compose
healthcheck uses:

```bash
curl http://localhost:9666/status
```

A synthesis request is not a health check. `/synthesize/...` can block on a cold
voice download and will mark a healthy instance as failed while that download runs.

## Resource sizing

Size the container for the voices you actually serve, not the full catalog. Each
loaded voice holds an ONNX Runtime session in memory in addition to its weight file
on disk; the batteries-included image also carries the full phonemizer and cloning
dependency set regardless of which voices you use. Disk needs one voice's weights per
prefetched voice, on the same volume as the cache so a restart does not lose them.

A catalog family sharing one underlying model — the `omnivoice` and `qwen3tts`
entries, for example — is still one voice-config entry per model artifact: each
cached voice id owns its own session, so loading several voices from the same
family loads several sessions, not one shared one. Weight size, not entry count, is
what dominates memory: a piper voice is on the order of tens of megabytes, an
omnivoice voice on the order of gigabytes.

The shipped image runs `ovos-tts-server` as a single ASGI app instance passed
directly to `uvicorn.run()`, with no `--workers` flag and no process manager — it is
one process, one worker, no fan-out. Memory multiplies with worker count only for
an operator who fronts the plugin's ASGI app with their own multi-worker deployment
(a `gunicorn`/`uvicorn` invocation using an import string, e.g.
`uvicorn app:server --workers 4`) instead of the bundled entrypoint: each worker
process there loads its own `PhoonnxTTSPlugin` and its own voice cache. This is
distinct from how many *clients* one worker can serve concurrently, which does not
multiply memory the same way.

## Memory budgeting

`max_loaded_bytes` (see [ovos_plugin.md](ovos_plugin.md#voice-caching)) bounds how
much voice weight the plugin keeps resident, evicting the least-recently-used
unpinned voice to make room for a new one. It is a soft, in-process limit — it has
no visibility into the phonemizer, the ONNX Runtime arena, or per-request buffers,
none of which it counts against the budget.

The container's cgroup memory limit is the real backstop, and it is a hard kill, not
an eviction. Set `max_loaded_bytes` well below that limit, leaving headroom for the
runtime itself, whichever phonemizer backends are loaded, and in-flight request
buffers. A budget set close to the cgroup limit does not fail safely: instead of
evicting a voice to stay under budget, the process gets OOM-killed by the kernel
mid-request. A budget with real headroom trades occasional cache misses (a voice
reloading from disk) for a server that never gets killed outright.

### Diagnosing OOM kills

In the shipped image the server execs as PID 1 — no supervisor, no worker children —
so an OOM kill terminates the container outright, and `docker inspect <container>
--format '{{.RestartCount}}'` under a `restart: unless-stopped` policy does
increment and is a valid signal there. `RestartCount` only undercounts in a
deployment where a supervisor or a multi-process server (see the multi-worker note
above) respawns children *inside* the container without the container itself
restarting — a kill there never touches `RestartCount`.

Either way, the kernel log is what distinguishes an OOM kill from any other crash or
exit code, and it works regardless of process model:

```bash
journalctl -k | grep -i "killed process"
```

or, without `journalctl`, `dmesg | grep -i "killed process"`. The kernel logs the
killed process by its `comm` — for the shipped image this is `python3.12` (the
interpreter behind the `ovos-tts-server` console-script shebang), not `phoonnx` or
`ovos-tts-server`, so name matching will not find it. Identify the victim by the
`memcg`/cgroup path in the same OOM report instead — it contains the container id.
Once confirmed, the fix is to lower `max_loaded_bytes`, raise the container's memory
limit, or both.

## Prosody enrichment degrades instead of failing

Voices that enrich text before synthesis — Arabic/Hebrew diacritization
(`add_diacritics`, `diacritizer_model`) and per-language script transforms such as
Russian stress marking (`stressonnx`) — never abort synthesis when that enrichment
fails or its backend is missing. Each falls back to the plain, unenriched text and
logs a warning:

```
diacritization failed for lang=ar: <error> — synthesizing unstressed text
stressonnx not installed — Russian stress skipped
```

The voice still speaks; it loses the stress or vocalization cues that make the
target script unambiguous. Grep the server logs for these warnings to see whether an
optional enrichment dependency is missing rather than assuming a silent voice
quality regression is a synthesis failure.

## Language codes and enrichment backends

A voice id carries a full regional code (`ru_RU`, `pt-PT`, ...), but the enrichment
backends above key off the base language only — `ru`, not `ru_RU` or `ru-RU`. What
determines whether a voice gets stress marking, script transforms, or diacritization
is whether its base language has an entry, not the region. When configuring or
auditing enrichment for a voice, check the base language rather than the full
regional code.

## Verify the configuration actually applied

The plugin reads `tts.ovos-tts-plugin-phoonnx` from `mycroft.conf`. Configuration
placed under any other key, or under the wrong module name, is silently ignored — the
server runs on the default voice and appears configured:

```json
{
  "tts": {
    "module": "ovos-tts-plugin-phoonnx",
    "ovos-tts-plugin-phoonnx": {
      "voice": "OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone"
    }
  }
}
```

Prove it applied by synthesizing and listening to (or inspecting) the result, rather
than trusting the mounted file:

```bash
curl "http://localhost:9666/synthesize/hello%20world" --output hello.wav
```

If the configured voice was never prefetched, this first call also pays the cold
download cost described above — expected, and separate from whether the config
applied.
