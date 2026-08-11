# Deployment

This page is for operators running the phoonnx TTS server as a public or long-lived
service. It covers persisting the voice cache, prefetching voice weights before serving
traffic, health checks, resource sizing, and verifying that `mycroft.conf` actually
applied. For the image itself and its quick start, see [docker.md](docker.md).

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
