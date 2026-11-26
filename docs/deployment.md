# Deployment

<!-- agent-hint
covers: how to run the phoonnx TTS server as a long-lived service: cache volume, prefetch, health checks, sizing, memory budget, OOM diagnosis, enrichment warnings and config checks
source: docker-entrypoint.sh, docker-compose.yml, phoonnx/voice_cache.py:VoiceCache, phoonnx/providers.py:SHARE_SESSIONS_ENV_VAR, phoonnx/config.py, phoonnx/lang_preprocess.py, phoonnx/tokenizer.py:ChatterboxMTLTokenizer
generated: no
verify: .venv/bin/python -c "import re,sys,glob; p=open('docs/deployment.md').read(); src=open('docker-entrypoint.sh').read()+open('docker-compose.yml').read()+''.join(open(f).read() for f in glob.glob('phoonnx/*.py')); block=p.split('logs a warning:',1)[1].split(chr(96)*3+'text',1)[1].split(chr(96)*3,1)[0]; need=[s for line in block.strip().splitlines() for s in re.split('<[a-z]+>',line)]+re.findall('PHOONNX_[A-Z_]+',p)+['[prefetch] FAILED','phoonnx-cache:/home/ovos/.cache','max_loaded_bytes']; bad=[s for s in need if s not in src]; print(bad); sys.exit(1 if bad else 0)"
see: docker.md, ovos_plugin.md, offline.md
-->

This page shows how to run the phoonnx TTS server as a public or long-lived service. It is for operators who keep the server up for real traffic. For the image and its first run, read [Docker TTS server](docker.md).

## Keep the cache on a volume

The voice index and the downloaded voice weights of the container live under `/home/ovos/.cache`. Mount that path as a named volume, as the bundled [`docker-compose.yml`](../docker-compose.yml) does:

```yaml
volumes:
  - phoonnx-cache:/home/ovos/.cache
```

With no volume, each new container starts from the voice index in the image, with no voice weights. The next request for each voice then downloads it again.

Docker fills a named volume from the image only when it creates the volume. An existing volume keeps its own copy of the voice index. After you pull a later image, refresh the index on the volume:

```bash
docker compose exec phoonnx-tts phoonnx-voices update-cache
```

## The first request for a voice waits for its download

The image runs `phoonnx-voices update-cache` at build time. That command fills the voice index: the voice ids, languages and download URLs. It downloads no ONNX weights. The model file of a voice downloads when the first synthesis request asks for that voice.

On a public server, the first caller for a voice waits for that download. This is not a fault. Download the voices that you serve before you send traffic to the server.

## Download voices before the server starts

Set `PHOONNX_PREFETCH_VOICES` to a comma-separated list of voice ids. The container entrypoint downloads each voice before it starts the TTS server:

```yaml
services:
  phoonnx-tts:
    environment:
      PHOONNX_PREFETCH_VOICES: "OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone,OpenVoiceOS/phoonnx_ca_miro_espeak"
```

Each listed voice must download. If one fails, the entrypoint stops and the server does not start. A server that started without the voice would answer requests for it with an error. The container log has a `[prefetch] FAILED` line for each voice that failed.

With a `restart` policy, a failed prefetch restarts the container, and the entrypoint tries again. A wrong voice id or a model host that you cannot reach then gives a restart loop. Remove the id from the list to start without that voice.

Prefetch the voices that `mycroft.conf` uses. A voice that is not in the list still downloads on its first request.

## Health checks

The `/status` endpoint answers without loading a voice model. Use it for a container health check or a load balancer. The health check of the bundled compose file uses it:

```bash
curl http://localhost:9666/status
```

Do not use a synthesis request as a health check. A request to `/synthesize/...` can wait for a voice download, and the health check then marks a working server as failed.

## Size the container

Size the container for the voices that you serve. Each loaded voice keeps an ONNX Runtime session in memory, and its weight file on disk. The image also holds every phonemizer and the cloning dependencies, whichever voices you use. Keep the voice weights on the cache volume, so a restart does not lose them.

Many index entries can name one model. For example, all `omnivoice` entries load one model, and all `qwen3tts` entries load another. The entries differ in language and engine options. Voices that load the same graph with the same providers share one ONNX Runtime session, and the memory budget of the plugin counts that model one time. Set `PHOONNX_SHARE_ONNX_SESSIONS=0` to give each load its own session. The weight size decides the memory use, not the number of entries: a Piper voice takes tens of megabytes, and an OmniVoice voice takes gigabytes.

The image runs `ovos-tts-server` as one process with one worker. Memory grows with the number of worker processes only when you serve the plugin with your own multi-worker server, such as `uvicorn app:server --workers 4`. Each worker then loads its own `PhoonnxTTSPlugin` and its own voice cache.

## Set a memory budget

The `max_loaded_bytes` setting (see [OVOS plugin](ovos_plugin.md#voice-caching)) limits the voice weights that the plugin keeps in memory. To make room for another voice, the plugin removes the unpinned voice with the oldest last use. The limit applies inside the process only. It does not count the phonemizers, the ONNX Runtime memory arena or the buffers of each request.

The memory limit of the container is a hard limit. When the process goes above it, the kernel kills the process. Set `max_loaded_bytes` well below the container limit, with room for the runtime, the loaded phonemizers and the requests in progress. A budget close to the container limit does not protect the server: the kernel kills the process in the middle of a request before the plugin removes a voice. A budget with room costs some voice reloads from disk.

### Find an OOM kill

In the image, the server runs as PID 1, with no supervisor and no worker processes. An OOM kill stops the container. With `restart: unless-stopped`, `docker inspect <container> --format '{{.RestartCount}}'` counts the restarts. A supervisor or a multi-process server that restarts processes inside the container does not change `RestartCount`.

The kernel log shows an OOM kill for each process model:

```bash
journalctl -k | grep -i "killed process"
```

With no `journalctl`, use `dmesg | grep -i "killed process"`. The kernel names the process `ovos-tts-server`, from the file name of the console script that the entrypoint starts. The `memcg` path in the same report holds the container id. To stop the kills, lower `max_loaded_bytes`, give the container more memory, or do both.

## Prosody enrichment degrades instead of failing

Some voices change the text before synthesis. Arabic voices and other voices with `add_diacritics` add diacritics. The multilingual Chatterbox tokenizer applies a script transform for Korean, Japanese, Russian and Chinese, such as Russian stress marks from `stressonnx`. When one of these steps fails, or its package is missing, phoonnx speaks the plain text and logs a warning:

```text
diacritization failed for lang=<lang>: <error> — synthesizing unstressed text
stressonnx not installed — Russian stress skipped
```

The voice still speaks, but it loses the stress or vowel marks. Search the server log for these warnings when a voice sounds worse than expected.

The script transforms use the base language of the voice: `ru-RU` and `ru_RU` both select the Russian transform.

## Check that the configuration applied

The plugin reads the `tts.ovos-tts-plugin-phoonnx` section of `mycroft.conf`. It ignores settings under any other key. The server then runs with the default voice for its language, and it looks configured:

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

To check the configuration, synthesize a sentence and listen to the result:

```bash
curl "http://localhost:9666/synthesize/hello%20world" --output hello.wav
```

The `/status` response also shows the configured voice in `default_voice`. If you did not prefetch the voice, this first request also waits for its download.

---

[← Docker TTS server](docker.md) · [Home](index.md) · [Benchmarking →](benchmarking.md)
