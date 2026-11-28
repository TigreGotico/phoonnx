# phoonnx documentation

<!-- agent-hint
covers: every hand-written docs page in reading order, grouped by reader task
source: docs/*.md, docs/training/*.md, docs/training/engines/*.md
generated: no
verify: .venv/bin/python -I -c "import re, sys, glob, os; os.chdir('docs'); order = [l for l in re.findall(r'^- \[[^]]+\]\(([^)#]+)\)', open('index.md').read(), re.M) if not l.startswith('..')]; pages = [p for p in glob.glob('*.md') + glob.glob('training/*.md') + glob.glob('training/engines/*.md') if p != 'index.md']; bad = sorted(set(pages) ^ set(order)) + [p for p in set(order) if order.count(p) > 1]; seq = ['index.md'] + order + [None]; rel = lambda q, p: os.path.relpath(q, os.path.dirname(p) or '.'); want = lambda i, p: ' · '.join(([re.escape('[← ') + '[^]]+' + re.escape('](' + rel(seq[i], p) + ')')] if i else []) + [re.escape('[Home](' + rel('index.md', p) + ')')] + ([r'\[[^]]+ →' + re.escape('](' + rel(seq[i + 2], p) + ')')] if seq[i + 2] else [])); bad += [p for i, p in enumerate(order) if os.path.exists(p) and not re.fullmatch(want(i, p), open(p).read().rstrip().splitlines()[-1])]; print(bad); sys.exit(bool(bad))"
see: ../README.md, quickstart.md, ../AGENTS.md
-->

This page lists every page of the phoonnx documentation in reading order. Start at the top for a first install, or jump to the group that matches your task.

phoonnx runs text-to-speech voices from many ecosystems through ONNX Runtime, and it ships a toolkit to train voices.

## Start

- [Installation](installation.md): install phoonnx, pick the extras your voices need, and set up a GPU runtime.
- [Quickstart](quickstart.md): download a voice and speak a WAV file from Python in about five minutes.

## Voices

- [Choose a voice](choose_voice.md): list the voices for a language, see which extra each one needs and how large it is, and hear it first.
- [Voice manager](voice_manager.md): find, download and load voices from Python with `TTSModelManager`.
- [CLI reference](cli.md): every `phoonnx-voices` command, with its output and exit status.
- [Precision variants](precision.md): load the smaller int8 files of a voice, and see how phoonnx picks full or int8 files.
- [Use phoonnx offline](offline.md): download voices while online, then speak with no network.
- [Pocket TTS](pockettts.md): the small multilingual Kyutai model for CPU.
- [Orpheus](orpheus.md): the large codec language model voices, their cost and their limits.
- [Galician phonemizer](galician.md): the Cotovia phonemizer and the Galician voices that use it.
- [BSC multispeaker StyleTTS2](bsc_multispeaker.md): the Spanish and Catalan speakers, where they come from and how their styles are made.

## Use

- [Usage guide](usage.md): load a voice, synthesize to a file or a stream, and tune each call from Python.
- [SSML](ssml.md): what each SSML element and attribute does, with the time rules, the bounds and the cost.
- [Voice cloning](cloning.md): clone a speaker from a reference clip.
- [Voice conversion](voice_conversion.md): convert the output of any voice to a target speaker.
- [Streaming](streaming.md): low-latency streaming with split encoder and decoder VITS voices.
- [Phoneme alignment](alignment.md): the timing of each phoneme, for lip-sync, karaoke and subtitles.

## Configure

- [Configuration reference](configuration.md): every `VoiceConfig` and `SynthesisConfig` field, the native `config.json`, execution providers and precision.
- [Phonemizers](phonemizers.md): every phonemizer backend and output alphabet, and how a voice selects one.
- [Alphabet conversion model](alphabet_model.md): how text becomes the token sequence that a model expects.

## Deploy

- [OVOS plugin](ovos_plugin.md): use phoonnx as the text-to-speech engine of OpenVoiceOS.
- [Docker TTS server](docker.md): run the published image as an HTTP text-to-speech server.
- [Deployment](deployment.md): run that server for real traffic, with a cache volume, prefetch, health checks and a memory budget.
- [Benchmarking](benchmarking.md): measure time to first audio and the real-time factor.
- [Voice sweep](voice_sweep.md): load and speak every index voice, check the audio with ASR, and compare two runs.
- [Voice QA](voices.md): the ASR result of the default voice of each leader language, through three load paths.

## Train

- [Training quickstart](training/quickstart.md): go from a folder of audio to a speaking ONNX voice, step by step.
- [Datasets](training/datasets.md): the input formats that preprocess accepts, and how to filter out bad clips.
- [Preprocess reference](training/preprocess.md): every `phoonnx_train.preprocess` flag and the files it writes.
- [Training reference](training/training.md): every `phoonnx_train.train` flag, the training engines and the multi-stage recipes.
- [Export to ONNX](training/export.md): convert a checkpoint to an ONNX voice and check it.

Each engine has its own page, with what it is, the extra it needs, how to get or train a voice, and an example:

- [Matcha-TTS](training/engines/matcha.md): run and train a Matcha-TTS voice.
- [GlowTTS](training/engines/glowtts.md): load, convert and train GlowTTS voices.
- [Mixer-TTS](training/engines/mixertts.md): load and train Mixer-TTS voices.
- [OptiSpeech](training/engines/optispeech.md): load and train OptiSpeech voices.
- [FastPitch](training/engines/fastpitch.md): load, convert and train FastPitch voices.
- [x-vits](training/engines/xvits.md): run the x-vits voice with its styles and stressed words.
- [ZipVoice](training/engines/zipvoice.md): clone with ZipVoice, and train and export a ZipVoice model.
- [Chatterbox](training/engines/chatterbox.md): clone a voice with Chatterbox and control its expressiveness.
- [Spark-TTS](training/engines/sparktts.md): the Spark-TTS preset speakers and cloning.
- [Qwen3-TTS](training/engines/qwen3tts.md): the Qwen3-TTS timbres and languages.
- [ArkTTS](training/engines/arktts.md): the ArkTTS checkpoints and voices.
- [F5-TTS](training/engines/f5tts.md): load an F5-TTS voice and clone from a clip and its transcription.
- [MOSS-TTS-Nano](training/engines/mosstts.md): fine-tune and export MOSS-TTS-Nano.
- [Shami](training/engines/shami.md): export, load and run the Levantine Arabic and English ShamiVITS voices.
- [OuteTTS](training/engines/outetts.md): the OuteTTS languages and speakers.
- [OmniVoice](training/engines/omnivoice.md): the OmniVoice engine and cloning with it.
- [SuperTonic](training/engines/supertonic.md): train a SuperTonic model stage by stage and export it.
- [Rápido](training/engines/rapido.md): train or fine-tune a Rápido voice and export it.
- [Indic Parler-TTS](training/engines/indic_parler.md): the Indic Parler-TTS engine and its voices.

## Reference

- [Architecture](architecture.md): how a string of text becomes audio, and which component owns each step.
- [Engine architecture](engines.md): the inference adapter and training engine registries, and how to add a back end.
- [Vocoders](vocoders.md): the vocoder registry for engines that produce mel spectrograms.
- [Python API](api.md): the names that phoonnx exports, as a walk-through and as a list.

Two files outside `docs/` hold data:

- [VOICES.md](../VOICES.md): a table of voice ids, languages, engines and phoneme types. `python scripts/voices_md.py` builds it from the bundled index, and a test fails while it is out of date. `phoonnx-voices list-available` lists the index itself.
- [CHANGELOG.md](../CHANGELOG.md): the release history, generated by the release workflow.

---

[← README](../README.md) · [Next: Installation →](installation.md)
