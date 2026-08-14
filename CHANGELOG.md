# Changelog

## [1.81.1a1](https://github.com/TigreGotico/phoonnx/tree/1.81.1a1) (2026-08-14)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.81.0a2...1.81.1a1)

**Merged pull requests:**

- fix: load one session per model, not one per voice [\#398](https://github.com/TigreGotico/phoonnx/pull/398) ([JarbasAl](https://github.com/JarbasAl))

## [1.81.0a2](https://github.com/TigreGotico/phoonnx/tree/1.81.0a2) (2026-08-14)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.81.0a1...1.81.0a2)

**Merged pull requests:**

- docs: memory budgeting and OOM diagnosis for self-hosted servers [\#406](https://github.com/TigreGotico/phoonnx/pull/406) ([JarbasAl](https://github.com/JarbasAl))
- fix: close every redirect hop and fix leading-silence bug in voice.py [\#403](https://github.com/TigreGotico/phoonnx/pull/403) ([JarbasAl](https://github.com/JarbasAl))
- Propagate providers to catalog-only engines [\#328](https://github.com/TigreGotico/phoonnx/pull/328) ([goldyfruit](https://github.com/goldyfruit))

## [1.81.0a1](https://github.com/TigreGotico/phoonnx/tree/1.81.0a1) (2026-08-14)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.80.1a1...1.81.0a1)

**Merged pull requests:**

- feat\(gl\): StyleTTS2 ONNX export recipe for the Proxecto Nós Galician voices [\#315](https://github.com/TigreGotico/phoonnx/pull/315) ([JarbasAl](https://github.com/JarbasAl))

## [1.80.1a1](https://github.com/TigreGotico/phoonnx/tree/1.80.1a1) (2026-08-14)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.80.0a2...1.80.1a1)

**Merged pull requests:**

- fix: degrade to unstressed text when diacritization fails [\#399](https://github.com/TigreGotico/phoonnx/pull/399) ([JarbasAl](https://github.com/JarbasAl))

## [1.80.0a2](https://github.com/TigreGotico/phoonnx/tree/1.80.0a2) (2026-08-13)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.80.0a1...1.80.0a2)

**Merged pull requests:**

- refactor: read tokenizer.model without sentencepiece [\#396](https://github.com/TigreGotico/phoonnx/pull/396) ([JarbasAl](https://github.com/JarbasAl))

## [1.80.0a1](https://github.com/TigreGotico/phoonnx/tree/1.80.0a1) (2026-08-13)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.79.5a1...1.80.0a1)

**Merged pull requests:**

- feat: bound the loaded-voice cache by bytes, not by voice count [\#388](https://github.com/TigreGotico/phoonnx/pull/388) ([JarbasAl](https://github.com/JarbasAl))

## [1.79.5a1](https://github.com/TigreGotico/phoonnx/tree/1.79.5a1) (2026-08-13)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.79.4a1...1.79.5a1)

**Merged pull requests:**

- fix: release the loading gate when caching the voice fails [\#393](https://github.com/TigreGotico/phoonnx/pull/393) ([JarbasAl](https://github.com/JarbasAl))

## [1.79.4a1](https://github.com/TigreGotico/phoonnx/tree/1.79.4a1) (2026-08-13)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.79.3a1...1.79.4a1)

**Merged pull requests:**

- fix: let a request choose the voice to clone [\#389](https://github.com/TigreGotico/phoonnx/pull/389) ([JarbasAl](https://github.com/JarbasAl))

## [1.79.3a1](https://github.com/TigreGotico/phoonnx/tree/1.79.3a1) (2026-08-13)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.79.2a1...1.79.3a1)

**Merged pull requests:**

- fix: stop a failed synthesis from poisoning the audio cache [\#390](https://github.com/TigreGotico/phoonnx/pull/390) ([JarbasAl](https://github.com/JarbasAl))

## [1.79.2a1](https://github.com/TigreGotico/phoonnx/tree/1.79.2a1) (2026-08-12)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.79.1a1...1.79.2a1)

## [1.79.1a1](https://github.com/TigreGotico/phoonnx/tree/1.79.1a1) (2026-08-12)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.79.0a1...1.79.1a1)

**Merged pull requests:**

- fix: decode ids back to text without the `tokenizers` package [\#385](https://github.com/TigreGotico/phoonnx/pull/385) ([JarbasAl](https://github.com/JarbasAl))
- fix: load a cold voice once, however many callers ask at the same time [\#384](https://github.com/TigreGotico/phoonnx/pull/384) ([JarbasAl](https://github.com/JarbasAl))
- fix: fetch voice files through the shared HuggingFace cache [\#379](https://github.com/TigreGotico/phoonnx/pull/379) ([JarbasAl](https://github.com/JarbasAl))

## [1.79.0a1](https://github.com/TigreGotico/phoonnx/tree/1.79.0a1) (2026-08-12)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.78.1a2...1.79.0a1)

**Merged pull requests:**

- feat: bound the loaded-voice cache and let voices be pinned in memory [\#380](https://github.com/TigreGotico/phoonnx/pull/380) ([JarbasAl](https://github.com/JarbasAl))

## [1.78.1a2](https://github.com/TigreGotico/phoonnx/tree/1.78.1a2) (2026-08-12)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.78.1a1...1.78.1a2)

**Merged pull requests:**

- Prefetch voice weights on TTS server startup [\#376](https://github.com/TigreGotico/phoonnx/pull/376) ([JarbasAl](https://github.com/JarbasAl))

## [1.78.1a1](https://github.com/TigreGotico/phoonnx/tree/1.78.1a1) (2026-08-12)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.78.0a2...1.78.1a1)

**Merged pull requests:**

- fix: read tokenizer.json without the tokenizers wheel [\#378](https://github.com/TigreGotico/phoonnx/pull/378) ([JarbasAl](https://github.com/JarbasAl))

## [1.78.0a2](https://github.com/TigreGotico/phoonnx/tree/1.78.0a2) (2026-08-10)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.78.0a1...1.78.0a2)

**Merged pull requests:**

- chore: add Apache-2.0 LICENSE [\#374](https://github.com/TigreGotico/phoonnx/pull/374) ([JarbasAl](https://github.com/JarbasAl))

## [1.78.0a1](https://github.com/TigreGotico/phoonnx/tree/1.78.0a1) (2026-08-04)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.77.0a1...1.78.0a1)

**Merged pull requests:**

- feat: Magpie-TTS engine \(NVIDIA multi-codebook AR\) [\#362](https://github.com/TigreGotico/phoonnx/pull/362) ([JarbasAl](https://github.com/JarbasAl))

## [1.77.0a1](https://github.com/TigreGotico/phoonnx/tree/1.77.0a1) (2026-08-04)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.76.0a1...1.77.0a1)

**Merged pull requests:**

- feat: Orpheus engine \(SNAC AR codec-LM\) [\#367](https://github.com/TigreGotico/phoonnx/pull/367) ([JarbasAl](https://github.com/JarbasAl))

## [1.76.0a1](https://github.com/TigreGotico/phoonnx/tree/1.76.0a1) (2026-08-04)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.75.1a1...1.76.0a1)

**Merged pull requests:**

- feat: Llasa engine \(XCodec2 AR codec-LM\) [\#366](https://github.com/TigreGotico/phoonnx/pull/366) ([JarbasAl](https://github.com/JarbasAl))

## [1.75.1a1](https://github.com/TigreGotico/phoonnx/tree/1.75.1a1) (2026-08-04)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.75.0a1...1.75.1a1)

**Merged pull requests:**

- fix: fetch onnx\_data sidecars for auxiliary graphs [\#364](https://github.com/TigreGotico/phoonnx/pull/364) ([JarbasAl](https://github.com/JarbasAl))

## [1.75.0a1](https://github.com/TigreGotico/phoonnx/tree/1.75.0a1) (2026-08-04)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.74.0a1...1.75.0a1)

**Merged pull requests:**

- feat: Indic-Parler-TTS engine [\#363](https://github.com/TigreGotico/phoonnx/pull/363) ([JarbasAl](https://github.com/JarbasAl))

## [1.74.0a1](https://github.com/TigreGotico/phoonnx/tree/1.74.0a1) (2026-08-04)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.73.0a1...1.74.0a1)

**Merged pull requests:**

- feat: KittenTTS voices via the StyleTTS2 adapter [\#360](https://github.com/TigreGotico/phoonnx/pull/360) ([JarbasAl](https://github.com/JarbasAl))

## [1.73.0a1](https://github.com/TigreGotico/phoonnx/tree/1.73.0a1) (2026-08-04)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.72.0a2...1.73.0a1)

**Merged pull requests:**

- feat: OmniVoice engine [\#358](https://github.com/TigreGotico/phoonnx/pull/358) ([JarbasAl](https://github.com/JarbasAl))

## [1.72.0a2](https://github.com/TigreGotico/phoonnx/tree/1.72.0a2) (2026-08-04)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.72.0a1...1.72.0a2)

**Merged pull requests:**

- docs: retroactive Habibi voice quality audit \(WER/CER/RTF\) [\#356](https://github.com/TigreGotico/phoonnx/pull/356) ([JarbasAl](https://github.com/JarbasAl))

## [1.72.0a1](https://github.com/TigreGotico/phoonnx/tree/1.72.0a1) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.71.0a1...1.72.0a1)

**Merged pull requests:**

- feat: BSC named speakers \(clean reland of \#349\) [\#353](https://github.com/TigreGotico/phoonnx/pull/353) ([JarbasAl](https://github.com/JarbasAl))

## [1.71.0a1](https://github.com/TigreGotico/phoonnx/tree/1.71.0a1) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.70.0a2...1.71.0a1)

**Merged pull requests:**

- feat: Galician StyleTTS2 voices \(ProxectoNos Celtia + Brais\) [\#345](https://github.com/TigreGotico/phoonnx/pull/345) ([JarbasAl](https://github.com/JarbasAl))

## [1.70.0a2](https://github.com/TigreGotico/phoonnx/tree/1.70.0a2) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.70.0a1...1.70.0a2)

**Merged pull requests:**

- test: harden scriptconv integration surface and bump floors [\#348](https://github.com/TigreGotico/phoonnx/pull/348) ([JarbasAl](https://github.com/JarbasAl))

## [1.70.0a1](https://github.com/TigreGotico/phoonnx/tree/1.70.0a1) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.69.0a1...1.70.0a1)

**Merged pull requests:**

- feat: ArkTTS engine \(Zortzi Basque + Audio8 multilingual\) [\#346](https://github.com/TigreGotico/phoonnx/pull/346) ([JarbasAl](https://github.com/JarbasAl))

## [1.69.0a1](https://github.com/TigreGotico/phoonnx/tree/1.69.0a1) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.68.0a1...1.69.0a1)

**Merged pull requests:**

- feat: OuteTTS engine [\#342](https://github.com/TigreGotico/phoonnx/pull/342) ([JarbasAl](https://github.com/JarbasAl))

## [1.68.0a1](https://github.com/TigreGotico/phoonnx/tree/1.68.0a1) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.67.0a1...1.68.0a1)

**Merged pull requests:**

- feat: Qwen3-TTS engine [\#343](https://github.com/TigreGotico/phoonnx/pull/343) ([JarbasAl](https://github.com/JarbasAl))

## [1.67.0a1](https://github.com/TigreGotico/phoonnx/tree/1.67.0a1) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.66.1a1...1.67.0a1)

**Merged pull requests:**

- feat: Spark-TTS engine \(BiCodec AR codec-LM\) [\#335](https://github.com/TigreGotico/phoonnx/pull/335) ([JarbasAl](https://github.com/JarbasAl))

## [1.66.1a1](https://github.com/TigreGotico/phoonnx/tree/1.66.1a1) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.66.0a1...1.66.1a1)

**Merged pull requests:**

- fix: point supertonic voice index at the OpenVoiceOS mirror [\#338](https://github.com/TigreGotico/phoonnx/pull/338) ([JarbasAl](https://github.com/JarbasAl))

## [1.66.0a1](https://github.com/TigreGotico/phoonnx/tree/1.66.0a1) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.65.0a1...1.66.0a1)

**Merged pull requests:**

- feat: Kyutai Pocket TTS engine [\#333](https://github.com/TigreGotico/phoonnx/pull/333) ([JarbasAl](https://github.com/JarbasAl))

## [1.65.0a1](https://github.com/TigreGotico/phoonnx/tree/1.65.0a1) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.64.2a1...1.65.0a1)

**Merged pull requests:**

- feat: NeuTTS/Akiti-TTS engine \(autoregressive NeuCodec LM\) [\#332](https://github.com/TigreGotico/phoonnx/pull/332) ([JarbasAl](https://github.com/JarbasAl))

## [1.64.2a1](https://github.com/TigreGotico/phoonnx/tree/1.64.2a1) (2026-08-03)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.64.1a2...1.64.2a1)

**Merged pull requests:**

- fix: follow scriptconv mantoq→halabi notation rename [\#334](https://github.com/TigreGotico/phoonnx/pull/334) ([JarbasAl](https://github.com/JarbasAl))

## [1.64.1a2](https://github.com/TigreGotico/phoonnx/tree/1.64.1a2) (2026-08-02)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.64.1a1...1.64.1a2)

**Merged pull requests:**

- docs: QA pass — verify docs against source, run all examples [\#330](https://github.com/TigreGotico/phoonnx/pull/330) ([JarbasAl](https://github.com/JarbasAl))

## [1.64.1a1](https://github.com/TigreGotico/phoonnx/tree/1.64.1a1) (2026-07-24)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.64.0a1...1.64.1a1)

**Merged pull requests:**

- fix: pad short token sequences for SpeedySpeech-style exports [\#318](https://github.com/TigreGotico/phoonnx/pull/318) ([JarbasAl](https://github.com/JarbasAl))

## [1.64.0a1](https://github.com/TigreGotico/phoonnx/tree/1.64.0a1) (2026-07-24)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.63.0a2...1.64.0a1)

## [1.63.0a2](https://github.com/TigreGotico/phoonnx/tree/1.63.0a2) (2026-07-24)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.63.0a1...1.63.0a2)

**Merged pull requests:**

- ci: bump docker workflow actions [\#313](https://github.com/TigreGotico/phoonnx/pull/313) ([JarbasAl](https://github.com/JarbasAl))

## [1.63.0a1](https://github.com/TigreGotico/phoonnx/tree/1.63.0a1) (2026-07-24)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.62.0a1...1.63.0a1)

**Merged pull requests:**

- feat: add uk\_UA piper voices tetiana/mykyta/oleksa \(high\) [\#312](https://github.com/TigreGotico/phoonnx/pull/312) ([JarbasAl](https://github.com/JarbasAl))

## [1.62.0a1](https://github.com/TigreGotico/phoonnx/tree/1.62.0a1) (2026-07-23)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.61.8a1...1.62.0a1)

## [1.61.8a1](https://github.com/TigreGotico/phoonnx/tree/1.61.8a1) (2026-07-23)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.61.7a1...1.61.8a1)

**Merged pull requests:**

- fix: resolve the voice at boot without fetching it [\#301](https://github.com/TigreGotico/phoonnx/pull/301) ([JarbasAl](https://github.com/JarbasAl))

## [1.61.7a1](https://github.com/TigreGotico/phoonnx/tree/1.61.7a1) (2026-07-23)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.61.6a1...1.61.7a1)

**Merged pull requests:**

- fix: select the Kokoro style row with the unpadded token count [\#308](https://github.com/TigreGotico/phoonnx/pull/308) ([JarbasAl](https://github.com/JarbasAl))

## [1.61.6a1](https://github.com/TigreGotico/phoonnx/tree/1.61.6a1) (2026-07-23)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.61.5a1...1.61.6a1)

**Merged pull requests:**

- fix: make downloads atomic and voices actually offline-ready [\#302](https://github.com/TigreGotico/phoonnx/pull/302) ([JarbasAl](https://github.com/JarbasAl))

## [1.61.5a1](https://github.com/TigreGotico/phoonnx/tree/1.61.5a1) (2026-07-23)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.61.4a1...1.61.5a1)

**Merged pull requests:**

- fix: degrade instead of crashing when alignments are unavailable [\#303](https://github.com/TigreGotico/phoonnx/pull/303) ([JarbasAl](https://github.com/JarbasAl))

## [1.61.4a1](https://github.com/TigreGotico/phoonnx/tree/1.61.4a1) (2026-07-23)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.61.3a1...1.61.4a1)

**Merged pull requests:**

- fix: honour length\_scale on every engine [\#304](https://github.com/TigreGotico/phoonnx/pull/304) ([JarbasAl](https://github.com/JarbasAl))

## [1.61.3a1](https://github.com/TigreGotico/phoonnx/tree/1.61.3a1) (2026-07-23)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.61.2a1...1.61.3a1)

## [1.61.2a1](https://github.com/TigreGotico/phoonnx/tree/1.61.2a1) (2026-07-23)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.61.1a1...1.61.2a1)

**Merged pull requests:**

- fix: never omit an ONNX input the graph requires [\#297](https://github.com/TigreGotico/phoonnx/pull/297) ([JarbasAl](https://github.com/JarbasAl))

## [1.61.1a1](https://github.com/TigreGotico/phoonnx/tree/1.61.1a1) (2026-07-23)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.61.0a3...1.61.1a1)

**Merged pull requests:**

- fix: always resolve a voice's alphabet [\#296](https://github.com/TigreGotico/phoonnx/pull/296) ([JarbasAl](https://github.com/JarbasAl))
- fix: honour an explicit engine over the detection heuristics [\#295](https://github.com/TigreGotico/phoonnx/pull/295) ([JarbasAl](https://github.com/JarbasAl))

## [1.61.0a3](https://github.com/TigreGotico/phoonnx/tree/1.61.0a3) (2026-07-23)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.61.0a2...1.61.0a3)

## [1.61.0a2](https://github.com/TigreGotico/phoonnx/tree/1.61.0a2) (2026-07-23)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.61.0a1...1.61.0a2)

## [1.61.0a1](https://github.com/TigreGotico/phoonnx/tree/1.61.0a1) (2026-07-22)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.60.0a1...1.61.0a1)

## [1.60.0a1](https://github.com/TigreGotico/phoonnx/tree/1.60.0a1) (2026-07-22)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.59.9a2...1.60.0a1)

## [1.59.9a2](https://github.com/TigreGotico/phoonnx/tree/1.59.9a2) (2026-07-21)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.59.9a1...1.59.9a2)

## [1.59.9a1](https://github.com/TigreGotico/phoonnx/tree/1.59.9a1) (2026-07-21)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.59.8a1...1.59.9a1)

## [1.59.8a1](https://github.com/TigreGotico/phoonnx/tree/1.59.8a1) (2026-07-21)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.59.7a1...1.59.8a1)

## [1.59.7a1](https://github.com/TigreGotico/phoonnx/tree/1.59.7a1) (2026-07-21)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.59.6a2...1.59.7a1)

## [1.59.6a2](https://github.com/TigreGotico/phoonnx/tree/1.59.6a2) (2026-07-21)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.3.4a1...1.59.6a2)

## [1.3.4a1](https://github.com/TigreGotico/phoonnx/tree/1.3.4a1) (2026-02-21)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.3.3...1.3.4a1)

## [1.3.3](https://github.com/TigreGotico/phoonnx/tree/1.3.3) (2026-02-16)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.3.3a2...1.3.3)

## [1.3.3a2](https://github.com/TigreGotico/phoonnx/tree/1.3.3a2) (2026-02-15)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.3.3a1...1.3.3a2)

## [1.3.3a1](https://github.com/TigreGotico/phoonnx/tree/1.3.3a1) (2026-02-15)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.3.2a5...1.3.3a1)

## [1.3.2a5](https://github.com/TigreGotico/phoonnx/tree/1.3.2a5) (2026-01-18)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.3.2a4...1.3.2a5)

## [1.3.2a4](https://github.com/TigreGotico/phoonnx/tree/1.3.2a4) (2026-01-14)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.3.2a3...1.3.2a4)

## [1.3.2a3](https://github.com/TigreGotico/phoonnx/tree/1.3.2a3) (2025-12-27)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.3.2a2...1.3.2a3)

## [1.3.2a2](https://github.com/TigreGotico/phoonnx/tree/1.3.2a2) (2025-12-27)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.3.2a1...1.3.2a2)

## [1.3.2a1](https://github.com/TigreGotico/phoonnx/tree/1.3.2a1) (2025-12-27)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.3.1a2...1.3.2a1)

## [1.3.1a2](https://github.com/TigreGotico/phoonnx/tree/1.3.1a2) (2025-12-27)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.3.1a1...1.3.1a2)

## [1.3.1a1](https://github.com/TigreGotico/phoonnx/tree/1.3.1a1) (2025-11-24)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.3.0a4...1.3.1a1)

## [1.3.0a4](https://github.com/TigreGotico/phoonnx/tree/1.3.0a4) (2025-11-24)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.3.0a3...1.3.0a4)

## [1.3.0a3](https://github.com/TigreGotico/phoonnx/tree/1.3.0a3) (2025-11-23)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.3.0a2...1.3.0a3)

## [1.3.0a2](https://github.com/TigreGotico/phoonnx/tree/1.3.0a2) (2025-11-23)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.3.0a1...1.3.0a2)

## [1.3.0a1](https://github.com/TigreGotico/phoonnx/tree/1.3.0a1) (2025-11-23)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.2.0a1...1.3.0a1)

## [1.2.0a1](https://github.com/TigreGotico/phoonnx/tree/1.2.0a1) (2025-11-12)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.1.0a1...1.2.0a1)

## [1.1.0a1](https://github.com/TigreGotico/phoonnx/tree/1.1.0a1) (2025-11-12)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.0.0a1...1.1.0a1)

## [1.0.0a1](https://github.com/TigreGotico/phoonnx/tree/1.0.0a1) (2025-11-12)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.5.4...1.0.0a1)

## [0.5.4](https://github.com/TigreGotico/phoonnx/tree/0.5.4) (2025-11-06)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.5.4a1...0.5.4)

## [0.5.4a1](https://github.com/TigreGotico/phoonnx/tree/0.5.4a1) (2025-11-06)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.5.3a1...0.5.4a1)

## [0.5.3a1](https://github.com/TigreGotico/phoonnx/tree/0.5.3a1) (2025-11-05)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.5.2...0.5.3a1)

## [0.5.2](https://github.com/TigreGotico/phoonnx/tree/0.5.2) (2025-10-16)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.5.2a2...0.5.2)

## [0.5.2a2](https://github.com/TigreGotico/phoonnx/tree/0.5.2a2) (2025-10-16)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.5.2a1...0.5.2a2)

## [0.5.2a1](https://github.com/TigreGotico/phoonnx/tree/0.5.2a1) (2025-10-16)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.5.1a1...0.5.2a1)

## [0.5.1a1](https://github.com/TigreGotico/phoonnx/tree/0.5.1a1) (2025-10-16)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.5.0a2...0.5.1a1)

## [0.5.0a2](https://github.com/TigreGotico/phoonnx/tree/0.5.0a2) (2025-10-16)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.5.0a1...0.5.0a2)

## [0.5.0a1](https://github.com/TigreGotico/phoonnx/tree/0.5.0a1) (2025-10-16)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.4.0a1...0.5.0a1)

## [0.4.0a1](https://github.com/TigreGotico/phoonnx/tree/0.4.0a1) (2025-10-16)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.3.0...0.4.0a1)

## [0.3.0](https://github.com/TigreGotico/phoonnx/tree/0.3.0) (2025-10-12)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.3.0a1...0.3.0)

## [0.3.0a1](https://github.com/TigreGotico/phoonnx/tree/0.3.0a1) (2025-10-12)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.2.7a1...0.3.0a1)

## [0.2.7a1](https://github.com/TigreGotico/phoonnx/tree/0.2.7a1) (2025-10-11)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.2.6...0.2.7a1)

## [0.2.6](https://github.com/TigreGotico/phoonnx/tree/0.2.6) (2025-10-11)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.2.6a2...0.2.6)

## [0.2.6a2](https://github.com/TigreGotico/phoonnx/tree/0.2.6a2) (2025-10-11)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.2.6a1...0.2.6a2)

## [0.2.6a1](https://github.com/TigreGotico/phoonnx/tree/0.2.6a1) (2025-10-05)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.2.5a1...0.2.6a1)

## [0.2.5a1](https://github.com/TigreGotico/phoonnx/tree/0.2.5a1) (2025-10-05)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.2.4...0.2.5a1)

## [0.2.4](https://github.com/TigreGotico/phoonnx/tree/0.2.4) (2025-10-05)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.2.4a1...0.2.4)

## [0.2.4a1](https://github.com/TigreGotico/phoonnx/tree/0.2.4a1) (2025-10-05)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.2.3...0.2.4a1)

## [0.2.3](https://github.com/TigreGotico/phoonnx/tree/0.2.3) (2025-10-04)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.2.3a1...0.2.3)

## [0.2.3a1](https://github.com/TigreGotico/phoonnx/tree/0.2.3a1) (2025-10-04)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.2.2a1...0.2.3a1)

## [0.2.2a1](https://github.com/TigreGotico/phoonnx/tree/0.2.2a1) (2025-10-04)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.2.1a1...0.2.2a1)

## [0.2.1a1](https://github.com/TigreGotico/phoonnx/tree/0.2.1a1) (2025-10-04)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.2.0...0.2.1a1)

## [0.2.0](https://github.com/TigreGotico/phoonnx/tree/0.2.0) (2025-10-04)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.2.0a2...0.2.0)

## [0.2.0a2](https://github.com/TigreGotico/phoonnx/tree/0.2.0a2) (2025-10-04)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.2.0a1...0.2.0a2)

## [0.2.0a1](https://github.com/TigreGotico/phoonnx/tree/0.2.0a1) (2025-10-04)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.1.1a1...0.2.0a1)

## [0.1.1a1](https://github.com/TigreGotico/phoonnx/tree/0.1.1a1) (2025-10-04)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.1.0...0.1.1a1)

## [0.1.0](https://github.com/TigreGotico/phoonnx/tree/0.1.0) (2025-10-03)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.1.0a3...0.1.0)

## [0.1.0a3](https://github.com/TigreGotico/phoonnx/tree/0.1.0a3) (2025-10-03)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.1.0a1...0.1.0a3)

## [0.1.0a1](https://github.com/TigreGotico/phoonnx/tree/0.1.0a1) (2025-08-05)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.0.2a2...0.1.0a1)

## [0.0.2a2](https://github.com/TigreGotico/phoonnx/tree/0.0.2a2) (2025-08-05)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.0.2a1...0.0.2a2)

## [0.0.2a1](https://github.com/TigreGotico/phoonnx/tree/0.0.2a1) (2025-08-03)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.0.1a1...0.0.2a1)

## [0.0.1a1](https://github.com/TigreGotico/phoonnx/tree/0.0.1a1) (2025-08-03)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.0.0...0.0.1a1)

## [0.0.0](https://github.com/TigreGotico/phoonnx/tree/0.0.0) (2025-08-03)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/0.0.0a2...0.0.0)

## [0.0.0a2](https://github.com/TigreGotico/phoonnx/tree/0.0.0a2) (2025-08-03)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/c98b63dedf62c824a9e0e85fca13ddd719550d82...0.0.0a2)



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
