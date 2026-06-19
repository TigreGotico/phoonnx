# Changelog

## [Unreleased](https://github.com/TigreGotico/phoonnx/tree/HEAD)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.3.4a1...HEAD)

**Implemented enhancements:**

- Itzune Basque voices \(antton + maider\) [\#174](https://github.com/TigreGotico/phoonnx/issues/174)

**Closed issues:**

- YourTTS inference engine \(zero-shot voice cloning\) [\#172](https://github.com/TigreGotico/phoonnx/issues/172)
- ZipVoice inference engine [\#170](https://github.com/TigreGotico/phoonnx/issues/170)
- OptiSpeech inference engine [\#167](https://github.com/TigreGotico/phoonnx/issues/167)
- Matcha-TTS inference engine [\#165](https://github.com/TigreGotico/phoonnx/issues/165)
- VITS training engine [\#164](https://github.com/TigreGotico/phoonnx/issues/164)
- StyleTTS2 inference engine [\#163](https://github.com/TigreGotico/phoonnx/issues/163)
- FastPitch inference engine [\#162](https://github.com/TigreGotico/phoonnx/issues/162)
- MixerTTS inference engine [\#161](https://github.com/TigreGotico/phoonnx/issues/161)
- GlowTTS inference engine [\#160](https://github.com/TigreGotico/phoonnx/issues/160)
- VITS inference engine [\#159](https://github.com/TigreGotico/phoonnx/issues/159)
- OptiSpeech support [\#134](https://github.com/TigreGotico/phoonnx/issues/134)
- Matcha-TTS support [\#133](https://github.com/TigreGotico/phoonnx/issues/133)
- Design: pluggable multi-engine architecture [\#132](https://github.com/TigreGotico/phoonnx/issues/132)
- Starting as an ovos-tts-server [\#98](https://github.com/TigreGotico/phoonnx/issues/98)

**Merged pull requests:**

- feat\(gl\): proxectonos Galician Matcha + extended VITS voices [\#197](https://github.com/TigreGotico/phoonnx/pull/197) ([JarbasAl](https://github.com/JarbasAl))
- feat: HiTZ multilingual \(gl/ca/es\) VITS voices [\#196](https://github.com/TigreGotico/phoonnx/pull/196) ([JarbasAl](https://github.com/JarbasAl))
- feat\(eu\): HiTZ StyleTTS2-eu + VITS voices in voice\_index [\#195](https://github.com/TigreGotico/phoonnx/pull/195) ([JarbasAl](https://github.com/JarbasAl))
- feat: AhoTTS \(pyahotts\) Basque phonemizer [\#193](https://github.com/TigreGotico/phoonnx/pull/193) ([JarbasAl](https://github.com/JarbasAl))
- feat\(pycotovia-gl\): replace cotovia binary shell-out with pycotovia [\#191](https://github.com/TigreGotico/phoonnx/pull/191) ([JarbasAl](https://github.com/JarbasAl))
- fix\(glowtts\): take larynx mel output 0 + denormalize \(Larynx noise\) [\#189](https://github.com/TigreGotico/phoonnx/pull/189) ([JarbasAl](https://github.com/JarbasAl))
- feat\(docker\): batteries-included OVOS TTS server image + publish workflow \(closes \#98\) [\#178](https://github.com/TigreGotico/phoonnx/pull/178) ([JarbasAl](https://github.com/JarbasAl))
- feat\(zipvoice\): native flow-matching engine — phoonnx's first iterative adapter [\#158](https://github.com/TigreGotico/phoonnx/pull/158) ([JarbasAl](https://github.com/JarbasAl))
- feat: add itzune Basque voices \(antton + maider\) [\#157](https://github.com/TigreGotico/phoonnx/pull/157) ([JarbasAl](https://github.com/JarbasAl))
- feat\(engines\): YourTTS engine + zero-shot voice cloning \(speaker encoder + registry\) [\#156](https://github.com/TigreGotico/phoonnx/pull/156) ([JarbasAl](https://github.com/JarbasAl))
- chore\(index\): migrate the last 3 voices to the OVOS mirror [\#155](https://github.com/TigreGotico/phoonnx/pull/155) ([JarbasAl](https://github.com/JarbasAl))
- chore\(index\): BCP-47 lang codes + regenerate VOICES.md [\#154](https://github.com/TigreGotico/phoonnx/pull/154) ([JarbasAl](https://github.com/JarbasAl))
- feat\(engines\): VITS2 + StyleTTS2 family \(pure StyleTTS2 + Kokoro, multilingual\) [\#153](https://github.com/TigreGotico/phoonnx/pull/153) ([JarbasAl](https://github.com/JarbasAl))
- feat\(voices\): coqui SpeedySpeech + en/vctk FastPitch \(108-spk\) [\#151](https://github.com/TigreGotico/phoonnx/pull/151) ([JarbasAl](https://github.com/JarbasAl))
- feat\(voices\): add ca-custom \(257-spk\) + fa-custom coqui VITS [\#150](https://github.com/TigreGotico/phoonnx/pull/150) ([JarbasAl](https://github.com/JarbasAl))
- feat\(voices\): coqui VITS engine + 36 voices across 33 languages [\#149](https://github.com/TigreGotico/phoonnx/pull/149) ([JarbasAl](https://github.com/JarbasAl))
- feat\(engines\): FastPitch engine + Arabic & coqui voices [\#148](https://github.com/TigreGotico/phoonnx/pull/148) ([JarbasAl](https://github.com/JarbasAl))
- feat\(engines\): Arabic Mixer-TTS voices \(tts\_arabic\) + matched vocoders [\#147](https://github.com/TigreGotico/phoonnx/pull/147) ([JarbasAl](https://github.com/JarbasAl))
- feat\(engines\): Mixer-TTS inference adapter + LJSpeech voices [\#145](https://github.com/TigreGotico/phoonnx/pull/145) ([JarbasAl](https://github.com/JarbasAl))
- test\(tokenization\): cross-framework golden tests + coqui vocab-order fix [\#144](https://github.com/TigreGotico/phoonnx/pull/144) ([JarbasAl](https://github.com/JarbasAl))
- feat\(engines\): GlowTTS / Larynx inference adapter [\#143](https://github.com/TigreGotico/phoonnx/pull/143) ([JarbasAl](https://github.com/JarbasAl))
- feat\(engines\): OptiSpeech inference adapter [\#142](https://github.com/TigreGotico/phoonnx/pull/142) ([JarbasAl](https://github.com/JarbasAl))
- refactor\(voices\): migrate community voice index to the OpenVoiceOS mirror collection [\#141](https://github.com/TigreGotico/phoonnx/pull/141) ([JarbasAl](https://github.com/JarbasAl))
- feat\(config\): native phoonnx config round-trip [\#139](https://github.com/TigreGotico/phoonnx/pull/139) ([JarbasAl](https://github.com/JarbasAl))
- feat\(engines\): Matcha-TTS inference + pluggable vocoder registry [\#138](https://github.com/TigreGotico/phoonnx/pull/138) ([JarbasAl](https://github.com/JarbasAl))
- feat\(engines\): pluggable multi-engine inference framework [\#131](https://github.com/TigreGotico/phoonnx/pull/131) ([JarbasAl](https://github.com/JarbasAl))
- fix: OVOS plugin config keys + KeyError, plugin tests, CI test\_path [\#124](https://github.com/TigreGotico/phoonnx/pull/124) ([JarbasAl](https://github.com/JarbasAl))

## [1.3.4a1](https://github.com/TigreGotico/phoonnx/tree/1.3.4a1) (2026-02-21)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.3.3...1.3.4a1)

**Merged pull requests:**

- Add docs: installation, configuration, usage guides, training notebook [\#113](https://github.com/TigreGotico/phoonnx/pull/113) ([JarbasAl](https://github.com/JarbasAl))



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
