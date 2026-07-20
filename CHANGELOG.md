# Changelog

## [Unreleased](https://github.com/TigreGotico/phoonnx/tree/HEAD)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.3.4a1...HEAD)

**Implemented enhancements:**

- Itzune Basque voices \(antton + maider\) [\#174](https://github.com/TigreGotico/phoonnx/issues/174)

**Closed issues:**

- Kabyle MMS model produces unintelligible audio : missing VITS blank tokens in tokenizer [\#203](https://github.com/TigreGotico/phoonnx/issues/203)
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

**Merged pull requests:**

- fix: missing base dependencies and wrong CLI script name in error messages [\#237](https://github.com/TigreGotico/phoonnx/pull/237) ([JarbasAl](https://github.com/JarbasAl))
- Add generic on-demand quality-metric filtering to preprocess [\#236](https://github.com/TigreGotico/phoonnx/pull/236) ([JarbasAl](https://github.com/JarbasAl))
- feat: opt-in validation audio-sample logging [\#234](https://github.com/TigreGotico/phoonnx/pull/234) ([JarbasAl](https://github.com/JarbasAl))
- fix\(train\): port VITS trainer to Lightning 2 + matcha gradient clip [\#233](https://github.com/TigreGotico/phoonnx/pull/233) ([JarbasAl](https://github.com/JarbasAl))
- feat\(train\): Vocos vocoder training, export and warm start [\#232](https://github.com/TigreGotico/phoonnx/pull/232) ([JarbasAl](https://github.com/JarbasAl))
- Streaming VITS engine \(split encoder/decoder\) + offline voice listing [\#231](https://github.com/TigreGotico/phoonnx/pull/231) ([JarbasAl](https://github.com/JarbasAl))
- fix: corpus-only phoneme map option and untrained-symbol warning [\#229](https://github.com/TigreGotico/phoonnx/pull/229) ([JarbasAl](https://github.com/JarbasAl))
- feat: checkpoint evaluation loop with UTMOS and speaker similarity [\#228](https://github.com/TigreGotico/phoonnx/pull/228) ([JarbasAl](https://github.com/JarbasAl))
- fix: replace monotonic\_align C extension with pure numpy implementation [\#227](https://github.com/TigreGotico/phoonnx/pull/227) ([JarbasAl](https://github.com/JarbasAl))
- fix: fraction error-handling test patched the wrong reference [\#226](https://github.com/TigreGotico/phoonnx/pull/226) ([JarbasAl](https://github.com/JarbasAl))
- feat: orthography2ipa lattice phonemizer backends \(o2i, arbtok, euskaphone, barranquenho\) + mwl fix [\#225](https://github.com/TigreGotico/phoonnx/pull/225) ([JarbasAl](https://github.com/JarbasAl))
- fix: register TTS plugin under opm.tts entry-point group [\#223](https://github.com/TigreGotico/phoonnx/pull/223) ([JarbasAl](https://github.com/JarbasAl))
- feat: espyak fallback for espeak phonemization [\#222](https://github.com/TigreGotico/phoonnx/pull/222) ([JarbasAl](https://github.com/JarbasAl))
- feat: add NAMAA-Saudi-TTS-V2 F5 voice \(namaa/ar-sa-v2\) [\#221](https://github.com/TigreGotico/phoonnx/pull/221) ([JarbasAl](https://github.com/JarbasAl))
- fix: honor voice config's add\_diacritics \(fixes F5-TTS Arabic output\) [\#220](https://github.com/TigreGotico/phoonnx/pull/220) ([JarbasAl](https://github.com/JarbasAl))
- fix: inline \[\[phoneme\]\] blocks crash on empty leading text [\#219](https://github.com/TigreGotico/phoonnx/pull/219) ([JarbasAl](https://github.com/JarbasAl))
- fix: tolerate offline sidecar probe when loading a cached voice [\#216](https://github.com/TigreGotico/phoonnx/pull/216) ([JarbasAl](https://github.com/JarbasAl))
- feat: configurable ONNX Runtime execution providers \(AMD/ROCm, DirectML, CoreML, ...\) [\#215](https://github.com/TigreGotico/phoonnx/pull/215) ([JarbasAl](https://github.com/JarbasAl))
- feat\(chatterbox\): restore lahgtna Arabic-dialect voices with fixed ONNX export [\#214](https://github.com/TigreGotico/phoonnx/pull/214) ([JarbasAl](https://github.com/JarbasAl))
- feat: SILMA TTS v1 \(Arabic+English\) voice via the f5tts engine [\#213](https://github.com/TigreGotico/phoonnx/pull/213) ([JarbasAl](https://github.com/JarbasAl))
- fix\(voice\_index\): dialect-accurate Arabic lang codes + regenerate VOICES.md [\#212](https://github.com/TigreGotico/phoonnx/pull/212) ([JarbasAl](https://github.com/JarbasAl))
- feat\(shami\): add Shami/HamsVITS engine for Levantine Arabic / English TTS [\#211](https://github.com/TigreGotico/phoonnx/pull/211) ([JarbasAl](https://github.com/JarbasAl))
- feat: FastPitch/SpeedySpeech training engine [\#210](https://github.com/TigreGotico/phoonnx/pull/210) ([JarbasAl](https://github.com/JarbasAl))
- feat: GlowTTS training engine [\#209](https://github.com/TigreGotico/phoonnx/pull/209) ([JarbasAl](https://github.com/JarbasAl))
- feat: StyleTTS2 training — every step trainable \(TTS + aligner + PL-BERT + pitch\) [\#208](https://github.com/TigreGotico/phoonnx/pull/208) ([JarbasAl](https://github.com/JarbasAl))
- feat: F5-TTS / Habibi-TTS engine adapter [\#207](https://github.com/TigreGotico/phoonnx/pull/207) ([JarbasAl](https://github.com/JarbasAl))
- feat\(train\): ZipVoice training engine \(Zipformer + flow matching\) [\#206](https://github.com/TigreGotico/phoonnx/pull/206) ([JarbasAl](https://github.com/JarbasAl))
- feat\(train\): YourTTS training engine — zero-shot cloning fine-tuning [\#205](https://github.com/TigreGotico/phoonnx/pull/205) ([JarbasAl](https://github.com/JarbasAl))
- fix: unbreak CI — drop py3.10 leg, exclude self from license check [\#204](https://github.com/TigreGotico/phoonnx/pull/204) ([JarbasAl](https://github.com/JarbasAl))
- feat\(es,ca\): BSC-LT StyleTTS2 multispeaker zero-shot cloning voices [\#201](https://github.com/TigreGotico/phoonnx/pull/201) ([JarbasAl](https://github.com/JarbasAl))
- feat\(opm\): speaker selection for multi-speaker voices [\#200](https://github.com/TigreGotico/phoonnx/pull/200) ([JarbasAl](https://github.com/JarbasAl))
- feat: HiTZ multilingual voices \(es/gl/ca VITS + eu StyleTTS2 emotional\) [\#198](https://github.com/TigreGotico/phoonnx/pull/198) ([JarbasAl](https://github.com/JarbasAl))
- feat\(gl\): proxectonos Galician Matcha + extended VITS voices [\#197](https://github.com/TigreGotico/phoonnx/pull/197) ([JarbasAl](https://github.com/JarbasAl))
- feat: HiTZ multilingual \(gl/ca/es\) VITS voices [\#196](https://github.com/TigreGotico/phoonnx/pull/196) ([JarbasAl](https://github.com/JarbasAl))
- feat\(eu\): HiTZ StyleTTS2-eu + VITS voices in voice\_index [\#195](https://github.com/TigreGotico/phoonnx/pull/195) ([JarbasAl](https://github.com/JarbasAl))
- feat: AhoTTS \(pyahotts\) Basque phonemizer [\#193](https://github.com/TigreGotico/phoonnx/pull/193) ([JarbasAl](https://github.com/JarbasAl))
- feat\(pycotovia-gl\): replace cotovia binary shell-out with pycotovia [\#191](https://github.com/TigreGotico/phoonnx/pull/191) ([JarbasAl](https://github.com/JarbasAl))
- refactor: delegate ARPA↔IPA, Buckwalter↔Arabic, MMS script-tag to scriptconv [\#190](https://github.com/TigreGotico/phoonnx/pull/190) ([JarbasAl](https://github.com/JarbasAl))
- fix\(glowtts\): take larynx mel output 0 + denormalize \(Larynx noise\) [\#189](https://github.com/TigreGotico/phoonnx/pull/189) ([JarbasAl](https://github.com/JarbasAl))
- feat\(alphabet\): unified alphabet model — phonemization as conversion [\#188](https://github.com/TigreGotico/phoonnx/pull/188) ([JarbasAl](https://github.com/JarbasAl))
- feat\(chatterbox\): autoregressive codec-LM engine — d-vector cloning + exaggeration [\#181](https://github.com/TigreGotico/phoonnx/pull/181) ([JarbasAl](https://github.com/JarbasAl))
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
- feat\(train\): Mixer-TTS training \(Lightning\) [\#146](https://github.com/TigreGotico/phoonnx/pull/146) ([JarbasAl](https://github.com/JarbasAl))
- feat\(engines\): Mixer-TTS inference adapter + LJSpeech voices [\#145](https://github.com/TigreGotico/phoonnx/pull/145) ([JarbasAl](https://github.com/JarbasAl))
- test\(tokenization\): cross-framework golden tests + coqui vocab-order fix [\#144](https://github.com/TigreGotico/phoonnx/pull/144) ([JarbasAl](https://github.com/JarbasAl))
- feat\(engines\): GlowTTS / Larynx inference adapter [\#143](https://github.com/TigreGotico/phoonnx/pull/143) ([JarbasAl](https://github.com/JarbasAl))
- feat\(engines\): OptiSpeech inference adapter [\#142](https://github.com/TigreGotico/phoonnx/pull/142) ([JarbasAl](https://github.com/JarbasAl))
- refactor\(voices\): migrate community voice index to the OpenVoiceOS mirror collection [\#141](https://github.com/TigreGotico/phoonnx/pull/141) ([JarbasAl](https://github.com/JarbasAl))
- feat\(config\): native phoonnx config round-trip [\#139](https://github.com/TigreGotico/phoonnx/pull/139) ([JarbasAl](https://github.com/JarbasAl))
- feat\(engines\): Matcha-TTS inference + pluggable vocoder registry [\#138](https://github.com/TigreGotico/phoonnx/pull/138) ([JarbasAl](https://github.com/JarbasAl))
- feat\(engines\): pluggable multi-engine inference framework [\#131](https://github.com/TigreGotico/phoonnx/pull/131) ([JarbasAl](https://github.com/JarbasAl))
- feat\(engines\): Matcha-TTS training adapter [\#128](https://github.com/TigreGotico/phoonnx/pull/128) ([JarbasAl](https://github.com/JarbasAl))
- fix: OVOS plugin config keys + KeyError, plugin tests, CI test\_path [\#124](https://github.com/TigreGotico/phoonnx/pull/124) ([JarbasAl](https://github.com/JarbasAl))

## [1.3.4a1](https://github.com/TigreGotico/phoonnx/tree/1.3.4a1) (2026-02-21)

[Full Changelog](https://github.com/TigreGotico/phoonnx/compare/1.3.3...1.3.4a1)

**Merged pull requests:**

- Add docs: installation, configuration, usage guides, training notebook [\#113](https://github.com/TigreGotico/phoonnx/pull/113) ([JarbasAl](https://github.com/JarbasAl))



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
