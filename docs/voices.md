# Voice QA

<!-- agent-hint
covers: the ASR round trip of the default voice of each leader language, through the manual, index and plugin load paths
source: tools/voice_qa.py, docs/data/voice_qa.json, tools/build_voice_table.py
generated: yes, by tools/build_voice_table.py from docs/data/voice_qa.json
verify: .venv/bin/python -I tools/build_voice_table.py --check
see: voice_sweep.md, choose_voice.md
-->

This table shows how well the default voice of each leader language speaks. `tools/voice_qa.py` measured it with phoonnx 1.139.0a1, 1.140.0a2 and wrote `docs/data/voice_qa.json`. The plugin picks each voice from the language tag alone. Each voice speaks two fixed sentences with a digit and punctuation through three load paths. A path cell gives the class and the mean WER of the two sentences. The result file holds the WER of each sentence.

The WER of each language comes from its own onnx-asr model, the one that the ovos-stt-plugin-onnx-asr registry names. The ASR floor is the WER of that model on two clips of real speech. A WER of 0.35 or less passes, and 0.5 or less for kab, hi-IN and bn-IN. When the floor is above 0.25, the cell is ASR-UNVERIFIED and gives no WER.

| Language | Voice | Engine | Phonemes / alphabet | ASR model | ASR floor | Manual | Index | Plugin | Note |
| :--- | :--- | :--- | :--- | :--- | ---: | :--- | :--- | :--- | :--- |
| pt-PT | `OpenVoiceOS/phoonnx_pt-PT_miro_unicode` | phoonnx | unicode / unicode | `OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-pt-onnx` | 0.02 | PASS 0.11 | PASS 0.11 | PASS 0.11 |  |
| pt-BR | `OpenVoiceOS/pipertts_pt-BR_miro` | piper | espeak / ipa | `OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-pt-onnx` | 0.02 | PASS 0.00 | PASS 0.00 | PASS 0.00 |  |
| es-CO | `OpenVoiceOS/phoonnx_es-CO_miro_espeak` | phoonnx | espeak / ipa | `OpenVoiceOS/stt_es_fastconformer_hybrid_large_pc_onnx` | 0.08 | PASS 0.12 | PASS 0.12 | PASS 0.12 |  |
| fr-FR | `OpenVoiceOS/pipertts_fr-FR_miro` | piper | espeak / ipa | `OpenVoiceOS/stt_fr_fastconformer_hybrid_large_pc_onnx` | 0.03 | PASS 0.09 | PASS 0.09 | PASS 0.09 |  |
| it-IT | `OpenVoiceOS/pipertts_it-IT_miro` | piper | espeak / ipa | `OpenVoiceOS/stt_it_fastconformer_hybrid_large_pc_onnx` | 0.02 | PASS 0.00 | PASS 0.00 | PASS 0.00 |  |
| de-DE | `OpenVoiceOS/pipertts_de-DE_miro` | piper | espeak / ipa | `OpenVoiceOS/stt_de_fastconformer_hybrid_large_pc_onnx` | 0.07 | PASS 0.06 | PASS 0.06 | PASS 0.06 |  |
| nl-NL | `OpenVoiceOS/pipertts_nl-NL_miro` | piper | espeak / ipa | `OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-nl-onnx` | 0.07 | PASS 0.00 | PASS 0.00 | PASS 0.00 |  |
| da-DK | `supertonic/F1/da` | supertonic | unicode / graphemes | `nemo-parakeet-tdt-0.6b-v3` | 0.25 | [FAIL (load error)](https://github.com/TigreGotico/phoonnx/issues/1186) | PASS 0.12 | PASS 0.12 |  |
| kab | `OpenVoiceOS/phoonnx_kab-DZ_miro_o2i` | phoonnx | orthography2ipa / ipa | `OpenVoiceOS/nvidia-kab-conformer-transducer-large-onnx` | 0.10 | PASS 0.33 | PASS 0.33 | PASS 0.33 | sentences not checked by a native speaker |
| hi-IN | `facebook/mms-tts-hin-Hindi` | transformers | graphemes / unicode | `OpenVoiceOS/ai4bharat-indicconformer-hi-onnx` | 0.02 | PASS 0.00 | PASS 0.00 | PASS 0.00 |  |
| bn-IN | `facebook/mms-tts-ben-Bengali` | transformers | graphemes / unicode | `OpenVoiceOS/ai4bharat-indicconformer-bn-onnx` | 0.15 | PASS 0.21 | PASS 0.21 | PASS 0.21 |  |
| mwr | NO-VOICE | NO-VOICE | NO-VOICE | NO-VOICE | NO-VOICE | NO-VOICE | NO-VOICE | NO-VOICE |  |

---

[← Voice sweep](voice_sweep.md) · [Home](index.md) · [Training quickstart →](training/quickstart.md)
