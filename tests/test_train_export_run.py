"""Train, export, run: one round trip per training engine.

Each test builds a tiny model of one architecture through its engine's real
``create_model``, trains it a few steps on the toy dataset of
tests/train_roundtrip.py, exports it with the engine's own ``export_onnx`` and
loads the export with ``TTSVoice.load``. The test then checks that the ids the
graph reads for a sentence are the ids training read for it, and that the raw
audio of the graph, with no peak normalisation, is long and loud enough: one
decoder hop per id at least, and above the floors of tests/train_roundtrip.py.
A tiny model makes noise, which is enough.
"""
import json
import math
import re
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from phoonnx.engines.vocoders import build_vocoder
from phoonnx.voice import SynthesisConfig, TTSVoice
from phoonnx_train.engines import get_engine
from phoonnx_train.engines.base import TrainingEngineConfig
from tests import train_roundtrip as rt


@pytest.fixture(autouse=True)
def _timed(request):
    start = time.monotonic()
    yield
    request.node.user_properties.append(("seconds", round(time.monotonic() - start, 1)))


@pytest.fixture(scope="session")
def toy_22k(tmp_path_factory):
    """The toy dataset at 22050 Hz, preprocessed once for the session."""
    return rt.preprocess(tmp_path_factory.mktemp("toy-22k"), 22050)


def _engine_config(data, **extra):
    config = rt.dataset_config(data)
    return TrainingEngineConfig(
        num_symbols=config["num_symbols"], num_speakers=config["num_speakers"],
        sample_rate=config["audio"]["sample_rate"],
        extra={"batch_size": 2, "num_workers": 0, "validation_split": 0.0, **extra})


def _text(utterance):
    return utterance["text"] if isinstance(utterance, dict) else utterance.text


def _training_ids(module, ids_of=lambda item: item.phoneme_ids):
    """The ids that the training split holds for the spoken sentence."""
    subset = module._train_set if hasattr(module, "_train_set") else module._train_dataset
    base = subset.dataset
    utterances = base.utterances if hasattr(base, "utterances") else base._inner.utterances
    index = next(i for i, utt in enumerate(utterances) if _text(utt) == rt.SPOKEN)
    assert index in subset.indices, "the spoken sentence is not in the training split"
    return [int(x) for x in ids_of(base[index])]


def _min_samples(voice, ids):
    """One decoder hop of the voice per id that training read. With a blank between
    the symbols, that is two frames per symbol."""
    return voice.config.hop_length * len(ids)


def _round_trip(engine_name, data, tmp_path, steps=2, voice_params=None, vocoder=None,
                training_ids=_training_ids, export_kwargs=None, syn_config=None, defect=None,
                **extra):
    """Build, train, export and load one engine, then check the ids and the audio."""
    engine = get_engine(engine_name)
    torch.manual_seed(0)
    module = engine.create_model(_engine_config(data, **extra), [data])
    checkpoint = rt.fit(module, steps=steps, root=tmp_path)
    ids = training_ids(module)
    out = tmp_path / "voice"
    out.mkdir()
    onnx_path = engine.export_onnx(checkpoint, data / "config.json", out, **(export_kwargs or {}))
    config_path = rt.export_config(out)
    voice = TTSVoice.load(onnx_path, config_path=config_path, engine_params=voice_params)
    if vocoder is not None:
        voice.adapter.vocoder = vocoder
    audio, fed = rt.speak(voice, syn_config=syn_config)
    with rt.known_defect(AssertionError, defect) if defect else nullcontext():
        rt.assert_round_trip(ids, fed, audio, _min_samples(voice, ids))
    return ids, onnx_path, config_path


# ---------------------------------------------------------------------- VITS
TINY_VITS = dict(
    inter_channels=16, hidden_channels=16, filter_channels=32, n_heads=2, n_layers=1,
    resblock="2", resblock_kernel_sizes=(3,), resblock_dilation_sizes=((1, 3),),
    upsample_initial_channel=16, segment_size=2048, num_test_examples=0,
)


def test_vits_round_trip(toy_22k, tmp_path):
    ids, onnx_path, config_path = _round_trip("vits", toy_22k, tmp_path, **TINY_VITS)

    # The control: a voice map with two ids swapped must fail the id check.
    config = json.loads(config_path.read_text(encoding="utf-8"))
    id_map = config["phoneme_id_map"]
    id_map["a"], id_map["t"] = id_map["t"], id_map["a"]
    swapped = tmp_path / "swapped.json"
    swapped.write_text(json.dumps(config), encoding="utf-8")
    audio, fed = rt.speak(TTSVoice.load(onnx_path, config_path=swapped))
    with pytest.raises(AssertionError, match="training read"):
        rt.assert_round_trip(ids, fed, audio, 0)


# ------------------------------------------------------- mel engines + vocoder
# The mel engines train on natural-log mels (phoonnx_train.vits.mel_processing).
# Griffin-Lim clips a mel to [-max_norm, max_norm], maps it linearly onto
# [min_level_db, 0] and reads amp = 10 ** ((db + ref_level_db) / spec_gain). With
# these values the map is db = mel - 4 and amp = exp(mel). The clip at 4 is above
# the loudest mel of speech, and it keeps the large mels of an untrained model
# from giving audio that only the final clip to [-1, 1] holds in range.
GRIFFIN_LIM = {"vocoder_type": "griffinlim",
               "vocoder_config": {"max_norm": 4.0, "min_level_db": -8.0, "ref_level_db": 4.0,
                                  "spec_gain": math.log(10), "mel_fmax": 8000.0}}


@pytest.fixture(scope="session")
def toy_22k_f0(tmp_path_factory):
    """The toy dataset with the F0 sidecars that FastPitch and Mixer-TTS read."""
    return rt.preprocess(tmp_path_factory.mktemp("toy-22k-f0"), 22050, engine="fastpitch")


def test_glowtts_round_trip(toy_22k, tmp_path):
    _round_trip("glowtts", toy_22k, tmp_path, voice_params=GRIFFIN_LIM, num_test_examples=0,
                hidden_channels=16, filter_channels=32, filter_channels_dp=16, n_heads=2,
                n_layers=1, prenet_n_layers=2, dec_hidden_channels=16, dec_n_blocks=1,
                dec_n_layers=1, gin_channels=0)


TINY_FORWARD = dict(hidden_channels=16, hidden_channels_ffn=32, encoder_num_layers=1,
                    decoder_num_layers=1, num_heads=1, num_test_examples=0)


def test_fastpitch_round_trip(toy_22k_f0, tmp_path):
    _round_trip("fastpitch", toy_22k_f0, tmp_path, voice_params=GRIFFIN_LIM, **TINY_FORWARD)


def test_speedyspeech_round_trip(toy_22k_f0, tmp_path):
    _round_trip("speedyspeech", toy_22k_f0, tmp_path, voice_params=GRIFFIN_LIM, **TINY_FORWARD)


def test_mixertts_round_trip(toy_22k_f0, tmp_path):
    # 20 steps at a high rate with no warm-up teach the duration predictor the frame
    # rate of the toy clips. Before that, every duration can round to 0 frames, and
    # the decoder cannot read an empty mel.
    _round_trip("mixertts", toy_22k_f0, tmp_path, steps=20, voice_params=GRIFFIN_LIM,
                symbols_embedding_dim=16, num_test_examples=0, learning_rate=1e-2,
                warmup_steps=1)


def test_matcha_round_trip(toy_22k, tmp_path):
    # The Matcha adapter builds a vocoder only from engine_params["vocoder_path"],
    # so the parametric Griffin-Lim vocoder, which has no file, is set on it here.
    vocoder = build_vocoder(vocoder_type="griffinlim", config=GRIFFIN_LIM["vocoder_config"])
    _round_trip("matcha", toy_22k, tmp_path, vocoder=vocoder,
                training_ids=lambda module: _training_ids(module, lambda item: item["x"]),
                encoder_channels=16, encoder_filter_channels=32,
                encoder_filter_channels_dp=16, encoder_n_heads=2, encoder_n_layers=1,
                decoder_channels=[16, 16], decoder_num_heads=1, decoder_attention_head_dim=8,
                decoder_num_mid_blocks=1)


def test_yourtts_round_trip(toy_22k, tmp_path):
    """YourTTS reads a d-vector per clip. Preprocess computes them with a speaker
    encoder model; here each clip gets a fixed random vector instead."""
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy(toy_22k / "config.json", data / "config.json")
    rows = [json.loads(line) for line in (toy_22k / "dataset.jsonl").read_text("utf-8").splitlines()]
    generator = torch.Generator().manual_seed(0)
    for i, row in enumerate(rows):
        row["d_vector_path"] = str(data / f"d{i}.pt")
        torch.save(torch.randn(512, generator=generator), row["d_vector_path"])
    (data / "dataset.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    _round_trip("yourtts", data, tmp_path, export_kwargs={"dataset_paths": [data / "dataset.jsonl"]},
                **TINY_VITS)


def _optispeech_ids(module):
    """OptiSpeech tokenizes each row's text again with its own TextProcessor."""
    dataset = module.train_dataloader().dataset
    index = next(i for i, row in enumerate(dataset.rows) if row["text"] == rt.SPOKEN)
    return [int(x) for x in dataset[index]["x"]]


@pytest.mark.xfail(strict=True, raises=rt.KnownDefect, reason=(
    "OptiSpeech trains on the ids of its own TextProcessor (a fixed IPA table, no blanks, no "
    "BOS/EOS), but the export writes the dataset's phoneme_id_map into model.json and the "
    "voice encodes with that map, so synthesis feeds ids the model never read"))
def test_optispeech_round_trip(toy_22k, tmp_path):
    sidecar_ids = re.escape(str(rt.dataset_ids(toy_22k)))
    _round_trip("optispeech", toy_22k, tmp_path, training_ids=_optispeech_ids,
                defect=f"the graph read .*{sidecar_ids}",
                dim=32, n_feats=40, hop_length=256, n_fft=256, win_length=256,
                encoder_intermediate_dim=48, decoder_intermediate_dim=48, encoder_num_layers=1,
                decoder_num_layers=1, vocoder_dim=32, vocoder_intermediate_dim=48,
                vocoder_num_layers=1, duration_intermediate_dim=32, pitch_intermediate_dim=32,
                energy_intermediate_dim=32, pitch_num_layers=1, segment_size=8, warmup_steps=1)


def test_rapido_round_trip(toy_22k, tmp_path, monkeypatch):
    """Rápido trains on AudioVAE latents at 25 frames a second. Preprocess encodes
    them with the released AudioVAE, a 377 MB download. Here each clip gets random
    latents of its own length, and the export decodes with a small random AudioVAE
    of the released frame rate."""
    from phoonnx_train.engines.rapido import _QUALITY_PRESETS
    from phoonnx_train.rapido.audio_vae import AudioVAE, AudioVAEConfig
    from phoonnx_train.rapido.dataset import latent_cache_path

    data = tmp_path / "data"
    data.mkdir()
    shutil.copy(toy_22k / "config.json", data / "config.json")
    rows = [json.loads(line) for line in (toy_22k / "dataset.jsonl").read_text("utf-8").splitlines()]
    generator = torch.Generator().manual_seed(0)
    latents = []
    for i, row in enumerate(rows):
        frames = math.ceil(rt.clip_seconds(row["text"]) * 25)
        latents.append(torch.randn(frames, 64, generator=generator) + torch.arange(64) / 8)
        row["audio_norm_path"] = str(data / f"u{i}.pt")
        torch.save(latents[-1], latent_cache_path(row["audio_norm_path"]))
    (data / "dataset.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")

    # Untrained, the duration predictor gives each token about 0.2 frame, and the
    # adapter's floor of one frame per word sets the length. After 50 steps at 1e-3
    # the durations of the spoken sentence add up to the 42 frames of its clip.
    engine = get_engine("rapido")
    torch.manual_seed(0)
    module = engine.create_model(
        _engine_config(data, **_QUALITY_PRESETS["low"], learning_rate=1e-3), [data])
    checkpoint = rt.fit(module, steps=50, root=tmp_path)
    training = module._train_dataset.dataset
    index = next(i for i, utt in enumerate(training.utterances) if utt["text"] == rt.SPOKEN)

    # A random decoder gives an RMS near 1e-9. Its last convolution is scaled so that
    # the training latents decode at an RMS of 0.3, the level of the toy clips. The
    # level check then shows that the latents the exported graphs make decode at
    # the level of the training latents.
    torch.manual_seed(0)
    vae = AudioVAE(AudioVAEConfig(encoder_dim=4, latent_dim=64, decoder_dim=64)).eval()
    with torch.no_grad():
        decoded = torch.cat([vae.decode(z.T[None]).reshape(-1) for z in latents])
        vae.decoder.model[-2].weight_g *= 0.3 / decoded.square().mean().sqrt()
    monkeypatch.setattr("phoonnx_train.rapido.audio_vae.load_audio_vae", lambda: vae)
    config_path = engine.export_onnx(checkpoint, data / "config.json", tmp_path / "voice")
    voice = TTSVoice.load(tmp_path / "voice" / "model.onnx", config_path=config_path)
    ids = training[index]["ids"].tolist()
    audio, fed = rt.speak(voice)
    rt.assert_round_trip(ids, fed, audio, _min_samples(voice, ids))


@pytest.mark.xfail(strict=True, raises=rt.KnownDefect, reason=(
    "SuperTonic export writes the four graphs, tts.json and unicode_indexer.json, "
    "but no per-voice style JSON and no phoonnx voice config, so TTSVoice cannot "
    "speak a model trained from zero"))
def test_supertonic_round_trip(toy_22k, tmp_path):
    from phoonnx_train.supertonic.text import CharTokenizer

    clips = toy_22k.parent
    filelist = tmp_path / "train.filelist"
    filelist.write_text("".join(f"u{i}.wav|{text}|en\n" for i, text in enumerate(rt.TEXTS)), "utf-8")
    tokenizer = CharTokenizer.build_from_texts(list(rt.TEXTS), ["en"])
    engine = get_engine("supertonic")
    config = rt.dataset_config(toy_22k)
    checkpoints = {}
    for stage in ("autoencoder", "text_to_latent", "duration_predictor"):
        extra = {"stage": stage, "quality": "low", "root_dir": str(clips), "batch_size": 2,
                 "num_workers": 0, "segment_seconds": 0.4, "tokenizer": tokenizer,
                 "ae_checkpoint": checkpoints.get("autoencoder")}
        torch.manual_seed(0)
        module = engine.create_model(TrainingEngineConfig(
            num_symbols=tokenizer.vocab_size, num_speakers=1,
            sample_rate=config["audio"]["sample_rate"], extra=extra), [filelist])
        root = tmp_path / stage
        root.mkdir()
        checkpoints[stage] = str(rt.fit(module, steps=1, root=root))
        if stage == "text_to_latent":
            training = module.train_dataloader().dataset.datasets[0]
            index = next(i for i, entry in enumerate(training.entries) if entry[1] == rt.SPOKEN)
            training_ids = [int(x) for x in training[index][1]]

    out = tmp_path / "voice"
    vector_estimator = engine.export_onnx(
        checkpoints["autoencoder"], toy_22k / "config.json", out,
        text_to_latent_ckpt=checkpoints["text_to_latent"],
        duration_predictor_ckpt=checkpoints["duration_predictor"])
    voice = TTSVoice.load(vector_estimator, config_path=out / "tts.json", lang_code="en", engine_params={
        "duration_predictor_path": str(out / "duration_predictor.onnx"),
        "text_encoder_path": str(out / "text_encoder.onnx"),
        "vocoder_path": str(out / "vocoder.onnx"),
        "unicode_indexer_path": str(out / "unicode_indexer.json"),
        "tts_config_path": str(out / "tts.json")})
    with rt.known_defect(RuntimeError, "missing style_path"):
        audio, fed = rt.speak(voice)
    rt.assert_round_trip(training_ids, fed, audio, _min_samples(voice, training_ids))


def test_zipvoice_round_trip(toy_22k, tmp_path):
    """ZipVoice clones in context from a reference clip and its text. The export
    writes the text encoder and the flow-matching decoder; the mel front end comes
    from scripts/conversion/zipvoice/export_mel.py, as for a released voice, and
    Griffin-Lim stands in for the Vocos vocoder."""
    import importlib.util

    script = Path(__file__).parent.parent / "scripts" / "conversion" / "zipvoice" / "export_mel.py"
    spec = importlib.util.spec_from_file_location("zipvoice_export_mel", script)
    export_mel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_mel)
    mel_path = tmp_path / "zipvoice_mel.onnx"
    torch.onnx.export(export_mel.ZipVoiceMel().eval(), torch.randn(1, 24000), str(mel_path),
                      input_names=["waveform"], output_names=["mel"],
                      dynamic_axes={"waveform": {0: "b", 1: "samples"}}, opset_version=17,
                      dynamo=False)
    vocoder = build_vocoder(vocoder_type="griffinlim", config={
        **GRIFFIN_LIM["vocoder_config"], "sample_rate": 24000, "num_mels": 100, "mel_fmax": 12000.0})
    out = tmp_path / "voice"
    _round_trip(
        "zipvoice", toy_22k, tmp_path, vocoder=vocoder,
        training_ids=lambda module: _training_ids(module, lambda item: item[0]),
        voice_params={"text_encoder_path": str(out / "text_encoder.onnx"),
                      "mel_path": str(mel_path), "num_step": 2},
        syn_config=SynthesisConfig(speaker_reference=str(toy_22k.parent / "u0.wav"),
                                   speaker_reference_text=rt.TEXTS[0]),
        quality="low", num_test_examples=0)


TINY_STYLETTS2 = {
    "dim_in": 16, "hidden_dim": 32, "max_conv_dim": 32, "n_layer": 1, "n_mels": 80,
    "n_token": 178, "max_dur": 50, "style_dim": 16, "dropout": 0.0,
    "diffusion": {"transformer": {"num_layers": 1, "num_heads": 2, "head_features": 8,
                                  "multiplier": 1}},
}
# The decoder trunk is 1024 channels wide whatever the model size, and its generator
# reads 512 channels, so upsample_initial_channel stays 512.
HIFIGAN = {"type": "hifigan", "resblock_kernel_sizes": [3], "upsample_rates": [10, 5, 3, 2],
           "upsample_initial_channel": 512, "resblock_dilation_sizes": [[1, 3, 5]],
           "upsample_kernel_sizes": [20, 10, 6, 4]}
ISTFTNET = {"type": "istftnet", "resblock_kernel_sizes": [3], "upsample_rates": [10, 6],
            "upsample_initial_channel": 512, "resblock_dilation_sizes": [[1, 3, 5]],
            "upsample_kernel_sizes": [20, 12], "gen_istft_n_fft": 20, "gen_istft_hop_size": 5}
TRAILING_PAD = pytest.mark.xfail(strict=True, raises=rt.KnownDefect, reason=(
    "StyleTTS2 training reads [0, *ids, 0] (meldataset.FilePathDataset._load_tensor pads both "
    "ends), but StyleTTS2Adapter pads the start only for a voice with no style pack, so the "
    "graph never reads the trailing pad that training read"))
ISTFT_EXPORT = pytest.mark.xfail(strict=True, raises=rt.KnownDefect, reason=(
    "the iSTFTNet decoder of the low and medium presets calls torch.stft with "
    "return_complex=True, and the TorchScript ONNX exporter stops with "
    "'Unknown number type: complex'"))


@pytest.mark.parametrize("decoder, defect", [
    pytest.param(HIFIGAN, "trailing pad", id="hifigan", marks=[TRAILING_PAD, pytest.mark.slow]),
    pytest.param(ISTFTNET, "complex stft", id="istftnet", marks=ISTFT_EXPORT),
])
def test_styletts2_round_trip(decoder, defect, tmp_path):
    """StyleTTS2 reads the upstream list layout at 24 kHz, and tokenizes each line
    with the fixed 178-symbol table of phoonnx_train.styletts2.meldataset."""
    from phoonnx_train.styletts2.meldataset import dicts as styletts2_symbols

    data = tmp_path / "data"
    (data / "wavs").mkdir(parents=True)
    rt.write_clips(data / "wavs", 24000)
    lines = [f"u{i}.wav|{text}|0" for i, text in enumerate(rt.TEXTS)]
    (data / "train_list.txt").write_text("\n".join(lines) + "\n", "utf-8")
    (data / "val_list.txt").write_text(lines[0] + "\n", "utf-8")
    model_params = {**TINY_STYLETTS2, "decoder": decoder}
    config = {"lang_code": "en", "phoneme_type": "graphemes", "alphabet": "graphemes",
              "keep_punctuation": True, "styletts2": {"model_params": model_params}}
    (data / "config.json").write_text(json.dumps(config), "utf-8")

    engine = get_engine("styletts2")
    torch.manual_seed(0)
    module = engine.create_model(TrainingEngineConfig(
        num_symbols=178, num_speakers=1, sample_rate=24000,
        extra={"stage": "first", "model_params": model_params, "max_len": 192, "batch_size": 2,
               "num_workers": 0, "use_slm": False, "use_slm_adv": False,
               "loss_params": {"tma_epoch": 10_000}}), [data])
    checkpoint = rt.fit(module, steps=2, root=tmp_path)
    training = module.train_dataloader().dataset
    index = next(i for i, row in enumerate(training.data_list) if row[1] == rt.SPOKEN)
    training_ids = training[index][2].tolist()

    out = tmp_path / "voice"
    istft = defect == "complex stft"
    with rt.known_defect(RuntimeError, "Unknown number type: complex") if istft else nullcontext():
        onnx_path = engine.export_onnx(checkpoint, data / "config.json", out,
                                       phoneme_id_map={k: [v] for k, v in styletts2_symbols.items()})
    voice = TTSVoice.load(onnx_path, config_path=out / "config.json", engine_params={
        "speaker_encoder_path": str(out / "style_encoder.onnx"),
        "speaker_encoder_type": "styletts2_style"})
    audio, fed = rt.speak(voice, syn_config=SynthesisConfig(
        speaker_reference=str(data / "wavs" / "u0.wav")))
    # The trailing-pad defect: the graph reads the training ids without the last pad.
    without_pad = f"the graph read .*{re.escape(str(training_ids[:-1]))}"
    with rt.known_defect(AssertionError, without_pad) if defect == "trailing pad" else nullcontext():
        rt.assert_round_trip(training_ids, fed, audio, _min_samples(voice, training_ids))
