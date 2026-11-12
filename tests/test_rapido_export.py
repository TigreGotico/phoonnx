"""Rápido torch modules and their ONNX export.

A tiny randomly-initialised model is exported and each graph is checked against the
torch module it came from; then the whole voice runs through ``TTSVoice`` and is
compared with the torch sampler on the same noise. The released weights are checked
against values computed by the upstream rapido-tts code (tests/fixtures).
"""
import copy
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime
import pytest
import torch

from phoonnx.config import SynthesisConfig
from phoonnx.engines.rapido import RapidoAdapter, frame_layout, word_frame_counts, word_layout
from phoonnx.voice import TTSVoice
from phoonnx_train.rapido.audio_vae import AudioVAE, AudioVAEConfig
from phoonnx_train.rapido.export_onnx import export_rapido
from phoonnx_train.rapido.model import RapidoConfig, RapidoModel

from tests.conftest import retry_download

TINY = RapidoConfig(d=32, n_heads=2, n_layers=2, text_conv=1, text_attn=1, align_heads=2, dur_hidden=16,
                    latent_dim=8)
TINY_VAE = AudioVAEConfig(encoder_dim=4, encoder_rates=(2, 2), latent_dim=8, decoder_dim=16, decoder_rates=(2, 2))
ID_MAP = {"<pad>": 0, " ": 1, "a": 2, "b": 3, "c": 4, ".": 5, "!": 6}
IDS = [2, 3, 6, 1, 4, 2, 1, 3, 3, 2, 5]   # "ab! ca bba."
BOUNDARIES = {1, 5, 6}


def _session(path):
    return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(0)
    model = RapidoModel(TINY, n_vocab=len(ID_MAP), n_speakers=3).eval()
    with torch.no_grad():
        model.net.chardur.out.bias.fill_(math.log(4.0))   # about 3 frames per token
        model.lat_mean.normal_()
        model.lat_std.uniform_(0.5, 2.0)
    vae = AudioVAE(TINY_VAE).eval()
    reference_vae = copy.deepcopy(vae)
    voice_config = {
        "lang_code": "pt-BR", "phoneme_type": "graphemes", "alphabet": "unicode", "keep_punctuation": True,
        "phoneme_id_map": ID_MAP, "speaker_id_map": {"x": 0, "y": 1, "z": 2}, "pad": "<pad>",
        "add_blank_char": False, "add_blank_word": False, "use_eos_bos": False,
        "blank_at_start": False, "blank_at_end": False, "word_sep_token": " ",
        "inference": {"add_diacritics": False}}
    config_path = export_rapido(model, voice_config, tmp_path_factory.mktemp("rapido"), vae=vae)
    return model, reference_vae, vae, config_path


def _layout(model, ids, length_scale=1.0):
    t = torch.tensor([ids])
    mask = torch.ones_like(t, dtype=torch.bool)
    h, dur = model.net.text_stage(t, mask)
    word, start = word_layout(ids, BOUNDARIES)
    frame_word, frame_pos = frame_layout(word_frame_counts(dur[0].detach().numpy() * length_scale, word))
    return t, mask, h, dur, word, start, frame_word, frame_pos


def test_config_is_a_native_rapido_voice(exported):
    _, _, _, config_path = exported
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["engine"] == "rapido"
    assert config["audio"]["sample_rate"] == 48000
    assert config["hop_length"] == 4                        # the tiny decoder's 2 * 2
    assert config["phoneme_id_map"] == ID_MAP
    assert config["num_speakers"] == 3 and config["num_symbols"] == len(ID_MAP)
    assert not config["add_blank_char"] and not config["use_eos_bos"]
    ep = config["engine_params"]
    assert all(Path(ep[k]).is_file() for k in ("text_encoder_path", "aligner_path", "decoder_path"))
    assert (ep["steps"], ep["guidance"], ep["seed"]) == (8, 4.0, 0)
    assert len(ep["latent_mean"]) == len(ep["latent_std"]) == TINY.latent_dim


@torch.no_grad()
def test_text_encoder_graph_matches_torch(exported):
    model, _, _, config_path = exported
    _, _, h, dur, *_ = _layout(model, IDS)
    features, durations = _session(config_path.parent / "text_encoder.onnx").run(
        None, {"ids": np.array([IDS], np.int64)})
    np.testing.assert_allclose(features, h.numpy(), atol=1e-5)
    np.testing.assert_allclose(durations, dur.numpy(), rtol=1e-5)
    assert durations.min() > 1.0     # the bias above makes the layout non-trivial


@torch.no_grad()
def test_aligner_graph_matches_torch(exported):
    model, _, _, config_path = exported
    _, mask, h, dur, word, start, frame_word, frame_pos = _layout(model, IDS)
    session = _session(config_path.parent / "aligner.onnx")
    for speaker in (0, 2):
        expected = model.condition(h, dur, mask, torch.from_numpy(word)[None], torch.from_numpy(start)[None],
                                   torch.from_numpy(frame_word)[None], torch.from_numpy(frame_pos)[None],
                                   torch.tensor([speaker]))
        got = session.run(None, {"features": h.numpy(), "durations": dur.numpy(), "word": word[None],
                                 "word_start": start[None], "frame_word": frame_word[None],
                                 "frame_pos": frame_pos[None], "speaker": np.array([speaker], np.int64)})[0]
        np.testing.assert_allclose(got, expected.numpy(), atol=1e-5)


@torch.no_grad()
def test_model_graph_folds_classifier_free_guidance(exported):
    model, _, _, config_path = exported
    session = _session(config_path.parent / "model.onnx")
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 13, TINY.latent_dim)).astype(np.float32)
    cond = rng.standard_normal((1, 13, TINY.d)).astype(np.float32)
    mask = torch.ones(1, 13, dtype=torch.bool)
    for t, guidance in ((0.25, 4.0), (0.75, 1.0)):
        tt = torch.full((1,), t)
        vc = model.velocity(torch.from_numpy(x), torch.from_numpy(cond), tt, mask)
        vu = model.velocity(torch.from_numpy(x), torch.zeros(1, 13, TINY.d), tt, mask)
        got = session.run(None, {"x": x, "condition": cond, "t": np.array([t], np.float32),
                                 "guidance": np.array([guidance], np.float32)})[0]
        np.testing.assert_allclose(got, (vu + guidance * (vc - vu)).numpy(), atol=1e-5)


@torch.no_grad()
def test_decoder_graph_matches_torch_with_weight_norm_folded(exported):
    _, reference_vae, folded_vae, config_path = exported
    z = torch.randn(1, TINY.latent_dim, 9)
    expected = reference_vae.decode(z)
    np.testing.assert_allclose(folded_vae.decode(z).numpy(), expected.numpy(), atol=1e-6)
    got = _session(config_path.parent / "decoder.onnx").run(None, {"latents": z.numpy()})[0]
    assert got.shape == (1, 1, 9 * 4)
    np.testing.assert_allclose(got, expected.numpy(), atol=1e-5)


@torch.no_grad()
def test_voice_end_to_end_matches_the_torch_sampler(exported):
    model, reference_vae, _, config_path = exported
    voice = TTSVoice.load(config_path.parent / "model.onnx", config_path=config_path)
    assert isinstance(voice.adapter, RapidoAdapter)
    assert voice.config.sample_rate == 48000
    for speaker, length_scale, seed in ((0, 1.0, 0), (2, 1.5, 4)):
        audio = voice.phoneme_ids_to_audio(IDS, SynthesisConfig(speaker_id=speaker, length_scale=length_scale,
                                                                 extra_params={"seed": seed}))
        z = model.generate(IDS, BOUNDARIES, speaker=speaker, length_scale=length_scale, seed=seed)
        expected = reference_vae.decode(z.T[None]).reshape(-1)
        assert audio.shape == expected.shape
        np.testing.assert_allclose(audio, expected.numpy(), atol=1e-4)


def test_audio_vae_round_trip_shapes():
    vae = AudioVAE(TINY_VAE).eval()
    with torch.no_grad():
        z = vae.encode(torch.randn(2, 1, 13), 16000)
        assert z.shape == (2, TINY_VAE.latent_dim, 4)      # 13 samples padded to 16, hop 4
        assert vae.decode(z).shape == (2, 1, 16)
    with pytest.raises(ValueError):
        vae.encode(torch.randn(1, 1, 8), 22050)


def test_released_weights_match_upstream():
    """rapido.pt and audiovae.pth through our modules give the values the upstream
    rapido-tts code computed for the same sentence and noise."""
    from huggingface_hub import hf_hub_download

    from phoonnx_train.rapido.audio_vae import load_audio_vae
    from phoonnx_train.rapido.model import load_checkpoint

    gold = json.loads((Path(__file__).parent / "fixtures" / "rapido_upstream_reference.json").read_text("utf-8"))
    checkpoint = load_checkpoint(retry_download(hf_hub_download, "edwixx/rapido-tts", "rapido.pt"))
    retry_download(hf_hub_download, "openbmb/VoxCPM2", "audiovae.pth")
    model = RapidoModel.from_checkpoint(checkpoint)
    vae = load_audio_vae()
    ids = gold["ids"]
    space = checkpoint["vocab"].index(" ")
    with torch.no_grad():
        t = torch.tensor([ids])
        _, dur = model.net.text_stage(t, torch.ones_like(t, dtype=torch.bool))
        np.testing.assert_allclose(dur[0].numpy(), gold["durations"], atol=1e-5)
        word, _ = word_layout(ids, {space})
        assert word_frame_counts(dur[0].numpy(), word).tolist() == gold["word_frames"]
        frames = sum(gold["word_frames"])
        for speaker, expected in gold["speakers"].items():
            noise = torch.randn(1, frames, 64, generator=torch.Generator().manual_seed(0))
            z = model.generate(ids, {space}, speaker=int(speaker), steps=gold["steps"],
                               guidance=gold["guidance"], noise=noise)
            np.testing.assert_allclose(z.norm(dim=1).numpy(), expected["latent_frame_norms"], rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(z[0].numpy(), expected["latent_first_frame"], atol=1e-4)
            audio = vae.decode(z.T[None]).reshape(-1, 1920)
            np.testing.assert_allclose(audio.pow(2).mean(1).sqrt().numpy(), expected["audio_frame_rms"], atol=1e-5)
