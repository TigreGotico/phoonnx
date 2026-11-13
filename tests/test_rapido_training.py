"""Tests for the Rápido training engine (phoonnx_train.engines.rapido).

Expected values are computed by hand from the layout rule (a word starts at a
non-boundary id after a boundary id; leading boundary ids join the first
word) and from the monotonic alignment of separable toy latents.
"""
import numpy as np
import pytest
import torch

from phoonnx.engines.rapido import layout_boundary_ids
from phoonnx.tokenizer import TTSTokenizer
from phoonnx_train.engines import get_engine, list_engines
from phoonnx_train.engines.base import BaseTrainingEngine
from phoonnx_train.engines.rapido import (
    _QUALITY_PRESETS,
    EMBEDDING_KEY,
    RapidoTrainingEngine,
    checkpoint_state,
)
from phoonnx_train.rapido.alignment import align, aligned_word_frames, batch_layout
from phoonnx_train.rapido.dataset import (
    RAPIDO_TOKENIZER_FLAGS,
    blank_id,
    collate_rapido,
    strip_blanks,
)
from phoonnx_train.rapido.lightning import RapidoModule
from phoonnx_train.rapido.model import RapidoConfig, RapidoModel, load_checkpoint
from tests.conftest import retry_download

# the special ids preprocess.py writes, then the symbols of the toy utterance
ID_MAP = {"_": 0, "^": 1, "$": 2, " ": 3, "o": 4, "l": 5, "a": 6, ",": 7,
          "t": 8, "u": 9, "d": 10, "b": 11, "ẽ": 12, "j": 13, "̃": 14,
          "?": 15, "s": 16, "ɐ": 17, "w": 18, "ˈ": 19}
CONFIG = {"phoneme_id_map": ID_MAP, "num_symbols": len(ID_MAP)}
# tugaphone output for "Olá, tudo bem? São" (keep_punctuation): no space after "?"
PHONEMES = ["o", "ˈ", "l", "a", ",", " ", "ˈ", "t", "u", "d", "u", " ",
            "ˈ", "b", "ẽ", "j", "̃", "?", "ˈ", "s", "ɐ", "̃", "w", "̃"]


def preprocess_ids(phonemes):
    return TTSTokenizer.from_phoonnx_config(CONFIG).tokenize(phonemes)


def tiny_module(**kwargs):
    torch.manual_seed(0)
    return RapidoModule(num_symbols=len(ID_MAP), num_speakers=2,
                        boundary_ids=sorted(layout_boundary_ids(ID_MAP)),
                        blank=0, sample_rate=22050, phoneme_id_map=ID_MAP,
                        model_params=_QUALITY_PRESETS["low"], **kwargs)


def toy_batch(seed=0):
    ids = strip_blanks(preprocess_ids(PHONEMES), 0)
    gen = torch.Generator().manual_seed(seed)
    items = [
        {"ids": torch.LongTensor(ids), "latents": torch.randn(20, 64, generator=gen), "speaker": 0},
        {"ids": torch.LongTensor(ids[:8]), "latents": torch.randn(9, 64, generator=gen), "speaker": 1},
    ]
    return collate_rapido(items)


# ----------------------------------------------------------------------
# Registry and presets
# ----------------------------------------------------------------------

def test_registered():
    assert "rapido" in list_engines()
    engine = get_engine("rapido")
    assert isinstance(engine, RapidoTrainingEngine)
    assert isinstance(engine, BaseTrainingEngine)


def test_base_preset_is_the_released_architecture():
    assert _QUALITY_PRESETS["base"] == {}
    assert RapidoConfig().d == 384 and RapidoConfig().n_layers == 8


def test_low_preset_builds_a_small_model():
    model = RapidoModel(RapidoConfig(**_QUALITY_PRESETS["low"]), n_vocab=20, n_speakers=1)
    assert sum(p.numel() for p in model.parameters()) < 1_000_000


# ----------------------------------------------------------------------
# Token stream
# ----------------------------------------------------------------------

def test_preprocess_interleaves_blanks_that_rapido_drops():
    ids = preprocess_ids(["o", "l"])
    assert ids == [1, 0, 4, 0, 5, 0, 2]
    assert strip_blanks(ids, blank_id(CONFIG)) == [1, 4, 5, 2]


def test_stripped_ids_equal_the_runtime_tokenizer_of_the_exported_voice():
    runtime = TTSTokenizer.from_phoonnx_config({**CONFIG, **RAPIDO_TOKENIZER_FLAGS})
    assert strip_blanks(preprocess_ids(PHONEMES), 0) == runtime.tokenize(PHONEMES)


def test_eval_text_to_ids_matches_the_training_ids():
    from phoonnx.config import Alphabet, PhonemeType, get_phonemizer
    from phoonnx.util import normalize

    text = "Olá, tudo bem? São às 21 horas."
    config = {"phoneme_type": "tugaphone", "alphabet": "ipa", "lang_code": "pt-BR",
              "keep_punctuation": True, "phoneme_id_map": {}}
    phonemizer = get_phonemizer(PhonemeType("tugaphone"), Alphabet.IPA, "",
                                keep_punctuation=True)
    phonemes = phonemizer.phonemize_to_list(normalize(text, "pt-BR"), "pt-BR")
    id_map = {"_": 0, "^": 1, "$": 2, " ": 3}
    for p in phonemes:
        id_map.setdefault(p, len(id_map))
    config["phoneme_id_map"] = id_map
    training = strip_blanks(TTSTokenizer.from_phoonnx_config(config).tokenize(phonemes), 0)

    assert RapidoTrainingEngine().eval_text_to_ids(text, config) == training
    assert 0 not in training


# ----------------------------------------------------------------------
# Word and frame layout
# ----------------------------------------------------------------------

def test_boundary_ids_are_separator_clause_tokens_bos_and_eos():
    assert layout_boundary_ids(ID_MAP) == {3, 7, 15, 1, 2}


def test_word_layout_from_ids_matches_hand_computed_values():
    ids = torch.LongTensor([strip_blanks(preprocess_ids(PHONEMES), 0)])
    # ^ o ˈ l a , ␠ | ˈ t u d u ␠ | ˈ b ẽ j ̃ ? | ˈ s ɐ ̃ w ̃ $
    words = [0] * 7 + [1] * 6 + [2] * 6 + [3] * 7
    starts = [0] * 7 + [7] * 6 + [13] * 6 + [19] * 7
    assert ids.shape[1] == len(words) == 26
    durations = torch.ones(1, 26) * 0.5          # 13 frames in all
    fmask = torch.ones(1, 13, dtype=torch.bool)
    mask = torch.ones(1, 26, dtype=torch.bool)
    cw, wstart, fw, fp = batch_layout(ids, mask, durations, fmask, layout_boundary_ids(ID_MAP))
    assert cw[0].tolist() == words
    assert wstart[0].tolist() == starts
    # word durations 3.5, 3, 3, 3.5 -> boundaries at rint(3.5, 6.5, 9.5, 13) = 4, 6, 10, 13
    assert fw[0].tolist() == [0] * 4 + [1] * 2 + [2] * 4 + [3] * 3
    assert fp[0].tolist() == pytest.approx([0, .25, .5, .75, 0, .5, 0, .25, .5, .75, 0, 1 / 3, 2 / 3])


def test_batch_layout_pads_with_minus_one():
    batch = toy_batch()
    durations = batch["mask"].float() * 0.3
    cw, _, fw, _ = batch_layout(batch["ids"], batch["mask"], durations, batch["fmask"],
                                layout_boundary_ids(ID_MAP))
    assert (cw[1, 8:] == -1).all() and (cw[1, :8] >= 0).all()
    assert (fw[1, 9:] == 0).all()


@pytest.mark.parametrize("durations, words, frames, expected", [
    ([0.25, 0.5, 0.25, 1.0, 3.0, 1.0, 0.25, 5.75], [0, 0, 0, 0, 1, 1, 1, 1], 12, [2, 10]),
    ([1.4, 1.4, 1.2], [0, 1, 2], 4, [1, 2, 1]),
    ([0.25, 0.25, 7.5], [0, 0, 1], 8, [1, 7]),
    ([0.1, 0.1, 0.1, 2.7], [0, 1, 2, 3], 3, [0, 0, 0, 3]),
])
def test_aligned_word_frames(durations, words, frames, expected):
    counts = aligned_word_frames(np.array(durations), np.array(words), frames)
    assert counts.tolist() == expected
    assert counts.sum() == frames


# ----------------------------------------------------------------------
# Monotonic alignment
# ----------------------------------------------------------------------

def test_alignment_recovers_the_durations_of_separable_latents():
    mu = torch.zeros(1, 3, 64)
    for i in range(3):
        mu[0, i, i] = 10.0
    true = [2, 3, 1]
    z = torch.cat([mu[0, i:i + 1].expand(n, 64) for i, n in enumerate(true)])[None]
    attn = align(mu, z, torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 6, dtype=torch.bool), 4)
    assert attn.sum(-1)[0].tolist() == [2.0, 3.0, 1.0]


def test_alignment_durations_sum_to_frames_and_are_monotonic():
    gen = torch.Generator().manual_seed(1)
    mu, z = torch.randn(2, 7, 64, generator=gen), torch.randn(2, 10, 64, generator=gen)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, 5:] = False
    fmask = torch.ones(2, 10, dtype=torch.bool)
    fmask[1, 6:] = False
    attn = align(mu, z, mask, fmask, 4)
    durations = attn.sum(-1)
    assert durations[0].sum().item() == pytest.approx(10)
    assert durations[1].sum().item() == pytest.approx(6)
    assert (durations[0] >= 0.25).all() and (durations[1, :5] >= 0.25).all()
    assert (durations[1, 5:] == 0).all() and (attn[1, :, 6:] == 0).all()
    # monotonic: the first letter holding each frame never moves backwards
    first = (attn[0] > 0).float().argmax(0)
    assert (first[1:] >= first[:-1]).all()


# ----------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------

def test_losses_on_a_toy_batch_are_finite_and_reach_every_stage():
    module = tiny_module()
    # the aligner starts with a zero frame query and a zero distance bias, so
    # its query, key, width and temperature get no gradient on step one;
    # trained values give every parameter a gradient
    aligner = module.model.net.aligner
    with torch.no_grad():
        aligner.frame_q.normal_()
        aligner.bias_w.fill_(1.0)
    out = module(toy_batch(), cond_drop=0.0)
    for name in ("flow_loss", "prior_loss", "dur_loss"):
        assert torch.isfinite(out[name]) and out[name] > 0
    assert out["durations"][0].sum().item() == pytest.approx(20)
    assert out["durations"][1].sum().item() == pytest.approx(9)
    module._total(out).backward()
    no_grad = [n for n, p in module.named_parameters() if p.grad is None or not p.grad.any()]
    assert no_grad == []


def test_flow_target_is_data_minus_noise_with_noise_at_t0():
    module = tiny_module()
    batch = toy_batch()
    z = module.normalize(batch["latents"])
    # at t = 0 the input is pure noise x0, so "z - x" is the exact velocity
    module.model.velocity = lambda x, cond, t, fmask: z - x
    out = module(batch, t=torch.zeros(2))
    assert out["flow_loss"].item() == pytest.approx(0.0, abs=1e-10)
    out = module(batch, t=torch.full((2,), 0.5))
    assert out["flow_loss"].item() > 0.1


def test_condition_drop_cuts_the_text_out_of_the_flow_loss():
    module = tiny_module()
    out = module(toy_batch(), cond_drop=1.0)
    out["flow_loss"].backward()
    aligner = [p.grad for n, p in module.named_parameters() if ".aligner." in n]
    assert all(g is None or not g.any() for g in aligner)


def test_duration_loss_is_zero_for_the_aligned_log_durations():
    module = tiny_module()
    batch = toy_batch()
    with torch.no_grad():
        h = module.model.net.text(batch["ids"], batch["mask"])
        expected = align(module.prior(h), module.normalize(batch["latents"]),
                         batch["mask"], batch["fmask"], 4).sum(-1)
    module.model.net.chardur.log_duration = lambda h, mask: torch.log1p(expected) * mask
    assert module(batch)["dur_loss"].item() == pytest.approx(0.0, abs=1e-12)


def test_normalization_uses_the_latent_buffers():
    module = tiny_module()
    module.model.lat_mean.fill_(2.0)
    module.model.lat_std.fill_(4.0)
    assert module.normalize(torch.full((1, 1, 64), 10.0)).unique().tolist() == [2.0]


# ----------------------------------------------------------------------
# Freezing
# ----------------------------------------------------------------------

def test_backbone_freeze_leaves_text_aligner_and_speakers_trainable():
    module = tiny_module(freeze_backbone_steps=5)
    trainable = {n for n, _ in module.named_parameters() if module._trainable(n)}
    assert "model.net.blocks.0.attn.qkv.weight" not in trainable
    assert "model.net.out_proj.weight" not in trainable
    assert {"model.net.text.emb.weight", "model.net.chardur.out.weight",
            "model.net.aligner.q.weight", "model.spk.weight", "prior.weight"} <= trainable


def test_text_only_trains_the_text_encoder_and_duration_predictor():
    module = tiny_module(text_only=True)
    trainable = {n for n, _ in module.named_parameters() if module._trainable(n)}
    assert all(n.startswith(("model.net.text.", "model.net.chardur.", "prior.")) for n in trainable)
    assert "model.net.text.emb.weight" in trainable


# ----------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------

def base_module(num_symbols, num_speakers):
    id_map = {f"s{i}": i for i in range(num_symbols)}
    return RapidoModule(num_symbols=num_symbols, num_speakers=num_speakers,
                        boundary_ids=[], blank=None, sample_rate=22050,
                        phoneme_id_map=id_map)


def test_load_released_rapido_checkpoint_counts(tmp_path):
    from huggingface_hub import hf_hub_download

    path = retry_download(hf_hub_download, "edwixx/rapido-tts", "rapido.pt")
    released = load_checkpoint(path)
    n_vocab, n_speakers = len(released["vocab"]), len(released["spk_order"])
    total = len(released["ema"])

    # IPA vocabulary of another size, one speaker: embedding and speaker table drop
    module = base_module(n_vocab + 30, 1)
    get_engine("rapido").load_checkpoint(module, path)
    report = module.load_report
    assert sorted(report["dropped_shape"]) == [EMBEDDING_KEY, "spk.weight"]
    assert len(report["loaded"]) == total - 2
    loaded = module.model.state_dict()["net.blocks.0.attn.qkv.weight"]
    assert torch.equal(loaded, released["ema"]["net.blocks.0.attn.qkv.weight"].float())

    # same vocabulary size, other symbols: the shape matches, the embedding still drops
    module = base_module(n_vocab, n_speakers)
    get_engine("rapido").load_checkpoint(module, path)
    assert module.load_report["dropped_shape"] == []
    assert len(module.load_report["loaded"]) == total - 1
    assert EMBEDDING_KEY not in module.load_report["loaded"]


def test_load_own_lightning_checkpoint_keeps_everything(tmp_path):
    module = tiny_module()
    for p in module.parameters():
        torch.nn.init.normal_(p)
    path = tmp_path / "own.ckpt"
    torch.save({"state_dict": module.state_dict(),
                "hyper_parameters": {"phoneme_id_map": ID_MAP}}, path)
    fresh = tiny_module()
    get_engine("rapido").load_checkpoint(fresh, path)
    assert len(fresh.load_report["loaded"]) == len(fresh.model.state_dict())
    assert torch.equal(fresh.prior.weight, module.prior.weight)
    assert torch.equal(fresh.model.net.text.emb.weight, module.model.net.text.emb.weight)

    fresh = tiny_module()
    get_engine("rapido").load_checkpoint(fresh, path, discard_encoder=True)
    assert EMBEDDING_KEY not in fresh.load_report["loaded"]


def test_checkpoint_state_strips_the_lightning_prefix():
    state = {"model.net.x": 1, "prior.weight": 2}
    assert checkpoint_state({"state_dict": state}) == {"net.x": 1}
    assert checkpoint_state({"ema": {"net.x": 3}}) == {"net.x": 3}
