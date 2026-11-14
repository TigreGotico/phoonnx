"""Tests for the SuperTonic training package (phoonnx_train.supertonic).

Everything runs on CPU at a tiny config in a second or two: module forward
shapes, loss and mask math, full-checkpoint save/resume round-trips (including
corrupt-file and grow-vocab handling), the four-graph ONNX export IO signature
run through onnxruntime, and the training-engine registry wiring. Adversarial
cases cover zero-length text, a resampled sample-rate mismatch, an empty indexer
and a truncated checkpoint.
"""
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phoonnx_train.supertonic.autoencoder import SpeechAutoencoder
from phoonnx_train.supertonic.checkpointing import (
    CheckpointError,
    load_state_dict_grow_vocab,
    resume_into,
    save_checkpoint,
)
from phoonnx_train.supertonic.config import SuperTonicConfig, load_model_config, tiny_config
from phoonnx_train.supertonic.duration_predictor import DurationPredictor, duration_loss
from phoonnx_train.supertonic.latent_utils import (
    compress,
    decompress,
    normalize_and_compress,
    sample_reference_crop,
)
from phoonnx_train.supertonic.layers import make_mask
from phoonnx_train.supertonic.text import AVAILABLE_LANGS, CharTokenizer, normalize_text
from phoonnx_train.supertonic.text_to_latent import TextToLatentModel, flow_matching_loss


@pytest.fixture()
def cfg():
    return tiny_config(vocab_size=64)


@pytest.fixture()
def ae(cfg):
    torch.manual_seed(0)
    return SpeechAutoencoder(cfg.ae)


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------

def test_compressed_dim_relation(cfg):
    assert cfg.ttl.compressed_dim == cfg.ttl.latent_dim * cfg.ttl.compress_factor


def test_load_model_config_overlays_ttsjson(tmp_path):
    p = tmp_path / "tts.json"
    p.write_text(json.dumps({"ae": {"sample_rate": 44100, "ldim": 32}}))
    cfg = load_model_config(str(p))
    assert cfg.ae.sample_rate == 44100 and cfg.ae.latent_dim == 32


def test_load_model_config_none_is_defaults():
    assert load_model_config(None) == SuperTonicConfig()


# ----------------------------------------------------------------------
# module forward shapes
# ----------------------------------------------------------------------

def test_autoencoder_roundtrip_shapes(ae):
    wav = torch.randn(2, 16000)
    recon, latent = ae(wav)
    assert recon.shape[0] == 2 and recon.dim() == 2
    assert latent.shape[:2] == (2, ae.cfg.latent_dim)


def test_text_to_latent_loss_backward(cfg, ae):
    torch.manual_seed(1)
    _, latent = ae(torch.randn(2, 16000))
    z1 = normalize_and_compress(ae, latent.detach(), cfg.ttl.compress_factor, cfg.ttl.normalizer_scale)
    lens = torch.tensor([z1.shape[-1], z1.shape[-1] - 1])
    lat_mask = make_mask(lens, z1.shape[-1])
    ref, ref_mask, rtm = sample_reference_crop(z1, lens, 10.0)
    text_ids = torch.randint(1, 64, (2, 7))
    text_mask = make_mask(torch.tensor([7, 5]), 7)
    ttl = TextToLatentModel(cfg.ttl, cfg.vocab_size)
    loss = flow_matching_loss(ttl, z1, lat_mask, text_ids, text_mask, ref, ref_mask, rtm,
                              n_expand=cfg.ttl.batch_expand)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in ttl.parameters())


def test_duration_predictor_positive(cfg, ae):
    _, latent = ae(torch.randn(2, 16000))
    z1 = normalize_and_compress(ae, latent.detach(), cfg.dp.compress_factor, cfg.dp.normalizer_scale)
    lens = torch.tensor([z1.shape[-1], z1.shape[-1]])
    ref, ref_mask, _ = sample_reference_crop(z1, lens, 10.0)
    text_ids = torch.randint(1, 64, (2, 7))
    text_mask = make_mask(torch.tensor([7, 6]), 7)
    dp = DurationPredictor(cfg.dp, cfg.vocab_size)
    dur = dp(text_ids, text_mask, ref, ref_mask)
    assert dur.shape == (2,) and (dur > 0).all()  # exp() output is strictly positive
    assert torch.isfinite(duration_loss(dur, torch.tensor([1.0, 1.2])))


# ----------------------------------------------------------------------
# mask / compression math
# ----------------------------------------------------------------------

def test_make_mask_values():
    m = make_mask(torch.tensor([2, 4]), 4)
    assert m.shape == (2, 1, 4)
    assert m[0, 0].tolist() == [1, 1, 0, 0]
    assert m[1, 0].tolist() == [1, 1, 1, 1]


def test_compress_decompress_roundtrip():
    x = torch.arange(2 * 3 * 12).float().reshape(2, 3, 12)
    c = compress(x, 4)
    assert c.shape == (2, 12, 3)
    back = decompress(c, 4, 3)
    assert torch.equal(back, x)


def test_compress_trims_to_multiple():
    x = torch.randn(1, 3, 13)
    assert compress(x, 4).shape == (1, 12, 3)  # 13 -> 12


def test_reference_crop_masks_align():
    z1 = torch.randn(3, 5, 40)
    lens = torch.tensor([40, 20, 10])
    ref, ref_mask, rtm = sample_reference_crop(z1, lens, frame_rate=10.0)
    assert ref.shape[0] == 3 and ref_mask.shape[1] == 1
    # the reference-time mask marks at least one frame per sample
    assert (rtm.sum(dim=(1, 2)) >= 1).all()


# ----------------------------------------------------------------------
# tokenizer
# ----------------------------------------------------------------------

def test_tokenizer_wraps_lang_tag():
    assert normalize_text("hello", "en").startswith("<en>") and normalize_text("hi", "en").endswith("</en>")


def test_tokenizer_adds_terminal_punctuation():
    assert normalize_text("hello", "en") == "<en>hello.</en>"
    assert normalize_text("hi!", "en") == "<en>hi!</en>"


def test_tokenizer_unknown_lang_raises():
    with pytest.raises(ValueError):
        normalize_text("hello", "xx")


def test_tokenizer_encode_and_indexer():
    tok = CharTokenizer.build_from_texts(["hello world"], ["en"])
    ids = tok.encode("hello", "en")
    assert all(i > 0 for i in ids)  # every char is known
    table = tok.to_indexer_list()
    assert len(table) == 65536
    assert table[ord("h")] == tok.char2id["h"]


def test_tokenizer_extend_keeps_existing_ids():
    tok = CharTokenizer.build_from_texts(["abc"], ["en"])
    grown = tok.extend_with_texts(["abcñ"], ["es"])
    for c, i in tok.char2id.items():
        assert grown.char2id[c] == i  # row-aligned
    assert grown.vocab_size >= tok.vocab_size


def test_all_langs_present():
    for lang in ("en", "ko", "ja", "ar", "pt"):
        assert lang in AVAILABLE_LANGS


# ----------------------------------------------------------------------
# adversarial: zero-length text, sample-rate mismatch
# ----------------------------------------------------------------------

def test_zero_length_text_batch(cfg):
    # a batch where one utterance tokenizes to a single (padded) id must not crash
    text_ids = torch.zeros(2, 1, dtype=torch.long)
    text_mask = make_mask(torch.tensor([1, 1]), 1)
    ttl = TextToLatentModel(cfg.ttl, cfg.vocab_size)
    style = ttl.style_encoder(torch.randn(2, cfg.ttl.compressed_dim, 6))
    out = ttl.text_encoder(text_ids, text_mask, style)
    assert out.shape == (2, cfg.ttl.char_dim, 1)


def test_dataset_resamples_mismatched_sample_rate(tmp_path):
    import soundfile as sf
    from phoonnx_train.supertonic.dataset import TextAudioDataset
    wav = np.random.randn(8000).astype(np.float32)
    sf.write(str(tmp_path / "a.wav"), wav, 8000)  # 8 kHz source
    (tmp_path / "list.txt").write_text("a.wav|hello there|en\n")
    tok = CharTokenizer.build_from_texts(["hello there"], ["en"])
    ds = TextAudioDataset(str(tmp_path / "list.txt"), str(tmp_path), tok, sample_rate=16000)
    audio, ids = ds[0]
    assert audio.shape[0] == pytest.approx(16000, abs=64)  # resampled 8k -> 16k
    assert ids.numel() > 0


# ----------------------------------------------------------------------
# checkpointing
# ----------------------------------------------------------------------

def test_checkpoint_save_resume_roundtrip(tmp_path, ae):
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    ae(torch.randn(1, 16000))[0].sum().backward()
    opt.step()
    before = {k: v.clone() for k, v in ae.state_dict().items()}
    path = tmp_path / "ckpt.pt"
    save_checkpoint(str(path), step=42, models={"ae": ae}, optimizers={"ae": opt})

    fresh = SpeechAutoencoder(ae.cfg)
    fresh_opt = torch.optim.Adam(fresh.parameters(), lr=1e-3)
    step = resume_into(str(path), models={"ae": fresh}, optimizers={"ae": fresh_opt})
    assert step == 42
    for k, v in fresh.state_dict().items():
        assert torch.equal(v, before[k])


def test_checkpoint_atomic_no_partial_on_replace(tmp_path, ae):
    path = tmp_path / "ckpt.pt"
    save_checkpoint(str(path), step=1, models={"ae": ae})
    # no leftover temp files in the directory
    assert [p.name for p in tmp_path.iterdir()] == ["ckpt.pt"]


def test_resume_missing_file_raises(tmp_path, ae):
    with pytest.raises(CheckpointError):
        resume_into(str(tmp_path / "nope.pt"), models={"ae": ae})


def test_resume_corrupt_checkpoint_raises(tmp_path, ae):
    path = tmp_path / "ckpt.pt"
    save_checkpoint(str(path), step=1, models={"ae": ae})
    with open(path, "r+b") as fh:  # truncate to half its bytes
        size = os.fstat(fh.fileno()).st_size
        fh.truncate(size // 2)
    with pytest.raises(CheckpointError):
        resume_into(str(path), models={"ae": SpeechAutoencoder(ae.cfg)})


def test_grow_vocab_load(cfg):
    small = TextToLatentModel(cfg.ttl, vocab_size=20)
    big = TextToLatentModel(cfg.ttl, vocab_size=30)
    load_state_dict_grow_vocab(big, small.state_dict())
    # the first 20 embedding rows come from the small model, the rest keep init
    assert torch.equal(big.text_encoder.embed.weight[:20], small.text_encoder.embed.weight)


def test_grow_vocab_rejects_shrink(cfg):
    big = TextToLatentModel(cfg.ttl, vocab_size=30)
    small = TextToLatentModel(cfg.ttl, vocab_size=20)
    with pytest.raises(CheckpointError):
        load_state_dict_grow_vocab(small, big.state_dict())


# ----------------------------------------------------------------------
# ONNX export IO signature
# ----------------------------------------------------------------------

def _build_all(cfg):
    from phoonnx_train.supertonic.text_to_latent import TextToLatentModel as TTL
    torch.manual_seed(0)
    return (SpeechAutoencoder(cfg.ae), TTL(cfg.ttl, cfg.vocab_size),
            DurationPredictor(cfg.dp, cfg.vocab_size))


def test_onnx_export_io_signature(tmp_path, cfg):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from phoonnx_train.supertonic.export_onnx import export_all
    ae, ttl, dp = _build_all(cfg)
    tok = CharTokenizer.build_from_texts(["hello world"], ["en"])
    paths = export_all(str(tmp_path), config=cfg, tokenizer=tok,
                       autoencoder=ae, text_to_latent=ttl, duration_predictor=dp)
    for key in ("duration_predictor", "text_encoder", "vector_estimator", "vocoder"):
        assert paths[key].is_file()

    def run(path, feed):
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, feed)

    text_ids = np.random.randint(1, 64, (1, 6)).astype(np.int64)
    text_mask = np.ones((1, 1, 6), np.float32)
    style_dp = np.random.randn(1, cfg.dp.n_style, cfg.dp.style_value_dim).astype(np.float32)
    style_ttl = np.random.randn(1, cfg.ttl.n_style, cfg.ttl.style_dim).astype(np.float32)

    dur = run(paths["duration_predictor"], {"text_ids": text_ids, "style_dp": style_dp, "text_mask": text_mask})[0]
    assert dur.shape == (1,) and dur.dtype == np.float32
    te = run(paths["text_encoder"], {"text_ids": text_ids, "style_ttl": style_ttl, "text_mask": text_mask})[0]
    assert te.shape == (1, cfg.ttl.char_dim, 6)
    L = 4
    noisy = np.random.randn(1, cfg.ttl.compressed_dim, L).astype(np.float32)
    ve = run(paths["vector_estimator"], {
        "noisy_latent": noisy, "text_emb": te, "style_ttl": style_ttl, "text_mask": text_mask,
        "latent_mask": np.ones((1, 1, L), np.float32),
        "current_step": np.array([1.0], np.float32), "total_step": np.array([8.0], np.float32)})[0]
    assert ve.shape == noisy.shape
    wav = run(paths["vocoder"], {"latent": noisy})[0]
    assert wav.shape[0] == 1 and wav.ndim == 2


def test_exported_config_and_indexer(tmp_path, cfg):
    pytest.importorskip("onnx")
    from phoonnx_train.supertonic.export_onnx import export_all
    ae, ttl, dp = _build_all(cfg)
    tok = CharTokenizer.build_from_texts(["hello"], ["en"])
    paths = export_all(str(tmp_path), config=cfg, tokenizer=tok,
                       autoencoder=ae, text_to_latent=ttl, duration_predictor=dp)
    tts = json.loads(paths["tts"].read_text())
    assert tts["ae"]["sample_rate"] == cfg.ae.sample_rate
    assert tts["ttl"]["chunk_compress_factor"] == cfg.ttl.compress_factor
    idx = json.loads(paths["unicode_indexer"].read_text())
    assert len(idx) == 65536 and idx[ord("h")] == tok.char2id["h"]


def test_empty_indexer_is_all_pad():
    tok = CharTokenizer()  # nothing built
    table = tok.to_indexer_list(size=128)
    assert set(table) == {0}


# ----------------------------------------------------------------------
# import-onnx utility
# ----------------------------------------------------------------------

def test_import_onnx_reads_initializers(tmp_path, cfg):
    pytest.importorskip("onnx")
    from phoonnx_train.supertonic.export_onnx import export_all
    from phoonnx_train.supertonic.import_onnx import load_onnx_initializers
    ae, ttl, dp = _build_all(cfg)
    tok = CharTokenizer.build_from_texts(["hello"], ["en"])
    paths = export_all(str(tmp_path), config=cfg, tokenizer=tok,
                       autoencoder=ae, text_to_latent=ttl, duration_predictor=dp)
    arrays = load_onnx_initializers(str(paths["vocoder"]))
    assert arrays  # at least the decoder weights are present
    assert all(hasattr(v, "shape") for v in arrays.values())


# ----------------------------------------------------------------------
# engine registry
# ----------------------------------------------------------------------

def test_engine_registered():
    from phoonnx_train.engines import get_engine, list_engines
    from phoonnx_train.engines.supertonic import SuperTonicTrainingEngine
    assert "supertonic" in list_engines()
    assert isinstance(get_engine("supertonic"), SuperTonicTrainingEngine)


def test_engine_presets():
    from phoonnx_train.engines import get_engine
    presets = get_engine("supertonic").quality_presets()
    assert "base" in presets and "low" in presets


def test_engine_unknown_stage_raises(tmp_path):
    from phoonnx_train.engines import get_engine
    from phoonnx_train.engines.base import TrainingEngineConfig
    eng = get_engine("supertonic")
    cfg = TrainingEngineConfig(num_symbols=64, sample_rate=16000,
                               extra={"quality": "low", "stage": "nonsense"})
    with pytest.raises(KeyError):
        eng.create_model(cfg, [])


# ----------------------------------------------------------------------
# Lightning training loops (short CPU fit + resume)
# ----------------------------------------------------------------------

def _write_corpus(tmp_path, n=4, seconds=0.6, sr=16000):
    import soundfile as sf
    lines = []
    for i in range(n):
        sf.write(str(tmp_path / f"u{i}.wav"), np.random.randn(int(seconds * sr)).astype(np.float32), sr)
        lines.append(f"u{i}.wav|hello world number {i}|en")
    (tmp_path / "list.txt").write_text("\n".join(lines) + "\n")
    return str(tmp_path / "list.txt")


def _trainer(tmp_path, **kw):
    import pytorch_lightning as pl
    return pl.Trainer(accelerator="cpu", devices=1, logger=False,
                      enable_checkpointing=False, enable_progress_bar=False,
                      enable_model_summary=False, **kw)


def _fitted_ae_checkpoint(tmp_path, cfg, flist, name="ae.pt"):
    """Train the autoencoder for one step and save it with its latent statistics."""
    from phoonnx_train.supertonic.lightning import AutoencoderModule
    mod = AutoencoderModule(config=cfg, dataset=[flist], root_dir=str(tmp_path),
                            batch_size=2, num_workers=0, segment_seconds=0.4)
    _trainer(tmp_path, max_steps=1).fit(mod)
    path = tmp_path / name
    torch.save({"state_dict": mod.state_dict()}, path)
    return str(path)


def test_autoencoder_lightning_fit(tmp_path):
    pl = pytest.importorskip("pytorch_lightning")
    from phoonnx_train.supertonic.lightning import AutoencoderModule
    flist = _write_corpus(tmp_path)
    mod = AutoencoderModule(config=tiny_config(64), dataset=[flist], root_dir=str(tmp_path),
                            batch_size=2, num_workers=0, segment_seconds=0.4)
    _trainer(tmp_path, max_steps=2).fit(mod)
    # a config + tokenizer ride along in every checkpoint
    ck = {}
    mod.on_save_checkpoint(ck)
    assert ck["supertonic_stage"] == "autoencoder" and "supertonic_config" in ck


def test_ttl_and_dp_lightning_fit_with_frozen_ae(tmp_path):
    pl = pytest.importorskip("pytorch_lightning")
    from phoonnx_train.supertonic.lightning import (
        DurationPredictorModule,
        TextToLatentModule,
    )
    flist = _write_corpus(tmp_path)
    cfg = tiny_config(64)
    tok = CharTokenizer.build_from_texts(["hello world number 0"], ["en"])

    ae_ckpt = _fitted_ae_checkpoint(tmp_path, cfg, flist)

    ttl = TextToLatentModule(config=cfg, tokenizer=tok, dataset=[flist], root_dir=str(tmp_path),
                             ae_checkpoint=ae_ckpt, batch_size=2, num_workers=0)
    _trainer(tmp_path, max_steps=2).fit(ttl)
    assert all(not p.requires_grad for p in ttl.frozen_ae.parameters())

    dp = DurationPredictorModule(config=cfg, tokenizer=tok, dataset=[flist], root_dir=str(tmp_path),
                                 ae_checkpoint=ae_ckpt, batch_size=2, num_workers=0)
    _trainer(tmp_path, max_steps=2).fit(dp)


def test_lightning_resume_from_checkpoint(tmp_path):
    pl = pytest.importorskip("pytorch_lightning")
    from phoonnx_train.supertonic.lightning import DurationPredictorModule
    flist = _write_corpus(tmp_path)
    cfg = tiny_config(64)
    tok = CharTokenizer.build_from_texts(["hello world number 0"], ["en"])
    ck_path = tmp_path / "dp.ckpt"
    ae_ckpt = _fitted_ae_checkpoint(tmp_path, cfg, flist)

    dp = DurationPredictorModule(config=cfg, tokenizer=tok, dataset=[flist], root_dir=str(tmp_path),
                                 ae_checkpoint=ae_ckpt, batch_size=2, num_workers=0)
    tr = _trainer(tmp_path, max_steps=2)
    tr.fit(dp)
    tr.save_checkpoint(str(ck_path))

    dp2 = DurationPredictorModule(config=cfg, tokenizer=tok, dataset=[flist], root_dir=str(tmp_path),
                                  ae_checkpoint=ae_ckpt, batch_size=2, num_workers=0)
    _trainer(tmp_path, max_steps=4).fit(dp2, ckpt_path=str(ck_path))
    assert dp2.global_step >= 2  # continued past the resumed step


def test_export_from_lightning_checkpoints(tmp_path):
    pl = pytest.importorskip("pytorch_lightning")
    pytest.importorskip("onnx")
    from phoonnx_train.supertonic.export_onnx import export_from_checkpoints
    from phoonnx_train.supertonic.lightning import (
        AutoencoderModule,
        DurationPredictorModule,
        TextToLatentModule,
    )
    flist = _write_corpus(tmp_path)
    cfg = tiny_config(64)
    tok = CharTokenizer.build_from_texts(["hello world number 0"], ["en"])

    def fit_and_save(mod, name):
        tr = _trainer(tmp_path, max_steps=1)
        tr.fit(mod)
        p = tmp_path / name
        tr.save_checkpoint(str(p))
        return str(p)

    ae_ck = fit_and_save(AutoencoderModule(config=cfg, dataset=[flist], root_dir=str(tmp_path),
                                           batch_size=2, num_workers=0, segment_seconds=0.4), "ae.ckpt")
    ttl_ck = fit_and_save(TextToLatentModule(config=cfg, tokenizer=tok, dataset=[flist],
                                             root_dir=str(tmp_path), ae_checkpoint=ae_ck,
                                             batch_size=2, num_workers=0), "ttl.ckpt")
    dp_ck = fit_and_save(DurationPredictorModule(config=cfg, tokenizer=tok, dataset=[flist],
                                                 root_dir=str(tmp_path), ae_checkpoint=ae_ck,
                                                 batch_size=2, num_workers=0), "dp.ckpt")

    paths = export_from_checkpoints(str(tmp_path / "out"), autoencoder_ckpt=ae_ck,
                                    text_to_latent_ckpt=ttl_ck, duration_predictor_ckpt=dp_ck)
    for key in ("duration_predictor", "text_encoder", "vector_estimator", "vocoder"):
        assert paths[key].is_file()


# ----------------------------------------------------------------------
# semantics: masking, flow-matching target, reference-crop exclusion
# ----------------------------------------------------------------------

def test_style_pool_ignores_batch_padding(cfg):
    from phoonnx_train.supertonic.layers import StylePool
    torch.manual_seed(3)
    pool = StylePool(cfg.ttl.compressed_dim, cfg.ttl.n_style, cfg.ttl.style_dim, cfg.ttl.style_heads)
    ref = torch.randn(1, cfg.ttl.compressed_dim, 7)
    base = pool(ref, make_mask(torch.tensor([7]), 7))
    for pad in (1, 5, 20):
        padded = torch.cat([ref, torch.zeros(1, cfg.ttl.compressed_dim, pad)], dim=-1)
        out = pool(padded, make_mask(torch.tensor([7]), 7 + pad))
        assert (out - base).abs().max() < 1e-5, f"padding by {pad} moved the style tokens"


class _OracleVectorField:
    """Reproduces the optimal-transport target from ``zt`` and ``t`` alone."""

    def __init__(self, cfg, z1, sign=1.0):
        self.cfg, self.z1, self.sign = cfg, z1, sign

    def __call__(self, zt, t, *args):
        sig = self.cfg.sigma_min
        tb = t.view(-1, 1, 1)
        z0 = (zt - tb * self.z1) / (1 - (1 - sig) * tb)
        return self.sign * (self.z1 - (1 - sig) * z0)


class _OracleModel:
    def __init__(self, cfg, z1, sign=1.0):
        self.cfg = cfg
        self.vector_field = _OracleVectorField(cfg, z1, sign)

    def conditions(self, text_ids, text_mask, ref, ref_mask):
        return torch.zeros(text_ids.shape[0], self.cfg.char_dim, text_ids.shape[-1]), \
            torch.zeros(text_ids.shape[0], self.cfg.n_style, self.cfg.style_dim)


def test_flow_matching_target_is_the_ot_velocity(cfg):
    torch.manual_seed(4)
    z1 = torch.randn(2, cfg.ttl.compressed_dim, 9)
    lat_mask = make_mask(torch.tensor([9, 7]), 9)
    text_ids = torch.randint(1, 64, (2, 5))
    text_mask = make_mask(torch.tensor([5, 4]), 5)
    ref, ref_mask, rtm = sample_reference_crop(z1, torch.tensor([9, 7]), 10.0)

    args = (z1, lat_mask, text_ids, text_mask, ref, ref_mask, rtm)
    right = flow_matching_loss(_OracleModel(cfg.ttl, z1), *args, n_expand=1)
    wrong = flow_matching_loss(_OracleModel(cfg.ttl, z1, sign=-1.0), *args, n_expand=1)
    assert right.item() < 1e-4, f"the exact OT velocity should cost nothing, got {right.item()}"
    assert wrong.item() > 0.5, f"the sign-flipped velocity should cost a lot, got {wrong.item()}"


def test_flow_matching_loss_excludes_the_reference_region(cfg):
    torch.manual_seed(5)
    z1 = torch.randn(2, cfg.ttl.compressed_dim, 12)
    lens = torch.tensor([12, 12])
    lat_mask = make_mask(lens, 12)
    text_ids = torch.randint(1, 64, (2, 5))
    text_mask = make_mask(torch.tensor([5, 5]), 5)
    ref, ref_mask, rtm = sample_reference_crop(z1, lens, 10.0)
    ttl = TextToLatentModel(cfg.ttl, cfg.vocab_size).eval()

    def loss_of(target_latent):
        torch.manual_seed(6)  # same noise draw and timestep for both calls
        return flow_matching_loss(ttl, target_latent, lat_mask, text_ids, text_mask,
                                  ref, ref_mask, rtm, n_expand=1).item()

    garbled = z1.clone()
    garbled[rtm.expand_as(z1) > 0.5] = 1000.0  # only the excluded positions
    assert loss_of(garbled) == pytest.approx(loss_of(z1), rel=1e-5)


# ----------------------------------------------------------------------
# latent statistics and checkpoint identity
# ----------------------------------------------------------------------

def test_stage_one_fits_latent_statistics(tmp_path):
    pytest.importorskip("pytorch_lightning")
    from phoonnx_train.supertonic.autoencoder import SpeechAutoencoder
    cfg = tiny_config(64)
    flist = _write_corpus(tmp_path)
    ae = SpeechAutoencoder(cfg.ae).eval()
    state = torch.load(_fitted_ae_checkpoint(tmp_path, cfg, flist), weights_only=False)["state_dict"]
    ae.load_state_dict({k[len("generator."):]: v for k, v in state.items()
                        if k.startswith("generator.")})
    assert bool(ae.stats_fitted)

    # the stage-2 flow target must match the unit-variance prior it is transported from
    wav, _ = torch.randn(2, 16000), None
    z1 = normalize_and_compress(ae, ae.encode(wav), cfg.ttl.compress_factor, 1.0)
    assert 0.5 < z1.std().item() < 2.0, f"flow target std {z1.std().item()} is far from the prior"


def test_stage_two_refuses_an_unfitted_autoencoder(tmp_path):
    pytest.importorskip("pytorch_lightning")
    from phoonnx_train.supertonic.autoencoder import SpeechAutoencoder
    from phoonnx_train.supertonic.lightning import TextToLatentModule
    cfg = tiny_config(64)
    path = tmp_path / "raw_ae.pt"
    torch.save({"state_dict": {f"generator.{k}": v
                               for k, v in SpeechAutoencoder(cfg.ae).state_dict().items()}}, path)
    with pytest.raises(ValueError, match="latent statistics"):
        TextToLatentModule(config=cfg, dataset=[], root_dir=str(tmp_path), ae_checkpoint=str(path))


def test_export_rejects_swapped_stage_checkpoints(tmp_path):
    pytest.importorskip("pytorch_lightning")
    from phoonnx_train.supertonic.export_onnx import export_from_checkpoints
    from phoonnx_train.supertonic.lightning import DurationPredictorModule, TextToLatentModule
    cfg = tiny_config(64)
    flist = _write_corpus(tmp_path)
    tok = CharTokenizer.build_from_texts(["hello world number 0"], ["en"])
    ae_ck = _fitted_ae_checkpoint(tmp_path, cfg, flist)

    def fit_and_save(mod, name):
        tr = _trainer(tmp_path, max_steps=1)
        tr.fit(mod)
        tr.save_checkpoint(str(tmp_path / name))
        return str(tmp_path / name)

    ttl_ck = fit_and_save(TextToLatentModule(config=cfg, tokenizer=tok, dataset=[flist],
                                             root_dir=str(tmp_path), ae_checkpoint=ae_ck,
                                             batch_size=2, num_workers=0), "ttl.ckpt")
    dp_ck = fit_and_save(DurationPredictorModule(config=cfg, tokenizer=tok, dataset=[flist],
                                                 root_dir=str(tmp_path), ae_checkpoint=ae_ck,
                                                 batch_size=2, num_workers=0), "dp.ckpt")
    with pytest.raises(ValueError, match="text_to_latent"):
        export_from_checkpoints(str(tmp_path / "out"), autoencoder_ckpt=ae_ck,
                                text_to_latent_ckpt=dp_ck, duration_predictor_ckpt=ttl_ck)


def test_exported_config_records_normalizer_scales(tmp_path, cfg):
    pytest.importorskip("onnx")
    from phoonnx_train.supertonic.export_onnx import export_all
    ae, ttl, dp = _build_all(cfg)
    paths = export_all(str(tmp_path), config=cfg, tokenizer=CharTokenizer.build_from_texts(["hi"], ["en"]),
                       autoencoder=ae, text_to_latent=ttl, duration_predictor=dp)
    tts = json.loads(paths["tts"].read_text())
    assert tts["ttl"]["normalizer"]["scale"] == cfg.ttl.normalizer_scale
    assert tts["dp"]["normalizer"]["scale"] == cfg.dp.normalizer_scale
    assert load_model_config(str(paths["tts"])).ttl.normalizer_scale == cfg.ttl.normalizer_scale
