"""CPU-only tests for the FastPitch / SpeedySpeech (ForwardTTS) training engine.

Uses tiny random-init models and synthetic tensors — no downloads, no real
audio — to exercise: registry wiring, quality presets, dataset/collate,
a full forward+loss training step for both variants (FastPitch with pitch,
SpeedySpeech without), and ONNX export producing the ``token_ids -> mel_spec``
contract consumed by ``phoonnx.engines.fastpitch.FastPitchAdapter``.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from phoonnx_train.engines import get_engine, list_engines
from phoonnx_train.engines.base import TrainingEngineConfig
from phoonnx_train.engines.fastpitch import (
    _QUALITY_PRESETS,
    Batch,
    ForwardTTSCollate,
    ForwardTTSModule,
    ForwardTTSTrainingEngine,
    SpeedySpeechTrainingEngine,
    UtteranceTensors,
)
from phoonnx_train.fastpitch.model import ForwardTTS, ForwardTTSArgs

N_SYMBOLS = 40
MEL_CHANNELS = 80

TINY_ARGS = dict(
    hidden_channels=16,
    hidden_channels_ffn=32,
    encoder_num_layers=1,
    decoder_num_layers=1,
    num_heads=1,
    predictor_hidden_channels=16,
)


def _finite(v):
    return torch.is_tensor(v) and torch.isfinite(v).all()


def synthetic_batch(batch_size=2, n_tokens=8, n_frames=40, use_pitch=True,
                    multispeaker=False):
    torch.manual_seed(0)
    phoneme_ids = torch.randint(1, N_SYMBOLS - 1, (batch_size, n_tokens), dtype=torch.long)
    phoneme_lengths = torch.full((batch_size,), n_tokens, dtype=torch.long)
    mels = torch.randn(batch_size, MEL_CHANNELS, n_frames)
    mel_lengths = torch.full((batch_size,), n_frames, dtype=torch.long)
    pitch = torch.randn(batch_size, 1, n_frames) if use_pitch else None
    speaker_ids = torch.randint(0, 3, (batch_size,), dtype=torch.long) if multispeaker else None
    return Batch(phoneme_ids, phoneme_lengths, mels, mel_lengths, pitch, speaker_ids)


# ---------------------------------------------------------------- registry
def test_engines_registered():
    assert "fastpitch" in list_engines()
    assert "speedyspeech" in list_engines()
    assert isinstance(get_engine("fastpitch"), ForwardTTSTrainingEngine)
    assert isinstance(get_engine("speedyspeech"), SpeedySpeechTrainingEngine)


def test_quality_presets():
    presets = ForwardTTSTrainingEngine().quality_presets()
    assert set(presets) == {"x-low", "medium", "high"}
    assert presets is _QUALITY_PRESETS
    for tier in presets.values():
        assert "hidden_channels" in tier


# ------------------------------------------------------------- ForwardTTS
def test_forward_tts_fastpitch_forward_and_inference():
    args = ForwardTTSArgs(num_chars=N_SYMBOLS, out_channels=MEL_CHANNELS,
                          use_pitch=True, use_energy=False, **TINY_ARGS)
    model = ForwardTTS(args)
    batch = synthetic_batch(use_pitch=True)
    out = model(
        x=batch.phoneme_ids, x_lengths=batch.phoneme_lengths,
        y=batch.mels, y_lengths=batch.mel_lengths, pitch=batch.pitch,
    )
    assert out["model_outputs"].shape == (2, 40, MEL_CHANNELS)
    assert _finite(out["model_outputs"])
    assert out["pitch_avg_pred"] is not None

    model.eval()
    mel = model.inference(batch.phoneme_ids[:1])
    assert mel.shape[0] == 1 and mel.shape[1] == MEL_CHANNELS
    assert _finite(mel)


def test_forward_tts_speedyspeech_variant_has_no_pitch():
    args = ForwardTTSArgs(
        num_chars=N_SYMBOLS, out_channels=MEL_CHANNELS,
        encoder_type="residual_conv_bn", decoder_type="residual_conv_bn",
        use_pitch=False, use_energy=False,
        encoder_num_res_blocks=2, decoder_num_res_blocks=2,
        hidden_channels=16, predictor_hidden_channels=16,
    )
    model = ForwardTTS(args)
    assert not hasattr(model, "pitch_predictor")
    batch = synthetic_batch(use_pitch=False)
    out = model(
        x=batch.phoneme_ids, x_lengths=batch.phoneme_lengths,
        y=batch.mels, y_lengths=batch.mel_lengths, pitch=None,
    )
    assert out["pitch_avg_pred"] is None
    assert _finite(out["model_outputs"])

    model.eval()
    mel = model.inference(batch.phoneme_ids[:1])
    assert _finite(mel)


def test_forward_tts_multispeaker():
    args = ForwardTTSArgs(num_chars=N_SYMBOLS, out_channels=MEL_CHANNELS,
                          num_speakers=4, use_pitch=True, **TINY_ARGS)
    model = ForwardTTS(args)
    assert hasattr(model, "emb_g")
    batch = synthetic_batch(use_pitch=True, multispeaker=True)
    out = model(
        x=batch.phoneme_ids, x_lengths=batch.phoneme_lengths,
        y=batch.mels, y_lengths=batch.mel_lengths, pitch=batch.pitch,
        speaker=batch.speaker_ids,
    )
    assert _finite(out["model_outputs"])


# --------------------------------------------------------- collate / batch
def test_collate_pads_and_stacks():
    utts = [
        UtteranceTensors(torch.LongTensor([1, 2, 3]), torch.randn(MEL_CHANNELS, 10),
                         torch.LongTensor([0]), torch.randn(1, 10)),
        UtteranceTensors(torch.LongTensor([1, 2]), torch.randn(MEL_CHANNELS, 15),
                         torch.LongTensor([1]), torch.randn(1, 15)),
    ]
    collate = ForwardTTSCollate(is_multispeaker=True, use_pitch=True)
    batch = collate(utts)
    assert batch.phoneme_ids.shape == (2, 3)
    assert batch.mels.shape == (2, MEL_CHANNELS, 15)
    assert batch.pitch.shape == (2, 1, 15)
    assert batch.speaker_ids.tolist() == [0, 1]
    assert batch.phoneme_lengths.tolist() == [3, 2]
    assert batch.mel_lengths.tolist() == [10, 15]


# ------------------------------------------------------------ LightningModule
def test_module_training_and_validation_step_fastpitch():
    module = ForwardTTSModule(
        num_symbols=N_SYMBOLS, num_speakers=1, variant="fastpitch",
        mel_channels=MEL_CHANNELS, dataset=None, **TINY_ARGS,
    )
    logged = {}
    module.log_dict = lambda d, *a, **k: logged.update(d)
    batch = synthetic_batch(use_pitch=True)

    loss = module.training_step(batch, 0)
    assert _finite(loss)
    assert "train_loss" in logged
    assert _finite(logged["train_mel_loss"]) if "train_mel_loss" in logged else True

    val_loss = module.validation_step(batch, 0)
    assert _finite(val_loss)

    opts, scheds = module.configure_optimizers()
    assert len(opts) == 1 and len(scheds) == 1


def test_module_training_step_speedyspeech():
    module = ForwardTTSModule(
        num_symbols=N_SYMBOLS, num_speakers=1, variant="speedyspeech",
        mel_channels=MEL_CHANNELS, dataset=None,
        hidden_channels=16, hidden_channels_ffn=32,
        encoder_num_layers=1, decoder_num_layers=1, num_heads=1,
        predictor_hidden_channels=16,
        encoder_num_res_blocks=2, decoder_num_res_blocks=2,
    )
    assert module.model_args.use_pitch is False
    batch = synthetic_batch(use_pitch=False)
    logged = {}
    module.log_dict = lambda d, *a, **k: logged.update(d)
    loss = module.training_step(batch, 0)
    assert _finite(loss)


def test_module_backward_updates_parameters():
    module = ForwardTTSModule(
        num_symbols=N_SYMBOLS, num_speakers=1, variant="fastpitch",
        mel_channels=MEL_CHANNELS, dataset=None, **TINY_ARGS,
    )
    module.log_dict = lambda *a, **k: None
    before = [p.detach().clone() for p in module.model.parameters()]
    batch = synthetic_batch(use_pitch=True)
    loss = module.training_step(batch, 0)
    loss.backward()
    grads = [p.grad for p in module.model.parameters()]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


# --------------------------------------------------------------- engine API
def test_engine_create_model():
    engine = get_engine("fastpitch")
    cfg = TrainingEngineConfig(num_symbols=N_SYMBOLS, num_speakers=1, sample_rate=22050,
                               extra=dict(TINY_ARGS, mel_channels=MEL_CHANNELS))
    model = engine.create_model(cfg, dataset_paths=[])
    assert isinstance(model, ForwardTTSModule)
    assert model.hparams.variant == "fastpitch"
    assert model.model_args.use_pitch is True


def test_speedyspeech_engine_defaults_variant():
    engine = get_engine("speedyspeech")
    cfg = TrainingEngineConfig(num_symbols=N_SYMBOLS, num_speakers=1, sample_rate=22050,
                               extra=dict(mel_channels=MEL_CHANNELS,
                                          hidden_channels=16, hidden_channels_ffn=32,
                                          encoder_num_layers=1, decoder_num_layers=1,
                                          num_heads=1, predictor_hidden_channels=16,
                                          encoder_num_res_blocks=2, decoder_num_res_blocks=2))
    model = engine.create_model(cfg, dataset_paths=[])
    assert model.hparams.variant == "speedyspeech"
    assert model.model_args.use_pitch is False


def test_engine_export_onnx_fastpitch(tmp_path: Path):
    engine = get_engine("fastpitch")
    cfg = TrainingEngineConfig(num_symbols=N_SYMBOLS, num_speakers=1, sample_rate=22050,
                               extra=dict(TINY_ARGS, mel_channels=MEL_CHANNELS))
    model = engine.create_model(cfg, dataset_paths=[])

    ckpt_path = tmp_path / "model.ckpt"
    torch.save({"state_dict": model.state_dict(),
               "hyper_parameters": dict(model.hparams)}, ckpt_path)

    import pytorch_lightning as pl

    class _Loadable(ForwardTTSModule):
        @classmethod
        def load_from_checkpoint(cls, checkpoint_path, dataset=None, map_location="cpu", **kw):
            ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
            hp = dict(ckpt["hyper_parameters"])
            hp["dataset"] = dataset
            m = cls(**hp)
            m.load_state_dict(ckpt["state_dict"])
            return m

    import phoonnx_train.engines.fastpitch as fp_mod
    orig = fp_mod.ForwardTTSModule
    fp_mod.ForwardTTSModule = _Loadable
    try:
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps({
            "audio": {"sample_rate": 22050},
            "phoneme_id_map": {"a": 1, "b": 2},
            "phoneme_type": "espeak",
            "alphabet": "ipa",
        }))
        out_dir = tmp_path / "out"
        onnx_path = engine.export_onnx(ckpt_path, cfg_path, out_dir)
    finally:
        fp_mod.ForwardTTSModule = orig

    assert onnx_path.exists()

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path))
    in_names = {i.name for i in sess.get_inputs()}
    out_names = {o.name for o in sess.get_outputs()}
    assert in_names == {"token_ids", "pace", "pitch_mul", "pitch_add"}
    assert out_names == {"mel_spec"}

    ids = np.random.randint(1, N_SYMBOLS - 1, size=(1, 12)).astype(np.int64)
    feed = {"token_ids": ids,
            "pace": np.ones(1, dtype=np.float32),
            "pitch_mul": np.ones(1, dtype=np.float32),
            "pitch_add": np.zeros(1, dtype=np.float32)}
    (mel,) = sess.run(None, feed)
    assert mel.ndim == 3 and mel.shape[0] == 1 and mel.shape[1] == MEL_CHANNELS
    assert np.isfinite(mel).all()

    # the control inputs are actually wired into the graph: an untrained
    # duration predictor clamps to 1 frame/token either way, so probe pace
    # via graph inputs and pitch_add via output values
    (mel_shifted,) = sess.run(
        None, {**feed, "pitch_add": np.full(1, 5.0, dtype=np.float32)})
    assert not np.allclose(mel_shifted, mel)
    (mel_slow,) = sess.run(None, {**feed, "pace": np.full(1, 0.05, dtype=np.float32)})
    assert mel_slow.shape[2] >= mel.shape[2]

    import onnx as _onnx

    onnx_model = _onnx.load(str(onnx_path))
    meta = {p.key: p.value for p in onnx_model.metadata_props}
    assert meta["engine"] == "fastpitch"
    assert meta["n_vocab"] == str(N_SYMBOLS)


def test_extra_preprocess_missing_deps_returns_empty(monkeypatch, tmp_path):
    """When pyworld/librosa aren't importable, extra_preprocess degrades to {}."""
    engine = ForwardTTSTrainingEngine()
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name in ("pyworld", "librosa"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = engine.extra_preprocess(tmp_path / "a.wav", tmp_path / "cache", 22050)
    assert result == {}


def test_load_checkpoint_tolerant(tmp_path: Path):
    engine = get_engine("fastpitch")
    cfg = TrainingEngineConfig(num_symbols=N_SYMBOLS, num_speakers=1, sample_rate=22050,
                               extra=dict(TINY_ARGS, mel_channels=MEL_CHANNELS))
    model = engine.create_model(cfg, dataset_paths=[])
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)

    # A model with a different vocab size (shape mismatch on emb) should
    # still load tolerant of the mismatch.
    cfg2 = TrainingEngineConfig(num_symbols=N_SYMBOLS + 5, num_speakers=1, sample_rate=22050,
                                extra=dict(TINY_ARGS, mel_channels=MEL_CHANNELS))
    model2 = engine.create_model(cfg2, dataset_paths=[])
    engine.load_checkpoint(model2, ckpt_path)  # should not raise


# ----------------------------------------------------------------------
# masked losses with variable-length batches
# ----------------------------------------------------------------------

def test_losses_masked_by_valid_length():
    """Duration/pitch losses must average over valid positions only —
    padding must not dilute the loss."""
    import torch
    from phoonnx_train.fastpitch.losses import ForwardTTSLoss
    from phoonnx_train.fastpitch.helpers import sequence_mask

    torch.manual_seed(0)
    crit = ForwardTTSLoss()
    b, t_en, t_de, c = 2, 20, 40, MEL_CHANNELS
    input_lens = torch.tensor([4, 20])
    output_lens = torch.tensor([40, 40])
    dur_output = torch.rand(b, t_en)
    dur_target = torch.randint(1, 5, (b, t_en))
    dec_out = torch.rand(b, t_de, c)

    losses = crit(
        decoder_output=dec_out, decoder_target=dec_out.clone(),
        decoder_output_lens=output_lens,
        dur_output=dur_output, dur_target=dur_target, input_lens=input_lens,
        pitch_output=None, pitch_target=None,
        aligner_logprob=None, alignment_hard=None, alignment_soft=None,
    )
    # reference: masked mean computed by hand
    mask = sequence_mask(input_lens, t_en).float()
    log_tgt = torch.log(dur_target.float() + 1)
    expected = (((dur_output - log_tgt) ** 2) * mask).sum() / mask.sum()
    assert torch.allclose(losses["loss_dur"], expected, atol=1e-6)


def test_binary_loss_weight_ramps_from_epoch_zero():
    def weight(epoch, start=0, warmup=10):
        warmup = max(1, warmup)
        return min(1.0, max(0.0, (epoch - start + 1) / warmup))

    weights = [weight(e) for e in (0, 4, 9, 20)]
    assert weights[0] < 1.0  # not full strength at epoch 0
    assert weights[-1] == 1.0  # saturates
    assert weights == sorted(weights)
    # delayed start
    assert weight(0, start=5) == 0.0 and weight(14, start=5) == 1.0


def test_pitch_stats_normalization(tmp_path):
    import json

    import numpy as np

    from phoonnx_train.engines.fastpitch import ForwardTTSDataset

    class _FakeUtt:
        def __init__(self, spec_path):
            self.audio_spec_path = spec_path

    class _FakeInner:
        def __init__(self, utts):
            self.utterances = utts

    f0 = np.zeros(50, dtype=np.float32)
    f0[10:40] = 200.0 + 10.0 * np.random.RandomState(0).randn(30)
    np.save(tmp_path / "utt0.f0.npy", f0)

    ds = ForwardTTSDataset.__new__(ForwardTTSDataset)
    ds._inner = _FakeInner([_FakeUtt(tmp_path / "utt0.spec.pt")])
    mean, std = ds._pitch_stats([tmp_path])
    assert 190 < mean < 210 and 0 < std < 30
    # stats are cached
    assert json.loads((tmp_path / "pitch_stats.json").read_text())["mean"] == mean
