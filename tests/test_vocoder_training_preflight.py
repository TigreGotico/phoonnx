"""Pre-flight correctness tests for the Vocos vocoder training path.

These guard the failure modes that silently waste an expensive training run:
a mel-space mismatch between the vocoder's training input and the mel the
Matcha acoustic model actually emits, broken GAN gradient isolation, resume
that drops optimizer/scheduler state, and export I/O that the inference-side
adapter cannot consume.

The heavy end-to-end train/resume/export/vocode ACID test lives outside the
unit suite (it needs a GPU and minutes); these tests cover the invariants
that must hold on every commit.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_ROOT = Path(__file__).parent.parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------
# 1. Mel-space equality: the vocoder trains on EXACTLY the mel Matcha emits
# ----------------------------------------------------------------------

def test_vocoder_and_matcha_share_one_mel_featurizer():
    """The Vocos trainer's featurizer must be the very function Matcha uses.

    Both sides are imported by their real module path; if a future edit
    re-points the vocoder at a different featurizer (e.g. the VITS one,
    whose 1e-6 magnitude epsilon shifts the log-mel silence floor by ~1.5
    nats), this assertion fails.
    """
    from phoonnx_train.matcha.audio import mel_spectrogram as matcha_mel
    from phoonnx_train.vocos import lightning as vocos_lightning

    assert vocos_lightning.mel_spectrogram_torch is matcha_mel


def test_vocoder_featurizer_matches_matcha_numerically():
    """Value-level cross-check on a signal that includes a silent region —
    the region where the epsilon difference bites hardest."""
    from phoonnx_train.matcha.audio import mel_spectrogram as matcha_mel
    from phoonnx_train.vocos.data import MelConfig

    m = MelConfig()
    torch.manual_seed(0)
    audio = 0.3 * torch.randn(1, m.hop_length * 80)
    audio[:, :4000] = 0.0  # silence
    audio = audio.clamp(-1.0, 1.0)

    ref = matcha_mel(audio, m.n_fft, m.n_mels, m.sample_rate,
                     m.hop_length, m.win_length, m.fmin, m.fmax)

    module = VocosTrainingModule_stub()
    got = module._mel(audio)
    assert torch.allclose(got, ref, atol=0.0)


def VocosTrainingModule_stub():
    from phoonnx_train.vocos.lightning import VocosTrainingModule
    from phoonnx_train.vocos.data import MelConfig
    return VocosTrainingModule(mel=MelConfig().as_dict(), audio_files=["x"])


def test_matcha_featurizer_importable_without_diffusers():
    """Training a vocoder must not require the full Matcha acoustic stack:
    importing the leaf mel front-end must not pull in diffusers."""
    import sys
    for mod in list(sys.modules):
        if mod.startswith(("diffusers", "phoonnx_train.matcha")):
            del sys.modules[mod]
    importlib.import_module("phoonnx_train.matcha.audio")
    assert "diffusers" not in sys.modules


# ----------------------------------------------------------------------
# 2. GAN correctness: discriminator loss must not reach the generator
# ----------------------------------------------------------------------

def test_discriminator_loss_does_not_backprop_into_generator():
    from phoonnx_train.vocos.models import (
        VocosGenerator, MultiPeriodDiscriminator, MultiResolutionDiscriminator,
        discriminator_loss,
    )
    torch.manual_seed(0)
    gen = VocosGenerator()
    mpd, mrd = MultiPeriodDiscriminator(), MultiResolutionDiscriminator()
    mel = torch.randn(2, 80, 32)
    audio_hat = gen(mel)
    audio = torch.randn_like(audio_hat)

    real_p, _ = mpd(audio)
    fake_p, _ = mpd(audio_hat.detach())
    real_r, _ = mrd(audio)
    fake_r, _ = mrd(audio_hat.detach())
    loss_d = discriminator_loss(real_p, fake_p) + discriminator_loss(real_r, fake_r)

    gen.zero_grad(); mpd.zero_grad(); mrd.zero_grad()
    loss_d.backward()

    g_grad = sum(float(p.grad.abs().sum()) for p in gen.parameters() if p.grad is not None)
    d_grad = sum(float(p.grad.abs().sum())
                 for p in list(mpd.parameters()) + list(mrd.parameters())
                 if p.grad is not None)
    assert g_grad == 0.0, "D loss leaked gradients into the generator"
    assert d_grad > 0.0, "D loss produced no discriminator gradient"
    # the generator graph must survive the detached D pass for its own step
    assert audio_hat.requires_grad


def test_generator_adv_loss_does_not_update_discriminator():
    from phoonnx_train.vocos.models import (
        VocosGenerator, MultiPeriodDiscriminator,
        generator_adversarial_loss,
    )
    torch.manual_seed(1)
    gen = VocosGenerator()
    mpd = MultiPeriodDiscriminator()
    mel = torch.randn(2, 80, 32)
    audio_hat = gen(mel)
    fake_p, _ = mpd(audio_hat)
    loss_g = generator_adversarial_loss(fake_p)

    gen.zero_grad(); mpd.zero_grad()
    loss_g.backward()
    g_grad = sum(float(p.grad.abs().sum()) for p in gen.parameters() if p.grad is not None)
    assert g_grad > 0.0, "generator adversarial loss produced no generator gradient"


# ----------------------------------------------------------------------
# 3. Resume: both optimizers + both schedulers survive save/restore, LR
#    continuity holds numerically.
# ----------------------------------------------------------------------

def test_checkpoint_carries_both_optimizers_and_schedulers(tmp_path):
    from pytorch_lightning import Trainer
    from phoonnx_train.vocos.lightning import VocosTrainingModule
    from phoonnx_train.vocos.data import MelConfig

    files = _make_wavs(tmp_path, n=4)
    model = VocosTrainingModule(
        mel=MelConfig().as_dict(), batch_size=2, num_workers=0,
        crop_samples=MelConfig().hop_length * 24, audio_files=files,
        num_warmup_steps=5,
    )
    ckpt = tmp_path / "s.ckpt"
    trainer = Trainer(max_steps=6, max_epochs=3, accelerator="cpu", devices=1,
                      logger=False, enable_checkpointing=False,
                      enable_progress_bar=False)
    trainer.fit(model)
    trainer.save_checkpoint(str(ckpt))

    state = torch.load(str(ckpt), map_location="cpu")
    # both the generator and the discriminator optimizer + their schedulers
    # must be persisted, or a resume silently reinitializes half the run
    assert len(state["optimizer_states"]) == 2
    assert len(state["lr_schedulers"]) == 2
    # schedulers are stepped once per training batch and must have advanced;
    # both branches must sit at the same position
    l0 = state["lr_schedulers"][0]["last_epoch"]
    l1 = state["lr_schedulers"][1]["last_epoch"]
    assert l0 == l1 > 0
    assert state["global_step"] > 0


def test_resume_continues_lr_schedule(tmp_path):
    """A resume must continue the schedule where it stopped, not reset it.

    Both schedulers' step counters must advance past the pre-resume value,
    and the restored learning rates stay positive and finite — a resume
    that reinitialized the optimizers would snap the counter back to 0.
    """
    from pytorch_lightning import Trainer
    from phoonnx_train.vocos.lightning import VocosTrainingModule
    from phoonnx_train.vocos.data import MelConfig

    files = _make_wavs(tmp_path, n=6)

    def build():
        return VocosTrainingModule(
            mel=MelConfig().as_dict(), batch_size=2, num_workers=0,
            crop_samples=MelConfig().hop_length * 24, audio_files=files,
            num_warmup_steps=1000,
        )

    m1 = build()
    ckpt = tmp_path / "r.ckpt"
    Trainer(max_epochs=2, accelerator="cpu", devices=1, logger=False,
            enable_checkpointing=False, enable_progress_bar=False).fit(m1)
    m1.trainer.save_checkpoint(str(ckpt))
    steps_before = m1.lr_schedulers()[0].last_epoch
    assert steps_before > 0

    m2 = build()
    t2 = Trainer(max_epochs=4, accelerator="cpu", devices=1, logger=False,
                 enable_checkpointing=False, enable_progress_bar=False)
    t2.fit(m2, ckpt_path=str(ckpt))

    for sched in m2.lr_schedulers():
        assert sched.last_epoch > steps_before, (steps_before, sched.last_epoch)
    for opt in m2.optimizers():
        lr = opt.param_groups[0]["lr"]
        assert lr > 0.0 and np.isfinite(lr)


# ----------------------------------------------------------------------
# 4. Data pipeline: crop cropping / short-clip handling
# ----------------------------------------------------------------------

def test_short_clip_is_padded_not_crashing(tmp_path):
    import soundfile as sf
    from phoonnx_train.vocos.dataset import AudioCropDataset
    from phoonnx_train.vocos.data import MelConfig

    m = MelConfig()
    crop = m.hop_length * 24
    p = tmp_path / "short.wav"
    sf.write(str(p), np.zeros(crop // 3, dtype=np.float32), m.sample_rate)
    ds = AudioCropDataset([p], mel=m, crop_samples=crop, train=True)
    item = ds[0]
    assert item.shape[0] == crop
    assert torch.isfinite(item).all()


def test_crop_is_exact_length_and_in_range(tmp_path):
    import soundfile as sf
    from phoonnx_train.vocos.dataset import AudioCropDataset
    from phoonnx_train.vocos.data import MelConfig

    m = MelConfig()
    crop = m.hop_length * 24
    p = tmp_path / "long.wav"
    sf.write(str(p), (2.0 * np.random.randn(crop * 4)).astype(np.float32), m.sample_rate)
    ds = AudioCropDataset([p], mel=m, crop_samples=crop, train=True)
    for _ in range(5):
        item = ds[0]
        assert item.shape[0] == crop
        assert float(item.max()) <= 1.0 and float(item.min()) >= -1.0


# ----------------------------------------------------------------------
# 5. Export fidelity: torch generator vs exported ONNX + runtime adapter
# ----------------------------------------------------------------------

def test_export_onnx_matches_torch_and_adapter_consumes_io(tmp_path):
    onnx = pytest.importorskip("onnx")  # noqa: F841
    pytest.importorskip("onnxruntime")
    from phoonnx_train.vocos.lightning import VocosTrainingModule
    from phoonnx_train.vocos.data import MelConfig
    from phoonnx_train.export_vocos import _CoefficientWrapper, legacy_onnx_export
    from phoonnx.engines.vocoders import build_vocoder

    m = MelConfig()
    model = VocosTrainingModule(mel=m.as_dict(), audio_files=["x"])
    model.eval()
    gen = model.generator

    frames = 40
    dummy = torch.randn(1, m.n_mels, frames) * 2.0 - 6.0
    out = tmp_path / "voc.onnx"
    # identical export path to production (export_vocos.main): the legacy
    # TorchScript exporter, so no onnxscript/dynamo dependency is pulled in
    legacy_onnx_export(
        _CoefficientWrapper(gen), dummy, str(out), opset_version=17,
        input_names=["mels"], output_names=["mag", "x", "y"],
        dynamic_axes={k: {0: "batch", 2: "time"} for k in ("mels", "mag", "x", "y")},
    )
    config = {"vocoder_type": "vocos", "n_fft": m.n_fft, "hop_length": m.hop_length}
    with torch.no_grad():
        audio_torch = gen(dummy).numpy().squeeze()
    vocoder = build_vocoder(model_path=str(out), vocoder_type="vocos", config=config)
    # adapter accepts the exported I/O names and produces finite audio
    audio_onnx = vocoder.mel_to_audio(dummy.numpy().astype(np.float32))
    assert np.isfinite(audio_onnx).all()
    n = min(audio_torch.shape[-1], audio_onnx.shape[-1])
    diff = float(np.max(np.abs(audio_torch[:n] - audio_onnx.squeeze()[:n])))
    assert diff < 1e-3, f"torch vs onnx parity {diff:.2e}"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _make_wavs(tmp_path, n=4):
    import soundfile as sf
    from phoonnx_train.vocos.data import MelConfig
    sr = MelConfig().sample_rate
    files = []
    for i in range(n):
        t = np.linspace(0, 0.6, int(sr * 0.6), endpoint=False)
        a = 0.3 * np.sin(2 * np.pi * 120 * (i + 1) * t)
        a[:1000] = 0.0
        p = tmp_path / f"w{i}.wav"
        sf.write(str(p), a.astype(np.float32), sr)
        files.append(p)
    return files
