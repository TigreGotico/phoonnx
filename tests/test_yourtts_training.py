"""
Tests for the YourTTS training engine (VITS + external d-vector conditioning).

Everything here runs on CPU with a tiny synthetic model/dataset and random
speaker embeddings — no downloads, no real speaker-encoder ONNX, no GPU.
"""
import json

import numpy as np
import onnxruntime
import pytest
import pytorch_lightning as pl
import torch

from phoonnx_train.engines import get_engine, list_engines
from phoonnx_train.engines.yourtts import YourttsTrainingEngine, _mean_dataset_d_vector
from phoonnx_train.vits.dataset import Utterance, UtteranceCollate, UtteranceTensors
from phoonnx_train.vits.models import SynthesizerTrn

TINY_KWARGS = dict(
    n_vocab=32,
    spec_channels=9,  # (filter_length // 2) + 1, filter_length=16
    segment_size=4,
    inter_channels=8,
    hidden_channels=8,
    filter_channels=16,
    n_heads=2,
    n_layers=1,
    kernel_size=3,
    p_dropout=0.0,
    resblock="2",
    resblock_kernel_sizes=(3, 5),
    resblock_dilation_sizes=((1, 2), (2, 6)),
    upsample_rates=(2, 2),
    upsample_initial_channel=16,
    upsample_kernel_sizes=(4, 4),
)
D_VECTOR_DIM = 16
GIN_CHANNELS = 8


def _tiny_model(**overrides):
    kwargs = dict(TINY_KWARGS)
    kwargs.update(overrides)
    return SynthesizerTrn(
        n_speakers=1,
        gin_channels=GIN_CHANNELS,
        external_speaker_embedding=True,
        speaker_embedding_dim=D_VECTOR_DIM,
        **kwargs,
    )


# ---------------------------------------------------------------- registry

def test_yourtts_engine_registered():
    assert "yourtts" in list_engines()
    assert isinstance(get_engine("yourtts"), YourttsTrainingEngine)


def test_quality_presets_mirror_vits_tiers():
    presets = YourttsTrainingEngine().quality_presets()
    assert set(presets) == {"x-low", "medium", "high"}
    for tier, params in presets.items():
        assert params["hidden_channels"] > 0
        assert params["inter_channels"] > 0


# ---------------------------------------------------- model construction

def test_model_constructs_with_d_vector_conditioning():
    model = _tiny_model()
    assert model.external_speaker_embedding is True
    assert model.gin_channels == GIN_CHANNELS
    # n_speakers==1 with external conditioning must NOT allocate a per-id table
    assert not hasattr(model, "emb_g")
    # dims differ (16 -> 8) so a projection layer must exist
    assert model.speaker_embedding_proj is not None
    assert model.speaker_embedding_proj.in_features == D_VECTOR_DIM
    assert model.speaker_embedding_proj.out_features == GIN_CHANNELS


def test_model_constructs_without_projection_when_dims_match():
    model = _tiny_model(**{})
    model2 = SynthesizerTrn(
        n_speakers=1,
        gin_channels=D_VECTOR_DIM,
        external_speaker_embedding=True,
        speaker_embedding_dim=D_VECTOR_DIM,
        **TINY_KWARGS,
    )
    assert model2.speaker_embedding_proj is None


def test_plain_multispeaker_vits_unaffected():
    """Backward compatibility: sid-based conditioning still works exactly as before."""
    model = SynthesizerTrn(n_speakers=3, gin_channels=GIN_CHANNELS, **TINY_KWARGS)
    assert hasattr(model, "emb_g")
    assert model.emb_g.num_embeddings == 3
    assert not model.external_speaker_embedding


def test_language_embedding_optional_and_additive():
    model = _tiny_model(n_langs=4)
    assert hasattr(model, "emb_l")

    speaker_embedding = torch.randn(1, D_VECTOR_DIM)
    g_speaker_only = model._compute_g(speaker_embedding=speaker_embedding)

    lid_a = torch.LongTensor([1])
    lid_b = torch.LongTensor([2])
    g_lang_a = model._compute_g(speaker_embedding=speaker_embedding, lid=lid_a)
    g_lang_b = model._compute_g(speaker_embedding=speaker_embedding, lid=lid_b)

    # the language embedding is additive on top of the speaker conditioning,
    # and different language ids move it differently.
    assert not torch.allclose(g_speaker_only, g_lang_a)
    assert not torch.allclose(g_lang_a, g_lang_b)
    assert torch.allclose(
        g_lang_a - g_speaker_only,
        model.emb_l(lid_a).unsqueeze(-1),
        atol=1e-6,
    )

    # sanity: still produces a finite, playable waveform end to end
    x = torch.randint(0, TINY_KWARGS["n_vocab"], (1, 6))
    x_lengths = torch.LongTensor([6])
    model.eval()
    with torch.no_grad():
        audio, *_ = model.infer(x, x_lengths, speaker_embedding=speaker_embedding, lid=lid_a)
    assert torch.isfinite(audio).all()


# --------------------------------------------------------------- training

def _synthetic_batch(batch_size=2, phoneme_len=6, spec_len=5, hop_length=4):
    from phoonnx_train.vits.dataset import Batch

    # audio must cover the full spectrogram in the waveform domain (spec_len *
    # hop_length) *and* be long enough for a random segment_size crop.
    audio_len = max(spec_len * hop_length, TINY_KWARGS["segment_size"] * 4) + hop_length
    return Batch(
        phoneme_ids=torch.randint(1, TINY_KWARGS["n_vocab"], (batch_size, phoneme_len)),
        phoneme_lengths=torch.LongTensor([phoneme_len] * batch_size),
        spectrograms=torch.randn(batch_size, TINY_KWARGS["spec_channels"], spec_len),
        spectrogram_lengths=torch.LongTensor([spec_len] * batch_size),
        audios=torch.randn(batch_size, 1, audio_len),
        audio_lengths=torch.LongTensor([audio_len] * batch_size),
        speaker_ids=None,
        d_vectors=torch.randn(batch_size, D_VECTOR_DIM),
        language_ids=None,
    )


def test_training_step_generator_forward_gives_finite_loss():
    model = _tiny_model()
    batch = _synthetic_batch()
    (
        y_hat, l_length, attn, ids_slice, x_mask, y_mask,
        (z, z_p, m_p, logs_p, m_q, logs_q),
    ) = model(
        batch.phoneme_ids, batch.phoneme_lengths, batch.spectrograms,
        batch.spectrogram_lengths, sid=None, speaker_embedding=batch.d_vectors,
    )
    assert torch.isfinite(y_hat).all()
    assert torch.isfinite(l_length).all()
    # sanity: mel-scale L1-ish proxy loss stays finite (mirrors training_step_g)
    loss = torch.sum(l_length.float()) + y_hat.abs().mean()
    assert torch.isfinite(loss)


def test_vitsmodel_lightning_training_step_with_d_vectors():
    """End-to-end through the shared VitsModel lightning module (generator + discriminator steps)."""
    from phoonnx_train.vits.lightning import VitsModel

    model = VitsModel(
        num_symbols=TINY_KWARGS["n_vocab"],
        num_speakers=1,
        sample_rate=2000,
        filter_length=16,
        hop_length=4,
        win_length=16,
        mel_channels=9,
        segment_size=TINY_KWARGS["segment_size"] * 4,
        inter_channels=TINY_KWARGS["inter_channels"],
        hidden_channels=TINY_KWARGS["hidden_channels"],
        filter_channels=TINY_KWARGS["filter_channels"],
        n_heads=TINY_KWARGS["n_heads"],
        n_layers=TINY_KWARGS["n_layers"],
        resblock=TINY_KWARGS["resblock"],
        resblock_kernel_sizes=TINY_KWARGS["resblock_kernel_sizes"],
        resblock_dilation_sizes=TINY_KWARGS["resblock_dilation_sizes"],
        upsample_rates=TINY_KWARGS["upsample_rates"],
        upsample_initial_channel=TINY_KWARGS["upsample_initial_channel"],
        upsample_kernel_sizes=TINY_KWARGS["upsample_kernel_sizes"],
        gin_channels=GIN_CHANNELS,
        external_speaker_embedding=True,
        speaker_embedding_dim=D_VECTOR_DIM,
        dataset=None,
        batch_size=2,
    )
    batch = _synthetic_batch(spec_len=8)
    g_loss = model.training_step_g(batch)
    assert torch.isfinite(g_loss)
    d_loss = model.training_step_d(batch)
    assert torch.isfinite(d_loss)


# ------------------------------------------------------------------ dataset

def test_collate_batches_d_vectors_and_language_ids(tmp_path):
    d_vec_path = tmp_path / "spk.pt"
    torch.save(torch.randn(D_VECTOR_DIM), d_vec_path)

    utt = Utterance(
        phoneme_ids=[1, 2, 3],
        audio_norm_path=tmp_path / "a.pt",
        audio_spec_path=tmp_path / "s.pt",
        d_vector_path=d_vec_path,
        language_id=1,
    )
    torch.save(torch.randn(1, 400), utt.audio_norm_path)
    torch.save(torch.randn(9, 5), utt.audio_spec_path)

    ut = UtteranceTensors(
        phoneme_ids=torch.LongTensor(utt.phoneme_ids),
        spectrogram=torch.load(utt.audio_spec_path),
        audio_norm=torch.load(utt.audio_norm_path),
        d_vector=torch.load(utt.d_vector_path).reshape(-1).float(),
        language_id=torch.LongTensor([utt.language_id]),
    )
    collate = UtteranceCollate(is_multispeaker=False, segment_size=4, has_d_vector=True, has_language_id=True)
    batch = collate([ut, ut])
    assert batch.d_vectors.shape == (2, D_VECTOR_DIM)
    assert batch.language_ids.tolist() == [1, 1]


def test_dataset_roundtrip_d_vector_path(tmp_path):
    d_vec_path = tmp_path / "spk.pt"
    torch.save(torch.randn(D_VECTOR_DIM), d_vec_path)
    line = json.dumps({
        "phoneme_ids": [1, 2, 3],
        "audio_norm_path": str(tmp_path / "a.pt"),
        "audio_spec_path": str(tmp_path / "s.pt"),
        "d_vector_path": str(d_vec_path),
        "language_id": 3,
    })
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(line + "\n", encoding="utf-8")

    from phoonnx_train.vits.dataset import PhoonnxDataset

    utts = list(PhoonnxDataset.load_dataset(dataset_path))
    assert len(utts) == 1
    assert utts[0].d_vector_path == d_vec_path
    assert utts[0].language_id == 3


# ------------------------------------------------------------- extra_preprocess

def test_extra_preprocess_requires_speaker_encoder_path(tmp_path):
    engine = YourttsTrainingEngine()
    with pytest.raises(ValueError):
        engine.extra_preprocess(tmp_path / "clip.wav", tmp_path, 16000)


def test_extra_preprocess_caches_d_vector(tmp_path, monkeypatch):
    """Mock the speaker encoder (no real ONNX download) and check caching behaviour."""
    import soundfile as sf

    audio_path = tmp_path / "clip.wav"
    sr = 16000
    sf.write(str(audio_path), np.random.randn(sr).astype(np.float32), sr)

    class _FakeEncoder:
        calls = 0

        def encode(self, audio, sample_rate):
            _FakeEncoder.calls += 1
            return np.random.randn(D_VECTOR_DIM).astype(np.float32)

    engine = YourttsTrainingEngine()
    monkeypatch.setattr(engine, "_get_speaker_encoder", lambda path: _FakeEncoder())

    cache_dir = tmp_path / "cache"
    out1 = engine.extra_preprocess(
        audio_path, cache_dir, sr,
        speaker_encoder_path="unused.onnx", language_id=2,
    )
    assert "d_vector_path" in out1
    assert out1["language_id"] == 2
    dvec_path = out1["d_vector_path"]
    assert torch.load(dvec_path).shape == (D_VECTOR_DIM,)

    # Second call must hit the cache, not recompute.
    out2 = engine.extra_preprocess(
        audio_path, cache_dir, sr, speaker_encoder_path="unused.onnx",
    )
    assert out2["d_vector_path"] == dvec_path
    assert _FakeEncoder.calls == 1


def test_mean_dataset_d_vector(tmp_path):
    vecs = [np.full(D_VECTOR_DIM, i, dtype=np.float32) for i in range(3)]
    paths = []
    for i, v in enumerate(vecs):
        p = tmp_path / f"d{i}.pt"
        torch.save(torch.from_numpy(v), p)
        paths.append(p)

    dataset_path = tmp_path / "dataset.jsonl"
    with open(dataset_path, "w", encoding="utf-8") as f:
        for i, p in enumerate(paths):
            f.write(json.dumps({
                "phoneme_ids": [1],
                "audio_norm_path": "x",
                "audio_spec_path": "y",
                "d_vector_path": str(p),
            }) + "\n")

    mean = _mean_dataset_d_vector([dataset_path])
    assert mean is not None
    raw = np.mean(np.stack(vecs), axis=0)
    expected = raw / np.linalg.norm(raw)  # renormalized after averaging
    assert np.allclose(mean, expected, atol=1e-5)
    assert abs(np.linalg.norm(mean) - 1.0) < 1e-5


def test_mean_dataset_d_vector_empty_returns_none(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(json.dumps({
        "phoneme_ids": [1], "audio_norm_path": "x", "audio_spec_path": "y",
    }) + "\n", encoding="utf-8")
    assert _mean_dataset_d_vector([dataset_path]) is None


# ------------------------------------------------------------------ export

def _train_tiny_lightning_module():
    from phoonnx_train.vits.lightning import VitsModel

    return VitsModel(
        num_symbols=TINY_KWARGS["n_vocab"],
        num_speakers=1,
        sample_rate=2000,
        filter_length=16,
        hop_length=4,
        win_length=16,
        mel_channels=9,
        segment_size=TINY_KWARGS["segment_size"] * 4,
        inter_channels=TINY_KWARGS["inter_channels"],
        hidden_channels=TINY_KWARGS["hidden_channels"],
        filter_channels=TINY_KWARGS["filter_channels"],
        n_heads=TINY_KWARGS["n_heads"],
        n_layers=TINY_KWARGS["n_layers"],
        resblock=TINY_KWARGS["resblock"],
        resblock_kernel_sizes=TINY_KWARGS["resblock_kernel_sizes"],
        resblock_dilation_sizes=TINY_KWARGS["resblock_dilation_sizes"],
        upsample_rates=TINY_KWARGS["upsample_rates"],
        upsample_initial_channel=TINY_KWARGS["upsample_initial_channel"],
        upsample_kernel_sizes=TINY_KWARGS["upsample_kernel_sizes"],
        gin_channels=GIN_CHANNELS,
        external_speaker_embedding=True,
        speaker_embedding_dim=D_VECTOR_DIM,
        n_langs=3,
        dataset=None,
        batch_size=1,
    )


def test_export_onnx_yourtts_loads_and_matches_adapter(tmp_path):
    """
    Export a tiny checkpoint end-to-end and verify:
    - onnxruntime can load it
    - YourTTSAdapter.detect() recognizes the exported voice config
    - the synthesis feed dict accepts a d_vector input matching the graph
    """
    model = _train_tiny_lightning_module()
    ckpt_path = tmp_path / "tiny.ckpt"
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False)
    trainer.strategy.connect(model)
    trainer.save_checkpoint(ckpt_path)

    config = {
        "audio": {"sample_rate": 2000, "quality": "test"},
        "inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8},
        "alphabet": "ipa",
        "phoneme_type": "espeak",
        "phonemizer_model": "",
        "phoneme_id_map": {str(i): i for i in range(TINY_KWARGS["n_vocab"])},
        "num_symbols": TINY_KWARGS["n_vocab"],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    engine = YourttsTrainingEngine()
    onnx_path = engine.export_onnx(
        checkpoint_path=ckpt_path,
        config_path=config_path,
        output_dir=tmp_path,
        default_d_vector=[0.1] * D_VECTOR_DIM,
    )
    assert onnx_path.exists()

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = {i.name for i in session.get_inputs()}
    assert {"input", "input_lengths", "scales", "d_vector", "langid"} <= input_names

    voice_json_path = tmp_path / f"{ckpt_path.stem}.json"
    assert voice_json_path.exists()
    voice_cfg = json.loads(voice_json_path.read_text(encoding="utf-8"))
    assert voice_cfg["engine"] == "yourtts"
    assert voice_cfg["engine_params"]["d_vector"] == [0.1] * D_VECTOR_DIM

    from phoonnx.engines.yourtts import YourTTSAdapter
    assert YourTTSAdapter.detect(config=voice_cfg) is True

    adapter = YourTTSAdapter(d_vector=np.array(voice_cfg["engine_params"]["d_vector"], np.float32))
    from phoonnx.engines.base import AdapterSynthesisRequest

    request = AdapterSynthesisRequest(
        phoneme_ids=np.array([[1, 2, 3, 4, 5]], dtype=np.int64),
        phoneme_lengths=np.array([5], dtype=np.int64),
        language_id=None, params={},
    )
    feed = adapter.build_feed_dict(request, session)
    assert feed["d_vector"].shape == (1, D_VECTOR_DIM)
    outputs = session.run(None, feed)
    result = adapter.parse_outputs(outputs, request)
    assert np.isfinite(result.audio).all()
    assert result.audio.size > 0


# ----------------------------------------------------------------------
# preprocess wiring: engine extras reach dataset.jsonl
# ----------------------------------------------------------------------

def test_utterance_carries_engine_extras(tmp_path):
    import json

    from phoonnx_train.preprocess import Utterance

    utt = Utterance(text="hi", audio_path=tmp_path / "a.wav")
    utt.d_vector_path = tmp_path / "dvec.pt"
    utt.language_id = 2
    data = utt.asdict()
    assert data["d_vector_path"] == str(tmp_path / "dvec.pt")
    assert data["language_id"] == 2
    # round-trips through the dataset loader
    from phoonnx_train.vits.dataset import Utterance as LoaderUtterance
    line = json.dumps({"phoneme_ids": [1, 2], "audio_norm_path": "x.pt",
                       "audio_spec_path": "y.pt", "text": "hi",
                       "d_vector_path": str(tmp_path / "dvec.pt"),
                       "language_id": 2})
    parsed = json.loads(line)
    loaded = LoaderUtterance(
        phoneme_ids=parsed["phoneme_ids"],
        audio_norm_path=parsed["audio_norm_path"],
        audio_spec_path=parsed["audio_spec_path"],
        d_vector_path=parsed.get("d_vector_path"),
        language_id=parsed.get("language_id"),
        text=parsed.get("text"),
    )
    assert loaded.d_vector_path and loaded.language_id == 2


def test_preprocess_cli_exposes_engine_options():
    import click.testing

    from phoonnx_train.preprocess import cli

    result = click.testing.CliRunner().invoke(cli, ["--help"])
    assert "--engine" in result.output
    assert "--speaker-encoder-path" in result.output
    assert "--language-id" in result.output


# ----------------------------------------------------------------------
# mean d-vector renormalization
# ----------------------------------------------------------------------

def test_mean_d_vector_renormalized():
    import numpy as np

    from phoonnx_train.engines.yourtts import _renormalize

    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    mean = np.mean(np.stack([a, b]), axis=0)  # norm ~0.707
    renorm = _renormalize(mean)
    assert abs(np.linalg.norm(renorm) - 1.0) < 1e-6
    assert np.allclose(_renormalize(np.zeros(2)), np.zeros(2))  # no div-by-0


# ----------------------------------------------------------------------
# speaker-consistency loss
# ----------------------------------------------------------------------

def test_speaker_encoder_forward_differentiable():
    import torch

    from phoonnx_train.vits.speaker_encoder import ResNetSpeakerEncoder

    enc = ResNetSpeakerEncoder()
    wav = torch.randn(2, 16000, requires_grad=True)
    emb = enc(wav)
    assert emb.shape == (2, 512)
    assert torch.allclose(emb.norm(dim=1), torch.ones(2), atol=1e-4)
    emb.sum().backward()
    assert wav.grad is not None and torch.isfinite(wav.grad).all()


def test_speaker_consistency_loss_in_generator(tmp_path):
    import torch

    from phoonnx_train.vits.lightning import VitsModel
    from phoonnx_train.vits.speaker_encoder import ResNetSpeakerEncoder

    model = VitsModel(num_symbols=32, num_speakers=1, dataset=None,
                      speaker_encoder_checkpoint="unused-lazy.pt", c_scl=9.0)
    # inject a random-init encoder (checkpoint loading is exercised
    # separately; the loss path must not require a download)
    model._speaker_encoder = ResNetSpeakerEncoder().eval()
    for p in model._speaker_encoder.parameters():
        p.requires_grad_(False)

    y = torch.randn(2, 1, 8192)
    y_hat = torch.randn(2, 1, 8192, requires_grad=True)
    loss = model._speaker_consistency_loss(y, y_hat)
    assert loss is not None and torch.isfinite(loss)
    assert -1.0 <= float(loss) <= 1.0  # negative cosine similarity
    loss.backward()
    assert y_hat.grad is not None  # gradients reach the generated audio

    model_off = VitsModel(num_symbols=32, num_speakers=1, dataset=None)
    assert model_off._speaker_consistency_loss(y, y_hat.detach()) is None
