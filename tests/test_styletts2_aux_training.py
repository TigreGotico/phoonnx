"""CPU-only tests for the StyleTTS2 auxiliary-model training engines
(text aligner, PL-BERT, pitch extractor) — tiny models, no downloads."""
import json
import math
import wave as wave_mod
from pathlib import Path

import numpy as np
import pytest
import torch

from phoonnx_train.engines import get_engine, list_engines
from phoonnx_train.engines.base import TrainingEngineConfig
from phoonnx_train.engines.styletts2_aligner import (AlignerConfig,
                                                     AlignerModule,
                                                     AlignerTrainingEngine)
from phoonnx_train.engines.styletts2_pitch import (PitchConfig, PitchModule,
                                                   PitchTrainingEngine)
from phoonnx_train.engines.styletts2_plbert import (PLBertConfig,
                                                    PLBertDataset,
                                                    PLBertModule,
                                                    PLBertTrainingEngine,
                                                    _collate)

TINY_ALIGNER = dict(n_mels=80, n_token=178, hidden_dim=32,
                    token_embedding_dim=32, n_layers=1)
TINY_PLBERT = dict(vocab_size=178, hidden_size=32, num_attention_heads=2,
                   intermediate_size=64, num_hidden_layers=1,
                   max_position_embeddings=64, max_seq_length=64)


def _write_wav(path: Path, seconds: float = 0.6, sr: int = 24000) -> None:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    samples = (0.3 * np.sin(2 * math.pi * 220 * t) * 32767).astype(np.int16)
    with wave_mod.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(samples.tobytes())


@pytest.fixture()
def dataset_dir(tmp_path: Path) -> Path:
    wavs = tmp_path / "wavs"
    wavs.mkdir()
    lines = []
    for i in range(3):
        _write_wav(wavs / f"utt{i}.wav")
        lines.append(f"utt{i}.wav|mɐɲˈɐ|0")
    (tmp_path / "train_list.txt").write_text("\n".join(lines) + "\n")
    (tmp_path / "val_list.txt").write_text(lines[0] + "\n")
    return tmp_path


@pytest.fixture()
def plbert_data_dir(tmp_path: Path) -> Path:
    rows = [
        {"phonemes": ["ˈɔlɐ", "mˈundu", "bˈɔm"], "words": ["olá", "mundo", "bom"]},
        {"phonemes": ["bˈɔm", "dˈiɐ", "ˈɔlɐ!"], "words": ["bom", "dia", "olá!"]},
    ] * 4
    with open(tmp_path / "data.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    token_maps = {"<sep>": 0, "<unk>": 1, "olá": 2, "mundo": 3, "bom": 4,
                  "dia": 5, "olá!": 6}
    (tmp_path / "token_maps.json").write_text(json.dumps(token_maps))
    return tmp_path


# ----------------------------------------------------------------------
# registry / presets
# ----------------------------------------------------------------------

def test_engines_registered():
    for name in ("styletts2-aligner", "styletts2-plbert", "styletts2-pitch"):
        assert name in list_engines()
        assert get_engine(name).quality_presets()


def test_aux_engines_do_not_export_onnx():
    for cls in (AlignerTrainingEngine, PLBertTrainingEngine, PitchTrainingEngine):
        with pytest.raises(NotImplementedError):
            cls().export_onnx(Path("x"), Path("y"), Path("z"))


# ----------------------------------------------------------------------
# aligner
# ----------------------------------------------------------------------

def _aligner_batch(bsz=2, n_frames=64, n_text=8):
    texts = torch.randint(3, 100, (bsz, n_text))
    text_lengths = torch.full((bsz,), n_text, dtype=torch.long)
    mels = torch.randn(bsz, 80, n_frames)
    mel_lengths = torch.full((bsz,), n_frames, dtype=torch.long)
    return texts, text_lengths, mels, mel_lengths


def test_aligner_training_step_runs():
    module = AlignerModule(AlignerConfig(**TINY_ALIGNER))
    loss = module.training_step(_aligner_batch(), 0)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in module.model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_aligner_checkpoint_consumable(tmp_path):
    from phoonnx_train.styletts2.models import load_ASR_models
    module = AlignerModule(AlignerConfig(**TINY_ALIGNER))
    ckpt = module.save_asr_checkpoint(tmp_path)
    model = load_ASR_models(str(ckpt), str(tmp_path / "config.yml"))
    ref = module.model.state_dict()
    for k, v in model.state_dict().items():
        assert torch.equal(v, ref[k])


def test_aligner_warm_start(tmp_path):
    module = AlignerModule(AlignerConfig(**TINY_ALIGNER))
    ckpt = module.save_asr_checkpoint(tmp_path)
    warm = AlignerModule(AlignerConfig(pretrained_path=str(ckpt), **TINY_ALIGNER))
    for k, v in warm.model.state_dict().items():
        assert torch.equal(v, module.model.state_dict()[k])


def test_aligner_engine_create_model(dataset_dir):
    cfg = TrainingEngineConfig(num_symbols=178, sample_rate=24000,
                               extra={**TINY_ALIGNER, "batch_size": 2})
    model = AlignerTrainingEngine().create_model(cfg, [dataset_dir])
    assert isinstance(model, AlignerModule)
    assert len(model.train_list) == 3 and len(model.val_list) == 1
    assert model.config.root_path.endswith("wavs")


def test_aligner_dataset_and_collate(dataset_dir):
    from phoonnx_train.styletts2.aligner_dataset import (AlignerCollater,
                                                         AuxMelDataset)
    lines = (dataset_dir / "train_list.txt").read_text().splitlines()
    ds = AuxMelDataset(lines, root_path=str(dataset_dir / "wavs"))
    mel, text = ds[0]
    assert mel.size(0) == 80 and text.dim() == 1
    texts, text_lengths, mels, mel_lengths = AlignerCollater()([ds[0], ds[1]])
    assert mels.size(0) == 2 and texts.size(0) == 2
    assert (mel_lengths >= text_lengths).all()
    # mel features were cached
    assert list((dataset_dir / "wavs").glob("*.mel.npy"))


# ----------------------------------------------------------------------
# PL-BERT
# ----------------------------------------------------------------------

def test_plbert_masking_dataset(plbert_data_dir):
    cfg = PLBertConfig(word_mask_prob=1.0, replace_prob=0.0, **TINY_PLBERT)
    ds = PLBertDataset(plbert_data_dir, cfg)
    phonemes, words, labels, masked = ds[0]
    assert phonemes.size(0) == words.size(0) == labels.size(0)
    assert masked.numel() > 0  # every word masked
    assert (phonemes[masked] == ds.mask_id).all()
    # labels keep ground truth under the mask
    assert not (labels[masked] == ds.mask_id).all()

    cfg = PLBertConfig(word_mask_prob=0.0, **TINY_PLBERT)
    _, _, _, masked = PLBertDataset(plbert_data_dir, cfg)[0]
    assert masked.numel() == 0


def test_plbert_prosodic_masking(plbert_data_dir):
    cfg = PLBertConfig(prosodic_masking=True, word_mask_prob=0.0,
                       prosodic_mark_mask_prob=1.0, replace_prob=0.0,
                       **TINY_PLBERT)
    ds = PLBertDataset(plbert_data_dir, cfg)
    assert ds._word_mask_prob("olá!") == 1.0
    assert ds._word_mask_prob("olá") == 0.0
    assert ds._word_mask_prob(",") == cfg.punct_mask_prob


@pytest.mark.parametrize("backbone", ["albert", "modernbert"])
def test_plbert_training_step_runs(plbert_data_dir, backbone):
    cfg = PLBertConfig(backbone=backbone, batch_size=2, **TINY_PLBERT)
    module = PLBertModule(cfg, data_dir=plbert_data_dir)
    ds = PLBertDataset(plbert_data_dir, cfg)
    batch = _collate([ds[0], ds[1]])
    loss_vocab, loss_token = module._losses(batch)
    loss = loss_vocab + loss_token
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.parametrize("backbone", ["albert", "modernbert"])
def test_plbert_dir_consumable(plbert_data_dir, tmp_path, backbone):
    from phoonnx_train.styletts2.Utils.PLBERT.util import load_plbert
    cfg = PLBertConfig(backbone=backbone, **TINY_PLBERT)
    module = PLBertModule(cfg, data_dir=plbert_data_dir)
    out = tmp_path / "plbert"
    module.save_plbert_dir(out, step=17)
    assert (out / "step_17.t7").is_file() and (out / "config.yml").is_file()
    bert = load_plbert(str(out))
    ref = module.model.encoder.state_dict()
    loaded = bert.state_dict()
    matched = [k for k in loaded if k in ref and torch.equal(loaded[k], ref[k])]
    assert len(matched) >= 0.9 * len(ref)
    # forward returns hidden states (CustomAlbert/CustomModernBert contract)
    ids = torch.randint(3, 100, (1, 8))
    out_hidden = bert(ids, attention_mask=torch.ones(1, 8, dtype=torch.int))
    assert out_hidden.shape == (1, 8, cfg.hidden_size)


def test_plbert_warm_start(plbert_data_dir, tmp_path):
    cfg = PLBertConfig(**TINY_PLBERT)
    module = PLBertModule(cfg, data_dir=plbert_data_dir)
    out = tmp_path / "plbert"
    module.save_plbert_dir(out, step=1)
    warm = PLBertModule(PLBertConfig(pretrained_dir=str(out), **TINY_PLBERT),
                        data_dir=plbert_data_dir)
    ref = module.model.encoder.state_dict()
    for k, v in warm.model.encoder.state_dict().items():
        if "position_ids" in k:
            continue
        assert torch.equal(v, ref[k]), k


def test_plbert_engine_create_model(plbert_data_dir):
    cfg = TrainingEngineConfig(extra={**TINY_PLBERT, "batch_size": 2})
    model = PLBertTrainingEngine().create_model(cfg, [plbert_data_dir])
    assert isinstance(model, PLBertModule)
    assert model.num_words == 7


# ----------------------------------------------------------------------
# pitch extractor
# ----------------------------------------------------------------------

def _pitch_batch(bsz=2, seq_len=192):
    mels = torch.randn(bsz, 80, seq_len)
    f0 = torch.rand(bsz, seq_len) * 200
    sil = (torch.rand(bsz, seq_len) > 0.5).float()
    return mels, f0, sil


def test_pitch_training_step_runs():
    module = PitchModule(PitchConfig())
    loss = module.training_step(_pitch_batch(), 0)
    assert torch.isfinite(loss)
    loss.backward()


def test_pitch_checkpoint_consumable(tmp_path):
    from phoonnx_train.styletts2.models import load_F0_models
    module = PitchModule(PitchConfig())
    ckpt = module.save_f0_checkpoint(tmp_path / "f0.t7")
    model = load_F0_models(str(ckpt))
    ref = module.model.state_dict()
    for k, v in model.state_dict().items():
        assert torch.equal(v, ref[k])


def test_pitch_engine_create_model(dataset_dir):
    cfg = TrainingEngineConfig(sample_rate=24000, extra={"batch_size": 2})
    model = PitchTrainingEngine().create_model(cfg, [dataset_dir])
    assert isinstance(model, PitchModule)
    assert len(model.train_list) == 3


def test_pitch_dataset_shapes(dataset_dir):
    from phoonnx_train.engines.styletts2_pitch import PitchSegmentDataset
    lines = (dataset_dir / "train_list.txt").read_text().splitlines()
    ds = PitchSegmentDataset(lines, root_path=str(dataset_dir / "wavs"),
                             sr=24000, seq_len=48)
    mel, f0, sil = ds[0]
    assert mel.shape == (80, 48) and f0.shape == (48,) and sil.shape == (48,)
    assert set(sil.unique().tolist()) <= {0.0, 1.0}


# ----------------------------------------------------------------------
# corpus phonemization
# ----------------------------------------------------------------------

def test_phonemize_corpus_list(tmp_path):
    from phoonnx_train.styletts2.phonemize_corpus import phonemize_list_file
    src = tmp_path / "raw_list.txt"
    src.write_text("utt0.wav|ola mundo|0\nutt1.wav|bom dia|1\n")
    out = tmp_path / "train_list.txt"
    phonemize_list_file(src, out, lang="pt", phonemizer="grapheme")
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("utt0.wav|") and lines[0].endswith("|0")
    assert lines[1].endswith("|1")


def test_phonemize_text_corpus(tmp_path):
    from phoonnx_train.styletts2.phonemize_corpus import phonemize_text_corpus
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("ola mundo lindo\nbom dia mundo\nxx\n")
    out = phonemize_text_corpus(corpus, tmp_path / "plbert", lang="pt",
                                phonemizer="grapheme")
    rows = [json.loads(l) for l in (out / "data.jsonl").read_text().splitlines()]
    assert len(rows) == 2  # the 1-word line is dropped
    assert all(len(r["phonemes"]) == len(r["words"]) for r in rows)
    token_maps = json.loads((out / "token_maps.json").read_text())
    assert token_maps["<sep>"] == 0 and token_maps["<unk>"] == 1
    assert "mundo" in token_maps


def test_symbol_coverage_warns_and_drops(caplog):
    from phoonnx_train.styletts2.phonemize_corpus import check_symbol_coverage
    with caplog.at_level("WARNING"):
        cleaned = check_symbol_coverage("mɐ⟨ɲ⟩ɐ")
    assert "⟨" not in cleaned and "⟩" not in cleaned
    assert "mɐɲɐ" == cleaned
    assert any("not in the StyleTTS2 table" in r.message for r in caplog.records)


# ----------------------------------------------------------------------
# downloads helper
# ----------------------------------------------------------------------

def test_download_helper_offline(tmp_path, monkeypatch):
    from phoonnx_train.styletts2 import downloads

    fetched = []

    def fake_download(url, dest, expected_size):
        fetched.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\0" * expected_size)

    monkeypatch.setattr(downloads, "_download", fake_download)
    paths = downloads.download_aux_models(cache_dir=str(tmp_path))
    assert set(paths) == {"asr_path", "asr_config", "f0_path", "plbert_dir"}
    assert len(fetched) == len(downloads._AUX_FILES)
    assert Path(paths["asr_path"]).is_file()
    # second call is fully cached — no downloads
    fetched.clear()
    downloads.download_aux_models(cache_dir=str(tmp_path))
    assert not fetched


def test_engine_does_not_download_when_paths_given(monkeypatch):
    from phoonnx_train.engines.styletts2 import (StyleTTS2Config,
                                                 _resolve_aux_paths)
    cfg = StyleTTS2Config(download_aux=False)
    _resolve_aux_paths(cfg)  # no network, no error
    assert cfg.asr_path is None

    called = []
    monkeypatch.setattr("phoonnx_train.styletts2.downloads.download_aux_models",
                        lambda **kw: called.append(1) or {
                            "asr_path": "a", "asr_config": "b",
                            "f0_path": "c", "plbert_dir": "d"})
    cfg = StyleTTS2Config(download_aux=True, asr_path="x", asr_config="y")
    _resolve_aux_paths(cfg)
    assert called and cfg.asr_path == "x" and cfg.f0_path == "c"


# ----------------------------------------------------------------------
# repo hygiene
# ----------------------------------------------------------------------

def test_no_vendored_binaries():
    utils_dir = (Path(__file__).parent.parent
                 / "phoonnx_train" / "styletts2" / "Utils")
    big = [p for p in utils_dir.rglob("*")
           if p.is_file() and p.stat().st_size > 1_000_000]
    assert not big, f"large binaries vendored under Utils/: {big}"
