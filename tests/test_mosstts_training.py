"""Unit tests for the vendored MOSS-TTS-Nano training pipeline."""
import json
import math

import numpy as np
import pytest
import torch

from phoonnx_train.mosstts.config import GPT2DecoderConfig, MossTTSNanoConfig
from phoonnx_train.mosstts.dataset import (
    IGNORE_INDEX,
    MossTTSNanoCollator,
    MossTTSNanoDataset,
    dump_jsonl,
    load_jsonl,
)
from phoonnx_train.mosstts.lightning import (
    MossTTSNanoModule,
    build_lr_lambda,
    parse_channelwise_loss_weight,
)
from phoonnx_train.mosstts.model import MossTTSNano
from phoonnx_train.mosstts.warmstart import normalize_source_keys, warm_start


class StubTokenizer:
    """Deterministic byte-ish tokenizer: enough for row-shape assertions, no model file."""

    def __init__(self, vocab_size: int = 64):
        self.vocab_size = vocab_size

    def encode(self, text):
        # ids 16.. keep the special ids (0-15) free
        return [16 + (ord(char) % (self.vocab_size - 16)) for char in text]

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)


@pytest.fixture
def config():
    return MossTTSNanoConfig.tiny()


@pytest.fixture
def tokenizer(config):
    return StubTokenizer(vocab_size=config.gpt2.vocab_size)


def make_record(frames=5, n_vq=4, codebook=32, text="hi", ref_frames=0):
    rng = np.random.default_rng(0)
    record = {
        "audio": "fake.wav",
        "text": text,
        "audio_codes": rng.integers(0, codebook, size=(frames, n_vq)).tolist(),
    }
    if ref_frames:
        record["ref_audio"] = "ref.wav"
        record["ref_audio_codes"] = rng.integers(0, codebook, size=(ref_frames, n_vq)).tolist()
    return record


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
def test_config_rejects_pad_inside_codebook():
    with pytest.raises(ValueError, match="audio_pad_token_id"):
        MossTTSNanoConfig(n_vq=2, audio_codebook_sizes=[1024, 1024], audio_pad_token_id=512)


def test_config_rejects_wrong_codebook_count():
    with pytest.raises(ValueError, match="length n_vq"):
        MossTTSNanoConfig(n_vq=3, audio_codebook_sizes=[1024, 1024])


def test_config_rejects_delay_pattern_architecture(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model_type": "moss_tts_nano", "model_architecture": "delay_pattern"}))
    with pytest.raises(ValueError, match="global-local"):
        MossTTSNanoConfig.from_json_file(path)


def test_config_roundtrip(tmp_path, config):
    path = config.save_json(tmp_path / "config.json")
    restored = MossTTSNanoConfig.from_json_file(path)
    assert restored.to_dict() == config.to_dict()


def test_upstream_config_shape_is_accepted(tmp_path):
    """The real checkpoint's ``config.json`` layout must load without transformers."""
    payload = {
        "model_type": "moss_tts_nano",
        "model_architecture": "global_local_transformer",
        "n_vq": 16,
        "audio_codebook_sizes": [1024] * 16,
        "audio_pad_token_id": 1024,
        "pad_token_id": 3,
        "im_start_token_id": 4,
        "im_end_token_id": 5,
        "audio_start_token_id": 6,
        "audio_end_token_id": 7,
        "audio_user_slot_token_id": 8,
        "audio_assistant_slot_token_id": 9,
        "local_transformer_layers": 1,
        "gpt2_config": {
            "vocab_size": 16384, "n_positions": 32768, "n_embd": 768, "n_layer": 12,
            "n_head": 12, "n_inner": 3072, "activation_function": "gelu_new",
            "position_embedding_type": "rope", "rope_base": 10000.0,
            "layer_norm_epsilon": 1e-5, "pad_token_id": 3,
            "tie_word_embeddings": True, "summary_type": "cls_index",
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    config = MossTTSNanoConfig.from_json_file(path)
    assert config.n_vq == 16
    assert config.row_width == 17
    assert config.gpt2.n_layer == 12
    assert config.local_gpt2().n_layer == 1
    assert config.local_gpt2().n_positions == 17


# ----------------------------------------------------------------------
# dataset / collation
# ----------------------------------------------------------------------
def test_rows_have_slot_and_code_columns(config, tokenizer):
    dataset = MossTTSNanoDataset([make_record()], tokenizer, config, max_length=256)
    example = dataset[0]
    rows = example["rows"]
    assert rows.shape[1] == config.row_width
    prompt_length = int(example["prompt_length"])
    # the prompt is all text rows: audio columns are pad
    assert torch.all(rows[:prompt_length, 1:] == config.audio_pad_token_id)
    # the target rows carry the assistant slot in column 0
    audio_rows = rows[prompt_length:-1]
    assert torch.all(audio_rows[:, 0] == config.audio_assistant_slot_token_id)
    assert torch.all(audio_rows[:, 1:] < config.audio_pad_token_id)
    # the sequence terminates with the audio-end text token
    assert int(rows[-1, 0]) == config.audio_end_token_id


def test_reference_codes_use_the_user_slot(config, tokenizer):
    dataset = MossTTSNanoDataset([make_record(ref_frames=3)], tokenizer, config, max_length=256)
    rows = dataset[0]["rows"]
    user_slot_rows = rows[rows[:, 0] == config.audio_user_slot_token_id]
    assert user_slot_rows.shape[0] == 3


def test_prompt_style_controls_the_im_end_token(config, tokenizer):
    record = make_record()
    inference = MossTTSNanoDataset([record], tokenizer, config, max_length=256, prompt_style="inference")
    finetuning = MossTTSNanoDataset([record], tokenizer, config, max_length=256, prompt_style="finetuning")
    inference_ids = inference[0]["rows"][:, 0].tolist()
    finetuning_ids = finetuning[0]["rows"][:, 0].tolist()
    assert config.im_end_token_id in inference_ids
    assert config.im_end_token_id not in finetuning_ids
    assert len(inference_ids) == len(finetuning_ids) + 1


def test_collate_masks_the_prompt_and_shifts_by_one(config, tokenizer):
    dataset = MossTTSNanoDataset([make_record(frames=4)], tokenizer, config, max_length=256)
    batch = MossTTSNanoCollator(config)([dataset[0]])
    rows = dataset[0]["rows"]
    prompt_length = int(dataset[0]["prompt_length"])

    assert batch["input_ids"].shape[1] == rows.shape[0] - 1
    assert batch["labels"].shape == batch["input_ids"].shape
    # labels are the next row, wherever supervision is live
    assert torch.equal(batch["input_ids"][0, 1], rows[1])
    assert torch.equal(batch["labels"][0, prompt_length - 1], rows[prompt_length])
    # nothing before the last prompt row is supervised
    assert torch.all(batch["labels"][0, : prompt_length - 1] == IGNORE_INDEX)
    # the first supervised step predicts the first audio frame
    assert int(batch["labels"][0, prompt_length - 1, 0]) == config.audio_assistant_slot_token_id


def test_collate_pads_a_ragged_batch(config, tokenizer):
    dataset = MossTTSNanoDataset(
        [make_record(frames=3), make_record(frames=9, text="a much longer sentence")],
        tokenizer, config, max_length=256,
    )
    batch = MossTTSNanoCollator(config)([dataset[0], dataset[1]])
    assert batch["input_ids"].shape[0] == 2
    lengths = batch["attention_mask"].sum(dim=1)
    assert lengths[0] != lengths[1]
    # padded positions never produce a target
    padded = ~batch["attention_mask"]
    assert torch.all(batch["labels"][padded] == IGNORE_INDEX)
    # padded rows carry pad ids, not stale data
    assert torch.all(batch["input_ids"][padded][:, 0] == config.pad_token_id)
    assert torch.all(batch["input_ids"][padded][:, 1:] == config.audio_pad_token_id)


def test_audio_pad_columns_are_never_targets(config, tokenizer):
    dataset = MossTTSNanoDataset([make_record()], tokenizer, config, max_length=256)
    batch = MossTTSNanoCollator(config)([dataset[0]])
    assert not bool((batch["labels"][:, :, 1:] == config.audio_pad_token_id).any())


def test_narrow_codes_are_padded_to_model_width(config, tokenizer):
    record = make_record(n_vq=2, codebook=config.audio_codebook_sizes[0])
    dataset = MossTTSNanoDataset([record], tokenizer, config, max_length=256)
    rows = dataset[0]["rows"]
    audio_rows = rows[rows[:, 0] == config.audio_assistant_slot_token_id]
    assert torch.all(audio_rows[:, 3:] == config.audio_pad_token_id)


def test_too_wide_codes_are_rejected(config, tokenizer):
    record = make_record(n_vq=config.n_vq + 1)
    dataset = MossTTSNanoDataset([record], tokenizer, config, max_length=256)
    with pytest.raises(ValueError, match="model expects at most"):
        dataset[0]


def test_missing_codes_are_rejected(config, tokenizer):
    dataset = MossTTSNanoDataset([{"text": "hi"}], tokenizer, config, max_length=256)
    with pytest.raises(ValueError, match="prepare_data"):
        dataset[0]


def test_reference_audio_without_codes_is_rejected(config, tokenizer):
    record = make_record()
    record["ref_audio"] = "ref.wav"
    dataset = MossTTSNanoDataset([record], tokenizer, config, max_length=256)
    with pytest.raises(ValueError, match="ref_audio_codes"):
        dataset[0]


def test_prompt_longer_than_max_length_is_rejected(config, tokenizer):
    dataset = MossTTSNanoDataset([make_record(text="x" * 200)], tokenizer, config, max_length=32)
    with pytest.raises(ValueError, match="max_length"):
        dataset[0]


def test_empty_dataset_is_rejected(config, tokenizer):
    with pytest.raises(ValueError, match="empty"):
        MossTTSNanoDataset([], tokenizer, config)


def test_jsonl_roundtrip(tmp_path):
    records = [make_record(), make_record(text="two")]
    path = dump_jsonl(records, tmp_path / "train.jsonl")
    assert load_jsonl(path) == records


# ----------------------------------------------------------------------
# model
# ----------------------------------------------------------------------
def test_heads_are_tied_to_their_embeddings(config):
    model = MossTTSNano(config)
    assert model.text_lm_head.weight is model.transformer.wte.weight
    for embedding, head in zip(model.audio_embeddings, model.audio_lm_heads):
        assert head.weight is embedding.weight


def test_local_transformer_has_no_wte_parameters(config):
    model = MossTTSNano(config)
    assert not any(key.startswith("local_transformer.wte") for key in model.state_dict())


def test_pad_columns_contribute_nothing_to_the_embedding(config):
    model = MossTTSNano(config).eval()
    rows = torch.full((1, 3, config.row_width), config.audio_pad_token_id, dtype=torch.long)
    rows[:, :, 0] = 5
    with torch.no_grad():
        embeds = model.build_inputs_embeds(rows)
        text_only = model.transformer.wte(rows[..., 0])
    assert torch.allclose(embeds, text_only)


def test_out_of_range_codes_are_rejected(config):
    model = MossTTSNano(config)
    rows = torch.full((1, 2, config.row_width), config.audio_pad_token_id, dtype=torch.long)
    rows[0, 0, 1] = config.audio_codebook_sizes[0] + 5  # above pad, so not masked out
    with pytest.raises(ValueError, match="out-of-range"):
        model.build_inputs_embeds(rows)


def test_local_teacher_forcing_uses_the_previous_channel(config):
    """Channel ``c`` must be conditioned on the *true* channel ``c-1``, not its own target."""
    model = MossTTSNano(config).eval()
    hidden = torch.randn(4, config.hidden_size)
    labels = torch.zeros((4, config.n_vq + 1), dtype=torch.long)
    local_inputs = model.build_local_inputs(hidden, labels)
    assert local_inputs.shape == (4, config.n_vq + 1, config.hidden_size)
    assert torch.allclose(local_inputs[:, 0, :], hidden)
    assert torch.allclose(local_inputs[:, 1, :], model.transformer.wte(labels[:, 0]))
    for channel in range(config.n_vq - 1):
        expected = model.audio_embeddings[channel](labels[:, channel + 1])
        assert torch.allclose(local_inputs[:, channel + 2, :], expected)


def test_ignored_labels_do_not_leak_into_the_local_inputs(config):
    model = MossTTSNano(config).eval()
    hidden = torch.randn(2, config.hidden_size)
    labels = torch.full((2, config.n_vq + 1), IGNORE_INDEX, dtype=torch.long)
    local_inputs = model.build_local_inputs(hidden, labels)
    # -100 targets are masked to a zero contribution on the audio channels
    assert torch.allclose(local_inputs[:, 2:, :], torch.zeros_like(local_inputs[:, 2:, :]))


def test_kv_cache_matches_a_full_forward(config):
    model = MossTTSNano(config).eval()
    rows = torch.full((1, 6, config.row_width), config.audio_pad_token_id, dtype=torch.long)
    rows[:, :, 0] = torch.arange(6) % 10
    mask = torch.ones((1, 6), dtype=torch.bool)
    with torch.no_grad():
        full, _ = model(input_ids=rows, attention_mask=mask)
        prefix, past = model(input_ids=rows[:, :5], attention_mask=mask[:, :5], use_cache=True)
        step, _ = model(
            input_ids=rows[:, 5:],
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
        )
    assert torch.allclose(full[:, -1], step[:, -1], atol=1e-4)


def test_eager_and_sdpa_agree(config):
    model = MossTTSNano(config).eval()
    rows = torch.full((1, 5, config.row_width), config.audio_pad_token_id, dtype=torch.long)
    rows[:, :, 0] = torch.arange(5)
    mask = torch.ones((1, 5), dtype=torch.bool)
    mask[0, -2:] = False
    with torch.no_grad():
        model.set_attention_implementation("eager")
        eager, _ = model(input_ids=rows, attention_mask=mask)
        model.set_attention_implementation("sdpa")
        sdpa, _ = model(input_ids=rows, attention_mask=mask)
    assert torch.allclose(eager, sdpa, atol=1e-4)


# ----------------------------------------------------------------------
# loss / lightning
# ----------------------------------------------------------------------
def test_channelwise_weight_shorthand():
    weights = parse_channelwise_loss_weight("1,32", 17)
    assert weights[0] == 1.0
    assert all(abs(weight - 2.0) < 1e-9 for weight in weights[1:])
    assert len(weights) == 17


def test_channelwise_weight_explicit_list():
    weights = parse_channelwise_loss_weight([1, 2, 3], 3)
    assert weights == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("spec", ["1,2,3", "", "-1,32"])
def test_channelwise_weight_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        parse_channelwise_loss_weight(spec, 17)


def test_lr_lambda_warms_up_then_decays():
    lr_lambda = build_lr_lambda("linear", num_warmup_steps=10, num_training_steps=100)
    assert lr_lambda(0) == 0.0
    assert lr_lambda(5) == pytest.approx(0.5)
    assert lr_lambda(10) == pytest.approx(1.0)
    assert lr_lambda(100) == pytest.approx(0.0, abs=1e-9)


def test_cosine_lr_lambda_is_monotone_after_warmup():
    lr_lambda = build_lr_lambda("cosine", 0, 100)
    values = [lr_lambda(step) for step in range(0, 101, 10)]
    assert all(later <= earlier + 1e-9 for earlier, later in zip(values, values[1:]))


def test_unknown_scheduler_is_rejected():
    with pytest.raises(ValueError, match="unsupported lr_scheduler_type"):
        build_lr_lambda("triangular", 0, 10)


def _tiny_batch(config, tokenizer, records=1):
    dataset = MossTTSNanoDataset(
        [make_record(frames=4, text=f"sample {i}") for i in range(records)],
        tokenizer, config, max_length=256,
    )
    return MossTTSNanoCollator(config)([dataset[i] for i in range(records)])


def test_training_step_produces_a_finite_gradient(config, tokenizer):
    module = MossTTSNanoModule(config=config)
    batch = _tiny_batch(config, tokenizer)
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in module.model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_loss_matches_a_hand_written_weighted_mean(config, tokenizer):
    """The reported loss must be ``sum(w_i CE_i) / sum(w_i)`` over live heads."""
    torch.manual_seed(0)
    module = MossTTSNanoModule(config=config).eval()
    batch = _tiny_batch(config, tokenizer)
    with torch.no_grad():
        outputs = module.compute_loss(batch)
    weights = module.channelwise_loss_weight
    manual_total = float(weights[0]) * float(outputs["loss_text"])
    manual_weight = float(weights[0])
    for channel in range(config.n_vq):
        key = f"loss_vq{channel}"
        if key not in outputs:
            continue
        manual_total += float(weights[channel + 1]) * float(outputs[key])
        manual_weight += float(weights[channel + 1])
    assert float(outputs["loss"]) == pytest.approx(manual_total / manual_weight, rel=1e-5)


def test_loss_ignores_padded_timesteps(config, tokenizer):
    """Padding a batch must not change the loss."""
    torch.manual_seed(0)
    module = MossTTSNanoModule(config=config).eval()
    dataset = MossTTSNanoDataset(
        [make_record(frames=4), make_record(frames=4)], tokenizer, config, max_length=256
    )
    collate = MossTTSNanoCollator(config)
    single = collate([dataset[0]])
    with torch.no_grad():
        alone = float(module.compute_loss(single)["loss"])
    padded = collate([dataset[0]])
    pad_rows = torch.full((1, 6, config.row_width), config.audio_pad_token_id, dtype=torch.long)
    pad_rows[:, :, 0] = config.pad_token_id
    padded["input_ids"] = torch.cat([padded["input_ids"], pad_rows], dim=1)
    padded["attention_mask"] = torch.cat(
        [padded["attention_mask"], torch.zeros((1, 6), dtype=torch.bool)], dim=1
    )
    padded["labels"] = torch.cat(
        [padded["labels"], torch.full((1, 6, config.row_width), IGNORE_INDEX, dtype=torch.long)], dim=1
    )
    with torch.no_grad():
        with_padding = float(module.compute_loss(padded)["loss"])
    assert with_padding == pytest.approx(alone, rel=1e-5)


def test_all_ignored_labels_raise(config, tokenizer):
    module = MossTTSNanoModule(config=config)
    batch = _tiny_batch(config, tokenizer)
    batch["labels"] = torch.full_like(batch["labels"], IGNORE_INDEX)
    with pytest.raises(RuntimeError, match="every label is ignored"):
        module.compute_loss(batch)


def test_optimizer_and_scheduler_follow_upstream_defaults(config):
    module = MossTTSNanoModule(config=config, max_train_steps=100)
    bundle = module.configure_optimizers()
    optimizer = bundle["optimizer"]
    assert optimizer.defaults["lr"] == 1e-5
    assert optimizer.defaults["betas"] == (0.9, 0.95)
    assert optimizer.defaults["weight_decay"] == 0.1
    assert bundle["lr_scheduler"]["interval"] == "step"


# ----------------------------------------------------------------------
# warm start
# ----------------------------------------------------------------------
def test_warm_start_from_a_saved_state_dict(tmp_path, config):
    torch.manual_seed(1)
    source = MossTTSNano(config)
    path = tmp_path / "source.pt"
    torch.save(source.state_dict(), path)

    torch.manual_seed(2)
    target = MossTTSNano(config)
    report = warm_start(target, path)
    assert report.missing == []
    assert report.shape_mismatch == []
    assert report.matched_fraction == pytest.approx(1.0)
    for key, tensor in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], tensor)


def test_warm_start_restores_tied_heads_from_the_embedding_alone(tmp_path, config):
    """A checkpoint that stores only ``transformer.wte`` still warm-starts ``text_lm_head``."""
    torch.manual_seed(1)
    source = MossTTSNano(config)
    state = {k: v for k, v in source.state_dict().items()
             if not k.startswith(("text_lm_head", "audio_lm_heads"))}
    path = tmp_path / "tied.pt"
    torch.save(state, path)

    target = MossTTSNano(MossTTSNanoConfig.tiny())
    report = warm_start(target, path)
    assert report.missing == []
    assert "text_lm_head.weight" in report.tied_from_source
    assert torch.equal(target.text_lm_head.weight, source.transformer.wte.weight)


def test_warm_start_reports_shape_mismatches(tmp_path, config):
    source = MossTTSNano(config)
    state = source.state_dict()
    state["transformer.wte.weight"] = torch.zeros(5, 5)
    path = tmp_path / "bad.pt"
    torch.save(state, path)
    target = MossTTSNano(MossTTSNanoConfig.tiny())
    report = warm_start(target, path)
    assert any(key == "transformer.wte.weight" for key, _, _ in report.shape_mismatch)
    assert report.matched_fraction < 1.0


def test_warm_start_strips_the_lightning_prefix(tmp_path, config):
    module = MossTTSNanoModule(config=config)
    path = tmp_path / "run.ckpt"
    torch.save({"state_dict": module.state_dict()}, path)
    target = MossTTSNano(MossTTSNanoConfig.tiny())
    report = warm_start(target, path)
    assert report.missing == []
    assert report.matched_fraction == pytest.approx(1.0)


def test_normalize_keys_leaves_bare_checkpoints_alone():
    state = {"transformer.wte.weight": torch.zeros(1), "audio_embeddings.0.weight": torch.zeros(1)}
    assert set(normalize_source_keys(state)) == set(state)


def test_warm_start_rejects_a_missing_checkpoint(tmp_path, config):
    with pytest.raises(FileNotFoundError):
        warm_start(MossTTSNano(config), tmp_path / "nope.safetensors")


# ----------------------------------------------------------------------
# prepare_data
# ----------------------------------------------------------------------
def test_fit_channels_duplicates_mono_and_averages_extra():
    from phoonnx_train.mosstts.prepare_data import fit_channels

    mono = np.ones((1, 8), np.float32)
    assert fit_channels(mono, 2).shape == (2, 8)
    stereo = np.stack([np.zeros(8), np.ones(8)]).astype(np.float32)
    assert np.allclose(fit_channels(stereo, 1), 0.5)
    assert fit_channels(stereo, 2) is stereo


def test_resample_changes_length_proportionally():
    from phoonnx_train.mosstts.prepare_data import resample

    audio = np.zeros((2, 1000), np.float32)
    assert resample(audio, 24000, 48000).shape[-1] == 2000
    assert resample(audio, 48000, 48000) is audio


def test_load_manifest_reads_ljspeech_csv(tmp_path):
    from phoonnx_train.mosstts.prepare_data import load_manifest

    csv_path = tmp_path / "metadata.csv"
    csv_path.write_text("0001|Bom dia.\n0002|Boa noite.\n", encoding="utf-8")
    records = load_manifest(csv_path, wav_dir=tmp_path / "wavs")
    assert [record["text"] for record in records] == ["Bom dia.", "Boa noite."]
    assert records[0]["audio"].endswith("wavs/0001.wav")


def test_load_manifest_rejects_a_line_without_text(tmp_path):
    from phoonnx_train.mosstts.prepare_data import load_manifest

    csv_path = tmp_path / "metadata.csv"
    csv_path.write_text("0001\n", encoding="utf-8")
    with pytest.raises(ValueError, match="separated text column"):
        load_manifest(csv_path)


def test_encode_records_caches_and_rejects_empty_codes():
    from phoonnx_train.mosstts.prepare_data import encode_records

    class CountingTokenizer:
        def __init__(self, frames):
            self.frames = frames
            self.calls = 0

        def encode_file(self, path, n_vq=None):
            self.calls += 1
            return [[0] * 4 for _ in range(self.frames)]

    tokenizer = CountingTokenizer(3)
    records = [{"audio": "a.wav", "text": "x"}, {"audio": "a.wav", "text": "y"}]
    encode_records(records, tokenizer)
    assert tokenizer.calls == 1  # the same path is encoded once
    assert len(records[0]["audio_codes"]) == 3

    with pytest.raises(ValueError, match="zero codec frames"):
        encode_records([{"audio": "b.wav", "text": "x"}], CountingTokenizer(0))


def test_encode_records_rejects_a_record_without_audio():
    from phoonnx_train.mosstts.prepare_data import encode_records

    with pytest.raises(ValueError, match="no `audio` path"):
        encode_records([{"text": "x"}], object())
