"""ONNX export round-trip for the vendored MOSS-TTS-Nano trainer.

A tiny randomly-initialised model is exported and then driven through exactly the graph
protocol :class:`phoonnx.engines.mosstts.MossTTSNanoAdapter` uses::

    global_hidden, past = prefill(rows, attention_mask)
    should_continue, frame = local_fixed_sampled_frame(global_hidden, ...)
    global_hidden, past = decode_step(row, past_valid_lengths, past)

so a change that breaks the exported layout fails here rather than at synthesis time.
"""
import numpy as np
import pytest
import torch

onnxruntime = pytest.importorskip("onnxruntime")

from phoonnx_train.mosstts.config import MossTTSNanoConfig
from phoonnx_train.mosstts.export_onnx import export_moss_tts_onnx
from phoonnx_train.mosstts.model import MossTTSNano


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(0)
    config = MossTTSNanoConfig.tiny(n_vq=4, codebook_size=32, vocab_size=64)
    model = MossTTSNano(config)
    output_dir = tmp_path_factory.mktemp("moss_onnx")
    meta_path = export_moss_tts_onnx(
        model, output_dir, opset=17, sample_seq_len=6, sample_past_len=6
    )
    return model, config, output_dir, meta_path


def _session(path):
    return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _named(session, outputs):
    return dict(zip([item.name for item in session.get_outputs()], outputs))


def _rows(config, length, rng):
    rows = np.full((1, length, config.row_width), config.audio_pad_token_id, dtype=np.int32)
    rows[0, :, 0] = rng.integers(0, config.gpt2.vocab_size, size=length)
    return rows


def test_metadata_describes_the_expected_layout(exported):
    import json

    _model, config, _output_dir, meta_path = exported
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert set(meta["files"]) == {
        "prefill", "decode_step", "local_decoder", "local_cached_step",
        "local_fixed_sampled_frame",
    }
    assert meta["model_config"]["row_width"] == config.row_width
    assert meta["model_config"]["n_vq"] == config.n_vq
    assert meta["onnx"]["prefill_output_names"][0] == "global_hidden"


def test_prefill_matches_torch(exported):
    model, config, output_dir, _meta = exported
    rng = np.random.default_rng(0)
    rows = _rows(config, 6, rng)
    mask = np.ones((1, 6), dtype=np.int32)

    session = _session(output_dir / "moss_tts_prefill.onnx")
    outputs = _named(session, session.run(None, {"input_ids": rows, "attention_mask": mask}))

    with torch.no_grad():
        model.set_attention_implementation("eager")
        reference, _ = model(
            input_ids=torch.as_tensor(rows, dtype=torch.long),
            attention_mask=torch.as_tensor(mask, dtype=torch.bool),
        )
    assert np.allclose(outputs["global_hidden"], reference.numpy(), atol=1e-4)
    assert outputs["present_key_0"].shape == (1, 6, config.gpt2.n_head, config.gpt2.head_dim)


def test_prefill_handles_a_different_length_than_it_was_traced_with(exported):
    _model, config, output_dir, _meta = exported
    rng = np.random.default_rng(1)
    session = _session(output_dir / "moss_tts_prefill.onnx")
    for length in (3, 11):
        rows = _rows(config, length, rng)
        outputs = _named(
            session, session.run(None, {"input_ids": rows, "attention_mask": np.ones((1, length), np.int32)})
        )
        assert outputs["global_hidden"].shape == (1, length, config.hidden_size)


def test_decode_step_continues_the_prefill(exported):
    """decode_step over the cache must reproduce a full prefill of the same rows."""
    _model, config, output_dir, _meta = exported
    rng = np.random.default_rng(2)
    rows = _rows(config, 7, rng)

    prefill = _session(output_dir / "moss_tts_prefill.onnx")
    decode = _session(output_dir / "moss_tts_decode_step.onnx")

    full = _named(prefill, prefill.run(
        None, {"input_ids": rows, "attention_mask": np.ones((1, 7), np.int32)}
    ))
    head = _named(prefill, prefill.run(
        None, {"input_ids": rows[:, :6], "attention_mask": np.ones((1, 6), np.int32)}
    ))
    past = {k.replace("present_", "past_"): v for k, v in head.items() if k.startswith("present_")}
    stepped = _named(decode, decode.run(None, {
        "input_ids": rows[:, 6:],
        "past_valid_lengths": np.asarray([6], np.int32),
        **past,
    }))
    assert np.allclose(
        stepped["global_hidden"].reshape(-1), full["global_hidden"][0, -1], atol=1e-4
    )


def test_local_fixed_frame_emits_a_full_frame(exported):
    _model, config, output_dir, _meta = exported
    session = _session(output_dir / "moss_tts_local_fixed_sampled_frame.onnx")
    rng = np.random.default_rng(3)
    outputs = _named(session, session.run(None, {
        "global_hidden": rng.standard_normal((1, config.hidden_size)).astype(np.float32),
        "repetition_seen_mask": np.zeros((1, config.n_vq, config.audio_codebook_sizes[0]), np.int32),
        "assistant_random_u": np.asarray([0.1], np.float32),
        "audio_random_u": np.full((1, config.n_vq), 0.3, np.float32),
    }))
    frame = np.asarray(outputs["frame_token_ids"]).reshape(-1)
    assert frame.shape[0] == config.n_vq
    assert frame.min() >= 0 and frame.max() < config.audio_codebook_sizes[0]
    assert int(np.asarray(outputs["should_continue"]).reshape(-1)[0]) in (0, 1)


def test_local_cached_step_matches_the_uncached_local_decoder(exported):
    """Walking the local transformer channel by channel must equal one full local pass."""
    _model, config, output_dir, _meta = exported
    cached = _session(output_dir / "moss_tts_local_cached_step.onnx")
    whole = _session(output_dir / "moss_tts_local_decoder.onnx")
    rng = np.random.default_rng(4)
    global_hidden = rng.standard_normal((1, config.hidden_size)).astype(np.float32)

    text_token = 3
    prefix = rng.integers(0, config.audio_codebook_sizes[0], size=config.n_vq - 1).astype(np.int32)

    reference = _named(whole, whole.run(None, {
        "global_hidden": global_hidden,
        "text_token_id": np.asarray([text_token], np.int32),
        "audio_prefix_token_ids": prefix.reshape(1, -1),
    }))

    layers = sum(1 for item in cached.get_inputs() if item.name.startswith("local_past_key_"))
    heads = int(cached.get_inputs()[6].shape[2])
    head_dim = int(cached.get_inputs()[6].shape[3])
    past = {
        name: np.zeros((1, 0, heads, head_dim), np.float32)
        for i in range(layers)
        for name in (f"local_past_key_{i}", f"local_past_value_{i}")
    }

    def step(text_id, audio_id, channel, step_type, valid):
        outputs = _named(cached, cached.run(None, {
            "global_hidden": global_hidden,
            "text_token_id": np.asarray([text_id], np.int32),
            "audio_token_id": np.asarray([audio_id], np.int32),
            "channel_index": np.asarray([channel], np.int32),
            "step_type": np.asarray([step_type], np.int32),
            "past_valid_lengths": np.asarray([valid], np.int32),
            **past,
        }))
        nxt = {k.replace("local_present_", "local_past_"): v
               for k, v in outputs.items() if k.startswith("local_present_")}
        past.update(nxt)
        return outputs

    first = step(0, 0, 0, 0, 0)
    assert np.allclose(first["text_logits"].reshape(-1), reference["text_logits"].reshape(-1), atol=1e-4)

    second = step(text_token, 0, 0, 1, 1)
    assert np.allclose(
        second["audio_logits"][0, 0], reference["audio_logits"][0, 0], atol=1e-4
    )
    for channel in range(1, config.n_vq):
        outputs = step(0, int(prefix[channel - 1]), channel - 1, 2, channel + 1)
        assert np.allclose(
            outputs["audio_logits"][0, channel], reference["audio_logits"][0, channel], atol=1e-4
        )


def test_full_adapter_loop_runs(exported):
    """The exact prefill -> local frame -> decode_step loop the phoonnx adapter drives."""
    _model, config, output_dir, _meta = exported
    prefill = _session(output_dir / "moss_tts_prefill.onnx")
    decode = _session(output_dir / "moss_tts_decode_step.onnx")
    local = _session(output_dir / "moss_tts_local_fixed_sampled_frame.onnx")
    rng = np.random.default_rng(5)

    rows = _rows(config, 5, rng)
    outputs = _named(prefill, prefill.run(
        None, {"input_ids": rows, "attention_mask": np.ones((1, 5), np.int32)}
    ))
    global_hidden = outputs["global_hidden"][:, -1, :]
    past = {k.replace("present_", "past_"): v for k, v in outputs.items() if k.startswith("present_")}
    valid_length = 5

    frames = []
    seen = np.zeros((1, config.n_vq, config.audio_codebook_sizes[0]), np.int32)
    for _ in range(4):
        frame_outputs = _named(local, local.run(None, {
            "global_hidden": global_hidden.astype(np.float32),
            "repetition_seen_mask": seen,
            "assistant_random_u": np.asarray([float(rng.random())], np.float32),
            "audio_random_u": rng.random((1, config.n_vq)).astype(np.float32),
        }))
        if not int(np.asarray(frame_outputs["should_continue"]).reshape(-1)[0]):
            break
        frame = np.asarray(frame_outputs["frame_token_ids"]).reshape(-1)
        for channel, token in enumerate(frame):
            seen[0, channel, int(token)] = 1
        frames.append(frame.tolist())

        row = np.full((1, 1, config.row_width), config.audio_pad_token_id, np.int32)
        row[0, 0, 0] = config.audio_assistant_slot_token_id
        row[0, 0, 1:] = frame
        outputs = _named(decode, decode.run(None, {
            "input_ids": row,
            "past_valid_lengths": np.asarray([valid_length], np.int32),
            **past,
        }))
        global_hidden = outputs["global_hidden"].reshape(1, -1)
        past = {k.replace("present_", "past_"): v for k, v in outputs.items() if k.startswith("present_")}
        valid_length += 1

    assert all(len(frame) == config.n_vq for frame in frames)


def test_external_data_layout(tmp_path):
    """``--external-data`` must produce the shared blobs the upstream layout uses."""
    torch.manual_seed(0)
    config = MossTTSNanoConfig.tiny()
    meta_path = export_moss_tts_onnx(
        MossTTSNano(config), tmp_path / "ext", opset=17,
        sample_seq_len=4, sample_past_len=4, external_data=True,
    )
    output_dir = meta_path.parent
    assert (output_dir / "moss_tts_global_shared.data").exists()
    assert (output_dir / "moss_tts_local_shared.data").exists()
    assert not (output_dir / "moss_tts_prefill.data").exists()
    session = _session(output_dir / "moss_tts_prefill.onnx")
    rows = np.full((1, 3, config.row_width), config.audio_pad_token_id, np.int32)
    rows[0, :, 0] = 1
    outputs = session.run(None, {"input_ids": rows, "attention_mask": np.ones((1, 3), np.int32)})
    assert outputs[0].shape == (1, 3, config.hidden_size)
