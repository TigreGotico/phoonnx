"""Export a Lokutor ito voice checkpoint to a phoonnx ONNX voice.

Wraps the installed ``ito`` package (``pip install ito-tts``; GPL-3.0 — an
export-time dependency of this developer tool, nothing is vendored) and
produces the phoonnx-native voice bundle:

    out_dir/model.onnx   single graph: input(int64[1,L]) + speed(f32[1])
                         -> wav / durations / mel / f0
    out_dir/config.json  phoonnx config (ito symbol table, engine "ito")

The ito voice model is CC BY-NC-SA 4.0 + Lokutor terms (non-commercial);
the exported artifacts inherit that license — do not redistribute them
without Lokutor's terms.

Usage:
    python export_ito_onnx.py --ckpt ito_female.pt --out-dir ./ito_female
    python export_ito_onnx.py --ckpt ito_female.pt --out-dir ./ito_female \
        --check-parity --wav parity.wav

The parity check synthesizes a few sentences through the ito PyTorch
reference (float64 prosody path, as ``ito.synth.Chain.infer_full``) and
through a float32-prosody torch twin of the export graph, then compares
the ONNX graph against both and reports SNR per stage.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from ito_onnx_graph import ItoOnnxGraph, snr_db  # noqa: E402


def load_ito(ckpt_path: str):
    """Load a checkpoint through the ito package; returns (front, voc, style, meta)."""
    from ito.front import Front
    from ito.vocoder import Vocoder

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    front = Front(ck["front"]["cfg"])
    front.load_state_dict(ck["front"]["model"])
    voc = Vocoder(**ck["vocoder"]["vcfg"])
    voc.load_state_dict(ck["vocoder"]["model"])
    style = ck["style"].float().reshape(1, -1)
    return front.eval(), voc.eval(), style, ck.get("meta") or {}


def build_symbol_map() -> dict:
    """ito's symbol table (the StyleTTS 2 table) as a char->id map."""
    from ito.text import SYMBOLS
    return {s: i for i, s in enumerate(SYMBOLS)}


def make_noise(length: int = 8192, seed: int = 0) -> torch.Tensor:
    """The deterministic excitation buffer baked into the graph."""
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.standard_normal(length).astype(np.float32)).reshape(1, -1)


def reference_wav(front, voc, style, tokens, noise, duration_scale=1.0,
                  prosody64=True):
    """Whole-utterance torch synthesis, the ito reference math.

    ``noise`` is the [1, M] buffer the graph tiles internally; the same
    tiling is applied here so both paths see identical excitation.
    prosody64=True replicates ``Chain.infer_full`` (float64 prosody);
    False runs the prosody net in float32 — the export graph's arithmetic.
    """
    with torch.no_grad():
        tmask = torch.ones(1, 1, tokens.size(1))
        s = front.style_proj(style)
        h = front.emb(tokens).transpose(1, 2) * tmask
        for layer in front.enc:
            h = layer(h, tmask)
        if front.text_rnn:
            lens = tmask[:, 0].sum(1).long().cpu()
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            pk = pack_padded_sequence(h.transpose(1, 2), lens, batch_first=True,
                                      enforce_sorted=False)
            o, _ = front.rnn(pk)
            o, _ = pad_packed_sequence(o, batch_first=True, total_length=h.size(-1))
            h = (h + front.rnn_proj(o).transpose(1, 2)) * tmask
        x = h
        for layer in front.dur_layers:
            x = layer(x, s, tmask)
        logd = front.dur_out(x)[:, 0]
        dur = (torch.round(torch.exp(logd) * duration_scale).clamp(min=1)
               * tmask[:, 0]).long()
        hf, _ = front.regulate(h, dur)

        if prosody64:
            import copy
            p64 = copy.deepcopy(torch.nn.ModuleDict(
                dict(pros_in=front.pros_in, pros=front.pros, pros_out=front.pros_out))).double()
            x = p64["pros_in"](hf.double())
            for layer in p64["pros"]:
                x = layer(x, s.double())
            pred = p64["pros_out"](x).to(torch.float32)
        else:
            x = front.pros_in(hf)
            for layer in front.pros:
                x = layer(x, s)
            pred = front.pros_out(x)
        lf0n, v, en, f0hz = front.pred_to_curves(pred)
        mel = front.mel_head(hf, s, lf0n, v, en)

        T = int(dur.sum())
        n_samples = T * voc.hop
        idx = torch.arange(n_samples) % noise.size(1)
        tiled = noise[:, idx]
        wav = voc(mel, f0hz, torch.full((1,), 0.25, dtype=torch.float64), tiled)
        return wav[0].float().numpy(), dur[0].numpy(), mel, f0hz


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="path to ito_*.pt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--voice", default=None,
                    help="voice label for config metadata (default: checkpoint stem)")
    ap.add_argument("--noise-len", type=int, default=8192)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--check-parity", action="store_true")
    ap.add_argument("--wav", default=None,
                    help="write the ONNX parity synthesis as a wav file")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    front, voc, style, meta = load_ito(args.ckpt)
    noise = make_noise(args.noise_len, args.noise_seed)
    graph = ItoOnnxGraph(front, voc, style, noise).eval()

    example = torch.tensor([[0, 104, 60, 83]], dtype=torch.int64)
    example_speed = torch.tensor([1.0], dtype=torch.float32)
    model_path = out_dir / "model.onnx"
    torch.onnx.export(
        graph, (example, example_speed), str(model_path),
        opset_version=args.opset, dynamo=False,
        input_names=["input", "speed"],
        output_names=["wav", "durations", "mel", "f0"],
        dynamic_axes={"input": {1: "L"}},
    )
    # the dynamo exporter writes weights as external data; fold them back
    # into a single portable file
    import onnx
    data_path = Path(str(model_path) + ".data")
    if data_path.exists():
        model = onnx.load_model(str(model_path), load_external_data=True)
        onnx.save_model(model, str(model_path), save_as_external_data=False)
        data_path.unlink()
    print(f"exported {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")

    config = {
        "phoonnx_version": "1.0",
        "engine": "ito",
        "phoneme_type": "styletts2",
        "alphabet": "ipa",
        "lang_code": "en-US",
        "audio": {"sample_rate": 24000},
        "num_symbols": len(build_symbol_map()),
        "num_speakers": 1,
        "num_langs": 1,
        "speaker_id_map": {},
        "lang_id_map": {},
        "phonemizer_model": None,
        "add_diacritics": False,
        "inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8},
        "phoneme_id_map": build_symbol_map(),
        # "$" is the table's pad (id 0): the ito front end prepends it to
        # every utterance, and the adapter uses pad_id for that
        "pad": "$",
        # no blanks, no BOS/EOS: the raw espeak char stream (the adapter
        # adds the single leading pad)
        "add_blank_char": False,
        "blank_at_start": False,
        "blank_at_end": False,
        "use_eos_bos": False,
        "hop_length": 300,
        "engine_params": {
            "ito_voice": args.voice or Path(args.ckpt).stem,
            "max_tokens": 400,
        },
    }
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    print(f"wrote {config_path}")

    if args.check_parity:
        check_parity(graph, model_path, front, voc, style, noise, args)


def check_parity(graph, model_path, front, voc, style, noise, args):
    import onnxruntime as ort
    from ito.text import text_to_ids

    sentences = [
        "Hello there!",
        "The package arrives on Friday.",
        "It's about twenty three degrees outside.",
        "I know it sounds strange, but I actually enjoy the quiet hours.",
    ]
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    worst = {"onnx_vs_f32": float("inf"), "onnx_vs_f64": float("inf")}
    for text in sentences:
        ids = text_to_ids(text)
        tokens = torch.tensor([ids], dtype=torch.int64)
        ref64, dur_ref, _, _ = reference_wav(front, voc, style, tokens, noise,
                                             prosody64=True)
        ref32, _, _, _ = reference_wav(front, voc, style, tokens, noise,
                                       prosody64=False)
        feed = {"input": tokens.numpy(), "speed": np.array([1.0], dtype=np.float32)}
        wav_onnx, dur_onnx, mel_onnx, f0_onnx = sess.run(None, feed)
        assert dur_onnx.reshape(-1).tolist() == dur_ref.tolist(), \
            f"durations differ for {text!r}"
        snr32 = snr_db(torch.from_numpy(wav_onnx.reshape(-1)),
                       torch.from_numpy(ref32))
        snr64 = snr_db(torch.from_numpy(wav_onnx.reshape(-1)),
                       torch.from_numpy(ref64))
        print(f"  {text!r}: tokens={tokens.size(1)} frames={int(dur_ref.sum())} "
              f"ONNX-vs-f32 {snr32:.1f} dB, ONNX-vs-f64 {snr64:.1f} dB")
        worst["onnx_vs_f32"] = min(worst["onnx_vs_f32"], snr32)
        worst["onnx_vs_f64"] = min(worst["onnx_vs_f64"], snr64)
        if args.wav:
            import soundfile as sf
            sf.write(args.wav, wav_onnx.reshape(-1), 24000, subtype="PCM_16")
            args.wav = None  # only the first sentence
    print(f"parity: worst ONNX-vs-f32 {worst['onnx_vs_f32']:.1f} dB, "
          f"ONNX-vs-f64 {worst['onnx_vs_f64']:.1f} dB")
    # the residual against either reference is float32 kernel arithmetic (the
    # ORT GRU/conv kernels vs torch's), not graph translation — a wrong op
    # lands orders of magnitude below this. The ito chip engine itself is
    # int8/int16-quantized against this same float reference.
    if min(worst["onnx_vs_f32"], worst["onnx_vs_f64"]) < 55:
        print("FAIL: ONNX deviates from the reference by more than 55 dB")
        sys.exit(1)


if __name__ == "__main__":
    main()
