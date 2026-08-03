"""Export an ArkTTS checkpoint to the three phoonnx ONNX graphs.

    python export_arktts_onnx.py --repo Audio8/Audio8-TTS-Preview-0.6b --out-dir ./onnx
    python export_arktts_onnx.py --repo itzune/zortzi-tts --out-dir ./onnx --fp16

The output layout and every tensor name match the official export at
``itzune/zortzi-tts-onnx``, so a mirror built from this script and the official one are
drop-in replacements for each other. See ``arktts_wrappers.py`` for the contract and why
the cache is a fixed window.

``--fp16`` writes a second copy of the two autoregressive graphs in half precision. The
codec decoder is written in single precision only. Its window transformer builds a rotary
table inside the graph, and the half-precision converter leaves that ``Einsum`` with one
single-precision and one half-precision operand, which ONNX Runtime rejects. This is not
worth solving here: both published ArkTTS checkpoints carry a byte-identical ``codec.pth``,
so the fp16 codec decoder published at ``itzune/zortzi-tts-onnx`` decodes either model's
codes, and ``verify_parity.py`` confirms it against whichever checkpoint you point it at.

Verify what comes out with ``verify_parity.py`` before mirroring it. Half precision is
cheap to produce and easy to get wrong; nothing here proves the result is usable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel

from arktts_wrappers import SLOW_CACHE_WIDTH, CodecDecoderWrapper, FastARWrapper, SlowARWrapper

OPSET = 17
"""Matches the official export; every operator used here is core opset 17."""


def cache_names(layers: int) -> tuple[list[str], list[str]]:
    """Input and output names for one model's cache, in the order the wrappers expect."""
    inputs = [f"cache_{kind}_{i}" for i in range(layers) for kind in ("key", "value")]
    outputs = [f"{kind}_delta_{i}" for i in range(layers) for kind in ("key", "value")]
    return inputs, outputs


def drop_norm_casts(model) -> None:
    """Remove the RMS norms' explicit float32 round-trip before tracing.

    ``ArkttsRMSNorm.forward`` writes ``x.float() * rsqrt(...)`` then ``.to(x.dtype)``. That
    protects bfloat16 checkpoints from accumulating the sum of squares in low precision.
    Here the module is already float32, so both casts are no-ops and the traced graph is
    numerically unchanged — but they survive as ``Cast`` nodes with hard-coded target
    types, and those are what make the half-precision converter emit a graph whose operands
    disagree and which ONNX Runtime then refuses to load.

    The consequence is real and worth stating: in the fp16 graph the norm accumulates in
    half precision rather than single. ``verify_parity.py`` measures exactly that.
    """
    import torch as torch_module

    def forward(self, x):
        return x * torch_module.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    type(model.norm).forward = forward


def export_slow_ar(model, path: Path) -> None:
    drop_norm_casts(model)
    wrapper = SlowARWrapper(model).eval()
    config = model.config
    layers = int(config.n_layer)
    cache_in, cache_out = cache_names(layers)
    width = 5
    codes = torch.zeros((1, config.num_codebooks + 1, width), dtype=torch.long)
    codes[0, 0] = torch.arange(width) + config.semantic_begin_id
    input_pos = torch.arange(width, dtype=torch.long)
    cache = [
        torch.zeros((1, config.n_local_heads, SLOW_CACHE_WIDTH, config.head_dim))
        for _ in range(2 * layers)
    ]
    torch.onnx.export(
        wrapper,
        (codes, input_pos, *cache),
        str(path),
        input_names=["codes", "input_pos", *cache_in],
        output_names=["logits", "slow_hidden", *cache_out],
        dynamic_axes={
            "codes": {2: "T"}, "input_pos": {0: "T"},
            "logits": {1: "T"}, "slow_hidden": {1: "T"},
            **{name: {2: "T"} for name in cache_out},
        },
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )


def export_fast_ar(model, path: Path) -> None:
    drop_norm_casts(model)
    wrapper = FastARWrapper(model).eval()
    config = model.config
    layers = int(config.n_fast_layer)
    cache_in, cache_out = cache_names(layers)
    cache = [
        torch.zeros((1, config.fast_n_local_heads, config.num_codebooks, config.fast_head_dim))
        for _ in range(2 * layers)
    ]
    torch.onnx.export(
        wrapper,
        (
            torch.zeros((1, 1, config.dim)),
            torch.zeros((1, 1), dtype=torch.long),
            torch.ones(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.long),
            *cache,
        ),
        str(path),
        input_names=["slow_hidden", "token_id", "use_slow_hidden", "input_pos", *cache_in],
        output_names=["logits", *cache_out],
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )


def simplify_causal_padding(codec) -> int:
    """Drop the codec's dynamic right-padding, which the ONNX tracer cannot lower.

    ``ArkttsCausalConv1d.forward`` calls ``_extra_padding``, whose ``math.ceil`` over a
    traced shape lowers to ``aten::__iand_`` and has no opset-17 equivalent. The helper
    is dead weight in this graph: with ``stride == 1`` it reduces to

        frames = (L - k + (k - 1)) / 1 + 1 = L,  ideal = (L - 1) + k - (k - 1) = L

    so the extra padding is exactly zero for every input length.

    The check is scoped to what ``ArkttsCodec.decode`` actually runs — the quantizer's
    codebook lookups, the post module, the upsampler and the decoder stack. The analysis
    encoder does contain strided causal convolutions, and it is not exported here: this
    engine never needs it, because voices ship as pre-encoded codes rather than as audio.
    A strided convolution *on the decode path* would make the simplification wrong rather
    than merely untraceable, so this raises instead of writing a subtly different model.
    """
    import torch.nn.functional as functional

    module = type(codec).__mro__[0].__module__
    causal = __import__(module, fromlist=["ArkttsCausalConv1d"]).ArkttsCausalConv1d
    on_decode_path = [
        (name, child)
        for root in (codec.decoder, codec.quantizer.post_module, codec.quantizer.upsample)
        for name, child in root.named_modules()
        if isinstance(child, causal)
    ]
    strided = [name for name, child in on_decode_path if child.stride != 1]
    if strided:
        raise RuntimeError(f"strided causal convolutions on the decode path: {strided}; "
                           "the zero-extra-padding simplification does not hold")

    def forward(self, x):
        return self.conv(functional.pad(x, (self.padding, 0))).contiguous()

    causal.forward = forward
    return len(on_decode_path)


def register_inplace_and() -> None:
    """Teach the tracer the in-place boolean ``and`` the codec's window mask uses.

    ``ArkttsCodecWindowTransformer.forward`` narrows its causal mask with ``mask &= ...``.
    The tracer records that as ``aten::__iand_``, for which opset 17 has no built-in
    lowering, so the export stops. Boolean ``And`` is the same operation — the only thing
    the in-place form adds is reusing the buffer, which has no meaning in a graph.
    """
    from torch.onnx import register_custom_op_symbolic

    register_custom_op_symbolic(
        "aten::__iand_", lambda g, self, other: g.op("And", self, other), OPSET)


def replace_complex_rope(codec) -> None:
    """Rebuild the codec's rotary table without complex numbers.

    ``_rope`` builds it with ``torch.polar``, which opset 17 cannot express because it has
    no complex tensors at all. ``polar(1, phase)`` is ``cos(phase) + i·sin(phase)``, and
    the function immediately splits the result back into its real and imaginary parts, so
    building those two directly is the identical computation with the complex step removed.

    The rotary table is left in float32 rather than the checkpoint's bfloat16. Casting a
    table of sines to bfloat16 only loses precision, and it would force a bfloat16 tensor
    into a graph that has none anywhere else.

    The codec's own norms and rotary application are rebuilt at the same time, for the
    reason :func:`drop_norm_casts` explains: their explicit float32 round-trips are no-ops
    in a float32 export and become ``Cast`` nodes that break the half-precision converter.
    """
    import torch as torch_module

    module = __import__(type(codec).__mro__[0].__module__,
                        fromlist=["_rope", "ArkttsCodecRMSNorm"])

    def rope(length: int, head_dim: int, base: float, device=None):
        frequencies = 1.0 / (
            base ** (torch_module.arange(0, head_dim, 2, device=device).float() / head_dim))
        phases = torch_module.outer(
            torch_module.arange(length, device=device).float(), frequencies)
        return torch_module.stack(
            (torch_module.cos(phases), torch_module.sin(phases)), dim=-1)

    def apply_rope(x, values):
        shaped = x.reshape(*x.shape[:-1], -1, 2)
        values = values.view(1, shaped.shape[1], 1, shaped.shape[3], 2)
        return torch_module.stack(
            (
                shaped[..., 0] * values[..., 0] - shaped[..., 1] * values[..., 1],
                shaped[..., 1] * values[..., 0] + shaped[..., 0] * values[..., 1],
            ),
            dim=-1,
        ).flatten(3)

    def norm_forward(self, x):
        return x * torch_module.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    module._rope = rope
    module._apply_rope = apply_rope
    module.ArkttsCodecRMSNorm.forward = norm_forward


def export_codec_decoder(model, path: Path) -> None:
    register_inplace_and()
    codec = model.load_codec(device="cpu")
    replace_complex_rope(codec)
    patched = simplify_causal_padding(codec)
    print(f"  simplified right-padding on {patched} causal convolutions")
    wrapper = CodecDecoderWrapper(codec).eval()
    codes = torch.zeros((1, model.config.num_codebooks, 16), dtype=torch.long)
    torch.onnx.export(
        wrapper,
        (codes,),
        str(path),
        input_names=["codes"],
        output_names=["audio"],
        dynamic_axes={"codes": {2: "T"}, "audio": {2: "samples"}},
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )


def consolidate(path: Path) -> None:
    """Rewrite a graph so its weights live in one sidecar file next to it.

    Past 2 GB ``torch.onnx.export`` spills every initializer into its own file named after
    the parameter, which turns one graph into hundreds of loose files that no mirror can
    sanely carry. This reloads the model and writes a single ``<name>.data`` beside it,
    the layout the official export uses.
    """
    import onnx

    model = onnx.load(str(path))
    deduplicate_initializers(model)
    for stale in path.parent.iterdir():
        if stale.is_file() and stale.suffix not in (".onnx", ".json", ".data"):
            stale.unlink()
    onnx.save(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
              location=path.name + ".data", size_threshold=1024)


def deduplicate_initializers(model) -> int:
    """Collapse initializers that hold identical bytes onto one copy.

    These checkpoints tie the output projection to the input embedding table, so the
    155776 x 896 matrix is reachable twice — once as the embedding and once as the
    projection's operand — and the tracer materialises it twice. On Audio8 that is an extra
    558 MB of single-precision weights for a tensor the graph already has. Merging them is
    safe because ONNX initializers are immutable.
    """
    seen: dict[tuple, str] = {}
    rename: dict[str, str] = {}
    keep = []
    for initializer in model.graph.initializer:
        key = (initializer.data_type, tuple(initializer.dims), initializer.raw_data)
        if key in seen:
            rename[initializer.name] = seen[key]
            continue
        seen[key] = initializer.name
        keep.append(initializer)
    if not rename:
        return 0
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name in rename:
                node.input[index] = rename[name]
    return len(rename)


def to_fp16(source: Path, target: Path) -> None:
    """Cast a graph's initializers to half precision.

    ONNX Runtime upcasts fp16 back to fp32 for CPU compute, so this buys disk and memory,
    not speed. It is still worth doing: the AR graphs are read once per token.
    """
    import onnx
    from onnxconverter_common import float16

    # The converter needs to know every intermediate's type so it can insert the casts that
    # keep an op's operands in one precision; without them the slow AR ends up with a
    # float32 tensor feeding a float16 Add and will not load. Its own shape inference
    # round-trips the model through a single protobuf, which the slow AR's weights alone
    # exceed, so inference runs file-to-file first and the converter is handed the result.
    inferred = target.with_suffix(".inferred.onnx")
    onnx.shape_inference.infer_shapes_path(str(source), str(inferred))
    model = onnx.load(str(inferred))
    converted = float16.convert_float_to_float16(model, keep_io_types=False,
                                                 disable_shape_infer=True)
    # The inferred types describe the *single-precision* graph. Leaving them in place makes
    # ONNX Runtime reject nodes whose real output is now half precision, so they are dropped
    # and the runtime re-infers them at load.
    del converted.graph.value_info[:]
    onnx.save(
        converted,
        str(target),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=target.name + ".data",
    )
    inferred.unlink(missing_ok=True)


def write_manifest(model, out_dir: Path, precisions: list[str]) -> None:
    """The runtime manifest, in the shape the official ONNX export publishes."""
    config = model.config
    manifest = {
        "model_family": "audio8_tts",
        "activation_dtype": "float32",
        "slow_logits_layout": "semantic_then_eos",
        "slow_logits_size": int(config.codebook_size) + 1,
        "kv_attention_layout": "valid_prefix",
        "max_seq_len": int(config.max_seq_len),
        "num_layers": int(config.n_layer),
        "num_fast_layers": int(config.n_fast_layer),
        "num_codebooks": int(config.num_codebooks),
        "n_local_heads": int(config.n_local_heads),
        "fast_n_local_heads": int(config.fast_n_local_heads),
        "head_dim": int(config.head_dim),
        "fast_head_dim": int(config.fast_head_dim),
        "fast_dim": int(config.fast_dim),
        "vocab_size": int(config.vocab_size),
        "codebook_size": int(config.codebook_size),
        "semantic_begin_id": int(config.semantic_begin_id),
        "semantic_end_id": int(config.semantic_end_id),
        "eos_token_id": int(config.eos_token_id),
        "pad_token_id": int(config.pad_token_id),
        "sample_rate": int(config.codec_sample_rate),
        "codec_sample_rate": int(config.codec_sample_rate),
        "codec_frame_size": int(config.codec_frame_size),
        "codec_hop_length": int(config.codec_frame_size),
        "ras_window_size": int(config.ras_window_size),
        "ras_top_p": float(config.ras_top_p),
        "ras_temperature": float(config.ras_temperature),
        "default_precision": precisions[0],
        "available_precisions": precisions,
        "slow_models": {p: f"slow_ar_{p}.onnx" for p in precisions},
        "fast_models": {p: f"fast_ar_{p}.onnx" for p in precisions},
        "default_codec_precision": "fp32",
        "available_codec_precisions": ["fp32"],
        "codec_models": {"fp32": "codec_decoder_fp32.onnx"},
    }
    (out_dir / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Hub id or local directory of the checkpoint")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--fp16", action="store_true", help="also write half-precision AR graphs")
    parser.add_argument("--drop-fp32", action="store_true",
                        help="delete the single-precision graphs once the fp16 ones exist")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(args.repo, dtype=torch.float32, trust_remote_code=True).eval()

    print("exporting slow_ar_fp32.onnx")
    export_slow_ar(model, args.out_dir / "slow_ar_fp32.onnx")
    consolidate(args.out_dir / "slow_ar_fp32.onnx")
    print("exporting fast_ar_fp32.onnx")
    export_fast_ar(model, args.out_dir / "fast_ar_fp32.onnx")
    print("exporting codec_decoder_fp32.onnx")
    export_codec_decoder(model, args.out_dir / "codec_decoder_fp32.onnx")

    precisions = ["fp32"]
    if args.fp16:
        precisions.append("fp16")
        for name in ("slow_ar", "fast_ar"):
            print(f"casting {name} to fp16")
            to_fp16(args.out_dir / f"{name}_fp32.onnx", args.out_dir / f"{name}_fp16.onnx")

    if args.drop_fp32 and args.fp16:
        precisions.remove("fp32")
        for name in ("slow_ar", "fast_ar"):
            for stale in args.out_dir.glob(f"{name}_fp32.onnx*"):
                stale.unlink()

    write_manifest(model, args.out_dir, precisions)
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
