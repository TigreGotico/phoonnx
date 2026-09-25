"""agbalu/Matoub-82M (Kabyle)  ->  phoonnx StyleTTS2 adapter config + ONNX.

Matoub-82M is a StyleTTS2 fine-tune of Kokoro-82M for Kabyle, one voice
``kab_male``. It already fits ``StyleTTS2Adapter``'s single-graph contract, so
no new engine is needed:

    input_ids(int64) + style(1,256) + speed(1)  ->  waveform @ 24kHz

Two things about this model are not what the adapter would guess, and both are
why this script exists rather than a line in a README.

**The style is in the weights, not in a voice file.** ``modeling_matoub.py``
registers a buffer ``voice`` of shape ``(1, style_dim * 2)``, which is
``(1, 256)``, and ``forward`` falls back to it when no ``voice`` tensor is
passed. That is why the model repository ships no ``voices/`` directory. The
export below writes that buffer out as a one-row style file, so the graph takes
``style`` as an input like any other Kokoro voice and the vector stays visible
instead of being frozen into the graph.

**It pads both ends.** ``tokenization_matoub.py`` wraps every id sequence in
the pad symbol on both sides:

    boundary = [vocab["$"]]
    merged = boundary + token_ids_0 + boundary

because ``meldataset`` wrapped every training target that way. The adapter
otherwise infers the padding from the style pack, and a single-row pack reads
as the plain StyleTTS2 lineage, which pads the start only. So the config sets
``engine_params["pad_both_ends"] = True`` explicitly. Without it the model is
fed a sequence off the distribution its durations were fitted on.

**The four substitutions this graph needs.** Each was checked against this
model rather than copied from the family recipe:

- ``dynamo=False``, the TorchScript exporter;
- ``TorchSTFT`` in real arithmetic, because complex tensors do not export;
- ``InstanceNorm1d/2d`` normalised by hand, because ``F.instance_norm`` wants
  a static channel count;
- PL-BERT on eager attention with no ``attention_mask``, because SDPA bakes
  the traced sequence length.

``export_bsc.py`` lists a fifth, dropping ``pack_padded_sequence``. This model
does not need it: its LSTMs never pack. The recipe asserts that rather than
assuming it, because an unnecessary monkeypatch is a silent behaviour change.

**The one that does not announce itself.** Without the Albert fix the export
SUCCEEDS and yields a graph that only works at the length it was traced with,
five tokens here. A smoke test that synthesises the same short phrase it
exported with would pass. Waveform parity on real sentences of other lengths
is what catches it, so a successful export is not the finish line.

**Requirements.** torch, transformers, numpy, and the 327 MB weights. The
export uses the TorchScript exporter (``dynamo=False``), so ``onnxscript`` is
not needed; the dynamo path wants it and then trips a data-dependent guard on
this graph anyway.

**The STFT substitution, checked.** The replacements below were verified
against a reference one-sided DFT at this decoder's own dimensions,
``n_fft=20``, ``hop=5``, ``win_length=20``, periodic Hann, on 2400 random
samples:

    STFT magnitude          max abs error  4.755e-07
    STFT phase, wrapped     max abs error  1.081e-06
    analysis then synthesis max abs error  9.083e-08

The raw phase difference reaches 6.283 on 156 of 5291 bins. That is 2*pi: the
two implementations put the same angle at -pi and at +pi. Wrapping the
difference into (-pi, pi] gives the 1.081e-06 above, and the round trip
confirms it, because a real phase error of 2*pi radians would not reconstruct
the waveform to 9e-08.

Run this on a host with torch. ``build_config`` needs nothing but the model's
own ``vocab.json`` and is covered by
``tests/test_export_matoub_config.py``; the ONNX export needs torch and the
327 MB weights file.

Usage:
    python export_matoub.py <model_dir_or_repo_id> <out_dir>
"""
import json
import sys
from pathlib import Path

MODEL_ID = "agbalu/Matoub-82M"
# The revision this recipe was written against. The sha256 of
# tokenization_matoub.py at this revision matches the value the model's own
# export.stats.json records for it.
MODEL_REVISION = "3d00056f3663d4d1d364e9b12c21230685a0ba9a"

SAMPLE_RATE = 24000        # config.json: sampling_rate
STYLE_WIDTH = 256          # config.json: style_dim 128, and forward wants 2x
PAD_SYMBOL = "$"


def build_config(vocab: dict) -> dict:
    """The phoonnx config for this voice, from the model's own ``vocab.json``.

    *vocab* is the symbol -> id map the model ships. It is used as-is: the ids
    are the embedding rows, so rebuilding them from a symbol list would be a
    second source of truth and a way to be silently wrong.
    """
    if PAD_SYMBOL not in vocab:
        raise ValueError(
            f"vocab.json has no {PAD_SYMBOL!r} row; the tokenizer wraps every "
            f"sequence in it, so the pad id cannot be resolved")
    # The embedding has one row per id, and the id space is NOT the symbol
    # count: this model maps 118 symbols over ids 0..177, leaving 60 ids
    # unmapped. They are Kokoro rows the Kabyle front end never emits.
    # len(vocab) would therefore understate the table by 60 rows and describe a
    # model that is not this one; the highest id plus one is the real size, and
    # it matches config.json's vocab_size of 178.
    num_symbols = max(vocab.values()) + 1
    return {
        "phoonnx_version": "1.0",
        "engine": "styletts2",
        # The Kabyle rules this model was fitted on, vendored in scriptconv as
        # the matoub_kab backend. NOT the africa-g2p kab route: that one does
        # not spirantise the initial t, writes gemination as a doubled symbol
        # rather than length, and does not back /a/, so its output carries
        # symbols this model has no embedding row for.
        "phoneme_type": "matoub_kab",
        "alphabet": "ipa",
        "lang_code": "kab",
        "audio": {"sample_rate": SAMPLE_RATE},
        "num_symbols": num_symbols,
        "num_speakers": 1,
        "num_langs": 1,
        "speaker_id_map": {},
        "lang_id_map": {},
        "phonemizer_model": None,
        "add_diacritics": False,
        "inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8},
        "phoneme_id_map": dict(vocab),
        "engine_params": {
            # Stated, not inferred. See the module docstring.
            "pad_both_ends": True,
            "style_path": "style.bin",
        },
        "source_model": MODEL_ID,
        "source_revision": MODEL_REVISION,
    }


def _stft_basis(n_fft: int, window):
    """cos/sin conv kernels for a one-sided DFT of length *n_fft*."""
    import math

    import torch

    k = torch.arange(n_fft // 2 + 1).unsqueeze(1).float()
    n = torch.arange(n_fft).unsqueeze(0).float()
    ang = 2 * math.pi * k * n / n_fft
    return (torch.cos(ang) * window).unsqueeze(1), (-torch.sin(ang) * window).unsqueeze(1)


def _stft_transform(self, waveform):
    """``torch.stft`` (center, reflect pad, onesided) without complex tensors."""
    import torch
    import torch.nn.functional as F

    n_fft, hop = self.filter_length, self.hop_length
    w = self._window(waveform)
    cos, sin = _stft_basis(n_fft, w)
    xp = F.pad(waveform.unsqueeze(1), (n_fft // 2, n_fft // 2), mode="reflect")
    re = F.conv1d(xp, cos.to(waveform.device), stride=hop)
    im = F.conv1d(xp, sin.to(waveform.device), stride=hop)
    return torch.sqrt(re ** 2 + im ** 2 + 1e-12), torch.atan2(im, re)


def _stft_inverse(self, magnitude, phase):
    """``torch.istft`` without complex tensors: IDFT matmul plus overlap-add."""
    import math

    import torch
    import torch.nn.functional as F

    n_fft, hop = self.filter_length, self.hop_length
    w = self._window(magnitude)
    re = magnitude * torch.cos(phase)
    im = magnitude * torch.sin(phase)
    k = torch.arange(n_fft // 2 + 1, device=magnitude.device).unsqueeze(0).float()
    n = torch.arange(n_fft, device=magnitude.device).unsqueeze(1).float()
    c = torch.full((1, n_fft // 2 + 1), 2.0, device=magnitude.device)
    c[0, 0] = c[0, -1] = 1.0            # DC and Nyquist are not mirrored
    ang = 2 * math.pi * k * n / n_fft
    frames = torch.matmul((c * torch.cos(ang)) / n_fft, re) + \
        torch.matmul((-c * torch.sin(ang)) / n_fft, im)
    frames = frames * w.view(1, -1, 1)
    eye = torch.eye(n_fft, device=magnitude.device).unsqueeze(1)
    y = F.conv_transpose1d(frames, eye, stride=hop)
    wsq = F.conv_transpose1d((w ** 2).view(1, -1, 1) * torch.ones_like(frames),
                             eye, stride=hop)
    y = y / (wsq + 1e-11)
    return y[:, :, n_fft // 2: n_fft // 2 + (frames.shape[-1] - 1) * hop]


def _instance_norm_forward(self, x):
    """``nn.InstanceNorm1d/2d`` without ``F.instance_norm``.

    The functional form needs a static channel count, and the exporter reports
    "ONNX export of instance_norm for unknown channel size". Normalising over
    the trailing dimensions by hand is the same arithmetic with no such
    requirement. ``export_nos_gl.py`` carries the identical replacement.
    """
    import torch

    dims = tuple(range(2, x.dim()))
    mean = x.mean(dims, keepdim=True)
    var = x.var(dims, keepdim=True, unbiased=False)
    xn = (x - mean) / torch.sqrt(var + self.eps)
    if getattr(self, "affine", False):
        shape = [1, -1] + [1] * (x.dim() - 2)
        xn = xn * self.weight.view(*shape) + self.bias.view(*shape)
    return xn


def _use_eager_albert_without_mask(model) -> None:
    """PL-BERT: eager attention, and no ``attention_mask``.

    This is the substitution that does not announce itself. SDPA builds its
    mask from the sequence length and bakes that length into the graph, so the
    export SUCCEEDS and returns a model that only works at the length it was
    traced with, which is the five dummy ids below. A smoke test that
    synthesises the same short phrase it exported with passes. Only parity on
    real sentences of other lengths catches it.

    ``_synthesise`` calls ``self.bert(input_ids, attention_mask=attention)``
    with an all-ones mask, so dropping the mask changes no arithmetic: every
    token is already attended.

    Both halves are verified below rather than assumed, because a silent
    no-op here is exactly the failure this function exists to prevent.
    """
    bert = model.bert
    setter = getattr(bert, "set_attn_implementation", None)
    if callable(setter):
        setter("eager")
    else:
        bert.config._attn_implementation = "eager"
    got = getattr(bert.config, "_attn_implementation", None)
    if got != "eager":
        raise RuntimeError(
            f"PL-BERT attention implementation is {got!r}, not 'eager'; SDPA "
            f"would bake the traced token length into the graph and the export "
            f"would still succeed")

    inner = bert.forward

    def _forward_without_mask(input_ids=None, attention_mask=None, **kwargs):
        kwargs.pop("attention_mask", None)
        return inner(input_ids, attention_mask=None, **kwargs)

    bert.forward = _forward_without_mask


def resolve_model_dir(model_dir: str) -> Path:
    """A local directory, or a repo id fetched into the shared cache.

    ``from_pretrained`` accepts either, so the usage line offers both. Only a
    directory has a ``vocab.json`` to read off disk, and resolving the id here
    rather than reading a path that cannot exist is what makes both work.
    The download goes through ``huggingface_hub``, so it lands in the shared
    cache and is not re-fetched per run.
    """
    path = Path(model_dir)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_dir, revision=MODEL_REVISION))


def load_vocab(model_dir: str) -> dict:
    """The model's own symbol-to-id map, from a directory or a repo id."""
    vocab_path = resolve_model_dir(model_dir) / "vocab.json"
    if not vocab_path.is_file():
        raise FileNotFoundError(
            f"{vocab_path} does not exist; the phoneme id map comes from the "
            f"model's own vocab.json and cannot be rebuilt")
    return json.loads(vocab_path.read_text(encoding="utf-8"))


def export(model_dir: str, out_dir: Path) -> None:
    """Write model.onnx, style.bin and config.json. Needs torch."""
    import numpy as np
    import torch
    from transformers import AutoModelForTextToWaveform

    # Resolve and read the vocab BEFORE anything is written. A repo id used to
    # pass from_pretrained and then fail here on Path("agbalu/Matoub-82M")/
    # "vocab.json", by which point model.onnx and style.bin were already on
    # disk: the caller was left with a partial output directory and a
    # traceback. Everything that can fail on the argument now fails first, so
    # a bad argument costs nothing.
    config = build_config(load_vocab(model_dir))

    model = AutoModelForTextToWaveform.from_pretrained(
        model_dir, trust_remote_code=True, revision=MODEL_REVISION).eval()

    style = model.voice.detach().cpu().numpy().astype(np.float32)
    if style.shape != (1, STYLE_WIDTH):
        raise ValueError(
            f"expected the baked voice buffer to be (1, {STYLE_WIDTH}), got "
            f"{style.shape}; the adapter's style contract is 256-wide")
    style.tofile(out_dir / "style.bin")

    # The decoder inverts a short-time transform, and torch.stft/torch.istft
    # carry complex tensors that the TorchScript exporter cannot represent
    # ("Unknown number type: complex"). Replace both with the real-arithmetic
    # equivalents before tracing. export_nos_gl.py does the same for the
    # Galician StyleTTS2 decoder; the window here is built per call by
    # TorchSTFT._window rather than held as self.window, so these read it from
    # there.
    import importlib

    istftnet = importlib.import_module(
        model.__class__.__module__.rsplit(".", 1)[0] + ".istftnet")
    istftnet.TorchSTFT.transform = _stft_transform
    istftnet.TorchSTFT.inverse = _stft_inverse

    # AdaIN1d holds an nn.InstanceNorm1d (istftnet.py:36) and every
    # AdaINResBlock1 holds two, so the whole decoder is built from them.
    torch.nn.InstanceNorm1d.forward = _instance_norm_forward
    torch.nn.InstanceNorm2d.forward = _instance_norm_forward

    _use_eager_albert_without_mask(model)

    # pack_padded_sequence is the third substitution export_bsc.py lists and
    # this model does not need it: its LSTMs never pack. Checked rather than
    # copied across, because an unnecessary monkeypatch is a silent behaviour
    # change.
    import inspect

    if "pack_padded_sequence" in inspect.getsource(type(model)):
        raise RuntimeError(
            "this model packs sequences after all; export_bsc.py's third "
            "substitution is needed and is not applied here")

    class _Graph(torch.nn.Module):
        """input_ids + style + speed -> waveform, the adapter's contract.

        The wrapper exists so ``style`` is a graph input rather than the baked
        buffer: the adapter passes it per call, and a later voice for the same
        model needs no re-export.

        It calls ``_synthesise`` rather than ``forward``. ``forward`` validates
        ``speed`` with ``if speed <= 0`` and takes it as a python float, and a
        traced tensor reaching either raises
        ``GuardOnDataDependentSymNode``. ``_synthesise`` only divides the
        predicted durations by it, which traces as an ordinary op, so ``speed``
        stays a real graph input instead of being baked to 1.0.
        """

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, style, speed):
            waveform, _frames = self.inner._synthesise(input_ids, style, speed)
            return waveform

    ids = torch.tensor([[0, 5, 9, 12, 0]], dtype=torch.long)
    torch.onnx.export(
        _Graph(model),
        (ids, torch.from_numpy(style), torch.tensor([1.0])),
        str(out_dir / "model.onnx"),
        input_names=["input_ids", "style", "speed"],
        output_names=["waveform"],
        dynamic_axes={"input_ids": {1: "tokens"}, "waveform": {0: "samples"}},
        opset_version=17,
        # The TorchScript exporter, not torch.export. The dynamo path wants
        # onnxscript installed and then trips its own data-dependent guards on
        # this graph. export_bsc.py passes the same flag.
        dynamo=False,
    )

    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


# The voice-index row to add once the mirror is published. It is written here
# rather than committed to phoonnx/voice_index/styletts2.json, because an entry
# whose URLs do not resolve yet is a voice the index offers and cannot load.
VOICE_INDEX_ENTRY = {
    "TigreGotico/phoonnx_kab_male_matoub": {
        "voice_id": "TigreGotico/phoonnx_kab_male_matoub",
        "model_url": "<mirror>/model.onnx",
        "config_url": "<mirror>/config.json",
        "vocab_url": None,
        "tokenizer_config_url": None,
        "tokens_url": None,
        "phoneme_map_url": None,
        "phoneme_type": "matoub_kab",
        "alphabet": "ipa",
        "engine": "styletts2",
        "lang": "kab",
    }
}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    export(sys.argv[1], out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
