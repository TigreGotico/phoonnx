"""Export Rápido to the four-graph voice directory the ``rapido`` inference engine reads.

* ``text_encoder.onnx`` : ``ids`` [1,L] -> ``features`` [1,L,D], ``durations`` [1,L]
* ``aligner.onnx``      : ``features``, ``durations``, ``word``, ``word_start`` [1,L],
  ``frame_word``, ``frame_pos`` [1,T], ``speaker`` [1] -> ``condition`` [1,T,D]
* ``model.onnx``        : ``x`` [1,T,64], ``condition``, ``t`` [1], ``guidance`` [1] ->
  ``velocity`` [1,T,64], classifier-free guidance folded in (a conditional and a
  zero-condition batch)
* ``decoder.onnx``      : ``latents`` [1,64,T] -> ``audio`` [1,1,T*1920], weight norm folded

plus a native phoonnx ``config.json``. The released checkpoint exports with its own
50-symbol letter vocabulary::

    python -m phoonnx_train.rapido.export_onnx --output-dir ./rapido-pt-BR
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import click
import torch

from phoonnx.config import VoiceConfig
from phoonnx_train.rapido.audio_vae import AudioVAE, load_audio_vae
from phoonnx_train.rapido.model import RapidoModel, load_checkpoint
from phoonnx_train.torch_compat import onnx_export_kwargs

_LOGGER = logging.getLogger("phoonnx_train.rapido.export_onnx")

OPSET = 17
DEFAULT_STEPS = 8
DEFAULT_GUIDANCE = 4.0
DEFAULT_SEED = 0
GRAPHS = {"text_encoder_path": "text_encoder.onnx", "aligner_path": "aligner.onnx",
          "decoder_path": "decoder.onnx"}


class _TextEncoderGraph(torch.nn.Module):
    def __init__(self, model: RapidoModel):
        super().__init__()
        self.net = model.net

    def forward(self, ids: torch.Tensor):
        return self.net.text_stage(ids, torch.ones_like(ids, dtype=torch.bool))


class _AlignerGraph(torch.nn.Module):
    def __init__(self, model: RapidoModel):
        super().__init__()
        self.model = model

    def forward(self, features, durations, word, word_start, frame_word, frame_pos, speaker):
        mask = torch.ones_like(word, dtype=torch.bool)
        return self.model.condition(features, durations, mask, word, word_start, frame_word, frame_pos, speaker)


class _VelocityGraph(torch.nn.Module):
    def __init__(self, model: RapidoModel):
        super().__init__()
        self.model = model

    def forward(self, x, condition, t, guidance):
        xb = torch.cat([x, x])
        cb = torch.cat([condition, torch.zeros_like(condition)])
        frame_mask = torch.ones_like(xb[..., 0], dtype=torch.bool)
        v = self.model.velocity(xb, cb, t.expand(2), frame_mask)
        conditional, unconditional = v[:1], v[1:]
        return unconditional + guidance * (conditional - unconditional)


class _DecoderGraph(torch.nn.Module):
    def __init__(self, vae: AudioVAE):
        super().__init__()
        self.decoder = vae.decoder
        self.sample_rate = vae.config.out_sample_rate

    def forward(self, latents):
        return self.decoder(latents, self.sample_rate)


def _export(module: torch.nn.Module, args: tuple, path: Path, inputs: Dict[str, Dict[int, str]],
            outputs: Dict[str, Dict[int, str]]) -> None:
    torch.onnx.export(module, args, str(path), opset_version=OPSET,
                      input_names=list(inputs), output_names=list(outputs),
                      dynamic_axes={k: v for k, v in {**inputs, **outputs}.items() if v},
                      **onnx_export_kwargs())


def _expm1(g, x):
    return g.op("Sub", g.op("Exp", x), g.op("Constant", value_t=torch.tensor(1.0)))


def export_graphs(model: RapidoModel, vae: AudioVAE, output_dir: Path) -> Dict[str, Path]:
    """Write the four ONNX graphs; the VAE's weight norm is folded in place."""
    # ONNX has no expm1; the duration predictor's exp(x) - 1 runs on x <= 6
    torch.onnx.register_custom_op_symbolic("aten::expm1", _expm1, OPSET)
    model = model.float().eval()
    vae.remove_weight_norm()
    vae = vae.float().eval()
    d, latent = model.config.d, model.config.latent_dim
    length, frames = 7, 11
    tokens, frame_axis = {1: "tokens"}, {1: "frames"}
    with torch.no_grad():
        _export(_TextEncoderGraph(model), (torch.ones(1, length, dtype=torch.long),),
                output_dir / GRAPHS["text_encoder_path"],
                {"ids": tokens}, {"features": tokens, "durations": tokens})
        _export(_AlignerGraph(model),
                (torch.randn(1, length, d), torch.rand(1, length) + 1,
                 torch.tensor([[0, 0, 0, 1, 1, 2, 2]]), torch.tensor([[0, 0, 0, 3, 3, 5, 5]]),
                 torch.tensor([[0] * 4 + [1] * 4 + [2] * 3]), torch.rand(1, frames),
                 torch.zeros(1, dtype=torch.long)),
                output_dir / GRAPHS["aligner_path"],
                {"features": tokens, "durations": tokens, "word": tokens, "word_start": tokens,
                 "frame_word": frame_axis, "frame_pos": frame_axis, "speaker": {}},
                {"condition": frame_axis})
        _export(_VelocityGraph(model),
                (torch.randn(1, frames, latent), torch.randn(1, frames, d), torch.full((1,), 0.5),
                 torch.full((1,), DEFAULT_GUIDANCE)),
                output_dir / "model.onnx",
                {"x": frame_axis, "condition": frame_axis, "t": {}, "guidance": {}},
                {"velocity": frame_axis})
        _export(_DecoderGraph(vae), (torch.randn(1, latent, frames),),
                output_dir / GRAPHS["decoder_path"],
                {"latents": {2: "frames"}}, {"audio": {2: "samples"}})
    return {"model": output_dir / "model.onnx",
            **{key: output_dir / name for key, name in GRAPHS.items()}}


def export_rapido(model: RapidoModel, voice_config: Dict[str, Any], output_dir: Path,
                  vae: Optional[AudioVAE] = None) -> Path:
    """Export ``model`` and the VoxCPM2 decoder into ``output_dir`` and write its config.json.

    ``voice_config`` gives the token side as a phoonnx config does: ``phoneme_id_map``,
    ``speaker_id_map``, ``lang_code``, ``phoneme_type``, ``alphabet``,
    ``keep_punctuation``, the special tokens and the tokenizer flags. The audio,
    engine and graph fields are set here. Returns the path of config.json.

    The graph paths in ``engine_params`` are absolute, as the voice manager writes them
    for a downloaded voice.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    vae = vae or load_audio_vae()
    paths = export_graphs(model, vae, output_dir)

    native = {
        **voice_config,
        "phoonnx_version": "1.0",
        "engine": "rapido",
        "audio": {"sample_rate": vae.config.out_sample_rate},
        "hop_length": vae.config.decode_hop_length,
        "num_symbols": model.net.text.emb.num_embeddings,
        "num_speakers": model.spk.num_embeddings,
        "engine_params": {
            **{key: str(paths[key]) for key in GRAPHS},
            "latent_mean": model.lat_mean.tolist(),
            "latent_std": model.lat_std.tolist(),
            "steps": DEFAULT_STEPS,
            "guidance": DEFAULT_GUIDANCE,
            "seed": DEFAULT_SEED,
        },
    }
    config = VoiceConfig.from_dict(native).to_native_dict()
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    _LOGGER.info("Rápido voice written to %s", output_dir)
    return config_path


def released_voice_config(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """The token side of the released checkpoint: one id per letter or mark of its
    vocabulary, no blanks, no BOS/EOS, words separated by a space."""
    return {
        "lang_code": "pt-BR",
        "phoneme_type": "graphemes",
        "alphabet": "unicode",
        "keep_punctuation": True,
        "inference": {"add_diacritics": False},
        "phoneme_id_map": {symbol: i for i, symbol in enumerate(checkpoint["vocab"])},
        "speaker_id_map": {speaker: i for i, speaker in enumerate(checkpoint["spk_order"])},
        "pad": "<pad>",
        "add_blank_char": False,
        "add_blank_word": False,
        "use_eos_bos": False,
        "blank_at_start": False,
        "blank_at_end": False,
        "word_sep_token": " ",
    }


@click.command()
@click.option("--checkpoint", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Released-format rapido.pt (default: edwixx/rapido-tts from the Hub)")
@click.option("--output-dir", required=True, type=click.Path(file_okay=False),
              help="Voice directory to write")
def main(checkpoint: Optional[str], output_dir: str) -> None:
    """Export a Rápido checkpoint to an ONNX voice directory."""
    logging.basicConfig(level=logging.INFO)
    ck = load_checkpoint(checkpoint)
    config_path = export_rapido(RapidoModel.from_checkpoint(ck), released_voice_config(ck), Path(output_dir))
    click.echo(str(config_path))


if __name__ == "__main__":
    main()
