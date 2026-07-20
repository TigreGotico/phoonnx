"""
Matcha-TTS training engine.

Uses the vendored ``phoonnx_train.matcha`` package (upstream Matcha-TTS
model code, MIT) so training runs on the same pytorch_lightning stack,
dataset format and CLI as every other phoonnx engine.
"""
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # heavy imports — only needed for type annotations
    import pytorch_lightning as pl

from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig

_LOG = logging.getLogger(__name__)

# ONNX opset used for export (matches upstream matcha/onnx/export.py)
OPSET_VERSION = 15

# Quality tier → Matcha hyper-param overrides
_QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "x-low": {
        "encoder_channels": 128,
        "encoder_filter_channels": 512,
        "encoder_filter_channels_dp": 192,
        "encoder_n_heads": 2,
        "encoder_n_layers": 4,
        "decoder_channels": [192, 192],
        "decoder_num_heads": 2,
        "decoder_num_mid_blocks": 2,
    },
    "medium": {
        "encoder_channels": 192,
        "encoder_filter_channels": 768,
        "encoder_filter_channels_dp": 256,
        "encoder_n_heads": 2,
        "encoder_n_layers": 6,
        "decoder_channels": [256, 256],
        "decoder_num_heads": 2,
        "decoder_num_mid_blocks": 2,
    },
    "high": {
        "encoder_channels": 256,
        "encoder_filter_channels": 1024,
        "encoder_filter_channels_dp": 384,
        "encoder_n_heads": 4,
        "encoder_n_layers": 8,
        "decoder_channels": [384, 384],
        "decoder_num_heads": 4,
        "decoder_num_mid_blocks": 2,
    },
}


def _dict_to_namespace(d: Dict[str, Any]) -> Any:
    """Recursively convert a dict to SimpleNamespace for attribute access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    return d


@dataclass
class MatchaEngineConfig:
    """Extended config for Matcha-TTS training."""

    n_vocab: int = 178
    n_spks: int = 1
    spk_emb_dim: int = 64
    n_feats: int = 80
    out_size: Optional[int] = None
    prior_loss: bool = True
    use_precomputed_durations: bool = False

    # Encoder
    encoder_type: str = "RoPE Encoder"
    encoder_channels: int = 192
    encoder_filter_channels: int = 768
    encoder_filter_channels_dp: int = 256
    encoder_n_heads: int = 2
    encoder_n_layers: int = 6
    encoder_kernel_size: int = 3
    encoder_p_dropout: float = 0.1
    encoder_prenet: bool = True

    # Decoder
    decoder_channels: List[int] = None  # type: ignore[assignment]
    decoder_dropout: float = 0.05
    decoder_attention_head_dim: int = 64
    decoder_n_blocks: int = 1
    decoder_num_mid_blocks: int = 2
    decoder_num_heads: int = 2
    decoder_act_fn: str = "snakebeta"
    decoder_down_block_type: str = "transformer"
    decoder_mid_block_type: str = "transformer"
    decoder_up_block_type: str = "transformer"

    # CFM
    cfm_name: str = "CFM"
    cfm_solver: str = "euler"
    cfm_sigma_min: float = 1e-4

    # Data (dataset statistics are computed and cached at first run
    # unless given explicitly)
    mel_mean: Optional[float] = None
    mel_std: Optional[float] = None

    # Training loop
    batch_size: int = 16
    num_workers: int = 1
    validation_split: float = 0.05
    learning_rate: float = 1e-4
    max_phoneme_ids: Optional[int] = None

    def __post_init__(self):
        if self.decoder_channels is None:
            self.decoder_channels = [256, 256]

    @classmethod
    def from_training_config(cls, cfg: TrainingEngineConfig) -> "MatchaEngineConfig":
        """Build from shared TrainingEngineConfig + extra overrides."""
        extra = dict(cfg.extra)
        preset_name = extra.pop("quality", "medium")
        params = dict(_QUALITY_PRESETS.get(preset_name, _QUALITY_PRESETS["medium"]))
        params.update(extra)

        known = {f.name for f in fields(cls)}
        ignored = set(params) - known
        if ignored:
            _LOG.debug("ignoring non-matcha config keys: %s", sorted(ignored))

        return cls(
            n_vocab=cfg.num_symbols,
            n_spks=cfg.num_speakers,
            **{k: v for k, v in params.items() if k in known},
        )


def _build_model_kwargs(mcfg: MatchaEngineConfig) -> Dict[str, Any]:
    """Assemble the nested config objects MatchaTTS expects."""
    encoder_cfg = _dict_to_namespace(
        {
            "encoder_type": mcfg.encoder_type,
            "encoder_params": {
                "n_feats": mcfg.n_feats,
                "n_channels": mcfg.encoder_channels,
                "filter_channels": mcfg.encoder_filter_channels,
                "filter_channels_dp": mcfg.encoder_filter_channels_dp,
                "n_heads": mcfg.encoder_n_heads,
                "n_layers": mcfg.encoder_n_layers,
                "kernel_size": mcfg.encoder_kernel_size,
                "p_dropout": mcfg.encoder_p_dropout,
                "spk_emb_dim": mcfg.spk_emb_dim,
                "n_spks": mcfg.n_spks,
                "prenet": mcfg.encoder_prenet,
            },
            "duration_predictor_params": {
                "filter_channels_dp": mcfg.encoder_filter_channels_dp,
                "kernel_size": mcfg.encoder_kernel_size,
                "p_dropout": mcfg.encoder_p_dropout,
            },
        }
    )

    decoder_cfg = {
        "channels": tuple(mcfg.decoder_channels),
        "dropout": mcfg.decoder_dropout,
        "attention_head_dim": mcfg.decoder_attention_head_dim,
        "n_blocks": mcfg.decoder_n_blocks,
        "num_mid_blocks": mcfg.decoder_num_mid_blocks,
        "num_heads": mcfg.decoder_num_heads,
        "act_fn": mcfg.decoder_act_fn,
        "down_block_type": mcfg.decoder_down_block_type,
        "mid_block_type": mcfg.decoder_mid_block_type,
        "up_block_type": mcfg.decoder_up_block_type,
    }

    cfm_cfg = _dict_to_namespace(
        {
            "name": mcfg.cfm_name,
            "solver": mcfg.cfm_solver,
            "sigma_min": mcfg.cfm_sigma_min,
        }
    )

    return {
        "n_vocab": mcfg.n_vocab,
        "n_spks": mcfg.n_spks,
        "spk_emb_dim": mcfg.spk_emb_dim,
        "n_feats": mcfg.n_feats,
        "encoder": encoder_cfg,
        "decoder": decoder_cfg,
        "cfm": cfm_cfg,
        "out_size": mcfg.out_size,
        "prior_loss": mcfg.prior_loss,
        "use_precomputed_durations": mcfg.use_precomputed_durations,
    }


class MatchaTrainingEngine(BaseTrainingEngine):

    def trainer_kwargs(self) -> Dict[str, Any]:
        # Upstream Matcha-TTS trains with gradient clipping at 5.0
        # (configs/trainer/default.yaml).  Safe here: MatchaTTS uses
        # automatic optimization.
        return {"gradient_clip_val": 5.0}

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    def create_model(
        self,
        config: TrainingEngineConfig,
        dataset_paths: List[Path],
        **kwargs: Any,
    ) -> "pl.LightningModule":
        """Build a MatchaTTS LightningModule from *config*."""
        from phoonnx_train.matcha import MatchaTTS

        mcfg = MatchaEngineConfig.from_training_config(config)

        if mcfg.mel_mean is None or mcfg.mel_std is None:
            if dataset_paths:
                from phoonnx_train.matcha.dataset import (
                    MatchaDataset,
                    load_or_compute_statistics,
                )

                stats_dataset = MatchaDataset(
                    dataset_paths,
                    n_feats=mcfg.n_feats,
                    sample_rate=config.sample_rate,
                )
                mcfg.mel_mean, mcfg.mel_std = load_or_compute_statistics(
                    dataset_paths, stats_dataset
                )
            else:
                mcfg.mel_mean, mcfg.mel_std = 0.0, 1.0
                _LOG.warning(
                    "no dataset paths and no mel statistics given — "
                    "using mel_mean=0.0 mel_std=1.0"
                )

        return MatchaTTS(
            **_build_model_kwargs(mcfg),
            data_statistics={"mel_mean": mcfg.mel_mean, "mel_std": mcfg.mel_std},
            dataset=[str(p) for p in dataset_paths],
            sample_rate=config.sample_rate,
            batch_size=mcfg.batch_size,
            num_workers=mcfg.num_workers,
            validation_split=mcfg.validation_split,
            learning_rate=mcfg.learning_rate,
            max_phoneme_ids=mcfg.max_phoneme_ids,
        )

    def export_onnx(
        self,
        checkpoint_path: Path,
        config_path: Path,
        output_dir: Path,
        **kwargs: Any,
    ) -> Path:
        """Export a Matcha checkpoint to ONNX (mel model only).

        Follows upstream ``matcha/onnx/export.py``: inputs
        ``x``/``x_lengths``/``scales`` (temperature, length_scale) and an
        optional ``spks``; outputs ``mel``/``mel_lengths``.
        """
        import torch

        from phoonnx_train.matcha import MatchaTTS

        n_timesteps = kwargs.get("n_timesteps", 5)

        from phoonnx_train.torch_compat import trusting_torch_load

        _LOG.info("Loading Matcha checkpoint from %s", checkpoint_path)
        # hyper_parameters include SimpleNamespace config objects, so the
        # checkpoint cannot be unpickled under weights_only=True. Loading a
        # local training checkpoint is trusted; force weights_only=False in a
        # version-portable way that does not rely on torch>=2.4 allowlisting
        # APIs (this project supports torch>=2.1).
        with trusting_torch_load():
            matcha = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location="cpu")
        matcha.eval()

        is_multi_speaker = matcha.n_spks > 1

        # Dummy inputs
        dummy_input_length = 50
        x = torch.randint(low=0, high=20, size=(1, dummy_input_length), dtype=torch.long)
        x_lengths = torch.LongTensor([dummy_input_length])
        scales = torch.Tensor([0.667, 1.0])

        model_inputs = [x, x_lengths, scales]
        input_names = ["x", "x_lengths", "scales"]

        if is_multi_speaker:
            spks = torch.LongTensor([1])
            model_inputs.append(spks)
            input_names.append("spks")

        # Monkey-patch forward for ONNX export
        def onnx_forward_func(x, x_lengths, scales, spks=None):
            temperature = scales[0]
            length_scale = scales[1]
            output = matcha.synthesise(
                x, x_lengths, n_timesteps, temperature, spks, length_scale
            )
            return output["mel"], output["mel_lengths"]

        matcha.forward = onnx_forward_func

        dynamic_axes = {
            "x": {0: "batch_size", 1: "time"},
            "x_lengths": {0: "batch_size"},
            "mel": {0: "batch_size", 2: "time"},
            "mel_lengths": {0: "batch_size"},
        }
        if is_multi_speaker:
            dynamic_axes["spks"] = {0: "batch_size"}

        output_path = output_dir / "model.onnx"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        from phoonnx_train.torch_compat import onnx_export_kwargs

        torch.onnx.export(
            matcha,
            tuple(model_inputs),
            str(output_path),
            input_names=input_names,
            output_names=["mel", "mel_lengths"],
            dynamic_axes=dynamic_axes,
            opset_version=OPSET_VERSION,
            export_params=True,
            do_constant_folding=True,
            **onnx_export_kwargs(),
        )
        _LOG.info("ONNX model exported to %s", output_path)
        return output_path

    def quality_presets(self) -> Dict[str, Dict[str, Any]]:
        return _QUALITY_PRESETS

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def load_checkpoint(
        self,
        model: "pl.LightningModule",
        checkpoint_path: Path,
        **kwargs: Any,
    ) -> "pl.LightningModule":
        """Load Matcha weights, tolerating vocab/speaker-count mismatches."""
        import torch

        from phoonnx_train.torch_compat import trusting_torch_load

        # weights_only=True rejects the SimpleNamespace config objects stored in
        # the checkpoint hyper_parameters; a local checkpoint is trusted.
        with trusting_torch_load():
            ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model_state = model.state_dict()
        filtered = {
            k: v for k, v in state_dict.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        dropped = set(state_dict) - set(filtered)
        if dropped:
            _LOG.warning("dropped %d mismatched checkpoint keys", len(dropped))
        model.load_state_dict(filtered, strict=False)
        return model

    # ------------------------------------------------------------------
    # Standalone evaluation synthesis
    # ------------------------------------------------------------------

    def eval_synthesize(
        self,
        checkpoint_path: Path,
        config: Optional[Dict[str, Any]],
        *,
        vocoder_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """Return ``callable(ids, scales, sid) -> np.ndarray`` for a Matcha checkpoint.

        Matcha only emits a mel spectrogram, so a vocoder stage is required
        to reach a waveform. When ``vocoder_path`` is given, that ONNX
        vocoder converts mel -> audio (matching production quality). With no
        vocoder, a Griffin-Lim fallback is used with the exact mel
        parameters MatchaTTS trains on (``phoonnx_train.matcha.dataset``
        module constants + ``n_feats``/``sample_rate`` read back from the
        checkpoint hyper-parameters) — comparable across checkpoints of this
        run, but not an absolute quality signal.

        ``scales`` is ``[temperature, length_scale]`` (matching
        ``MatchaTTS.synthesise``); a third CFM step-count element may be
        supplied via ``config["n_timesteps"]`` (default 10).
        """
        import contextlib

        import numpy as np
        import torch

        from phoonnx_train.matcha import MatchaTTS

        safe_globals = getattr(torch.serialization, "safe_globals", None)
        ctx = safe_globals([SimpleNamespace]) if safe_globals else contextlib.nullcontext()
        with ctx:
            matcha = MatchaTTS.load_from_checkpoint(
                str(checkpoint_path), map_location=device
            )
        matcha = matcha.to(device)
        matcha.eval()

        n_timesteps = int((config or {}).get("n_timesteps", 10))
        is_multi_speaker = matcha.n_spks > 1

        vocoder = None
        if vocoder_path:
            from phoonnx.engines.vocoders import build_vocoder

            vocoder = build_vocoder(
                model_path=str(vocoder_path),
                config=(config or {}).get("vocoder", {}),
            )
        else:
            _warn_griffinlim_once()

        n_feats = int(matcha.n_feats)
        sample_rate = int(matcha.hparams.sample_rate)

        def _synth(ids: List[int], scales: List[float], sid: Optional[int]) -> "np.ndarray":
            x = torch.LongTensor(ids).unsqueeze(0).to(device)
            x_lengths = torch.LongTensor([len(ids)]).to(device)
            temperature = float(scales[0]) if len(scales) > 0 else 0.667
            length_scale = float(scales[1]) if len(scales) > 1 else 1.0

            spks = None
            if is_multi_speaker:
                spks = torch.LongTensor([sid if sid is not None else 0]).to(device)

            with torch.no_grad():
                out = matcha.synthesise(
                    x, x_lengths, n_timesteps, temperature, spks, length_scale
                )
            mel = out["mel"].detach().cpu().numpy().astype(np.float32)  # [1, n_feats, T]

            if vocoder is not None:
                wav = vocoder.mel_to_audio(mel)
            else:
                wav = _griffinlim_from_matcha_mel(
                    mel, n_feats=n_feats, sample_rate=sample_rate
                )

            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            if not np.isfinite(wav).all():
                raise RuntimeError(
                    "Matcha eval_synthesize produced non-finite samples "
                    "(NaN/Inf) — checkpoint or vocoder is likely broken"
                )
            return wav

        return _synth


_GRIFFINLIM_WARNED = False


def _warn_griffinlim_once() -> None:
    global _GRIFFINLIM_WARNED
    if not _GRIFFINLIM_WARNED:
        _LOG.warning(
            "Matcha eval_synthesize: no vocoder_path given — falling back to "
            "Griffin-Lim. Griffin-Lim scores are only meaningful relative to "
            "each other (e.g. comparing checkpoints of the same run); they "
            "are NOT comparable to absolute/neural-vocoder quality scores."
        )
        _GRIFFINLIM_WARNED = True


def _griffinlim_from_matcha_mel(mel, *, n_feats: int, sample_rate: int):
    """Invert a MatchaTTS natural-log mel (as returned by ``synthesise``) with
    Griffin-Lim, using the exact STFT params Matcha trains on."""
    import librosa
    import numpy as np
    import torch

    from phoonnx_train.matcha.dataset import F_MAX, F_MIN, HOP_LENGTH, N_FFT, WIN_LENGTH

    m = np.asarray(mel, dtype=np.float64)
    if m.ndim == 3:
        m = m[0]  # [n_feats, T]

    # MatchaTTS.synthesise() already denormalizes (mel_mean/mel_std) but the
    # mel itself is a *natural*-log magnitude (spectral_normalize_torch ==
    # ln, see phoonnx_train.matcha.audio.dynamic_range_compression).
    mel_linear = np.exp(m)

    mel_basis = librosa.filters.mel(
        sr=sample_rate, n_fft=N_FFT, n_mels=n_feats, fmin=F_MIN, fmax=F_MAX
    )
    inv_mel_basis = np.linalg.pinv(mel_basis)
    linear_mag = np.maximum(1e-10, np.dot(inv_mel_basis, mel_linear))

    # librosa.griffinlim seeds its random initial phase from the *global*
    # numpy RNG, which torch.manual_seed does not touch — pin it to the
    # active torch seed explicitly so eval_synthesize is reproducible under
    # a fixed torch seed, as the rest of the callable already is.
    rng = np.random.RandomState(torch.initial_seed() % (2**32))
    audio = librosa.griffinlim(
        linear_mag.astype(np.float32),
        n_iter=60,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_fft=N_FFT,
        random_state=rng,
    )
    return audio.astype(np.float32)
