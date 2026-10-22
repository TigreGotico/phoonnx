#!/usr/bin/env python3
"""Per-stage TTS latency benchmark.

Mirrors the exact call sequence of ``phoonnx.voice.TTSVoice.synthesize`` and
times each stage with ``time.perf_counter``, plus the wall-clock time to the
first yielded ``AudioChunk`` (time-to-first-audio, TTFA) and the total
synthesis time. No internals are patched: every stage below calls the same
public methods ``synthesize()`` itself calls, in the same order.
"""
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import click
import numpy as np
import onnxruntime

from phoonnx.config import SynthesisConfig
from phoonnx.voice import AudioChunk, TTSVoice

DEFAULT_TEXT_1SENT = "The quick brown fox jumps over the lazy dog."
DEFAULT_TEXT_5SENT = (
    "The quick brown fox jumps over the lazy dog. "
    "A rainbow is a meteorological phenomenon caused by reflection and refraction of light. "
    "Voice assistants rely on low latency to feel responsive. "
    "This sentence exists purely to pad out the paragraph for benchmarking. "
    "Synthesis quality and speed both matter for a good user experience."
)


def timed(fn: Callable[[], Any]) -> tuple:
    """Run ``fn`` and return ``(result, elapsed_seconds)`` using a monotonic clock."""
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def resolve_model(model: str, config: Optional[str]) -> tuple:
    """Resolve MODEL to a local (model_path, config_path) pair.

    ``model`` may be a filesystem path to a ``.onnx`` file, or a voice id
    downloadable via ``TTSModelManager`` (e.g. ``piper/en_US-amy-low``).
    """
    model_path = Path(model)
    if model_path.is_file():
        return str(model_path), config

    import os

    import phoonnx.model_manager as mm

    manager = mm.TTSModelManager()
    if not manager.download_voice_by_id(model):
        raise click.ClickException(f"Could not resolve or download voice '{model}'")

    # download_voice_by_id() resolves bundled-index voices without registering
    # them in manager.voices; look it up the same way it did internally.
    info = manager.voices.get(model)
    if info is None:
        base_path = Path(os.path.dirname(mm.__file__)) / "voice_index"
        for index_file in base_path.glob("*.json"):
            data = json.loads(index_file.read_text(encoding="utf-8"))
            if model in data:
                info = mm.TTSModelInfo(**data[model])
                break
    if info is None:
        raise click.ClickException(f"Downloaded '{model}' but could not reload its metadata")
    local_model_path = info.download_model()
    local_config_path = local_model_path.parent / "model.json"
    if not local_config_path.is_file():
        info.download_config()
    return str(local_model_path), str(local_config_path)


def load_voice_timed(model_path: str, config_path: Optional[str]) -> Dict[str, Any]:
    """Load a voice, separately timing config parse from ONNX session creation.

    Mirrors ``TTSVoice.load`` (see phoonnx/voice.py) rather than calling it as
    one opaque unit, so config-parse cost and session-creation cost are
    reported separately.
    """
    from phoonnx.config import VoiceConfig
    from phoonnx.providers import make_session, resolve_providers

    if config_path is None:
        config_path = f"{model_path}.json"

    def _parse_config():
        import os

        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        else:
            config_dict = {"phoneme_type": "unicode", "alphabet": "unicode"}
        return VoiceConfig.from_dict(config_dict), config_dict

    (voice_config, config_dict), t_config_parse = timed(_parse_config)

    resolved_providers = resolve_providers(None, use_cuda=False)

    session, t_session_create = timed(lambda: make_session(model_path, providers=resolved_providers))

    from phoonnx.engines import detect_engine

    adapter = detect_engine(config=config_dict, session=session)
    voice = TTSVoice(config=voice_config, session=session, adapter=adapter)

    return {
        "voice": voice,
        "t_config_parse": t_config_parse,
        "t_session_create": t_session_create,
    }


def warmup_session(voice: TTSVoice, text: str = "hi") -> float:
    """Run one throwaway synthesis pass and time the ONNX session's cold
    ``session.run`` (graph optimization / kernel selection happens here)."""
    _, elapsed = timed(lambda: list(voice.synthesize(text)))
    return elapsed


def bench_stages(voice: TTSVoice, text: str, syn_config: SynthesisConfig) -> Dict[str, Any]:
    """Time each stage of the synthesize() pipeline honestly, by calling the
    same public methods in the same order voice.synthesize() does."""
    stages: Dict[str, float] = {}

    working_text = text
    if voice.phonetic_spellings and syn_config.enable_phonetic_spellings:
        working_text, stages["text_normalize"] = timed(
            lambda: voice.phonetic_spellings.apply(working_text)
        )
    else:
        stages["text_normalize"] = 0.0

    do_diacritics = syn_config.add_diacritics
    if do_diacritics is None:
        do_diacritics = voice.config.add_diacritics
    if do_diacritics:
        diacritizer_model = syn_config.diacritizer_model or voice.config.diacritizer_model
        working_text, stages["diacritics"] = timed(
            lambda: voice.phonemizer.add_diacritics(
                working_text, voice.config.lang_code, model=diacritizer_model
            )
        )
    else:
        stages["diacritics"] = 0.0

    sentence_phonemes, stages["phonemize"] = timed(lambda: voice.phonemize(working_text))

    all_ids, stages["tokenize"] = timed(
        lambda: [voice.phonemes_to_ids(p) for p in sentence_phonemes if p]
    )

    run_times: List[float] = []
    post_times: List[float] = []
    audio_seconds = 0.0
    for phoneme_ids in all_ids:
        if not phoneme_ids:
            continue
        audio, run_elapsed = timed(lambda: voice.phoneme_ids_to_audio(phoneme_ids, syn_config))
        run_times.append(run_elapsed)

        def _postprocess():
            a = audio
            if syn_config.normalize_audio:
                max_val = np.max(np.abs(a))
                a = np.zeros_like(a) if max_val < 1e-8 else a / max_val
            if syn_config.volume != 1.0:
                a = a * syn_config.volume
            a = np.clip(a, -1.0, 1.0).astype(np.float32)
            return AudioChunk(sample_rate=voice.config.sample_rate, sample_width=2, sample_channels=1,
                               audio_float_array=a)

        chunk, post_elapsed = timed(_postprocess)
        post_times.append(post_elapsed)
        audio_seconds += len(chunk.audio_float_array) / chunk.sample_rate

    stages["session_run_total"] = sum(run_times)
    stages["session_run_per_sentence"] = run_times
    stages["postprocess_total"] = sum(post_times)
    stages["audio_seconds"] = audio_seconds
    return stages


def bench_ttfa(voice: TTSVoice, text: str, syn_config: SynthesisConfig) -> Dict[str, float]:
    """Wall-clock time from calling synthesize() to the first yielded
    AudioChunk (TTFA), and total time to exhaust the generator."""
    start = time.perf_counter()
    gen = voice.synthesize(text, syn_config=syn_config)
    first_chunk = next(gen)
    ttfa = time.perf_counter() - start
    audio_seconds = len(first_chunk.audio_float_array) / first_chunk.sample_rate
    for chunk in gen:
        audio_seconds += len(chunk.audio_float_array) / chunk.sample_rate
    total = time.perf_counter() - start
    rtf = total / audio_seconds if audio_seconds > 0 else float("nan")
    return {"ttfa": ttfa, "total": total, "audio_seconds": audio_seconds, "rtf": rtf}


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"median": float("nan"), "p95": float("nan"), "min": float("nan"), "max": float("nan")}
    ordered = sorted(values)
    p95_idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "median": statistics.median(values),
        "p95": ordered[p95_idx],
        "min": ordered[0],
        "max": ordered[-1],
    }


@click.command()
@click.argument("model")
@click.option("--config", "config_path", default=None, help="Path to the voice's JSON config (defaults to MODEL + '.json').")
@click.option("--runs", default=5, show_default=True, help="Number of warm runs to measure (median/p95).")
@click.option("--warmup/--no-warmup", default=False, show_default=True,
              help="Run an extra untimed warmup pass before the cold run (the cold run is always reported separately).")
@click.option("--text", default=None, help="Override both the 1-sentence and 5-sentence benchmark texts (used verbatim for both).")
@click.option("--text-file", default=None, type=click.Path(exists=True), help="Read benchmark text from a file instead of the built-in defaults.")
@click.option("--json-out", default=None, type=click.Path(), help="Write machine-readable results as JSON to this path.")
def main(model, config_path, runs, warmup, text, text_file, json_out):
    """Measure per-stage TTS latency and time-to-first-audio for MODEL.

    MODEL is a path to a local ONNX voice, or a voice id downloadable via
    TTSModelManager (e.g. 'piper/en_US-amy-low').
    """
    if text_file:
        text = Path(text_file).read_text(encoding="utf-8").strip()

    text_1sent = text or DEFAULT_TEXT_1SENT
    text_5sent = text or DEFAULT_TEXT_5SENT

    click.echo(f"Resolving model '{model}'...")
    model_path, resolved_config_path = resolve_model(model, config_path)

    click.echo("Loading voice (cold)...")
    load_result = load_voice_timed(model_path, resolved_config_path)
    voice = load_result["voice"]
    providers = onnxruntime.get_available_providers()

    if warmup:
        warmup_session(voice, text_1sent)

    click.echo("Timing cold warmup run (first session.run)...")
    t_warmup_run = warmup_session(voice, text_1sent)

    syn_config = SynthesisConfig()

    results: Dict[str, Any] = {
        "model": model,
        "model_path": model_path,
        "sample_rate": voice.config.sample_rate,
        "onnxruntime_providers_available": providers,
        "load": {
            "config_parse_s": load_result["t_config_parse"],
            "session_create_s": load_result["t_session_create"],
        },
        "cold_warmup_run_s": t_warmup_run,
        "runs": runs,
    }

    for label, sample_text in (("1sentence", text_1sent), ("5sentence", text_5sent)):
        stage_samples: Dict[str, List[float]] = {}
        ttfa_samples: List[float] = []
        total_samples: List[float] = []
        rtf_samples: List[float] = []
        audio_seconds = None

        click.echo(f"Benchmarking '{label}' ({runs} warm runs)...")
        for _ in range(runs):
            stages = bench_stages(voice, sample_text, syn_config)
            for key in ("text_normalize", "diacritics", "phonemize", "tokenize",
                        "session_run_total", "postprocess_total"):
                stage_samples.setdefault(key, []).append(stages[key])
            audio_seconds = stages["audio_seconds"]

            ttfa_result = bench_ttfa(voice, sample_text, syn_config)
            ttfa_samples.append(ttfa_result["ttfa"])
            total_samples.append(ttfa_result["total"])
            rtf_samples.append(ttfa_result["rtf"])

        results[label] = {
            "stages": {k: summarize(v) for k, v in stage_samples.items()},
            "ttfa": summarize(ttfa_samples),
            "total": summarize(total_samples),
            "rtf": summarize(rtf_samples),
            "audio_seconds": audio_seconds,
        }

    _print_table(results)

    if json_out:
        Path(json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        click.echo(f"\nWrote JSON results to {json_out}")


def _print_table(results: Dict[str, Any]) -> None:
    click.echo("\n=== Load (cold, one-shot) ===")
    click.echo(f"{'stage':<20}{'seconds':>12}")
    click.echo(f"{'config_parse':<20}{results['load']['config_parse_s']:>12.4f}")
    click.echo(f"{'session_create':<20}{results['load']['session_create_s']:>12.4f}")
    click.echo(f"{'cold_warmup_run':<20}{results['cold_warmup_run_s']:>12.4f}")

    for label in ("1sentence", "5sentence"):
        r = results[label]
        click.echo(f"\n=== {label} (warm, N={results['runs']}) ===")
        click.echo(f"{'stage':<24}{'median':>10}{'p95':>10}{'min':>10}{'max':>10}")
        for stage_name, stats in r["stages"].items():
            click.echo(
                f"{stage_name:<24}{stats['median']:>10.4f}{stats['p95']:>10.4f}{stats['min']:>10.4f}{stats['max']:>10.4f}"
            )
        for extra in ("ttfa", "total", "rtf"):
            stats = r[extra]
            click.echo(
                f"{extra:<24}{stats['median']:>10.4f}{stats['p95']:>10.4f}{stats['min']:>10.4f}{stats['max']:>10.4f}"
            )
        click.echo(f"audio_seconds: {r['audio_seconds']:.3f}")


if __name__ == "__main__":
    main()
