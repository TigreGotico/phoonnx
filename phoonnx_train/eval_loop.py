"""Sidecar checkpoint-evaluation scoreboard (thin CLI over the evaluation pkg).

Watches a training directory for new lightning checkpoints (``epoch=*.ckpt``),
synthesizes a fixed set of held-out sentences on CPU, scores each clip (UTMOS at
least, and speaker similarity against a reference-speaker centroid when a
``--speaker-ref-dir`` is given), appends a summary row to metrics.csv plus
per-utterance scores under samples/, and maintains best.ckpt/best.json for the
similarity-gated best epoch so far.

Idempotent: already-scored (and permanently failed) epochs are skipped. Ordering
is by epoch number parsed from the filename, never wall-clock. With
``--early-stop-patience`` set, once that many scored epochs pass with no
improvement a ``stop.flag`` is written into the training directory — the training
process's ``StopFileCallback`` picks it up and stops (sidecar -> trainer bridge).

The heavy lifting lives in ``phoonnx_train.evaluation``; this module is the
watch loop, CLI surface and idempotent-restart glue.
"""
import json
import logging
import time
from pathlib import Path

import click

from phoonnx_train.engines import get_engine, list_engines
from phoonnx_train.eval_utils import (find_checkpoints, read_sentences,
                                      size_stable)
from phoonnx_train.evaluation import CheckpointScorer, MetricsTracker, SelectionPolicy
from phoonnx_train.evaluation.scorer import (build_encoder, text_to_ids,
                                             write_epoch_perutt)

_LOGGER = logging.getLogger(__package__)


def resolve_engine(name: str, config: dict) -> str:
    """Resolve ``--engine`` (default 'auto'): the config's ``engine`` key when
    present, else 'vits'."""
    if name and name != "auto":
        return name
    engine = config.get("engine")
    if engine and engine in list_engines():
        return engine
    return "vits"


def evaluate_one(scorer, selection, tracker, ckpt_path, epoch, train_dir):
    """Score a single checkpoint into the scoreboard. Returns True when scored,
    False when deferred (unstable) or failed."""
    _LOGGER.info("evaluating epoch %d: %s", epoch, ckpt_path)
    if not size_stable(ckpt_path):
        _LOGGER.warning("checkpoint %s not stable yet, deferring", ckpt_path)
        return False
    work_dir = tracker.output_dir / "_work" / f"epoch{epoch}"
    try:
        row = scorer.score(ckpt_path, epoch, work_dir=work_dir)
    except Exception:
        _LOGGER.exception("failed to score epoch %d", epoch)
        tracker.record_failure(epoch)
        return False

    best = selection.read_best(tracker.output_dir)
    tracker.append(row.to_csv_row())
    # Keep a per-epoch per-utterance file for EVERY epoch (overfit diagnosis:
    # per-sentence trends across epochs, not just the best epoch's samples).
    write_epoch_perutt(tracker.output_dir, row, scorer.metrics)
    _LOGGER.info("epoch %d: %s", epoch, row.aggregates)
    if selection.is_improvement(row, best):
        selection.commit_best(row, tracker.output_dir, work_dir=work_dir)
    return True


@click.command()
@click.option('--train-dir', required=True, type=click.Path(exists=True, file_okay=False), help='Training directory watched for epoch=*.ckpt checkpoints')
@click.option('--config', 'config_path', required=True, type=click.Path(exists=True, dir_okay=False), help='config.json produced by preprocessing')
@click.option('--sentences', 'sentences_path', required=True, type=click.Path(exists=True, dir_okay=False), help='Held-out sentences, one per line')
@click.option('--output-dir', required=True, type=click.Path(file_okay=False), help='Where metrics.csv, perutt/ and samples/ are written')
@click.option('--speaker-ref-dir', type=click.Path(exists=True, file_okay=False), default=None, help='Directory of reference wavs of the target speaker (optional; enables speaker similarity)')
@click.option('--num-ref-wavs', type=int, default=60, help='Number of largest reference wavs averaged into the speaker centroid (default: 60)')
@click.option('--poll', type=float, default=60.0, help='Seconds between checkpoint scans (default: 60)')
@click.option('--once', is_flag=True, help='Single pass then exit')
@click.option('--dry-run', is_flag=True, help='Load config, phonemize and encode the sentences, no synthesis')
@click.option('--noise-scale', type=float, default=None, help='Override inference noise_scale from config')
@click.option('--length-scale', type=float, default=None, help='Override inference length_scale from config')
@click.option('--noise-w', type=float, default=None, help='Override inference noise_w from config')
@click.option('--seed', type=int, default=1234, help='Random seed (default: 1234)')
@click.option('--early-stop-patience', type=int, default=0, help='Write stop.flag into --train-dir after this many scored epochs with no improvement (0=off)')
@click.option('--min-spk-sim', type=float, default=None, help='Similarity gate: best checkpoint must have mean speaker similarity >= this (when speaker scoring is active)')
@click.option('--speaker-id', type=int, default=None, help='Speaker id (sid) to synthesize for multi-speaker models')
@click.option('--engine', 'engine_name', type=str, default='auto', help="Training engine (default: auto from config.json, falling back to vits)")
@click.option('--vocoder', 'vocoder_path', type=click.Path(exists=True, dir_okay=False), default=None, help='Optional vocoder passed to the engine synth (engine-dependent)')
def main(train_dir, config_path, sentences_path, output_dir, speaker_ref_dir,
         num_ref_wavs, poll, once, dry_run, noise_scale, length_scale,
         noise_w, seed, early_stop_patience, min_spk_sim, speaker_id,
         engine_name, vocoder_path):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    train_dir = Path(train_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences = read_sentences(sentences_path)
    if not sentences:
        raise click.ClickException(f"no sentences in {sentences_path}")
    _LOGGER.info("%d held-out sentences", len(sentences))

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if dry_run:
        ph, tokenizer, lang, sample_rate, scales = build_encoder(
            config, noise_scale, length_scale, noise_w)
        _LOGGER.info("lang=%s sample_rate=%d scales=%s vocab=%d", lang,
                     sample_rate, scales, len(config.get("phoneme_id_map", {})))
        for i, text in enumerate(sentences):
            phonemes, ids = text_to_ids(text, ph, tokenizer, lang)
            _LOGGER.info("utt%02d ids=%d phon=%d | %s", i, len(ids),
                         len(phonemes), text[:40])
        _LOGGER.info("dry-run OK: config loaded, all sentences encoded")
        return

    engine = get_engine(resolve_engine(engine_name, config))
    if speaker_ref_dir is None:
        _LOGGER.warning("--speaker-ref-dir not given; scoring UTMOS only")

    scorer = CheckpointScorer(
        engine, config, sentences,
        speaker_ref_dir=Path(speaker_ref_dir) if speaker_ref_dir else None,
        num_ref_wavs=num_ref_wavs,
        speaker_id=speaker_id,
        seed=seed,
        vocoder_path=Path(vocoder_path) if vocoder_path else None,
        scales_override=(noise_scale, length_scale, noise_w),
    )
    selection = SelectionPolicy(metric="utmos_mean", min_spk_sim=min_spk_sim)
    tracker = MetricsTracker(output_dir)

    while True:
        try:
            skip = tracker.skip_epochs()
            ckpts = find_checkpoints(train_dir)
            todo = sorted(e for e in ckpts if e not in skip)
            if not todo:
                _LOGGER.info("no new checkpoints (skip=%d, found=%d)",
                             len(skip), len(ckpts))
            for epoch in todo:
                evaluate_one(scorer, selection, tracker, ckpts[epoch], epoch,
                             train_dir)

            best = selection.read_best(output_dir)
            best_epoch = best.epoch if best is not None else None
            if early_stop_patience and tracker.maybe_stop(best_epoch, early_stop_patience):
                # write stop.flag into the training dir so the trainer sees it
                flag = train_dir / "stop.flag"
                flag.write_text(
                    f"early stopping: no improvement for >= {early_stop_patience} "
                    f"epochs (best epoch {best_epoch})\n", encoding="utf-8")
                _LOGGER.warning("wrote %s", flag)
        except Exception:
            _LOGGER.exception("scan iteration failed; continuing")
        if once:
            break
        time.sleep(poll)


if __name__ == "__main__":
    main()
