"""Held-out evaluation loop for VITS training checkpoints.

Watches a training directory for new lightning checkpoints (epoch=*.ckpt),
synthesizes a fixed set of held-out sentences on CPU, scores each clip with
UTMOS (SpeechMOS utmos22_strong) and, when available, speaker similarity
(cosine against a reference-speaker centroid), then appends a summary row to
metrics.csv plus per-utterance scores to perutt/epoch<N>.csv. Wavs of the
best-scoring epoch so far are kept under samples/epoch<N>/.

Idempotent: already-scored epochs are read back from metrics.csv on start and
skipped. Ordering is by epoch number parsed from the filename, never
wall-clock.
"""
import csv
import json
import logging
import shutil
import time
from pathlib import Path

import click
import numpy as np
import soundfile as sf
import torch
import torchaudio

from phoonnx.config import PhonemeType, Alphabet, get_phonemizer
from phoonnx.tokenizer import TTSTokenizer
from phoonnx.util import normalize
from phoonnx_train.eval_utils import (append_metrics, best_mean_so_far,
                                      done_epochs, find_checkpoints,
                                      largest_wavs, parse_step,
                                      read_sentences, size_stable)
from phoonnx_train.vits.lightning import VitsModel

_LOGGER = logging.getLogger(__package__)

# torch >=2.6 defaults torch.load(weights_only=True); lightning checkpoints
# embed pathlib.PosixPath (and full pickled objects), which that rejects.
# These are our own trusted checkpoints, so restore weights_only=False.
_ORIG_TORCH_LOAD = torch.load


def _torch_load(*a, **k):
    k.setdefault("weights_only", False)
    return _ORIG_TORCH_LOAD(*a, **k)


torch.load = _torch_load


def build_encoder(config, noise_scale, length_scale, noise_w):
    """Return (phonemizer, tokenizer, lang, sample_rate, scales) matching
    preprocess/train exactly."""
    phoneme_type = PhonemeType(config["phoneme_type"])
    alphabet = Alphabet(config.get("alphabet") or "ipa")
    ph = get_phonemizer(phoneme_type, alphabet, config.get("phonemizer_model"))
    tokenizer = TTSTokenizer.from_phoonnx_config(config)
    lang = config.get("lang_code", "")
    sample_rate = int(config.get("audio", {}).get("sample_rate", 22050))
    inf = config.get("inference", {})
    scales = [noise_scale if noise_scale is not None else float(inf.get("noise_scale", 0.667)),
              length_scale if length_scale is not None else float(inf.get("length_scale", 1.0)),
              noise_w if noise_w is not None else float(inf.get("noise_w", 0.8))]
    return ph, tokenizer, lang, sample_rate, scales


def text_to_ids(text, ph, tokenizer, lang):
    """Replicate preprocess.py: normalize -> phonemize_to_list (drop '\\n')
    -> tokenizer.tokenize (intersperse pad + BOS/EOS)."""
    utt = normalize(text, lang)
    phonemes = [p for p in ph.phonemize_to_list(utt, lang) if p != "\n"]
    return phonemes, tokenizer.tokenize(phonemes)


def load_model(ckpt_path):
    model = VitsModel.load_from_checkpoint(str(ckpt_path), dataset=None,
                                           map_location="cpu")
    model = model.to("cpu")
    model.eval()
    with torch.no_grad():
        model.model_g.dec.remove_weight_norm()
    return model


def synth(model, ids, scales):
    """Run VitsModel.forward on CPU; return 1-D float32 waveform."""
    text = torch.LongTensor(ids).unsqueeze(0)
    text_lengths = torch.LongTensor([len(ids)])
    scales_t = torch.FloatTensor(scales)
    with torch.no_grad():
        audio = model(text, text_lengths, scales_t, sid=None)
    return audio.detach().cpu().float().squeeze().numpy().astype(np.float32)


def load_speaker_reference(ref_dir, num_ref_wavs):
    """Mean (unit-norm) speaker embedding over the largest reference wavs.

    Returns (embedder, centroid) or (None, None) when speaker similarity
    cannot be computed (speakeronnx missing or no wavs found).
    """
    try:
        from speakeronnx import SpeakerEmbedder
    except ImportError:
        _LOGGER.warning("speakeronnx not installed; scoring UTMOS only")
        return None, None
    wavs = largest_wavs(ref_dir, num_ref_wavs)
    if not wavs:
        _LOGGER.warning("no reference wavs under %s; scoring UTMOS only", ref_dir)
        return None, None
    emb = SpeakerEmbedder()
    vecs = []
    for w in wavs:
        try:
            v = np.asarray(emb.embed(str(w)), dtype=np.float32)
            vecs.append(v / (np.linalg.norm(v) + 1e-9))
        except Exception:
            _LOGGER.exception("failed to embed reference %s", w)
    if not vecs:
        _LOGGER.warning("no reference wavs could be embedded; scoring UTMOS only")
        return None, None
    ref = np.mean(vecs, axis=0)
    ref /= np.linalg.norm(ref) + 1e-9
    _LOGGER.info("speaker reference centroid from %d clips", len(vecs))
    return emb, ref


def speaker_similarity(emb, ref, wav_path):
    v = np.asarray(emb.embed(str(wav_path)), dtype=np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return float(np.dot(v, ref))


def load_utmos():
    _LOGGER.info("loading UTMOS (SpeechMOS utmos22_strong)...")
    predictor = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong",
                               trust_repo=True)
    predictor.eval()
    return predictor


def utmos_score(predictor, wav, sr):
    """wav: 1-D np.float32 at sr. Resample to 16k mono and score."""
    t = torch.from_numpy(np.ascontiguousarray(wav)).float()
    if sr != 16000:
        t = torchaudio.functional.resample(t, sr, 16000)
    with torch.no_grad():
        score = predictor(t.unsqueeze(0), 16000)
    return float(score.squeeze().item())


def evaluate_checkpoint(ckpt_path, epoch, sentences, ph, tokenizer, lang,
                        sample_rate, scales, predictor, emb, ref, output_dir):
    _LOGGER.info("evaluating epoch %d: %s", epoch, ckpt_path)
    if not size_stable(ckpt_path):
        _LOGGER.warning("checkpoint %s not stable yet, deferring", ckpt_path)
        return False
    try:
        model = load_model(ckpt_path)
    except Exception:
        _LOGGER.exception("failed to load %s", ckpt_path)
        return False

    metrics_csv = output_dir / "metrics.csv"
    tmp = output_dir / "_work" / f"epoch{epoch}"
    tmp.mkdir(parents=True, exist_ok=True)
    perutt = []
    for i, text in enumerate(sentences):
        try:
            _, ids = text_to_ids(text, ph, tokenizer, lang)
            wav = synth(model, ids, scales)
            wpath = tmp / f"utt{i:02d}.wav"
            sf.write(wpath, wav, sample_rate)
            score = utmos_score(predictor, wav, sample_rate)
            sim = speaker_similarity(emb, ref, wpath) if emb is not None else None
            _LOGGER.info("  utt%02d dur=%.2fs utmos=%.3f spk_sim=%s  %s",
                         i, len(wav) / sample_rate, score,
                         f"{sim:.3f}" if sim is not None else "-", text[:30])
            perutt.append((text, score, sim))
        except Exception:
            _LOGGER.exception("  failed utt %d: %s", i, text)

    del model
    if not perutt:
        _LOGGER.error("no utterances scored for epoch %d", epoch)
        return False

    scores = np.array([s for _, s, _ in perutt], dtype=np.float64)
    sims = np.array([m for _, _, m in perutt if m is not None], dtype=np.float64)

    perutt_dir = output_dir / "perutt"
    perutt_dir.mkdir(parents=True, exist_ok=True)
    with open(perutt_dir / f"epoch{epoch}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sentence", "utmos", "spk_sim"])
        for text, s, m in perutt:
            w.writerow([text, f"{s:.4f}", f"{m:.4f}" if m is not None else ""])

    row = {
        "epoch": epoch,
        "step": parse_step(ckpt_path.name),
        "checkpoint": str(ckpt_path),
        "utmos_mean": f"{scores.mean():.4f}",
        "utmos_std": f"{scores.std():.4f}",
        "utmos_min": f"{scores.min():.4f}",
        "utmos_max": f"{scores.max():.4f}",
        "spk_sim_mean": f"{sims.mean():.4f}" if sims.size else "",
        "spk_sim_std": f"{sims.std():.4f}" if sims.size else "",
        "spk_sim_min": f"{sims.min():.4f}" if sims.size else "",
        "n_sentences": len(scores),
    }

    prev_best = best_mean_so_far(metrics_csv)
    append_metrics(metrics_csv, row)
    _LOGGER.info("epoch %d: utmos=%.4f±%.4f n=%d", epoch, scores.mean(),
                 scores.std(), len(scores))

    if prev_best is None or scores.mean() > prev_best:
        dest = output_dir / "samples" / f"epoch{epoch}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(tmp, dest)
        _LOGGER.info("new best epoch %d (mean=%.4f); wavs kept at %s",
                     epoch, scores.mean(), dest)
    shutil.rmtree(tmp, ignore_errors=True)
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
def main(train_dir, config_path, sentences_path, output_dir, speaker_ref_dir,
         num_ref_wavs, poll, once, dry_run, noise_scale, length_scale,
         noise_w, seed):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(seed)

    train_dir = Path(train_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences = read_sentences(sentences_path)
    if not sentences:
        raise click.ClickException(f"no sentences in {sentences_path}")
    _LOGGER.info("%d held-out sentences", len(sentences))

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    ph, tokenizer, lang, sample_rate, scales = build_encoder(
        config, noise_scale, length_scale, noise_w)
    _LOGGER.info("lang=%s sample_rate=%d scales=%s vocab=%d", lang,
                 sample_rate, scales, len(config.get("phoneme_id_map", {})))

    if dry_run:
        for i, text in enumerate(sentences):
            phonemes, ids = text_to_ids(text, ph, tokenizer, lang)
            _LOGGER.info("utt%02d ids=%d phon=%d | %s", i, len(ids),
                         len(phonemes), text[:40])
        _LOGGER.info("dry-run OK: config loaded, all sentences encoded")
        return

    predictor = load_utmos()
    emb, ref = load_speaker_reference(speaker_ref_dir, num_ref_wavs) \
        if speaker_ref_dir else (None, None)
    if speaker_ref_dir is None:
        _LOGGER.warning("--speaker-ref-dir not given; scoring UTMOS only")

    metrics_csv = output_dir / "metrics.csv"
    while True:
        try:
            done = done_epochs(metrics_csv)
            ckpts = find_checkpoints(train_dir)
            todo = sorted(e for e in ckpts if e not in done)
            if not todo:
                _LOGGER.info("no new checkpoints (done=%d, found=%d)",
                             len(done), len(ckpts))
            for epoch in todo:
                evaluate_checkpoint(ckpts[epoch], epoch, sentences, ph,
                                    tokenizer, lang, sample_rate, scales,
                                    predictor, emb, ref, output_dir)
        except Exception:
            _LOGGER.exception("scan iteration failed; continuing")
        if once:
            break
        time.sleep(poll)


if __name__ == "__main__":
    main()
