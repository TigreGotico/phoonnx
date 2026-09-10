"""Fetch the reference clips ``export_bsc_speakers.py`` turns into per-speaker styles.

Neither BSC-LT multispeaker repository ships reference audio, so the reference clips
come from the corpora the checkpoints were trained on:

* **ca** -- ``projecte-aina/festcat_trimmed_denoised`` (11 named speakers). The other
  Catalan training corpus, ``openslr-slr69-ca-trimmed-denoised``, carries anonymous
  ids only and contributes no named voices.
* **es** -- ``ylacombe/cml-tts`` config ``spanish``. Speaker ids are numeric. Six
  speakers hold essentially the whole corpus; the rest have minutes each and are not
  worth a preset voice.

Clips are read straight out of the HuggingFace parquet exports with DuckDB, which
projects only the columns and row groups it needs -- downloading the full corpora
(hundreds of GB for CML-TTS) to keep four clips per speaker is not necessary.

Writes ``<out>/<speaker>_<n>.wav`` plus the matching ``.txt`` transcript, which the
WER check uses as its reference text.

Usage::

    pip install duckdb soundfile
    python fetch_bsc_speaker_refs.py ca /refs/ca
    python fetch_bsc_speaker_refs.py es /refs/es
"""
import argparse
import io
import json
import os
import urllib.request
from collections import defaultdict

N_CLIPS = 4
DUR_RANGE = (3.0, 12.0)   # long enough for a stable style, short enough to stay clean

FESTCAT = "projecte-aina/festcat_trimmed_denoised"
CMLTTS = "ylacombe/cml-tts"

CA_SPEAKERS = ["bet", "eli", "eva", "jan", "mar", "ona", "pau", "pep", "pol", "teo", "uri"]
ES_SPEAKERS = [3946, 8882, 9972, 10246, 11797, 12367]


def parquet_urls(dataset: str, config: str, split: str = "train"):
    url = f"https://huggingface.co/api/datasets/{dataset}/parquet/{config}/{split}"
    return json.load(urllib.request.urlopen(url))


def _connect():
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    return con


def _write(out, speaker, n, audio_bytes, text):
    import soundfile as sf

    y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if not (DUR_RANGE[0] <= len(y) / sr <= DUR_RANGE[1]):
        return False
    sf.write(f"{out}/{speaker}_{n}.wav", y, sr)
    with open(f"{out}/{speaker}_{n}.txt", "w") as f:
        f.write(text)
    return True


def fetch_ca(out):
    con = _connect()
    urls = parquet_urls(FESTCAT, "default")
    got = defaultdict(int)
    for url in urls:
        if all(got[s] >= N_CLIPS for s in CA_SPEAKERS):
            break
        # cheap: speaker_id only, so the audio column is never fetched
        present = {r[0] for r in con.execute(
            f"SELECT DISTINCT speaker_id FROM read_parquet('{url}')").fetchall()}
        todo = [s for s in CA_SPEAKERS if s in present and got[s] < N_CLIPS]
        if not todo:
            continue
        quoted = ",".join(f"'{s}'" for s in todo)
        rows = con.execute(f"SELECT speaker_id, audio, transcription FROM read_parquet('{url}') "
                           f"WHERE speaker_id IN ({quoted}) LIMIT 400").fetchall()
        for sid, audio, text in rows:
            if got[sid] < N_CLIPS and _write(out, sid, got[sid], audio["bytes"], text):
                got[sid] += 1
    return dict(got)


def fetch_es(out):
    con = _connect()
    urls = parquet_urls(CMLTTS, "spanish")
    got = defaultdict(int)
    for url in urls:
        if all(got[s] >= N_CLIPS for s in ES_SPEAKERS):
            break
        present = {r[0] for r in con.execute(
            f"SELECT DISTINCT speaker_id FROM read_parquet('{url}')").fetchall()}
        todo = [s for s in ES_SPEAKERS if s in present and got[s] < N_CLIPS]
        if not todo:
            continue
        ids = ",".join(str(s) for s in todo)
        rows = con.execute(
            f"SELECT speaker_id, audio, text FROM read_parquet('{url}') "
            f"WHERE speaker_id IN ({ids}) AND duration BETWEEN {DUR_RANGE[0]} AND {DUR_RANGE[1]} "
            f"LIMIT 400").fetchall()
        for sid, audio, text in rows:
            if got[sid] < N_CLIPS and _write(out, sid, got[sid], audio["bytes"], text):
                got[sid] += 1
    return dict(got)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lang", choices=["ca", "es"])
    ap.add_argument("out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    got = fetch_ca(a.out) if a.lang == "ca" else fetch_es(a.out)
    print({str(k): v for k, v in sorted(got.items(), key=lambda kv: str(kv[0]))})
