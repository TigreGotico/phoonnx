"""Tests for scripts/voice_sweep.py.

The voices are fakes that return synthetic numpy audio, and the `run` tests replace the
worker process with a stand-in. Nothing loads a model or downloads a file.
"""
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import voice_sweep as vs  # noqa: E402

RATE = 22050
# 41 and 22 letters or digits.
SENTENCES = [{"id": "plain", "text": "The weather is nice today, and I want to go for a walk.", "native": True},
             {"id": "digits", "text": "I have 23 apples and 7 pears.", "native": True}]
ENTRY = {"model_url": "https://example.org/model.onnx", "phoneme_type": "espeak", "lang": "en-US"}


def speechlike(seconds: float) -> np.ndarray:
    """A 180 Hz tone under a 3 Hz envelope: voiced, centred, peak 0.3."""
    t = np.arange(int(seconds * RATE)) / RATE
    return (0.3 * np.sin(2 * np.pi * 180 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def synthesise(monkeypatch, audio_for, sentences=SENTENCES) -> dict:
    """Run synthesise_voice on a fake voice whose n-th sentence is `audio_for(n)`."""
    import phoonnx.model_manager

    class FakeVoice:
        config = SimpleNamespace(lang_code="en-US", phoneme_type="espeak", alphabet="ipa", sample_rate=RATE)
        adapter = object()

        def __init__(self):
            self.calls = 0

        def synthesize(self, text, syn):
            audio = audio_for(self.calls)
            self.calls += 1
            return iter([SimpleNamespace(audio_float_array=audio, sample_rate=RATE)])

    class FakeInfo:
        def __init__(self, **entry):
            pass

        def load(self):
            return FakeVoice()

    monkeypatch.setattr(phoonnx.model_manager, "TTSModelInfo", FakeInfo)
    record = {"voice": "fake"}
    vs.synthesise_voice(record, ENTRY, sentences, SimpleNamespace(reference_dir=None, audio_dir=None))
    return record


class TestAudioStatus:
    def test_speech_like_audio_is_ok(self, monkeypatch):
        record = synthesise(monkeypatch, lambda n: speechlike(3))
        assert record["status"] == "ok"

    def test_silence(self, monkeypatch):
        record = synthesise(monkeypatch, lambda n: np.zeros(3 * RATE, np.float32))
        assert record["status"] == "silent"
        assert [s["problems"] for s in record["sentences"]] == [["silent"], ["silent"]]

    def test_nan(self, monkeypatch):
        record = synthesise(monkeypatch, lambda n: np.full(3 * RATE, np.nan, np.float32))
        assert record["status"] == "nonfinite"

    def test_inf(self, monkeypatch):
        record = synthesise(monkeypatch, lambda n: np.full(3 * RATE, np.inf, np.float32))
        assert record["status"] == "nonfinite"

    def test_one_sample(self, monkeypatch):
        record = synthesise(monkeypatch, lambda n: np.array([0.05], np.float32))
        assert record["status"] == "too_short"

    def test_too_short_for_the_text(self, monkeypatch):
        # 10 ms for 41 characters is 0.0002 s per character.
        record = synthesise(monkeypatch, lambda n: speechlike(0.01))
        assert record["status"] == "too_short"

    def test_dc_offset(self, monkeypatch):
        record = synthesise(monkeypatch, lambda n: speechlike(3) + 0.4)
        assert record["status"] == "dc_offset"

    def test_clipped_square_wave(self, monkeypatch):
        square = np.sign(np.sin(2 * np.pi * 200 * np.arange(3 * RATE) / RATE)).astype(np.float32)
        record = synthesise(monkeypatch, lambda n: square)
        assert record["status"] == "clipped"

    def test_white_noise(self, monkeypatch):
        rng = np.random.default_rng(0)
        record = synthesise(monkeypatch, lambda n: rng.uniform(-0.5, 0.5, 3 * RATE).astype(np.float32))
        assert record["status"] == "noise"

    def test_one_silent_sentence_among_speech(self, monkeypatch):
        record = synthesise(monkeypatch, lambda n: speechlike(3) if n == 0 else np.zeros(3 * RATE, np.float32))
        assert record["status"] == "partial"
        assert record["error"] == "digits: silent"

    def test_language_with_no_sentences(self, monkeypatch):
        record = synthesise(monkeypatch, lambda n: speechlike(3), sentences=[])
        assert record["status"] == "no_sentences"
        assert "error" not in record


class TestWerLimit:
    def score(self, tmp_path, monkeypatch, transcript):
        wav = tmp_path / "plain.wav"
        sf.write(wav, speechlike(3), RATE)
        floor = tmp_path / "floor.json"
        floor.write_text(json.dumps({"en": {"verified": True, "model": "fake", "sentences": []}}))
        monkeypatch.setattr(vs, "spoken", lambda text, lang: text)
        monkeypatch.setattr(vs.Recognisers, "transcribe", lambda self, name, lang, audio, rate: transcript)
        done = {"v": {"voice": "v", "lang": "en-US", "status": "ok",
                      "sentences": [{"id": "plain", "text": "the weather is nice today", "native": True,
                                     "wav": str(wav)}]}}
        out = tmp_path / "run.jsonl"
        args = SimpleNamespace(asr_floor=str(floor), threads=1, jobs=1, out=str(out), keep_audio=True,
                               max_wer=0.5, max_cer=0.35)
        code = vs.score_run(args, {"v": ("x.json", {"lang": "en-US"})}, {"v": None}, done)
        return code, json.loads(out.read_text().splitlines()[0])

    def test_wer_over_the_limit_fails_the_voice(self, tmp_path, monkeypatch):
        code, record = self.score(tmp_path, monkeypatch, "banana helicopter purple")
        assert record["wer"] == 1.0
        assert record["status"] == "unintelligible"
        assert code == 1

    def test_wer_under_the_limit_stays_ok(self, tmp_path, monkeypatch):
        code, record = self.score(tmp_path, monkeypatch, "the weather is nice today")
        assert record["wer"] == 0.0
        assert record["status"] == "ok"
        assert code == 0


class TestDiff:
    def diff(self, tmp_path, capsys, a, b):
        for name, rows in (("a", a), ("b", b)):
            (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        code = vs.diff(SimpleNamespace(a=str(tmp_path / "a.jsonl"), b=str(tmp_path / "b.jsonl"), wer_margin=0.15))
        return code, capsys.readouterr().out

    def test_ok_to_partial_is_a_regression(self, tmp_path, capsys):
        code, out = self.diff(tmp_path, capsys, [{"voice": "v", "status": "ok"}],
                              [{"voice": "v", "status": "partial", "error": "digits: silent"}])
        assert code == 1
        assert "regressions: 1" in out

    def test_voice_missing_from_b_is_a_regression(self, tmp_path, capsys):
        code, out = self.diff(tmp_path, capsys, [{"voice": "v", "status": "ok"}, {"voice": "w", "status": "ok"}],
                              [{"voice": "v", "status": "ok"}])
        assert code == 1
        assert "regressions: 1" in out

    def test_wer_missing_from_b_is_a_regression(self, tmp_path, capsys):
        code, out = self.diff(tmp_path, capsys, [{"voice": "v", "status": "ok", "wer": 0.05, "asr": {"lang": "en"}}],
                              [{"voice": "v", "status": "ok", "asr": {"lang": "en"}}])
        assert code == 1
        assert "regressions: 1" in out

    def test_no_sentences_in_b_is_not_a_regression(self, tmp_path, capsys):
        code, out = self.diff(tmp_path, capsys, [{"voice": "v", "status": "ok"}],
                              [{"voice": "v", "status": "no_sentences"}])
        assert code == 0
        assert "regressions: 0" in out


# ---------------------------------------------------------------- run

def voices_with_own_repo(count: int) -> list[str]:
    """Index voices that each download from one Phoonnx repo that no other of them uses."""
    chosen, seen = [], set()
    for voice_id, (_, entry) in sorted(vs.read_index().items()):
        repos = vs.repos(entry)
        if len(repos) == 1 and next(iter(repos)).startswith("Phoonnx/") and not repos & seen:
            chosen.append(voice_id)
            seen |= repos
        if len(chosen) == count:
            return chosen
    raise AssertionError("the index has too few voices with a repo of their own")


def make_repo(cache: Path, repo: str, commit: str | None = None) -> Path:
    """A cached repo in the layout huggingface_hub writes: blob, ref and snapshot link."""
    folder = cache / f"models--{repo.replace('/', '--')}"
    commit, blob = commit or hashlib.sha1(repo.encode()).hexdigest(), "1" * 64
    (folder / "blobs").mkdir(parents=True)
    (folder / "blobs" / blob).write_bytes(b"onnx")
    (folder / "refs").mkdir()
    (folder / "refs" / "main").write_text(commit)
    (folder / "snapshots" / commit).mkdir(parents=True)
    (folder / "snapshots" / commit / "model.onnx").symlink_to(f"../../blobs/{blob}")
    return folder


def test_evict_removes_the_named_repos_when_commits_match(tmp_path):
    # A moved repo has the same commit hashes under its new and its old id.
    commit = "0" * 40
    moved, other, old = (make_repo(tmp_path, repo, commit)
                         for repo in ("Phoonnx/x", "Phoonnx/y", "OpenVoiceOS/x"))
    vs.evict({"Phoonnx/x", "Phoonnx/y"}, str(tmp_path))
    assert not moved.exists()
    assert not other.exists()
    assert old.is_dir()


# Stands in for `voice_sweep.py worker`: it "downloads" each voice's repo into the cache
# the way the loader would (under the old id when the cache holds that one) and reports
# the voice as ok.
FAKE_WORKER = """#!{python}
import json, sys
from pathlib import Path
sys.path.insert(0, {tests!r})
from huggingface_hub.constants import HF_HUB_CACHE
from test_voice_sweep import make_repo, vs
cache = Path(HF_HUB_CACHE)
index = vs.read_index()
for line in sys.stdin:
    voice = line.strip()
    print(json.dumps({{"event": "start", "voice": voice}}), flush=True)
    for repo in vs.repos(index[voice][1]):
        if not any((cache / f"models--{{i.replace('/', '--')}}").exists()
                   for i in (repo, repo.replace("Phoonnx/", "OpenVoiceOS/"))):
            make_repo(cache, repo)
    print(json.dumps({{"event": "result", "record": {{"voice": voice, "lang": index[voice][1]["lang"],
                                                     "status": "ok", "sentences": []}}}}), flush=True)
"""


def sweep(tmp_path, *argv, executable=None, timeout=120):
    """`voice_sweep.py run ...` in a child process, offline, with its own Hugging Face cache."""
    code = (f"import sys\nsys.path.insert(0, {str(SCRIPTS)!r})\nimport voice_sweep\n"
            + (f"sys.executable = {str(executable)!r}\n" if executable else "")
            + f"sys.argv = ['voice_sweep.py', 'run', *{list(argv)!r}]\nsys.exit(voice_sweep.main())\n")
    env = {**os.environ, "HF_HUB_CACHE": str(tmp_path / "hub"), "HF_HUB_OFFLINE": "1"}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout,
                          env=env, cwd=tmp_path)


@pytest.fixture
def run_files(tmp_path):
    (tmp_path / "hub").mkdir()
    sentences = tmp_path / "sentences.tsv"
    sentences.write_text("lang\tid\tcategory\ttext\tsource\n"
                         "en-US\tplain\tplain\tThe weather is nice today.\ttest\n", encoding="utf-8")
    return {"sentences": str(sentences), "out": str(tmp_path / "run.jsonl"), "voices": tmp_path / "voices.txt"}


class TestRun:
    def test_unknown_voice_id_is_an_error(self, tmp_path, run_files):
        known = voices_with_own_repo(1)[0]
        run_files["voices"].write_text(f"{known}\nOpenVoiceOS/pipertts_en-US_mira\n")
        result = sweep(tmp_path, "--sentences", run_files["sentences"], "--voices", str(run_files["voices"]),
                       "--out", run_files["out"], "--nice", "0")
        assert result.returncode != 0
        assert "not in the index: OpenVoiceOS/pipertts_en-US_mira" in result.stderr
        assert not Path(run_files["out"] + ".worker.log").exists()

    def test_missing_sentence_file_stops_before_any_worker(self, tmp_path, run_files):
        run_files["voices"].write_text(voices_with_own_repo(1)[0] + "\n")
        missing = str(tmp_path / "missing.tsv")
        result = sweep(tmp_path, "--sentences", missing, "--voices", str(run_files["voices"]),
                       "--out", run_files["out"], "--nice", "0", timeout=60)
        assert result.returncode != 0
        assert missing in result.stderr
        assert not Path(run_files["out"] + ".worker.log").exists()

    def test_worker_that_dies_before_its_first_voice_ends_the_run(self, tmp_path, run_files):
        voices = voices_with_own_repo(2)
        run_files["voices"].write_text("\n".join(voices) + "\n")
        result = sweep(tmp_path, "--sentences", run_files["sentences"], "--voices", str(run_files["voices"]),
                       "--out", run_files["out"], "--nice", "0", "--timeout", "30",
                       executable="/bin/false", timeout=60)
        records = {r["voice"]: r for r in map(json.loads, Path(run_files["out"]).read_text().splitlines())}
        assert {v: r["status"] for v, r in records.items()} == {v: "crash" for v in voices}
        assert all("before it started" in r["error"] for r in records.values())
        assert result.returncode == 1

    def test_evict_keeps_repos_cached_before_the_run(self, tmp_path, run_files):
        hub = tmp_path / "hub"
        cached, old_id, downloaded = voices_with_own_repo(3)
        index = vs.read_index()
        repo_of = {v: next(iter(vs.repos(index[v][1]))) for v in (cached, old_id, downloaded)}
        kept = make_repo(hub, repo_of[cached])
        kept_old = make_repo(hub, repo_of[old_id].replace("Phoonnx/", "OpenVoiceOS/"))
        worker = tmp_path / "fake_worker.py"
        worker.write_text(FAKE_WORKER.format(python=sys.executable, tests=str(Path(__file__).parent)))
        worker.chmod(worker.stat().st_mode | stat.S_IEXEC)
        run_files["voices"].write_text(f"{cached}\n{old_id}\n{downloaded}\n")
        result = sweep(tmp_path, "--sentences", run_files["sentences"], "--voices", str(run_files["voices"]),
                       "--out", run_files["out"], "--nice", "0", "--evict", executable=worker)
        assert result.returncode == 0, result.stdout + result.stderr
        assert kept.is_dir()
        assert kept_old.is_dir()
        assert not (hub / f"models--{repo_of[downloaded].replace('/', '--')}").exists()
