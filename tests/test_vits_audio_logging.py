"""Tests for the opt-in validation audio-sample logging guard in VitsModel.

Constructing a real ``VitsModel`` builds a full SynthesizerTrn and needs a real
dataset + batch, so the guarded logic in ``validation_step`` is exercised in
isolation: a lightweight harness borrows the *real* ``VitsModel.validation_step``
function (so the actual guard code runs), while the heavy collaborators
(loss steps, forward synthesis, ``self.log``) are stubbed. This lets the three
guard branches be driven deterministically with no model, dataset, or GPU.
"""
import types

import torch

from phoonnx_train.vits.lightning import VitsModel


class _FakeUtt:
    """Minimal stand-in for a dataset utterance."""

    def __init__(self, phoneme_ids, text):
        self.phoneme_ids = torch.tensor(phoneme_ids, dtype=torch.long)
        self.text = text
        self.speaker_id = None
        self.d_vector = None
        self.language_id = None


class _Harness:
    """Borrows the real validation_step; stubs everything it depends on."""

    # the code under test, unmodified
    validation_step = VitsModel.validation_step

    def __init__(self, logger, dataset, log_audio_samples):
        self.logger = logger
        self._test_dataset = dataset
        self.device = torch.device("cpu")
        self.hparams = types.SimpleNamespace(
            sample_rate=22050,
            log_audio_samples=log_audio_samples,
        )
        self.synth_count = 0
        self.logged_val_loss = None

    # stubbed heavy collaborators
    def training_step_g(self, batch):
        return torch.tensor(1.0)

    def training_step_d(self, batch):
        return torch.tensor(0.5)

    def log(self, name, value):
        if name == "val_loss":
            self.logged_val_loss = value

    def __call__(self, text, text_lengths, scales, sid=None,
                 speaker_embedding=None, lid=None):
        # stand-in for generator inference; a non-zero tensor so the loudness
        # scaling in validation_step is exercised without a div-by-zero
        self.synth_count += 1
        return torch.tensor([[[0.1, -0.2, 0.3, -0.4]]])


class _RecordingExperiment:
    """A logger.experiment that supports add_audio (like TensorBoard)."""

    def __init__(self):
        self.calls = []

    def add_audio(self, tag, audio, sample_rate):
        self.calls.append((tag, sample_rate))


class _RecordingLogger:
    def __init__(self):
        self.experiment = _RecordingExperiment()


class _NoAudioLogger:
    """A logger whose experiment has no add_audio (like CSVLogger)."""

    def __init__(self):
        self.experiment = types.SimpleNamespace()  # deliberately no add_audio


def _dataset():
    return [_FakeUtt([1, 2, 3], "hello"), _FakeUtt([4, 5], "world")]


def test_disabled_does_not_synthesize_or_log():
    """Default (flag off): no synthesis, no add_audio, even at batch_idx 0."""
    logger = _RecordingLogger()
    h = _Harness(logger, _dataset(), log_audio_samples=False)

    out = h.validation_step(batch=object(), batch_idx=0)

    assert h.synth_count == 0  # zero wasted generator inference
    assert logger.experiment.calls == []
    assert h.logged_val_loss is not None  # normal path still ran
    assert float(out) == 1.5


def test_enabled_without_add_audio_does_not_raise_or_log():
    """Flag on but logger lacks add_audio (CSVLogger): guard skips the loop."""
    logger = _NoAudioLogger()
    h = _Harness(logger, _dataset(), log_audio_samples=True)

    # must not raise AttributeError despite add_audio being absent
    out = h.validation_step(batch=object(), batch_idx=0)

    assert h.synth_count == 0  # nothing synthesized when it cannot be logged
    assert float(out) == 1.5


def test_enabled_with_add_audio_logs_once_per_epoch():
    """Flag on + TensorBoard-like logger: add_audio called for each utt."""
    logger = _RecordingLogger()
    dataset = _dataset()
    h = _Harness(logger, dataset, log_audio_samples=True)

    h.validation_step(batch=object(), batch_idx=0)

    assert h.synth_count == len(dataset)
    tags = [c[0] for c in logger.experiment.calls]
    assert tags == ["hello", "world"]
    assert all(c[1] == 22050 for c in logger.experiment.calls)


def test_enabled_skips_non_first_batch():
    """Only batch_idx == 0 logs; later validation batches must not."""
    logger = _RecordingLogger()
    h = _Harness(logger, _dataset(), log_audio_samples=True)

    h.validation_step(batch=object(), batch_idx=1)

    assert h.synth_count == 0
    assert logger.experiment.calls == []
