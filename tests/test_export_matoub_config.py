"""The Matoub export recipe's config, checked against the model's own vocab.

The ONNX export itself needs torch and the weights, so it runs on a host. The
part that decides what the adapter is told is pure, and it is the part that can
be silently wrong, so it is tested here against the real ``vocab.json`` shipped
by ``agbalu/Matoub-82M`` at revision 3d00056f.
"""
import importlib.util
import inspect
import json
import pathlib
import tempfile
import unittest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "export_matoub",
    _ROOT / "scripts" / "conversion" / "styletts2" / "export_matoub.py")
export_matoub = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_matoub)

_VOCAB = json.loads(
    (_ROOT / "tests" / "fixtures" / "matoub_vocab.json").read_text(encoding="utf-8"))


class TestMatoubConfig(unittest.TestCase):
    def setUp(self):
        self.cfg = export_matoub.build_config(_VOCAB)

    def test_the_padding_is_stated_not_inferred(self):
        # The whole reason this recipe exists. The adapter would otherwise read
        # a single-row style as the plain StyleTTS2 lineage and pad one side.
        self.assertIs(self.cfg["engine_params"]["pad_both_ends"], True)

    def test_num_symbols_is_the_embedding_size_not_the_symbol_count(self):
        # 118 symbols are mapped over ids 0..177: 60 ids are Kokoro rows the
        # Kabyle front end never emits. len(vocab) would understate the
        # embedding by 60 rows.
        self.assertEqual(len(_VOCAB), 118)
        self.assertEqual(max(_VOCAB.values()), 177)
        self.assertEqual(self.cfg["num_symbols"], 178)
        self.assertNotEqual(self.cfg["num_symbols"], len(_VOCAB))

    def test_the_id_map_is_the_model_s_own(self):
        # Copied, never rebuilt from a symbol list: the ids are embedding rows.
        self.assertEqual(self.cfg["phoneme_id_map"], _VOCAB)
        self.assertEqual(self.cfg["phoneme_id_map"]["$"], 0)

    def test_it_names_the_matoub_route_not_africa_g2p(self):
        # Feeding this model africa-g2p's Kabyle output gives it symbols it has
        # no embedding row for.
        self.assertEqual(self.cfg["phoneme_type"], "matoub_kab")
        self.assertEqual(self.cfg["lang_code"], "kab")

    def test_the_audio_matches_the_model_config(self):
        self.assertEqual(self.cfg["audio"]["sample_rate"], 24000)
        self.assertEqual(self.cfg["engine"], "styletts2")

    def test_provenance_is_recorded(self):
        self.assertEqual(self.cfg["source_model"], "agbalu/Matoub-82M")
        self.assertEqual(self.cfg["source_revision"],
                         "3d00056f3663d4d1d364e9b12c21230685a0ba9a")

    def test_a_vocab_without_the_pad_symbol_is_refused(self):
        # The pad id cannot be guessed: the tokenizer wraps every sequence in
        # it, and a wrong pad is audible as an extra syllable.
        broken = {k: v for k, v in _VOCAB.items() if k != "$"}
        with self.assertRaises(ValueError):
            export_matoub.build_config(broken)

    def test_the_voice_index_row_is_not_committed_to_the_index(self):
        # It names an unpublished mirror. The row lives in the recipe until the
        # mirror exists, so the shipped index never offers a voice that cannot
        # load.
        from phoonnx.model_manager import TTSModelManager  # noqa: F401

        entry = export_matoub.VOICE_INDEX_ENTRY
        (voice_id,) = entry
        self.assertEqual(entry[voice_id]["phoneme_type"], "matoub_kab")
        self.assertEqual(entry[voice_id]["lang"], "kab")
        index = json.loads(
            (_ROOT / "phoonnx" / "voice_index" / "styletts2.json").read_text(
                encoding="utf-8"))
        self.assertNotIn(voice_id, index)


class TestArgumentFailsBeforeAnythingIsWritten(unittest.TestCase):
    """A bad first argument must cost nothing.

    ml-ops hit this on a real run: a repo id passed ``from_pretrained`` and
    then died in ``build_config`` on ``Path("agbalu/Matoub-82M")/"vocab.json"``,
    after ``model.onnx`` and ``style.bin`` were already written. The output
    directory was left half-populated beside a traceback.
    """

    def test_a_local_directory_resolves_to_itself(self):
        self.assertEqual(
            export_matoub.resolve_model_dir(str(_ROOT / "tests")),
            _ROOT / "tests")

    def test_a_directory_without_a_vocab_is_named_not_guessed(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(FileNotFoundError) as caught:
                export_matoub.load_vocab(empty)
            self.assertIn("vocab.json", str(caught.exception))

    def test_the_vocab_is_read_before_the_exporter_is_reached(self):
        # export() calls load_vocab first, so a bad argument raises before any
        # file is created. Proven by the ordering in the source rather than by
        # running the export, which needs torch and the weights: the config
        # build must precede every write and the model load.
        source = inspect.getsource(export_matoub.export)
        first_read = source.index("build_config(load_vocab(")
        for later in ("AutoModelForTextToWaveform.from_pretrained(",
                      "style.tofile(", "torch.onnx.export(", "write_text("):
            with self.subTest(step=later):
                self.assertLess(first_read, source.index(later),
                                f"{later} happens before the vocab is read")

    def test_the_usage_line_matches_what_is_accepted(self):
        # The old line said "<hf_model_dir_or_id>" while the code only handled
        # a directory. Both are handled now, and the line says so.
        self.assertIn("<model_dir_or_repo_id>", export_matoub.__doc__)


if __name__ == "__main__":
    unittest.main()
