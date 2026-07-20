"""PhoneticSpellings.from_path parsing edge cases."""
import logging

from phoonnx.voice import PhoneticSpellings


def test_from_path_skips_blank_lines_comments_and_trailing_newline(tmp_path):
    spellings_file = tmp_path / "phonetic_spellings.txt"
    spellings_file.write_text(
        "foo:fuh\n"
        "\n"
        "# a comment line\n"
        "bar:bahr\n"
    )
    ps = PhoneticSpellings.from_path(str(spellings_file))
    assert ps.replacements == {"foo": "fuh", "bar": "bahr"}


def test_from_path_skips_malformed_line_and_warns(tmp_path, caplog):
    spellings_file = tmp_path / "phonetic_spellings.txt"
    spellings_file.write_text(
        "foo:fuh\n"
        "this line has no colon\n"
        "bar:bahr\n"
    )
    with caplog.at_level(logging.WARNING):
        ps = PhoneticSpellings.from_path(str(spellings_file))
    assert ps.replacements == {"foo": "fuh", "bar": "bahr"}
    assert any("malformed" in r.message.lower() for r in caplog.records)
