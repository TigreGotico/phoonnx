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


def test_from_path_skips_malformed_line_and_warns(tmp_path):
    from unittest.mock import patch

    spellings_file = tmp_path / "phonetic_spellings.txt"
    spellings_file.write_text(
        "foo:fuh\n"
        "this line has no colon\n"
        "bar:bahr\n"
    )
    with patch("phoonnx.voice.LOG") as mock_log:
        ps = PhoneticSpellings.from_path(str(spellings_file))
    assert ps.replacements == {"foo": "fuh", "bar": "bahr"}
    assert mock_log.warning.called
    assert "malformed" in str(mock_log.warning.call_args).lower()


# -- user spellings directory ------------------------------------------------------------

def test_user_spellings_are_applied_over_the_packaged_ones(tmp_path):
    packaged = PhoneticSpellings.from_lang("en-US").replacements
    assert packaged, "the packaged English file is not empty"
    word = next(iter(packaged))
    (tmp_path / "en.txt").write_text(f"{word}: my spelling\nThalovant: Talovant\n")
    ps = PhoneticSpellings.from_lang("en-US", user_dir=str(tmp_path))
    assert ps.replacements[word] == "my spelling"
    assert ps.replacements["Thalovant"] == "Talovant"
    assert set(packaged) <= set(ps.replacements), "the rest of the packaged file is kept"
    assert ps.apply("Say Thalovant now") == "Say Talovant now"


def test_the_full_tag_overrides_the_primary_language(tmp_path):
    (tmp_path / "en.txt").write_text("tomato: generic\nfoo: only primary\n")
    (tmp_path / "en-US.txt").write_text("tomato: american\n")
    ps = PhoneticSpellings.from_lang("en-US", user_dir=str(tmp_path))
    assert ps.replacements["tomato"] == "american"
    assert ps.replacements["foo"] == "only primary"
    assert PhoneticSpellings.from_lang("en-GB", user_dir=str(tmp_path)).replacements["tomato"] == "generic"


def test_a_language_with_only_a_user_file_still_gets_spellings(tmp_path):
    (tmp_path / "xx.txt").write_text("hello: bonjour\n")
    ps = PhoneticSpellings.from_lang("xx-YY", user_dir=str(tmp_path))
    assert ps.replacements == {"hello": "bonjour"}


def test_the_environment_variable_points_at_the_directory(tmp_path, monkeypatch):
    (tmp_path / "en.txt").write_text("zzword: zed word\n")
    monkeypatch.setenv("PHOONNX_PHONETIC_SPELLINGS_DIR", str(tmp_path))
    assert PhoneticSpellings.from_lang("en-US").replacements["zzword"] == "zed word"
    monkeypatch.delenv("PHOONNX_PHONETIC_SPELLINGS_DIR")
    assert "zzword" not in PhoneticSpellings.from_lang("en-US").replacements


def test_a_language_with_nothing_anywhere_is_still_not_found(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        PhoneticSpellings.from_lang("xx-YY", user_dir=str(tmp_path))


def test_a_user_entry_overrides_a_packaged_entry_that_differs_only_in_case(tmp_path):
    assert "OpenVoiceOS" in PhoneticSpellings.from_lang("en-US").replacements
    (tmp_path / "en.txt").write_text("openvoiceos: user wins\n")
    ps = PhoneticSpellings.from_lang("en-US", user_dir=str(tmp_path))
    assert ps.apply("I use OpenVoiceOS daily") == "I use user wins daily"
    assert "OpenVoiceOS" not in ps.replacements


def test_an_undecodable_user_file_is_skipped_and_the_packaged_spellings_stay(tmp_path):
    from unittest.mock import patch

    (tmp_path / "en.txt").write_bytes(b"caf\xe9: kafay\n")
    with patch("phoonnx.voice.LOG") as mock_log:
        ps = PhoneticSpellings.from_lang("en-US", user_dir=str(tmp_path))
    assert ps.replacements == PhoneticSpellings.from_lang("en-US").replacements
    assert mock_log.warning.called
    assert "en.txt" in str(mock_log.warning.call_args)


def test_a_user_file_saved_with_a_byte_order_mark_still_overrides(tmp_path):
    (tmp_path / "en.txt").write_bytes("zzword: zed word\n".encode("utf-8-sig"))
    ps = PhoneticSpellings.from_lang("en-US", user_dir=str(tmp_path))
    assert ps.apply("say zzword") == "say zed word"


def test_a_user_entry_removes_only_the_entries_the_case_insensitive_match_would_hit(tmp_path):
    packaged = tmp_path / "packaged" / "de"
    packaged.mkdir(parents=True)
    (packaged / "phonetic_spellings.txt").write_text("Straße: Schtrasse\n")
    user = tmp_path / "user"
    user.mkdir()
    (user / "de.txt").write_text("STRASSE: Strasse\n")
    ps = PhoneticSpellings.from_lang("de", locale_path=str(tmp_path / "packaged"), user_dir=str(user))
    assert ps.replacements == {"Straße": "Schtrasse", "STRASSE": "Strasse"}
    assert ps.apply("die Straße") == "die Schtrasse"
