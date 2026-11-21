"""The default voice for a language must not depend on the cache's history.

``get_lang_voices`` breaks ties by registry order. A cache written by an older
release holds the voices it knew; ``merge_default_voices`` must still rank the
bundled voices in index order, as on a fresh install.
"""
import json

from phoonnx.model_manager import TTSModelManager

MMS_KAB = "facebook/mms-tts-kab-Amazigh"
OVOS_KAB = "OpenVoiceOS/phoonnx_kab-DZ_miro_o2i"
USER_VOICE = "someone/my-own-voice"


def _bundled_entry(voice_id):
    for index_file in TTSModelManager.voice_index_files():
        entries = json.loads(index_file.read_text())
        if voice_id in entries:
            return entries[voice_id]
    raise KeyError(voice_id)


def _manager(tmp_path, name, cache):
    path = tmp_path / name
    path.write_text(json.dumps(cache))
    manager = TTSModelManager(cache_path=str(path))
    manager.load()
    manager.merge_default_voices()
    return manager


def test_old_cache_ranks_bundled_voices_as_a_fresh_install(tmp_path):
    fresh = _manager(tmp_path, "fresh.json", {})
    user_entry = dict(_bundled_entry(MMS_KAB), voice_id=USER_VOICE)
    old = _manager(tmp_path, "old.json", {USER_VOICE: user_entry,
                                          MMS_KAB: _bundled_entry(MMS_KAB)})

    assert [v for v in old.voices if v != USER_VOICE] == list(fresh.voices)
    assert list(old.voices)[-1] == USER_VOICE


def test_old_cache_keeps_the_ovos_kabyle_voice_as_default(tmp_path):
    old = _manager(tmp_path, "old.json", {MMS_KAB: _bundled_entry(MMS_KAB)})

    assert old.get_lang_voices("kab")[0].voice_id == OVOS_KAB
    assert old.get_lang_voices("kab-DZ")[0].voice_id == OVOS_KAB


def test_bundled_entry_replaces_the_cached_copy(tmp_path):
    stale = dict(_bundled_entry(MMS_KAB), lang="xx")
    old = _manager(tmp_path, "old.json", {MMS_KAB: stale})

    assert old.voices[MMS_KAB].lang == "kab"
