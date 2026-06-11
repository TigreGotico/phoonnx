"""ARPABET → IPA conversion (scriptconv.notation)."""

from scriptconv.notation import arpa_to_ipa, _ARPA_TO_IPA as arpa_to_ipa_lookup

__all__ = ["arpa_to_ipa", "arpa_to_ipa_lookup"]
