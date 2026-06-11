# ARPABET → IPA conversion — delegated to scriptconv.notation.
# scriptconv is the org-wide notation-transcoding library; all mapping tables
# live there so fixes apply across the stack.
#
# Vendored tables removed; backward-compat symbols re-exported so call sites
# that imported `arpa_to_ipa_lookup` or `arpa_to_ipa` keep working unchanged.

from scriptconv.notation import arpa_to_ipa, _ARPA_TO_IPA as arpa_to_ipa_lookup

__all__ = ["arpa_to_ipa", "arpa_to_ipa_lookup"]
