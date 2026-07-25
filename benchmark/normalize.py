"""Text normalization for WER/CER comparison.

The normalization here is deliberately light: it removes the annotation tags
defined in ``annotation_methodology.md`` and applies the standard
case-folding / punctuation-stripping that any WER computation needs so that
scoring reflects lexical recognition rather than casing or punctuation
conventions. It intentionally does NOT do morphological normalization — that
distinction is already captured by the two reference layers (verbatim vs
normalized) stored in ``metadata.csv``.
"""

import re
import unicodedata

# Inline annotation tags from the methodology (Section 3).
#   [false_start] : marks a truncated self-correction fragment. The fragment
#                   itself (e.g. "шы-") is real speech and is kept; only the
#                   literal tag token is removed.
#   [unclear]     : marks an unintelligible segment; clips containing it are
#                   excluded from metric aggregation (Section 3.2).
_TAG_RE = re.compile(r"\[(?:false_start|unclear)\]")

# Any remaining bracketed token, as a safety net for unforeseen tags.
_BRACKET_RE = re.compile(r"\[[^\]]*\]")

# Punctuation to drop. We keep letters, digits, whitespace and the intra-word
# hyphen is handled below. Everything else (.,!?;:"«»…—) is removed.
_PUNCT_RE = re.compile(r"[^\w\s-]", flags=re.UNICODE)

_WS_RE = re.compile(r"\s+")


def has_unclear(text: str) -> bool:
    """Return True if the reference contains an ``[unclear]`` segment."""
    return "[unclear]" in text


def strip_tags(text: str) -> str:
    """Remove annotation tags, leaving surrounding speech intact."""
    text = _TAG_RE.sub(" ", text)
    text = _BRACKET_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def normalize_for_wer(text: str, normalize_numbers: bool = False) -> str:
    """Normalize a transcript (reference or hypothesis) for WER/CER scoring.

    Steps: Unicode NFC → strip annotation tags → lowercase → (optionally) spell
    digits as Kazakh words → drop punctuation (keeping the trailing hyphen of
    truncated fragments) → collapse whitespace.

    ``normalize_numbers`` converts digit runs to spelled-out Kazakh cardinals so
    that a service emitting "40" is not penalized against a reference that spells
    "қырық". References contain no digits, so this only ever affects hypotheses.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = strip_tags(text)
    text = text.lower()
    if normalize_numbers:
        from .numbers import digits_to_words_kk
        text = digits_to_words_kk(text)
    # Remove punctuation but keep hyphens (they mark truncated fragments such
    # as "шы-"). A hyphen that is left dangling at a word boundary is harmless
    # for token comparison.
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text
