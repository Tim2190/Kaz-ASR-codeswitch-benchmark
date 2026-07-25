"""Kazakh code-switching ASR benchmark: scoring and evaluation utilities."""

from .normalize import normalize_for_wer, strip_tags, has_unclear
from .numbers import spell_integer_kk, digits_to_words_kk
from .scoring import (
    clip_metrics,
    aggregate,
    ReferenceLayers,
    ClipScore,
)

__all__ = [
    "normalize_for_wer",
    "strip_tags",
    "has_unclear",
    "spell_integer_kk",
    "digits_to_words_kk",
    "clip_metrics",
    "aggregate",
    "ReferenceLayers",
    "ClipScore",
]
