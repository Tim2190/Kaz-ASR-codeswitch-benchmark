"""Kazakh code-switching ASR benchmark: scoring and evaluation utilities."""

from .normalize import normalize_for_wer, strip_tags, has_unclear
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
    "clip_metrics",
    "aggregate",
    "ReferenceLayers",
    "ClipScore",
]
