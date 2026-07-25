"""Per-clip and aggregate ASR metrics against the two reference layers.

Metric design follows ``annotation_methodology.md`` Section 6: every hypothesis
is scored against both the *verbatim* and the *normalized* reference, and the
better (lower-error) of the two is reported as ``*_best``. Scoring against both
layers separates genuine recognition errors from a service's tendency to
auto-normalize morphology.

Both word-level (WER) and character-level (CER) error rates are produced.
CER is reported because, for an agglutinative language like Kazakh, a single
mis-recognized suffix inflates WER far more than it does CER — the same
observation Perle AI make about transliteration penalties.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable

import jiwer

from .normalize import normalize_for_wer, has_unclear


@dataclass
class ReferenceLayers:
    """The two reference transcripts for one clip."""

    verbatim: str
    normalized: str


@dataclass
class _EditCounts:
    """Raw edit-operation counts, used for micro-averaging across clips."""

    ref_len: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        return self.errors / self.ref_len if self.ref_len else 0.0

    def __iadd__(self, other: "_EditCounts") -> "_EditCounts":
        self.ref_len += other.ref_len
        self.substitutions += other.substitutions
        self.deletions += other.deletions
        self.insertions += other.insertions
        self.hits += other.hits
        return self


def _word_counts(reference: str, hypothesis: str) -> _EditCounts:
    out = jiwer.process_words(reference, hypothesis)
    ref_len = out.substitutions + out.deletions + out.hits  # == len(reference words)
    return _EditCounts(
        ref_len=ref_len,
        substitutions=out.substitutions,
        deletions=out.deletions,
        insertions=out.insertions,
        hits=out.hits,
    )


def _char_counts(reference: str, hypothesis: str) -> _EditCounts:
    out = jiwer.process_characters(reference, hypothesis)
    ref_len = out.substitutions + out.deletions + out.hits
    return _EditCounts(
        ref_len=ref_len,
        substitutions=out.substitutions,
        deletions=out.deletions,
        insertions=out.insertions,
        hits=out.hits,
    )


@dataclass
class ClipScore:
    """All metrics for one (clip, service) pair."""

    audio_id: str
    service: str
    raw_output: str
    excluded: bool = False  # True if [unclear] present -> dropped from aggregates

    wer_vs_verbatim: float = 0.0
    wer_vs_normalized: float = 0.0
    wer_best: float = 0.0
    cer_vs_verbatim: float = 0.0
    cer_vs_normalized: float = 0.0
    cer_best: float = 0.0

    # Which layer gave the best WER: "verbatim" | "normalized" | "tie"
    best_layer: str = "tie"

    # Retained for micro-averaging; not serialized to the per-clip CSV.
    _wer_counts_best: _EditCounts = field(default_factory=_EditCounts, repr=False)
    _cer_counts_best: _EditCounts = field(default_factory=_EditCounts, repr=False)

    def as_row(self) -> dict:
        row = {
            k: v
            for k, v in asdict(self).items()
            if not k.startswith("_")
        }
        return row


def clip_metrics(audio_id: str, service: str, raw_output: str,
                 refs: ReferenceLayers, normalize_numbers: bool = False) -> ClipScore:
    """Compute WER/CER of ``raw_output`` against both reference layers."""
    hyp = normalize_for_wer(raw_output, normalize_numbers)

    excluded = has_unclear(refs.verbatim) or has_unclear(refs.normalized)

    ref_v = normalize_for_wer(refs.verbatim, normalize_numbers)
    ref_n = normalize_for_wer(refs.normalized, normalize_numbers)

    wc_v = _word_counts(ref_v, hyp)
    wc_n = _word_counts(ref_n, hyp)
    cc_v = _char_counts(ref_v, hyp)
    cc_n = _char_counts(ref_n, hyp)

    wer_v, wer_n = wc_v.rate, wc_n.rate
    cer_v, cer_n = cc_v.rate, cc_n.rate

    if wer_v < wer_n:
        best_layer = "verbatim"
    elif wer_n < wer_v:
        best_layer = "normalized"
    else:
        best_layer = "tie"

    # For the pooled (micro) aggregate we keep the edit counts of whichever
    # layer minimizes the per-clip WER, mirroring the reported wer_best.
    wc_best = wc_v if wer_v <= wer_n else wc_n
    cc_best = cc_v if cer_v <= cer_n else cc_n

    return ClipScore(
        audio_id=audio_id,
        service=service,
        raw_output=raw_output,
        excluded=excluded,
        wer_vs_verbatim=round(wer_v, 4),
        wer_vs_normalized=round(wer_n, 4),
        wer_best=round(min(wer_v, wer_n), 4),
        cer_vs_verbatim=round(cer_v, 4),
        cer_vs_normalized=round(cer_n, 4),
        cer_best=round(min(cer_v, cer_n), 4),
        best_layer=best_layer,
        _wer_counts_best=wc_best,
        _cer_counts_best=cc_best,
    )


@dataclass
class Aggregate:
    """Summary metrics over a set of clips for one service and subset."""

    service: str
    subset: str
    n_clips: int
    # macro = mean of per-clip rates; micro = pooled edits / pooled ref length.
    wer_best_macro: float
    wer_best_micro: float
    cer_best_macro: float
    cer_best_micro: float
    wer_verbatim_macro: float
    wer_normalized_macro: float

    def as_row(self) -> dict:
        return asdict(self)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(scores: Iterable[ClipScore], service: str,
              subset: str = "all") -> Aggregate:
    """Aggregate per-clip scores into macro/micro summaries.

    Clips flagged ``excluded`` (an ``[unclear]`` segment) are dropped, per the
    methodology's rule that such segments leave comparison.
    """
    kept = [s for s in scores if not s.excluded]

    wer_best = [s.wer_best for s in kept]
    cer_best = [s.cer_best for s in kept]
    wer_v = [s.wer_vs_verbatim for s in kept]
    wer_n = [s.wer_vs_normalized for s in kept]

    micro_w = _EditCounts()
    micro_c = _EditCounts()
    for s in kept:
        micro_w += s._wer_counts_best
        micro_c += s._cer_counts_best

    return Aggregate(
        service=service,
        subset=subset,
        n_clips=len(kept),
        wer_best_macro=round(_mean(wer_best), 4),
        wer_best_micro=round(micro_w.rate, 4),
        cer_best_macro=round(_mean(cer_best), 4),
        cer_best_micro=round(micro_c.rate, 4),
        wer_verbatim_macro=round(_mean(wer_v), 4),
        wer_normalized_macro=round(_mean(wer_n), 4),
    )
