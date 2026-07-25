"""Optional semantic scoring with BERTScore.

WER over-penalizes recognition that is semantically correct but orthographically
different (the transliteration effect Perle AI describe, where WER inflates the
apparent quality gap ~3x). BERTScore against a multilingual model complements
WER by rewarding semantic agreement.

This is optional: it pulls a large multilingual model from Hugging Face, so it
is imported lazily and only when ``--bertscore`` is requested.

    pip install bert-score torch
"""

from __future__ import annotations

from typing import Sequence

# A multilingual model that covers both Kazakh (Cyrillic) and Russian.
DEFAULT_MODEL = "bert-base-multilingual-cased"


def bertscore_f1(hypotheses: Sequence[str], references: Sequence[str],
                 model_type: str = DEFAULT_MODEL,
                 lang: str = "kk") -> list[float]:
    """Return a per-item BERTScore F1 for each (hypothesis, reference) pair."""
    from bert_score import score  # lazy import

    _, _, f1 = score(
        list(hypotheses),
        list(references),
        model_type=model_type,
        lang=lang,
        verbose=False,
        rescale_with_baseline=False,
    )
    return [float(x) for x in f1]
