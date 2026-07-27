#!/usr/bin/env python3
"""Statistical analysis of the benchmark: bootstrap CIs, error profiles, and
per-clip difficulty correlation. Backs STATISTICAL-ANALYSIS.md.

    python scripts/significance.py

All numbers are reproducible (bootstrap seed = 42). Requires numpy + scipy.
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jiwer
from scipy.stats import spearmanr

from benchmark.normalize import normalize_for_wer
from benchmark.scoring import ReferenceLayers, clip_metrics

SEED = 42
N_BOOT = 10_000

SERVICES = [
    ("Fine-tuned", "results/predictions_hf_shyngys879_kazakh-whisper-large-v3-turbo.jsonl"),
    ("Yandex", "results/predictions_yandex_stt.jsonl"),
    ("Gemini", "results/predictions_gemini_gemini-2.5-flash.jsonl"),
    ("Base Whisper", "results/predictions_whisper_local_large-v3.jsonl"),
    ("Google", "results/predictions_google_stt.jsonl"),
]

# Kazakh-specific letters folded to their nearest Russian counterparts, to
# detect the "orthographic tic" (writing и/к for і/қ, etc.).
_FOLD = str.maketrans("іқғңөұүһә", "икгноууха")


def load_predictions(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                d[r["audio_id"]] = r.get("raw_output", "")
    return d


def main():
    rng = np.random.default_rng(SEED)
    meta = {
        r["audio_id"]: ReferenceLayers(r["transcript_verbatim"],
                                       r["transcript_normalized_written"])
        for r in csv.DictReader(open("metadata.csv", encoding="utf-8"))
    }
    ids = list(meta)
    names = [n for n, _ in SERVICES]
    preds = {n: load_predictions(p) for n, p in SERVICES}

    # Per-clip wer_best for each system.
    W = {
        n: np.array([clip_metrics(a, n, preds[n][a], meta[a],
                                  normalize_numbers=True).wer_best for a in ids])
        for n in names
    }

    # --- 1. Bootstrap confidence intervals -------------------------------
    print("=" * 64)
    print(f"1. Bootstrap 95% CI on WER ({N_BOOT} resamples, seed={SEED})")
    print("=" * 64)
    idx = rng.integers(0, len(ids), size=(N_BOOT, len(ids)))
    boot = {n: W[n][idx].mean(axis=1) for n in names}
    for n in names:
        lo, hi = np.percentile(boot[n], [2.5, 97.5])
        print(f"  {n:<13} WER={W[n].mean()*100:5.1f}%   95% CI [{lo*100:4.1f}, {hi*100:4.1f}]")

    print("\n  Pairwise differences (paired bootstrap; significant if CI excludes 0)")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = boot[names[i]] - boot[names[j]]
            lo, hi = np.percentile(d, [2.5, 97.5])
            sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s."
            print(f"    {names[i]:<13} vs {names[j]:<13}: "
                  f"delta={d.mean()*100:+5.1f}%  CI[{lo*100:+5.1f},{hi*100:+5.1f}]  {sig}")

    # --- 2. Error profiles ----------------------------------------------
    print("\n" + "=" * 64)
    print("2. Error profile (edit types on the closer reference layer)")
    print("=" * 64)

    def best_align(hyp, refs):
        h = normalize_for_wer(hyp, True)
        ov = jiwer.process_words(normalize_for_wer(refs.verbatim, True), h)
        on = jiwer.process_words(normalize_for_wer(refs.normalized, True), h)
        return ov if ov.wer <= on.wer else on

    print(f"  {'system':<13}{'sub%':>6}{'del%':>6}{'ins%':>6}{'orth.tic':>10}")
    for n in names:
        S = D = I = quirk = 0
        for a in ids:
            o = best_align(preds[n][a], meta[a])
            S += o.substitutions
            D += o.deletions
            I += o.insertions
            ref = " ".join(o.references[0]).split()
            hyp = " ".join(o.hypotheses[0]).split()
            for ch in o.alignments[0]:
                if ch.type == "substitute":
                    for r_i, h_i in zip(range(ch.ref_start_idx, ch.ref_end_idx),
                                        range(ch.hyp_start_idx, ch.hyp_end_idx)):
                        if ref[r_i].translate(_FOLD) == hyp[h_i].translate(_FOLD):
                            quirk += 1
        tot = S + D + I
        print(f"  {n:<13}{S/tot*100:>5.0f}%{D/tot*100:>5.0f}%{I/tot*100:>5.0f}%"
              f"{quirk/max(S,1)*100:>8.0f}% of subs")

    # --- 3. Difficulty correlation --------------------------------------
    print("\n" + "=" * 64)
    print("3. Per-clip difficulty correlation (Spearman rho)")
    print("=" * 64)
    M = np.array([W[n] for n in names])
    print("        " + " ".join(f"{n[:6]:>7}" for n in names))
    for i, n in enumerate(names):
        row = " ".join(f"{spearmanr(M[i], M[j]).correlation:>7.2f}" for j in range(len(names)))
        print(f"  {n[:6]:<6}" + row)
    mean_w = M.mean(axis=0)
    hardest = [ids[k] for k in np.argsort(-mean_w)[:5]]
    print("\n  Hardest for everyone (highest mean WER):", ", ".join(hardest))


if __name__ == "__main__":
    main()
