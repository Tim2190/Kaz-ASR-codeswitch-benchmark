#!/usr/bin/env python3
"""Render the benchmark leaderboard from results/summary.csv to a PNG.

    python scripts/plot_results.py            # -> results/leaderboard.png

Produces a two-panel horizontal bar chart: overall WER and CER per system
(subset = all), sorted best-first. Regenerate after adding a new service.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Display names + order preference is derived from the data (sorted by WER).
DISPLAY = {
    "hf_shyngys879_kazakh-whisper-large-v3-turbo": "Fine-tuned Kazakh\nWhisper",
    "yandex_stt": "Yandex SpeechKit",
    "gemini_gemini-2.5-flash": "Gemini 2.5 Flash",
    "whisper_local_large-v3": "Whisper large-v3\n(zero-shot)",
    "google_stt": "Google Cloud STT",
}


def load_overall(summary_path):
    rows = []
    with open(summary_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["subset"] == "all":
                rows.append((
                    DISPLAY.get(r["service"], r["service"]),
                    float(r["wer_best_macro"]) * 100,
                    float(r["cer_best_macro"]) * 100,
                ))
    rows.sort(key=lambda x: x[1], reverse=True)  # worst first -> best ends on top
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary.csv")
    ap.add_argument("--out", default="results/leaderboard.png")
    args = ap.parse_args()

    rows = load_overall(args.summary)
    names = [r[0] for r in rows]
    wer = [r[1] for r in rows]
    cer = [r[2] for r in rows]
    y = range(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    # Data is worst-first, so index 0 (bottom) is worst -> red, top is best -> green.
    colors = plt.cm.RdYlGn([i / max(1, len(names) - 1) for i in range(len(names))])

    for ax, vals, title in ((ax1, wer, "Word Error Rate (%)"),
                            (ax2, cer, "Character Error Rate (%)")):
        ax.barh(list(y), vals, color=colors)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        for i, v in zip(y, vals):
            ax.text(v + max(vals) * 0.01, i, f"{v:.1f}", va="center", fontsize=9)
        ax.set_xlim(0, max(vals) * 1.15)

    ax1.set_yticks(list(y))
    ax1.set_yticklabels(names, fontsize=9)
    fig.suptitle("Kazakh code-switching ASR — lower is better (31 clips)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
