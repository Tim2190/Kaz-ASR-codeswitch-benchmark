#!/usr/bin/env python3
"""Word-level error analysis on the barbarism clips.

For the clips flagged ``has_barbarisms=1``, this splits every reference word into
two buckets — the Russian insertion (barbarism) itself vs. the surrounding
Kazakh words — and reports each system's error rate on each bucket. It answers
"do the errors land on the borrowed Russian word, or on the Kazakh around it?".

    python scripts/barbarism_analysis.py

NOTE: the set of barbarism tokens below is hand-annotated from the normalized
references (which word is the Russian insertion is a judgement call, e.g. the
conjunction "и"). Edit ``BARBARISM_WORDS`` to refine it; the aggregate picture
(errors concentrate on the Russian words for every system) is robust to small
changes, but exact percentages will shift.
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jiwer

from benchmark.normalize import normalize_for_wer

# Hand-annotated Russian insertions per clip, in normalized-layer spelling.
BARBARISM_WORDS = {
    "kzaudio_1": {"вложение", "вложениены", "конечно", "и", "в", "общем"},
    "kzaudio_2": {"короче", "тупой", "и"},
    "kzaudio_3": {"специально"},
    "kzaudio_4": {"специально"},
    "kzaudio_5": {"ощущение"},
    "kzaudio_6": {"в", "общем", "званием", "звание"},
    "kzaudio_10": {"так", "стоп"},
    "kzaudio_12": {"самый", "потолок", "сестрасы"},
    "kzaudio_14": {"короче"},
    "kzaudio_16": {"просто", "недостойномын"},
    "kzaudio_25": {"интерес", "спонсор"},
}

SERVICES = {
    "fine-tuned": "results/predictions_hf_shyngys879_kazakh-whisper-large-v3-turbo.jsonl",
    "yandex": "results/predictions_yandex_stt.jsonl",
    "gemini": "results/predictions_gemini_gemini-2.5-flash.jsonl",
    "base-whisper": "results/predictions_whisper_local_large-v3.jsonl",
    "google": "results/predictions_google_stt.jsonl",
}


def load_predictions(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                d[r["audio_id"]] = r.get("raw_output", "")
    return d


def correct_flags(ref_tokens, hyp_tokens):
    """Return a bool per reference token: was it recognized correctly (a 'hit')?"""
    out = jiwer.process_words(" ".join(ref_tokens), " ".join(hyp_tokens))
    flags = [False] * len(ref_tokens)
    for chunk in out.alignments[0]:
        if chunk.type == "equal":
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                flags[i] = True
    return flags


def main():
    meta = {r["audio_id"]: r for r in csv.DictReader(open("metadata.csv", encoding="utf-8"))}

    print(f"{'system':<14}{'err % on Russian':>18}{'err % on Kazakh':>18}")
    for name, path in SERVICES.items():
        if not os.path.exists(path):
            continue
        preds = load_predictions(path)
        b_hit = b_tot = k_hit = k_tot = 0
        for aid, barb in BARBARISM_WORDS.items():
            ref = normalize_for_wer(meta[aid]["transcript_normalized_written"], True).split()
            hyp = normalize_for_wer(preds.get(aid, ""), True).split()
            for tok, ok in zip(ref, correct_flags(ref, hyp)):
                if tok in barb:
                    b_tot += 1
                    b_hit += ok
                else:
                    k_tot += 1
                    k_hit += ok
        b_err = (1 - b_hit / b_tot) * 100 if b_tot else 0
        k_err = (1 - k_hit / k_tot) * 100 if k_tot else 0
        print(f"{name:<14}{b_err:>17.0f}%{k_err:>17.0f}%")

    print("\n(For most systems errors fall harder on the Russian insertions "
          "— catastrophically so for Google — but not universally: Gemini errs "
          "more on the Kazakh and the fine-tune is roughly even. See README.)")


if __name__ == "__main__":
    main()
