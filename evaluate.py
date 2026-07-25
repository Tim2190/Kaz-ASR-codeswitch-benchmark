#!/usr/bin/env python3
"""Score ASR predictions against the Kazakh code-switching benchmark.

Usage
-----
    python evaluate.py \
        --metadata metadata.csv \
        --predictions results/predictions_openai.jsonl results/predictions_google.jsonl \
        --outdir results

Each predictions file lists one hypothesis per clip. Two formats are accepted:

  * JSONL: one object per line, ``{"audio_id": "kzaudio_1", "raw_output": "..."}``
  * CSV:   a header with at least ``audio_id`` and ``raw_output`` columns

The service name is taken from a ``service`` field/column if present, otherwise
from the file name (``predictions_<service>.jsonl`` -> ``<service>``).

Outputs (written to ``--outdir``):
  * ``scores_<service>.csv`` — per-clip metrics for each service
  * ``summary.csv``          — macro/micro aggregates, overall and per flag
  * a leaderboard printed to stdout
"""

import argparse
import csv
import json
import os
import sys
from collections import OrderedDict

from benchmark.scoring import ReferenceLayers, clip_metrics, aggregate

FLAG_COLUMNS = ["has_contraction", "has_dialect_slang", "has_barbarisms", "has_propers"]


def load_metadata(path):
    """Return OrderedDict[audio_id] -> {'refs': ReferenceLayers, 'flags': {...}}."""
    meta = OrderedDict()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aid = row["audio_id"].strip()
            meta[aid] = {
                "refs": ReferenceLayers(
                    verbatim=row["transcript_verbatim"],
                    normalized=row["transcript_normalized_written"],
                ),
                "flags": {c: row.get(c, "0").strip() == "1" for c in FLAG_COLUMNS},
            }
    return meta


def load_predictions(path):
    """Return (service_name, dict[audio_id] -> raw_output)."""
    base = os.path.basename(path)
    service_from_name = base
    for prefix in ("predictions_", "preds_"):
        if service_from_name.startswith(prefix):
            service_from_name = service_from_name[len(prefix):]
            break
    service_from_name = os.path.splitext(service_from_name)[0]

    preds = OrderedDict()
    service = None
    if path.endswith(".jsonl") or path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            content = f.read().strip()
        # Support both JSONL and a single JSON array.
        records = []
        if content.startswith("["):
            records = json.loads(content)
        else:
            records = [json.loads(line) for line in content.splitlines() if line.strip()]
        for rec in records:
            preds[str(rec["audio_id"]).strip()] = rec.get("raw_output", "") or ""
            service = service or rec.get("service")
    else:  # CSV/TSV
        delim = "\t" if path.endswith(".tsv") else ","
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=delim):
                preds[str(row["audio_id"]).strip()] = row.get("raw_output", "") or ""
                service = service or row.get("service")
    return (service or service_from_name), preds


def write_scores_csv(path, scores, bert_by_id=None):
    if not scores:
        return
    bert_by_id = bert_by_id or {}
    fields = list(scores[0].as_row().keys())
    if bert_by_id:
        fields.append("bertscore_f1")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in scores:
            row = s.as_row()
            if bert_by_id:
                row["bertscore_f1"] = round(bert_by_id.get(s.audio_id, float("nan")), 4)
            w.writerow(row)


def subsets_for(meta):
    """Yield (subset_name, predicate(audio_id)) pairs for aggregation."""
    yield "all", lambda aid: True
    for flag in FLAG_COLUMNS:
        yield f"{flag}=1", (lambda aid, fl=flag: meta[aid]["flags"][fl])
        yield f"{flag}=0", (lambda aid, fl=flag: not meta[aid]["flags"][fl])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", default="metadata.csv")
    ap.add_argument("--predictions", nargs="+", required=True,
                    help="One or more predictions files (jsonl/json/csv/tsv).")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--bertscore", action="store_true",
                    help="Also compute BERTScore F1 vs the normalized layer "
                         "(needs bert-score + a large HF model; optional).")
    ap.add_argument("--bertscore-model", default=None,
                    help="Override the BERTScore model_type.")
    args = ap.parse_args(argv)

    meta = load_metadata(args.metadata)
    os.makedirs(args.outdir, exist_ok=True)

    all_summaries = []
    leaderboard = []
    bert_scores_by_id = {}

    for pred_path in args.predictions:
        service, preds = load_predictions(pred_path)

        missing = [aid for aid in meta if aid not in preds]
        extra = [aid for aid in preds if aid not in meta]
        if missing:
            print(f"[{service}] WARNING: {len(missing)} clips missing a prediction "
                  f"(treated as empty output): {', '.join(missing[:5])}"
                  f"{'...' if len(missing) > 5 else ''}", file=sys.stderr)
        if extra:
            print(f"[{service}] WARNING: {len(extra)} predictions have no matching "
                  f"clip and are ignored: {', '.join(extra[:5])}"
                  f"{'...' if len(extra) > 5 else ''}", file=sys.stderr)

        scores = []
        for aid, info in meta.items():
            raw = preds.get(aid, "")
            scores.append(clip_metrics(aid, service, raw, info["refs"]))

        bert_by_id = {}
        if args.bertscore:
            from benchmark.normalize import normalize_for_wer
            from benchmark.semantic import bertscore_f1, DEFAULT_MODEL
            ids = [s.audio_id for s in scores if not s.excluded]
            hyps = [normalize_for_wer(preds.get(aid, "")) for aid in ids]
            refs = [normalize_for_wer(meta[aid]["refs"].normalized) for aid in ids]
            f1s = bertscore_f1(hyps, refs, model_type=args.bertscore_model or DEFAULT_MODEL)
            bert_by_id = dict(zip(ids, f1s))

        write_scores_csv(os.path.join(args.outdir, f"scores_{service}.csv"),
                         scores, bert_by_id)

        by_id = {s.audio_id: s for s in scores}
        bert_scores_by_id[service] = bert_by_id
        for subset_name, pred in subsets_for(meta):
            subset_scores = [by_id[aid] for aid in meta if pred(aid)]
            agg = aggregate(subset_scores, service=service, subset=subset_name)
            all_summaries.append(agg)
            if subset_name == "all":
                leaderboard.append(agg)

    # Write combined summary.
    summary_path = os.path.join(args.outdir, "summary.csv")
    if all_summaries:
        fields = list(all_summaries[0].as_row().keys())
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for a in all_summaries:
                w.writerow(a.as_row())

    # Print leaderboard (overall, sorted by wer_best_macro).
    def mean_bert(service):
        vals = list(bert_scores_by_id.get(service, {}).values())
        return sum(vals) / len(vals) if vals else None

    leaderboard.sort(key=lambda a: a.wer_best_macro)
    print("\n=== Leaderboard (overall, subset=all) ===")
    header = (f"{'service':<28} {'n':>3} {'WER_best':>9} {'WER_micro':>10} "
              f"{'CER_best':>9} {'CER_micro':>10}")
    if args.bertscore:
        header += f" {'BERT_F1':>8}"
    print(header)
    for a in leaderboard:
        line = (f"{a.service:<28} {a.n_clips:>3} {a.wer_best_macro:>9.3f} "
                f"{a.wer_best_micro:>10.3f} {a.cer_best_macro:>9.3f} "
                f"{a.cer_best_micro:>10.3f}")
        if args.bertscore:
            mb = mean_bert(a.service)
            line += f" {mb:>8.3f}" if mb is not None else f" {'-':>8}"
        print(line)
    print(f"\nWrote per-clip scores and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
