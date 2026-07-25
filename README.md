# Kazakh Code-Switching ASR Benchmark

A benchmark for evaluating commercial and open ASR services (OpenAI Whisper,
Google Speech-to-Text, Yandex SpeechKit, Hugging Face models, …) on **natural
Kazakh speech that code-switches with Russian** — the everyday Kazakh-Russian
mixing found in stand-up, interviews, and vlogs, not scripted read speech.

## Why this benchmark exists

Kazakh-Russian code-switching is essentially off the industry radar for speech:

- The closest methodological analogue, Perle AI's *"Benchmarking Commercial ASR
  Systems on Code-Switching Speech: Arabic, Persian, German"*, covers five
  providers across four language pairs — but **not** Kazakh-Russian.
- Gladia's 2026 industry survey of code-switching ASR does not mention the
  Kazakh-Russian pair at all.
- Existing Kazakh-Russian NLP resources (KazMMLU, Qorǵau, KazSAn-DRA, the
  kino.kz review corpus, MT corpora) are **text-only** — none target speech/ASR.
- Published Kazakh ASR baselines are read/clean-speech: base Whisper WER >40%
  (FLEURS) / >55% (KSC) without fine-tuning; a fine-tuned Whisper reaches
  WER 14.5% / CER 3.39%; wav2vec2.0 reaches WER 8.7% / CER 2.8%.

This benchmark measures how those systems hold up on **spontaneous,
code-switched** Kazakh — the setting where they are actually used and where they
are expected to struggle most.

## What's in the dataset

| | |
|---|---|
| Clips | 31 |
| Total audio | ~3.7 min (clips ≤ 20 s) |
| Sources | 2 YouTube channels (Qazaq StandUp; Dinara Satzhan), CC BY |
| Reference layers | 2 per clip (verbatim + normalized) |

Each clip in [`metadata.csv`](metadata.csv) carries **two independent reference
transcripts** and a set of binary phenomenon flags:

| Flag | Clips | Meaning |
|---|---:|---|
| `has_contraction` | 20 | word forms reshaped by fast/connected speech |
| `has_barbarisms` | 11 | Russian lexical insertion where a native Kazakh word exists |
| `has_propers` | 5 | proper nouns present |
| `has_dialect_slang` | 0 | dialectal / slang lexis (none in the current set) |

The full annotation scheme — the verbatim vs normalized distinction, the
`[false_start]` / `[unclear]` tags, and the normalization rules — is documented
in [`annotation_methodology.md`](annotation_methodology.md). Read it before
interpreting the numbers; the two-layer design is the core idea.

### Why two reference layers

A gap between ASR output and a single reference can come from two very different
places: a genuine recognition error, or a system's tendency to auto-normalize
morphology (e.g. restoring a contracted form). Scoring against **both** the
verbatim layer (what was literally said) and the normalized layer (dictionary
word forms) — and reporting the better of the two as `wer_best` — separates
these. See methodology §6.

## Metrics

`evaluate.py` computes, per `(clip, service)`:

- **WER** vs verbatim, vs normalized, and `wer_best = min(...)`
- **CER** vs verbatim, vs normalized, and `cer_best` — reported because for an
  agglutinative language one wrong suffix inflates WER far more than CER
- **BERTScore F1** (optional, `--bertscore`) vs the normalized layer — a
  semantic metric that does not punish correct-but-differently-spelled output.
  Perle AI show WER overstates the quality gap ~3× versus semantic scoring.

Aggregates are reported both **macro** (mean of per-clip rates) and **micro**
(pooled edits ÷ pooled reference length), overall and **per phenomenon flag**,
so you can read e.g. WER on barbarism-bearing clips separately.

## Repository layout

```
metadata.csv                 # 31 clips: 2 reference layers + flags + source links
annotation_methodology.md    # annotation scheme (read this first)
audio/                       # 31 WAV clips (see "Audio format" note below)
evaluate.py                  # scorer: predictions -> WER/CER/(BERTScore) + summary
benchmark/                   # scoring library (normalize, metrics, aggregation)
runners/                     # one script per ASR provider -> predictions_<svc>.jsonl
results/                     # generated predictions & score tables
requirements.txt             # core scoring deps
requirements-runners.txt     # per-provider runner deps
```

## Quickstart

The workflow is two steps: **run a provider** to get transcripts, then **score**
them. The scorer is model-agnostic — if you already have transcripts from any
system, you can skip straight to scoring.

### 1. Score existing predictions

```bash
pip install -r requirements.txt

python evaluate.py --predictions results/predictions_myservice.jsonl
```

A predictions file is one hypothesis per clip, as JSONL:

```json
{"audio_id": "kzaudio_1", "raw_output": "біз өзімізге ..."}
```

or a CSV with `audio_id,raw_output` columns. Pass several files at once to build
a leaderboard. Outputs land in `results/`: `scores_<service>.csv` (per clip) and
`summary.csv` (aggregates, overall + per flag).

### 2. Run an ASR provider

```bash
pip install -r requirements-runners.txt   # or just the SDK you need

# OpenAI audio API
export OPENAI_API_KEY=sk-...
python runners/run_openai_whisper.py --model whisper-1

# Google Cloud Speech-to-Text
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
python runners/run_google_stt.py --language kk-KZ

# Yandex SpeechKit
export YANDEX_API_KEY=...  YANDEX_FOLDER_ID=...
python runners/run_yandex_stt.py --language kk-KZ

# Local Whisper (CPU/GPU, no API key)
python runners/run_whisper_local.py --model large-v3 --device cpu

# Any Hugging Face ASR model, incl. the fine-tuned Kazakh Whisper
python runners/run_hf_transformers.py --model shyngys879/kazakh-whisper-large-v3-turbo
```

Each runner writes `results/predictions_<service>.jsonl`. Then score everything:

```bash
python evaluate.py --predictions results/predictions_*.jsonl
```

Add `--bertscore` to also compute the semantic metric (pulls a large
multilingual model; needs `bert-score` + network).

Use `--limit N` on any runner for a quick smoke test on the first N clips.

## Notes and limitations

- **Audio format.** The distributed clips in `audio/` are **48 kHz stereo**
  16-bit PCM, while `annotation_methodology.md` §6 specifies 16 kHz mono. The
  runners resample to 16 kHz mono on the fly, so this does not affect scoring;
  `runners/prep_audio.py` can also write a canonical 16 kHz mono copy to
  `audio_16k/`. (Reconcile the spec and the data before a formal release.)
- **Small set.** 31 clips is enough for a directional comparison, not for tight
  confidence intervals — treat differences between close services with caution.
- **`has_dialect_slang` is empty.** The flag is defined in the methodology but
  no current clip triggers it.
- **Number/formatting normalization** is intentionally minimal: a system that
  emits digits ("5") where the reference spells them out ("бес") is penalized.
  This is a known WER artifact; BERTScore is less sensitive to it.

## License & attribution

Code and annotations: see [`LICENSE`](LICENSE). Source audio is derived from
CC BY YouTube material; the originating channel and title for every clip are
recorded in the `audio_source_link` column of `metadata.csv` for attribution.
