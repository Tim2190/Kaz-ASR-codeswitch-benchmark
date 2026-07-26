# Analysis & Findings

This note goes beyond the leaderboard in [`README.md`](README.md) and reports
three deeper analyses of the 31-clip Kazakh code-switching benchmark: how the
systems treat spontaneous speech, where their errors actually fall on
code-switched words, and how the reference annotation was quality-controlled.

All numbers use `wer_best`/`cer_best` with number normalization, computed with
the harness in this repository (`evaluate.py`, `scripts/`).

## 1. Do the systems transcribe literally, or "clean up" the speech?

The dataset carries two reference layers: **verbatim** (exactly as spoken, with
reductions and contractions) and **normalized** (dictionary word forms). Scoring
each hypothesis against both reveals a system's *style*.

| System | WER vs verbatim | WER vs normalized | gap | clips closer to norm / verb / tie |
|---|---:|---:|---:|---:|
| Fine-tuned Kazakh Whisper | 19.8% | 11.9% | +7.9 | 23 / 0 / 8 |
| Yandex SpeechKit | 24.3% | 17.8% | +6.5 | 19 / 3 / 9 |
| Gemini 2.5 Flash | 36.3% | 29.7% | +6.6 | 19 / 1 / 11 |
| Base Whisper large-v3 | 47.6% | 43.0% | +4.5 | 15 / 3 / 13 |
| Google Cloud STT | 66.0% | 65.3% | +0.7 | 8 / 4 / 19 |

**Every system scores lower against the normalized layer** — they output
canonical dictionary forms rather than the reduced forms actually spoken (e.g. a
speaker says *боп*, the model writes *болып*). This is expected: ASR models are
trained largely on clean/written-style transcripts.

But the *degree* differs and separates two styles:
- **The fine-tuned model is an "editor"**: it normalizes aggressively (0 of 31
  clips are closer to the literal layer) — which is why it aligns so tightly to
  the normalized reference.
- **Yandex is a "stenographer"**: it more often keeps the literal spoken form
  (*боп*, *болам*, *бола*), so it diverges from the normalized layer more, while
  matching the verbatim layer.

**Methodological consequence:** scoring against a single reference would
mis-attribute 4–8 WER points of "error" to every system that is actually correct
normalization, not misrecognition. Reporting the better of the two layers
(`wer_best`) neutralizes this. Google's near-zero gap (+0.7) is *not* literal
transcription — it is simply so inaccurate that it is equally far from both
layers (19 of 31 clips are ties).

## 2. Where do the errors fall on code-switched speech?

The headline finding is that Russian insertions (barbarisms) raise WER. But WER
is clip-level — does the error land on the *Russian word* itself, or on the
Kazakh around it? A word-level alignment on the 11 barbarism clips separates the
two buckets (25 Russian insertion words vs. 169 Kazakh words):

| System | error on Russian insertions | error on Kazakh words |
|---|---:|---:|
| Fine-tuned Kazakh | 19% | 17% |
| Yandex | 22% | 15% |
| Gemini | 26% | 36% |
| Base Whisper | 52% | 47% |
| Google Cloud STT | **100%** | 66% |

For most systems the errors fall **harder on the Russian insertions** — and for
Google **catastrophically**: it mis-recognizes *every one* of the 25 Russian
words, turning them into unrelated tokens (*специально* → *мышеловки*,
*спонсор* → *телефон*). This is code-switching failure localized to the exact
switch point.

Two nuances the data forces:
- **It is not universal.** Gemini errs *more* on the Kazakh (36%) than on the
  Russian (26%) — its weakness is general Kazakh, not the switch. The fine-tuned
  model is roughly even (19% vs 17%).
- **The hardest case is the hybrid word** — a Russian root with a Kazakh suffix
  (*вложениены*, *недостойномын*). Even the strong systems miss these most,
  because the language switch happens *inside a single token*.

(Reproduce: `python scripts/barbarism_analysis.py`. The set of barbarism tokens
is hand-annotated; the aggregate picture is robust to small changes, exact
percentages will shift.)

## 3. Annotation quality control

The normalized layer was audited for consistency, not just eyeballed. Two checks
were used:

1. **Manual re-inspection** against the methodology's normalization rules.
2. **Model-assisted candidate flagging** — surfacing every reference word where
   the two strongest systems *independently agree* on a form that differs from
   the reference (folding away Yandex's і/и, қ/к orthographic quirk). Such
   agreement is a strong signal that the reference, not the models, is the
   outlier.

The audit flagged 11 raw candidates; on linguistic review, **3 were genuine
missed normalizations** and were corrected (verbatim layer left untouched):

| Clip | Was (normalized) | Corrected to | Rule |
|---|---|---|---|
| kzaudio_1, kzaudio_6 | вообщем | в общем | #3 (standardize borrowing spelling) |
| kzaudio_15 | берейн | берейін | #1 (restore reduced dictionary form) |
| kzaudio_9 | сүйтіп | сөйтіп | #1 (standard literary form) |

The remaining 8 candidates were cases where the reference was correct and the
models happened to share the same reduction or error — so they were left as-is.

Notably, these missed normalizations had **penalized the strongest models**,
which all output the standard form (*в общем*, *берейін*, *сөйтіп*). After the
fix the fine-tuned model improved from 13.0% to 11.9% WER. Out of ~500 words in
the normalized layer, only 3 needed correction — the annotation was already
consistent.

## Limitations

- **Scale.** 31 clips (~3.7 min) is a pilot. Rankings are clear, but per-flag
  subgroups are small (barbarisms n=11, proper nouns n=5), so those deltas are
  directional, not statistically tight.
- **Diversity.** Two sources and a handful of speakers; the `has_dialect_slang`
  flag is currently empty. Broader speaker/register coverage is the priority for
  a v1.0.
- **Barbarism token set** in the word-level analysis is hand-annotated and small.
