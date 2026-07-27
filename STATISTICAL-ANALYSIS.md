# Statistical Analysis

This note complements [`ANALYSIS.md`](ANALYSIS.md) with significance testing and
error-type profiling. It asks not just *which system is better*, but *how much of
the ranking is statistically real* given only 31 clips, and *what kind* of errors
each system makes. Reproduce with `python scripts/significance.py` (bootstrap
seed = 42).

## 1. How reliable are the numbers? (Bootstrap confidence intervals)

With only 31 clips, a single WER number can be misleading. To quantify the
uncertainty, the 31 clips were **resampled with replacement 10,000 times**
(bootstrap); each resample yields a mean WER, and the middle 95% of those means
is the confidence interval (CI).

| System | WER | 95% CI |
|---|---:|---|
| Fine-tuned Kazakh Whisper | 11.9% | [8.4, 15.6] |
| Yandex SpeechKit | 17.1% | [11.4, 23.8] |
| Gemini 2.5 Flash | 29.5% | [23.1, 36.2] |
| Base Whisper large-v3 | 42.5% | [36.1, 49.6] |
| Google Cloud STT | 64.9% | [59.1, 70.5] |

The intervals are wide (±6–7 points) — the honest consequence of a 31-clip pilot.

**Which differences are statistically significant** (paired bootstrap; a
difference is significant when its 95% CI excludes zero):

- **Fine-tuned vs Yandex: NOT significant** (Δ = −5.3%, CI [−11.8, +0.6]). On
  this data we *cannot* claim the dedicated fine-tune beats an off-the-shelf
  commercial API.
- **All nine other pairs: significant.**

So the systems fall into a statistically-grounded ordering: a top tier of
**{Fine-tuned, Yandex}** that are indistinguishable from each other, then
significant steps down to Gemini, Base Whisper, and Google. The most useful
takeaway is not "the fine-tune wins" but **"Yandex is statistically on par with
the dedicated fine-tune, and clearly ahead of everything else."**

## 2. What kind of errors does each system make?

Aligning each hypothesis to its closer reference layer and counting edit
operations gives an error *profile* (share of substitutions / deletions /
insertions), plus an "orthographic-tic" count — substitutions that differ from
the reference by a Kazakh-specific letter only (і↔и, қ↔к …).

| System | subs | dels | ins | orthographic tic |
|---|---:|---:|---:|---:|
| Fine-tuned | 66% | 23% | 11% | 2% of subs |
| Yandex | 68% | 19% | 12% | **8% of subs** |
| Gemini | 79% | 19% | 3% | 5% |
| Base Whisper | 82% | 16% | 2% | 4% |
| Google Cloud STT | 44% | **55%** | 1% | 1% |

Two signatures stand out:
- **Google mostly *omits* words** — 55% of its errors are deletions (it produced
  193 deletions vs. 155 substitutions). It does not just mis-hear; it drops
  roughly half the content. Combined with its 100% failure on Russian insertions
  (see `ANALYSIS.md`), this is a distinctive failure mode, not generic noise.
- **Yandex has a systematic orthographic tic** — ~8% of its substitutions differ
  from the reference by a Kazakh-only letter (writing и/к for і/қ). Its *true*
  recognition quality is therefore marginally better than its 17.1% WER suggests.

## 3. Is difficulty shared across systems?

Per-clip WER correlations (Spearman) between systems are **moderate (0.17–0.64)**,
not high — difficulty is partly intrinsic to a clip and partly system-specific.
Notably, **Google correlates weakly with the fine-tuned model (0.17)**: it fails
on different clips than the strong systems do. Gemini and Base Whisper correlate
most (0.64) — the two "generalist" models struggle in similar places.

The clips that are hard for *everyone* (highest mean WER) are the fast, mumbled,
heavily code-switched ones (`kzaudio_31`, `kzaudio_6`, `kzaudio_15`, `kzaudio_3`,
`kzaudio_16`).

## Reproduction & caveats

- `python scripts/significance.py` — bootstrap CIs, error profiles, correlation.
- Bootstrap: 10,000 resamples, seed = 42, paired design (same resampled clip
  indices across systems for pairwise tests).
- With n = 31, confidence intervals are inherently wide; results are a
  well-quantified *pilot*, not tight population estimates. Expanding the dataset
  would narrow every interval.
