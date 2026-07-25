# Annotation Methodology

## 1. Purpose

This document describes the transcription and annotation principles applied to the audio clips used in this benchmark. It is intended for users of the dataset — researchers and developers who will use the data to evaluate automatic speech recognition (ASR) systems or for their own research. Its purpose is to ensure the annotation process is reproducible and that every field can be interpreted unambiguously.

## 2. Transcription Structure

Each audio clip is annotated with two independent text representations.

### 2.1. Verbatim Transcript

A literal phonetic record of what was said, with no grammatical, orthographic, or other corrections. It preserves:

- word order exactly as spoken;
- reduced, contracted, or phonetically incomplete word forms;
- lexical borrowings and foreign-language insertions in their original (non-standardized) spelling;
- interjections and discourse markers, including hesitation fillers ("аа", "ммм", and similar);
- cases where the speaker produced a word that exists in the language but differs from what was presumably intended in context (left uncorrected).

The verbatim layer serves as the reference for computing ASR quality metrics against the actually spoken signal.

### 2.2. Normalized Transcript

A text derived from the verbatim transcript with a limited, explicitly defined set of edits applied. Syntactic structure (word order, relations between words) is never altered under any circumstances: this layer reflects normalization at the level of individual word forms, not a full grammatical error correction (GEC) task at the sentence level.

Edits applied:

| # | Phenomenon | Normalization rule |
|---|---|---|
| 1 | A reduced/contracted word form that is not itself an independent lexical item | The full word form is restored |
| 2 | An utterance that matches an existing, independent word of the language (even if it likely does not match the speaker's communicative intent) | Left unchanged — kept as in the verbatim layer |
| 3 | Spelling of lexical borrowings | Standardized to the conventional spelling of the source language; the lexical item itself is not replaced with a target-language equivalent |
| 4 | Interjections and discourse markers | Left unchanged |

Criterion for rule #1: this edit is applied only when the spoken form cannot be identified as an independent lexical item of the language — that is, when it is purely a phonetically truncated variant of the standard word form. If the spoken form is itself an existing word, rule #2 applies instead, and the presumably intended word is never reconstructed. Such reconstruction would require interpreting the speaker's communicative intent, which falls outside the scope of morphological normalization and introduces irreducible subjectivity.

## 3. Annotation of Incomplete and Unintelligible Segments

The dataset uses two special-purpose tags for segments that cannot be transcribed under the standard rules above.

### 3.1. The `[false_start]` Tag

Used for speaker self-correction: a word form that is begun but not completed, immediately followed by the speaker producing the corrected version within the same utterance.

- In the verbatim layer, the incomplete fragment is transcribed phonetically up to the point of interruption, immediately followed by the tag. The fragment is not omitted, as it constitutes part of the acoustic signal that must be compared against ASR system output when computing metrics.
- In the normalized layer, the incomplete fragment together with the tag is removed entirely. No reconstruction or other interpretation of the fragment is performed.

### 3.2. The `[unclear]` Tag

Used when the content of a segment cannot be reliably determined by ear.

- The tag is placed in both transcription layers — verbatim and normalized — in place of text. The presumed content is not reconstructed in either layer.
- When computing text-comparison metrics (WER and similar), segments marked `[unclear]` are excluded from comparison in both layers.

## 4. Relationship Between the Two Layers

Importantly, whether the verbatim and normalized layers are identical cannot be inferred directly from the binary annotation flags (Section 5). Even when none of the flags listed in Section 5 are set, the normalized layer may still differ from verbatim due to the removal of `[false_start]` fragments (Section 3.1). Flags and inline tags are independent sources of divergence between the two layers and should not be conflated during data processing.

## 5. Annotation Flags

Each clip is annotated with a set of independent binary flags. The flags are not mutually exclusive: a clip may exhibit any combination of them simultaneously.

| Flag | Description |
|---|---|
| `has_contraction` | Reduced/contracted word forms are present (corresponds to rule #1 in Section 2.2) |
| `has_dialect_slang` | Dialectal or colloquial/slang lexical items are present |
| `has_barbarisms` | The Kazakh-language utterance contains lexical items from another language (typically Russian), regardless of the size of the insertion (a single word, a phrase, or a fragment of a sentence), provided the utterance as a whole remains Kazakh speech in a Kazakh-language context |
| `has_propers` | Proper nouns are present |

A clip for which all flags above are set to zero and which contains none of the tags described in Section 3 is defined as requiring no normalization; in this case the normalized layer is identical to the verbatim layer.

## 6. Data Format

**Audio:** WAV, 16 kHz sample rate, mono, 16-bit PCM, clip duration ≤10 seconds.

**Annotation table** (one row per clip):

```
id, audio_id, transcript_verbatim, has_contraction, has_dialect_slang,
has_barbarisms, has_propers, transcript_normalized_written, audio_source_link
```

The `audio_source_link` field identifies the source (channel/author, title of the material) for attribution purposes, in accordance with the terms of the Creative Commons (CC BY) license under which the source audio materials are distributed.

**ASR evaluation results table** (one row per clip × recognition service pair):

```
audio_id, service, raw_output, wer_vs_verbatim, wer_vs_normalized, wer_best
```

The use of two reference layers (verbatim and normalized) reflects the fact that a discrepancy between ASR output and the reference may not stem from a recognition error, but from a given system's tendency to automatically normalize the morphology of the recognized text. Computing the metric against both layers separately makes it possible to distinguish between these two sources of discrepancy.

## 7. Annotation Quality Control

Consistency in applying the rules set out in Sections 2–5 is verified through intra-annotator reliability checks: a subsample of clips is re-annotated by the same annotator after a time interval, and the results are compared. Persistent discrepancies between iterations are treated as grounds for refining the wording of this document.
