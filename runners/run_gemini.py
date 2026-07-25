#!/usr/bin/env python3
"""Transcribe the benchmark clips with Google's Gemini API.

Gemini is a multimodal LLM that accepts audio, so it works as an ASR service
with a plain Gemini API key (the kind from Google AI Studio) — this is a
*different* product from Google Cloud Speech-to-Text (`run_google_stt.py`),
which needs a GCP service account.

    pip install google-genai soundfile scipy
    export GEMINI_API_KEY=...            # or GOOGLE_API_KEY
    python runners/run_gemini.py --model gemini-2.5-flash

Because it is an LLM, output is prompt-sensitive; the prompt below asks for a
verbatim, code-switching-preserving transcript with no translation or
commentary. Service name: ``gemini_<model>``.
"""

import argparse
import os
import re
import sys
import time

from common import (
    add_common_args, iter_clips, write_predictions,
    load_audio_16k_mono, pcm16_wav_bytes,
)

PROMPT = (
    "Transcribe this audio exactly as spoken, word for word. "
    "The speaker is talking in Kazakh and may mix in Russian words or phrases — "
    "keep every word in the language it was actually spoken, in its original "
    "script (Cyrillic). Do not translate, do not paraphrase, do not add or remove "
    "words, do not add punctuation, and do not add commentary. Output only the "
    "raw transcription text."
)

_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?[:\s]+['\"]?(\d+(?:\.\d+)?)")


def _suggested_delay(err: Exception, default: float) -> float:
    """Pull the server's suggested retry delay out of a 429 error message."""
    m = _RETRY_DELAY_RE.search(str(err))
    return float(m.group(1)) + 1.0 if m else default


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--service-name", default=None)
    ap.add_argument("--rpm", type=float, default=5.0,
                    help="Requests-per-minute throttle to respect the free-tier "
                         "quota (gemini-2.5-flash free tier = 5 RPM). Set higher "
                         "on a paid tier.")
    ap.add_argument("--max-retries", type=int, default=6,
                    help="Retries per clip on a 429 (rate-limit) response.")
    args = ap.parse_args()

    interval = 60.0 / args.rpm if args.rpm > 0 else 0.0

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("Set GEMINI_API_KEY (or GOOGLE_API_KEY).")

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        sys.exit("Install the SDK first:  pip install google-genai")

    client = genai.Client(api_key=api_key)
    service = args.service_name or f"gemini_{args.model}".replace("/", "_")

    clips = list(iter_clips(args.metadata, args.audio_dir))
    if args.limit:
        clips = clips[: args.limit]

    records = []
    for i, clip in enumerate(clips, 1):
        samples, sr = load_audio_16k_mono(clip.path)
        wav_bytes = pcm16_wav_bytes(samples, sr)
        audio_part = types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")

        text = ""
        for attempt in range(args.max_retries + 1):
            if interval and (i > 1 or attempt > 0):
                time.sleep(interval)  # base throttle to stay under the RPM cap
            try:
                resp = client.models.generate_content(
                    model=args.model, contents=[audio_part, PROMPT],
                )
                text = (resp.text or "").strip()
                break
            except Exception as e:  # noqa: BLE001
                is_rate = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                if is_rate and attempt < args.max_retries:
                    delay = _suggested_delay(e, default=interval or 20.0)
                    print(f"[{clip.audio_id}] rate-limited, waiting {delay:.0f}s "
                          f"(retry {attempt + 1}/{args.max_retries})", file=sys.stderr)
                    time.sleep(delay)
                    continue
                print(f"[{clip.audio_id}] ERROR: {e}", file=sys.stderr)
                break
        records.append({"audio_id": clip.audio_id, "raw_output": text})
        print(f"({i}/{len(clips)}) {clip.audio_id}: {text[:60]!r}")

    path = write_predictions(service, records, args.results_dir)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
