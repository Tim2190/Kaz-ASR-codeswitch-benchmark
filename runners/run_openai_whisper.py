#!/usr/bin/env python3
"""Transcribe the benchmark clips with the OpenAI audio API.

Requires the ``openai`` package and an ``OPENAI_API_KEY`` environment variable.

    pip install openai
    export OPENAI_API_KEY=sk-...
    python runners/run_openai_whisper.py --model whisper-1

``--model`` accepts any OpenAI transcription model, e.g. ``whisper-1``,
``gpt-4o-transcribe``, or ``gpt-4o-mini-transcribe``. The service name written
to the predictions file is ``openai_<model>`` so different models don't collide.
"""

import argparse
import os
import sys

from common import (
    add_common_args, iter_clips, write_predictions,
    load_audio_16k_mono, pcm16_wav_bytes,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--model", default="whisper-1")
    ap.add_argument("--language", default="kk", help="ISO-639-1 hint (kk = Kazakh).")
    ap.add_argument("--service-name", default=None)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("Install the SDK first:  pip install openai")

    client = OpenAI()
    service = args.service_name or f"openai_{args.model}"

    records = []
    clips = list(iter_clips(args.metadata, args.audio_dir))
    if args.limit:
        clips = clips[: args.limit]

    for i, clip in enumerate(clips, 1):
        # Re-encode to 16 kHz mono PCM WAV for a consistent, small upload.
        samples, sr = load_audio_16k_mono(clip.path)
        wav_bytes = pcm16_wav_bytes(samples, sr)
        import io
        buf = io.BytesIO(wav_bytes)
        buf.name = f"{clip.audio_id}.wav"
        try:
            resp = client.audio.transcriptions.create(
                model=args.model,
                file=buf,
                language=args.language,
                response_format="text",
            )
            text = resp if isinstance(resp, str) else getattr(resp, "text", "")
        except Exception as e:  # noqa: BLE001 - record the failure, keep going
            text = ""
            print(f"[{clip.audio_id}] ERROR: {e}", file=sys.stderr)
        records.append({"audio_id": clip.audio_id, "raw_output": text})
        print(f"({i}/{len(clips)}) {clip.audio_id}: {text[:60]!r}")

    path = write_predictions(service, records, args.results_dir)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
