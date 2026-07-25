#!/usr/bin/env python3
"""Transcribe the benchmark clips with Google Cloud Speech-to-Text (v1 sync).

    pip install google-cloud-speech soundfile scipy
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
    python runners/run_google_stt.py --language kk-KZ

Clips are ≤20 s, so the synchronous ``recognize`` endpoint is used. Audio is
re-encoded to 16 kHz mono LINEAR16 in-memory. Service name: ``google_stt``.
"""

import argparse
import sys

from common import (
    add_common_args, iter_clips, write_predictions,
    load_audio_16k_mono, pcm16_wav_bytes,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--language", default="kk-KZ")
    ap.add_argument("--model", default="default",
                    help="Google recognition model (e.g. default, latest_long).")
    ap.add_argument("--service-name", default="google_stt")
    args = ap.parse_args()

    try:
        from google.cloud import speech
    except ImportError:
        sys.exit("Install the SDK first:  pip install google-cloud-speech")

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        audio_channel_count=1,
        language_code=args.language,
        model=args.model,
        enable_automatic_punctuation=False,
    )

    clips = list(iter_clips(args.metadata, args.audio_dir))
    if args.limit:
        clips = clips[: args.limit]

    records = []
    for i, clip in enumerate(clips, 1):
        try:
            samples, sr = load_audio_16k_mono(clip.path)
            content = pcm16_wav_bytes(samples, sr)
            audio = speech.RecognitionAudio(content=content)
            resp = client.recognize(config=config, audio=audio)
            text = " ".join(
                r.alternatives[0].transcript for r in resp.results if r.alternatives
            ).strip()
        except Exception as e:  # noqa: BLE001
            text = ""
            print(f"[{clip.audio_id}] ERROR: {e}", file=sys.stderr)
        records.append({"audio_id": clip.audio_id, "raw_output": text})
        print(f"({i}/{len(clips)}) {clip.audio_id}: {text[:60]!r}")

    path = write_predictions(args.service_name, records, args.results_dir)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
