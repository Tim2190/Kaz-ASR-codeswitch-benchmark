#!/usr/bin/env python3
"""Transcribe the benchmark clips with Yandex SpeechKit (STT v1, short audio).

    pip install requests soundfile scipy
    export YANDEX_API_KEY=...           # a service-account API key
    export YANDEX_FOLDER_ID=...         # your cloud folder id
    python runners/run_yandex_stt.py --language kk-KZ

Uses the synchronous short-audio REST endpoint (clips are ≤20 s / well under the
1 MB & 30 s limits). Audio is sent as 16 kHz mono LPCM. Service name: ``yandex_stt``.
"""

import argparse
import os
import sys
import time

from common import (
    add_common_args, iter_clips, write_predictions, read_predictions_map,
    load_audio_16k_mono,
)

ENDPOINT = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--language", default="kk-KZ")
    ap.add_argument("--service-name", default="yandex_stt")
    ap.add_argument("--max-retries", type=int, default=5,
                    help="Retries per clip on a transient error (intermittent "
                         "401 / SSL drops seen on the free tier).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-transcribe every clip, ignoring any saved output.")
    args = ap.parse_args()

    api_key = os.environ.get("YANDEX_API_KEY")
    folder_id = os.environ.get("YANDEX_FOLDER_ID")
    if not api_key:
        sys.exit("YANDEX_API_KEY is not set.")

    try:
        import requests
        import numpy as np
    except ImportError:
        sys.exit("Install deps first:  pip install requests numpy")

    headers = {"Authorization": f"Api-Key {api_key}"}
    params = {"lang": args.language, "format": "lpcm", "sampleRateHertz": "16000"}
    if folder_id:
        params["folderId"] = folder_id

    clips = list(iter_clips(args.metadata, args.audio_dir))
    if args.limit:
        clips = clips[: args.limit]
    clip_order = [c.audio_id for c in clips]

    # Resume: keep any already-transcribed clips so a re-run only fills the gaps
    # (the free tier intermittently 401s, so a second pass usually completes it).
    done = {} if args.overwrite else read_predictions_map(args.service_name, args.results_dir)
    results = dict(done)
    if done:
        got = sum(1 for v in done.values() if v)
        print(f"Resuming: {got} clips already done, {len(clip_order) - got} to go.")

    def flush():
        ordered = [{"audio_id": aid, "raw_output": results.get(aid, "")}
                   for aid in clip_order]
        write_predictions(args.service_name, ordered, args.results_dir)

    for i, clip in enumerate(clips, 1):
        if results.get(clip.audio_id):
            print(f"({i}/{len(clips)}) {clip.audio_id}: [skip, already done]")
            continue

        samples, _ = load_audio_16k_mono(clip.path)
        lpcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

        text = ""
        for attempt in range(args.max_retries + 1):
            try:
                resp = requests.post(ENDPOINT, params=params, headers=headers,
                                     data=lpcm, timeout=60)
                resp.raise_for_status()
                text = resp.json().get("result", "").strip()
                break
            except Exception as e:  # noqa: BLE001
                if attempt < args.max_retries:
                    delay = 2 ** attempt  # 1, 2, 4, 8, 16s backoff
                    print(f"[{clip.audio_id}] {type(e).__name__}, retry "
                          f"{attempt + 1}/{args.max_retries} in {delay}s",
                          file=sys.stderr)
                    time.sleep(delay)
                    continue
                print(f"[{clip.audio_id}] ERROR: {e}", file=sys.stderr)

        if text:
            results[clip.audio_id] = text
            flush()  # crash-safe: save after every successful clip
        print(f"({i}/{len(clips)}) {clip.audio_id}: {text[:60]!r}")

    flush()
    path = write_predictions(args.service_name,
                             [{"audio_id": aid, "raw_output": results.get(aid, "")}
                              for aid in clip_order], args.results_dir)
    missing = [aid for aid in clip_order if not results.get(aid)]
    print(f"\nWrote {path} — {len(clip_order) - len(missing)}/{len(clip_order)} done.")
    if missing:
        print(f"Still missing {len(missing)}: {', '.join(missing)}. Re-run to fill.")


if __name__ == "__main__":
    main()
