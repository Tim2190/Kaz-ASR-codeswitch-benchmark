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

from common import add_common_args, iter_clips, write_predictions, load_audio_16k_mono

ENDPOINT = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--language", default="kk-KZ")
    ap.add_argument("--service-name", default="yandex_stt")
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

    records = []
    for i, clip in enumerate(clips, 1):
        try:
            samples, _ = load_audio_16k_mono(clip.path)
            lpcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            resp = requests.post(ENDPOINT, params=params, headers=headers,
                                 data=lpcm, timeout=60)
            resp.raise_for_status()
            text = resp.json().get("result", "").strip()
        except Exception as e:  # noqa: BLE001
            text = ""
            print(f"[{clip.audio_id}] ERROR: {e}", file=sys.stderr)
        records.append({"audio_id": clip.audio_id, "raw_output": text})
        print(f"({i}/{len(clips)}) {clip.audio_id}: {text[:60]!r}")

    path = write_predictions(args.service_name, records, args.results_dir)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
