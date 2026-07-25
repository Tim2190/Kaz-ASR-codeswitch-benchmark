#!/usr/bin/env python3
"""Transcribe the benchmark clips with a locally-run Whisper model.

Uses faster-whisper (CTranslate2) which runs efficiently on CPU or GPU and
downloads weights from Hugging Face on first use.

    pip install faster-whisper
    python runners/run_whisper_local.py --model large-v3 --device cpu

``--model`` accepts faster-whisper size names (``tiny``..``large-v3``) or a
CTranslate2 model directory. The service name is ``whisper_local_<model>``.

Note: model download needs network access to Hugging Face. In restricted
environments, pre-download the weights and pass a local path to ``--model``.
"""

import argparse
import sys

from common import add_common_args, iter_clips, write_predictions


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--compute-type", default="int8",
                    help="e.g. int8 (CPU), float16 (GPU).")
    ap.add_argument("--language", default="kk")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--service-name", default=None)
    args = ap.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("Install the backend first:  pip install faster-whisper")

    model = WhisperModel(args.model, device=args.device,
                         compute_type=args.compute_type)
    service = args.service_name or f"whisper_local_{args.model}".replace("/", "_")

    clips = list(iter_clips(args.metadata, args.audio_dir))
    if args.limit:
        clips = clips[: args.limit]

    records = []
    for i, clip in enumerate(clips, 1):
        try:
            segments, _ = model.transcribe(
                clip.path, language=args.language, beam_size=args.beam_size,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as e:  # noqa: BLE001
            text = ""
            print(f"[{clip.audio_id}] ERROR: {e}", file=sys.stderr)
        records.append({"audio_id": clip.audio_id, "raw_output": text})
        print(f"({i}/{len(clips)}) {clip.audio_id}: {text[:60]!r}")

    path = write_predictions(service, records, args.results_dir)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
