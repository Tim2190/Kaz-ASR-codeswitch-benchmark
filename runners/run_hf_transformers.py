#!/usr/bin/env python3
"""Transcribe the benchmark clips with a Hugging Face ASR model.

Works with any ``transformers`` automatic-speech-recognition pipeline, including
the fine-tuned Kazakh Whisper model referenced in the README:

    pip install "transformers[torch]" soundfile scipy
    python runners/run_hf_transformers.py --model shyngys879/kazakh-whisper-large-v3-turbo
    python runners/run_hf_transformers.py --model openai/whisper-large-v3 --language kk

The service name is ``hf_<model>`` (slashes replaced). Weights download from
Hugging Face on first use, so this needs network access to huggingface.co.
"""

import argparse
import sys

from common import add_common_args, iter_clips, write_predictions, load_audio_16k_mono


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--model", required=True,
                    help="HF model id or local path (e.g. openai/whisper-large-v3).")
    ap.add_argument("--language", default="kk",
                    help="Language hint for Whisper-family models; ignored otherwise.")
    ap.add_argument("--device", default="cpu",
                    help="'cpu', 'cuda', 'cuda:0', or a device index string.")
    ap.add_argument("--service-name", default=None)
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
        from transformers import pipeline
    except ImportError:
        sys.exit('Install the backend first:  pip install "transformers[torch]"')

    device = -1 if args.device == "cpu" else args.device
    asr = pipeline("automatic-speech-recognition", model=args.model, device=device)

    is_whisper = "whisper" in args.model.lower()
    generate_kwargs = {"language": args.language, "task": "transcribe"} if is_whisper else {}

    service = args.service_name or "hf_" + args.model.replace("/", "_")

    clips = list(iter_clips(args.metadata, args.audio_dir))
    if args.limit:
        clips = clips[: args.limit]

    records = []
    for i, clip in enumerate(clips, 1):
        try:
            samples, sr = load_audio_16k_mono(clip.path)
            out = asr({"array": samples, "sampling_rate": sr},
                      generate_kwargs=generate_kwargs)
            text = (out.get("text") if isinstance(out, dict) else str(out)).strip()
        except Exception as e:  # noqa: BLE001
            text = ""
            print(f"[{clip.audio_id}] ERROR: {e}", file=sys.stderr)
        records.append({"audio_id": clip.audio_id, "raw_output": text})
        print(f"({i}/{len(clips)}) {clip.audio_id}: {text[:60]!r}")

    path = write_predictions(service, records, args.results_dir)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
