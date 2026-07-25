#!/usr/bin/env python3
"""Produce a canonical 16 kHz / mono / 16-bit PCM copy of the audio.

The distributed clips in ``audio/`` are 48 kHz stereo, whereas
``annotation_methodology.md`` specifies 16 kHz mono. This script writes a
normalized copy to a separate directory (default ``audio_16k/``) without
touching the originals, so you can either ship the canonical set or feed it to
tools that expect the documented format.

    pip install soundfile scipy numpy
    python runners/prep_audio.py --out audio_16k
"""

import argparse
import os

from common import iter_clips, load_audio_16k_mono, pcm16_wav_bytes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", default="metadata.csv")
    ap.add_argument("--audio-dir", default="audio")
    ap.add_argument("--out", default="audio_16k")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n = 0
    for clip in iter_clips(args.metadata, args.audio_dir):
        samples, sr = load_audio_16k_mono(clip.path)
        wav = pcm16_wav_bytes(samples, sr)
        dst = os.path.join(args.out, f"{clip.audio_id}.wav")
        with open(dst, "wb") as f:
            f.write(wav)
        n += 1
        print(f"{clip.audio_id}: {len(samples)/sr:.2f}s -> {dst}")
    print(f"\nWrote {n} canonical 16 kHz mono clips to {args.out}/")


if __name__ == "__main__":
    main()
