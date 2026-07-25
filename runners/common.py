"""Shared helpers for ASR provider runners.

Every runner:
  1. iterates over the clips in ``metadata.csv`` (or an ``audio/`` directory),
  2. sends each clip to a service and captures the raw transcript,
  3. writes ``results/predictions_<service>.jsonl`` in the format that
     ``evaluate.py`` consumes.

Runners are intentionally thin and dependency-light; heavy/provider-specific
imports happen lazily inside each runner so that installing one provider's SDK
is enough to use it.
"""

from __future__ import annotations

import csv
import io
import json
import os
import wave
from dataclasses import dataclass
from typing import Iterator


AUDIO_DIR_DEFAULT = "audio"
METADATA_DEFAULT = "metadata.csv"
RESULTS_DIR_DEFAULT = "results"


@dataclass
class Clip:
    audio_id: str
    path: str


def iter_clips(metadata: str = METADATA_DEFAULT,
               audio_dir: str = AUDIO_DIR_DEFAULT) -> Iterator[Clip]:
    """Yield clips in the order they appear in metadata.csv."""
    if os.path.exists(metadata):
        with open(metadata, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                aid = row["audio_id"].strip()
                yield Clip(aid, os.path.join(audio_dir, f"{aid}.wav"))
    else:  # fall back to scanning the audio directory
        for name in sorted(os.listdir(audio_dir)):
            if name.lower().endswith(".wav"):
                aid = os.path.splitext(name)[0]
                yield Clip(aid, os.path.join(audio_dir, name))


def load_audio_16k_mono(path: str):
    """Return (float32 mono samples in [-1, 1], sample_rate=16000).

    Uses soundfile + scipy so it correctly handles the dataset's 48 kHz stereo
    WAVs (see README note on the format discrepancy). Falls back to the stdlib
    ``wave`` module for plain 16-bit PCM if soundfile is unavailable.
    """
    target_sr = 16000
    try:
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)  # downmix to mono
        if sr != target_sr:
            from scipy.signal import resample_poly
            from math import gcd

            g = gcd(int(sr), target_sr)
            mono = resample_poly(mono, target_sr // g, int(sr) // g)
        return mono.astype("float32"), target_sr
    except ImportError:
        # Minimal stdlib path: assumes 16-bit PCM; performs naive downmix and
        # decimation only when sr is an integer multiple of 16 kHz.
        import numpy as np  # numpy is a hard dependency of jiwer anyway

        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            raw = w.readframes(w.getnframes())
        samples = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
        if ch > 1:
            samples = samples.reshape(-1, ch).mean(axis=1)
        if sr != target_sr:
            if sr % target_sr == 0:
                samples = samples[:: sr // target_sr]
            else:
                raise RuntimeError(
                    f"Cannot resample {sr} Hz -> {target_sr} Hz without scipy; "
                    "install soundfile+scipy (see requirements-runners.txt)."
                )
        return samples, target_sr


def pcm16_wav_bytes(samples, sr: int = 16000) -> bytes:
    """Encode float32 mono samples as 16-bit PCM WAV bytes (for API uploads)."""
    import numpy as np

    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


def predictions_path(service: str, results_dir: str = RESULTS_DIR_DEFAULT) -> str:
    return os.path.join(results_dir, f"predictions_{service}.jsonl")


def write_predictions(service: str, records: list[dict],
                      results_dir: str = RESULTS_DIR_DEFAULT) -> str:
    """Write predictions to results/predictions_<service>.jsonl and return path."""
    os.makedirs(results_dir, exist_ok=True)
    path = predictions_path(service, results_dir)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            rec.setdefault("service", service)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def read_predictions_map(service: str,
                         results_dir: str = RESULTS_DIR_DEFAULT) -> "dict[str, str]":
    """Load an existing predictions file into {audio_id: raw_output}.

    Used to resume a partially-completed run: clips already transcribed (with a
    non-empty output) can be skipped so a re-run only fills the gaps. Returns an
    empty dict if no file exists yet.
    """
    path = predictions_path(service, results_dir)
    out: dict[str, str] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[str(rec["audio_id"]).strip()] = rec.get("raw_output", "") or ""
    return out


def add_common_args(ap):
    ap.add_argument("--metadata", default=METADATA_DEFAULT)
    ap.add_argument("--audio-dir", default=AUDIO_DIR_DEFAULT)
    ap.add_argument("--results-dir", default=RESULTS_DIR_DEFAULT)
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N clips (0 = all); useful for a smoke test.")
    return ap
