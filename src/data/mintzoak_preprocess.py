"""
Mintzoak speech preprocessing: VAD, resampling, normalization, town-disjoint splits.

Adapted from the Ahotsak pipeline. Mintzoak files are already 16kHz mono WAV,
spread across worker_*/ subdirectories with a parent manifest.jsonl.

Usage:
    python -m src.data.mintzoak_preprocess \
      --audio-dir data/raw/speech/mintzoak/audio \
      --output data/processed/speech/mintzoak
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf

logger = logging.getLogger(__name__)

# Import shared preprocessing functions from the main speech module
from src.data.speech_preprocessing import (  # noqa: E402
    normalize_volume,
    remove_silence,
    resample_audio,
    vad_split,
)


def load_manifest(manifest_path: Path) -> list[dict]:
    """Load the combined manifest.jsonl."""
    entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def find_audio_files(audio_dir: Path, manifest: list[dict]) -> dict[str, str]:
    """Build mapping: passage_key → absolute WAV path via manifest 'filename' field."""
    file_map = {}
    for entry in manifest:
        filename = entry.get("filename", "")
        audio_path = entry.get("audio_path", "")
        passage_id = entry.get("passage_id", "")

        # Try absolute path first, then relative to audio_dir
        if audio_path and Path(audio_path).exists():
            file_map[passage_id] = audio_path
        elif filename:
            # Search in worker subdirectories
            candidate = audio_dir / filename
            if candidate.exists():
                file_map[passage_id] = str(candidate)
            else:
                # Search recursively
                for w in sorted(audio_dir.glob(f"worker_*/{filename}")):
                    file_map[passage_id] = str(w)
                    break
    return file_map


def main():
    parser = argparse.ArgumentParser(description="Preprocess Mintzoak speech data")
    parser.add_argument(
        "--audio-dir", required=True, type=Path, help="Mintzoak audio directory"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output directory for segments"
    )
    parser.add_argument(
        "--max-files", type=int, default=0, help="Limit number of files (0=all)"
    )
    parser.add_argument(
        "--min-dur", type=float, default=1.0, help="Min segment duration"
    )
    parser.add_argument(
        "--max-dur", type=float, default=15.0, help="Max segment duration"
    )
    parser.add_argument("--vad-method", default="energy", choices=["energy", "silero"])
    parser.add_argument(
        "--no-resume", action="store_true", help="Don't skip existing segments"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    manifest_path = args.audio_dir / "manifest.jsonl"
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    logger.info(f"Loading manifest: {manifest_path}")
    manifest = load_manifest(manifest_path)
    logger.info(f"  {len(manifest)} entries")

    # Build audio file map
    logger.info("Scanning audio files...")
    file_map = find_audio_files(args.audio_dir, manifest)
    logger.info(f"  {len(file_map)} audio files found")

    # Filter: only take entries where we have the audio file
    valid = [e for e in manifest if e.get("passage_id", "") in file_map]
    logger.info(f"  {len(valid)} entries with audio files")

    if args.max_files > 0:
        valid = valid[: args.max_files]
        logger.info(f"  Limited to {len(valid)} files")

    # Output dirs
    segments_dir = args.output / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    # Process each file
    processed = []
    errors = 0
    total_seg_count = 0

    for i, entry in enumerate(valid):
        passage_id = entry["passage_id"]
        dialect = entry.get("dialect", "unknown")
        town = entry.get("town_slug", "unknown")
        region = entry.get("region", "unknown")
        audio_path = file_map.get(passage_id, "")

        if (i + 1) % 500 == 0 or i == 0:
            logger.info(
                f"  [{i + 1}/{len(valid)}] total_segments={total_seg_count} errors={errors}"
            )

        # Check for existing segments (resume support)
        if not args.no_resume:
            existing = list(segments_dir.glob(f"{passage_id.replace('/', '_')}_*.wav"))
            if existing:
                for seg_path in existing:
                    try:
                        info = sf.info(seg_path)
                        duration = info.duration
                    except Exception:
                        duration = 0
                    processed.append(
                        {
                            "path": str(seg_path),
                            "filename": seg_path.name,
                            "dialect": dialect,
                            "town": town,
                            "region": region,
                            "passage_id": passage_id,
                            "duration_sec": round(duration, 2),
                        }
                    )
                    total_seg_count += 1
                continue

        try:
            audio, sr = sf.read(audio_path)

            # Convert to mono
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Resample if needed (Mintzoak is already 16kHz, but just in case)
            if sr != 16000:
                audio = resample_audio(audio, sr, 16000)
                sr = 16000

            # Remove silence
            audio = remove_silence(audio, sr)

            # Normalize volume
            audio = normalize_volume(audio)

            # VAD split
            segments = vad_split(audio, sr, args.min_dur, args.max_dur, args.vad_method)

            # Save segments
            safe_id = passage_id.replace("/", "_")
            for j, segment in enumerate(segments):
                seg_filename = f"{safe_id}_{j:03d}.wav"
                seg_path = segments_dir / seg_filename
                duration = len(segment) / sr
                sf.write(seg_path, segment, sr)

                processed.append(
                    {
                        "path": str(seg_path),
                        "filename": seg_filename,
                        "dialect": dialect,
                        "town": town,
                        "region": region,
                        "passage_id": passage_id,
                        "duration_sec": round(duration, 2),
                    }
                )
                total_seg_count += 1

        except Exception as e:
            logger.debug(f"  Error processing {passage_id}: {e}")
            errors += 1

    logger.info(
        f"Processed {len(valid)} files → {total_seg_count} segments ({errors} errors)"
    )

    # ── Town-disjoint splits ──
    random.seed(42)

    town_samples = defaultdict(list)
    for s in processed:
        town_samples[s["town"]].append(s)

    towns = list(town_samples.keys())
    random.shuffle(towns)
    n = len(towns)
    n_train = int(n * 0.60)
    n_val = int(n * 0.15)

    train_towns = set(towns[:n_train])
    val_towns = set(towns[n_train : n_train + n_val])

    splits = {"train": [], "val": [], "test": []}
    for town in towns:
        if town in train_towns:
            splits["train"].extend(town_samples[town])
        elif town in val_towns:
            splits["val"].extend(town_samples[town])
        else:
            splits["test"].extend(town_samples[town])

    for split_name, samples in splits.items():
        csv_path = args.output / f"{split_name}.csv"
        with open(csv_path, "w") as f:
            f.write("path,filename,dialect,town,region,passage_id,duration_sec\n")
            for s in samples:
                f.write(
                    f"{s['path']},{s['filename']},{s['dialect']},"
                    f"{s['town']},{s['region']},{s['passage_id']},"
                    f"{s['duration_sec']}\n"
                )
        town_count = len(set(s["town"] for s in samples))
        logger.info(f"  {split_name}: {len(samples)} segments, {town_count} towns")

    # Summary
    total_duration = sum(s["duration_sec"] for s in processed) / 3600
    by_dialect = defaultdict(int)
    for s in processed:
        by_dialect[s["dialect"]] += 1

    stats = {
        "total_files": len(valid),
        "total_segments": total_seg_count,
        "total_hours": round(total_duration, 2),
        "by_dialect": dict(by_dialect),
        "train_segments": len(splits["train"]),
        "val_segments": len(splits["val"]),
        "test_segments": len(splits["test"]),
        "num_towns": n,
        "errors": errors,
    }

    stats_path = args.output / "preprocessing_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats → {stats_path}")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
