"""
Speech preprocessing pipeline: VAD, resampling, normalization, splitting.

Produces Hugging Face Dataset-compatible output with speaker-disjoint splits.

Usage:
    uv run python -m src.data.speech_preprocessing run \
      --audio-dir data/raw/speech/ahotsak/audio \
      --passages data/raw/speech/ahotsak/ahotsak_passages_20260608_213742.jsonl \
      --output data/processed/speech/ahotsak \
      --config configs/speech/preprocessing.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import yaml

logger = logging.getLogger(__name__)


def load_passage_metadata(passages_path: Path) -> dict[str, dict]:
    """Load passage JSONL and build lookup by passage_id.

    Also loads the manifest.jsonl if available (has filename mapping).
    Returns dict[passage_id, metadata].
    """
    metadata = {}
    with open(passages_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            pid = d.get("passage_id", "")
            if pid:
                metadata[pid] = d
    return metadata


def load_manifest(manifest_path: Path) -> dict[str, dict]:
    """Load downloaded file manifest.

    Returns dict[filename, metadata].
    """
    if not manifest_path.exists():
        return {}

    manifest = {}
    with open(manifest_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            manifest[d["filename"]] = d
    return manifest


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000,
) -> np.ndarray:
    """Resample audio to target sample rate using librosa."""
    import librosa
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    return librosa.resample(audio.astype(np.float64), orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)


def remove_silence(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -40,
) -> np.ndarray:
    """Trim leading and trailing silence using energy threshold."""
    import librosa

    # Compute RMS energy
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # Find non-silent frames
    non_silent = rms_db > threshold_db
    if not non_silent.any():
        return audio  # All silent, don't trim

    # Convert frames to samples
    hop = 512
    first_frame = np.argmax(non_silent)
    last_frame = len(non_silent) - np.argmax(non_silent[::-1]) - 1

    start = max(0, first_frame * hop)
    end = min(len(audio), (last_frame + 1) * hop)

    return audio[start:end]


def normalize_volume(
    audio: np.ndarray,
    target_db: float = -23,
) -> np.ndarray:
    """Normalize audio to target RMS level."""
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-8:
        return audio
    rms_db = 20 * np.log10(rms)
    gain = 10 ** ((target_db - rms_db) / 20)
    normalized = audio * gain
    # Clip to avoid distortion
    max_val = np.max(np.abs(normalized))
    if max_val > 0.99:
        normalized = normalized / max_val * 0.99
    return normalized


def vad_split(
    audio: np.ndarray,
    sr: int,
    min_duration: float = 1.0,
    max_duration: float = 15.0,
    vad_method: str = "energy",
) -> list[np.ndarray]:
    """Split audio into speech segments using VAD.

    Methods:
    - 'energy': Simple energy-based VAD (fast, no deps)
    - 'silero': Silero VAD (more accurate, needs torch)

    Returns list of audio arrays.
    """
    if vad_method == "silero":
        return _vad_split_silero(audio, sr, min_duration, max_duration)
    else:
        return _vad_split_energy(audio, sr, min_duration, max_duration)


def _vad_split_energy(
    audio: np.ndarray,
    sr: int,
    min_duration: float,
    max_duration: float,
) -> list[np.ndarray]:
    """Simple energy-based VAD split."""
    import librosa

    # Compute RMS energy
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # Detect speech segments
    threshold_db = -35  # dB threshold for speech
    is_speech = rms_db > threshold_db

    # Find contiguous speech regions
    min_speech_frames = int(min_duration * sr / hop_length)
    max_speech_frames = int(max_duration * sr / hop_length)

    # Smooth: fill small gaps (< 0.3s) and remove short segments
    gap_frames = int(0.3 * sr / hop_length)

    segments = []
    in_speech = False
    segment_start = 0

    for i in range(len(is_speech)):
        if is_speech[i] and not in_speech:
            segment_start = i
            in_speech = True
        elif not is_speech[i] and in_speech:
            # Look ahead to see if gap is small enough to bridge
            gap_end = min(i + gap_frames, len(is_speech))
            if np.any(is_speech[i:gap_end]):
                continue  # Bridge the gap
            else:
                segment_end = i
                duration_frames = segment_end - segment_start
                if duration_frames >= min_speech_frames:
                    s = segment_start * hop_length
                    e = min(segment_end * hop_length, len(audio))
                    # Split long segments into max_duration chunks
                    chunk = audio[s:e]
                    chunk_samples = int(max_duration * sr)
                    for offset in range(0, len(chunk), chunk_samples):
                        subchunk = chunk[offset:offset + chunk_samples]
                        if len(subchunk) >= int(min_duration * sr):
                            segments.append(subchunk)
                in_speech = False

    # Handle case where audio ends during speech
    if in_speech:
        segment_end = len(is_speech) - 1
        duration_frames = segment_end - segment_start
        if duration_frames >= min_speech_frames:
            s = segment_start * hop_length
            e = min(segment_end * hop_length, len(audio))
            chunk = audio[s:e]
            chunk_samples = int(max_duration * sr)
            for offset in range(0, len(chunk), chunk_samples):
                subchunk = chunk[offset:offset + chunk_samples]
                if len(subchunk) >= int(min_duration * sr):
                    segments.append(subchunk)

    # Edge case: no segments found (maybe short audio, or threshold too high)
    if not segments and len(audio) >= int(min_duration * sr):
        # Return the whole thing, capped at max_duration
        chunk_samples = int(max_duration * sr)
        segments = [
            audio[i:i + chunk_samples]
            for i in range(0, len(audio), chunk_samples)
            if len(audio[i:i + chunk_samples]) >= int(min_duration * sr)
        ]

    return segments


def _vad_split_silero(
    audio: np.ndarray,
    sr: int,
    min_duration: float,
    max_duration: float,
) -> list[np.ndarray]:
    """Silero VAD-based split (needs torch)."""
    try:
        import torch
    except ImportError:
        logger.warning("Silero VAD requires torch. Falling back to energy VAD.")
        return _vad_split_energy(audio, sr, min_duration, max_duration)

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
    )
    (get_speech_timestamps, _, _, _, _) = utils

    # Silero expects float32 tensor, 16 kHz
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio.astype(np.float64), orig_sr=sr, target_sr=16000).astype(np.float32)
        sr = 16000

    audio_tensor = torch.from_numpy(audio).float()

    speech_timestamps = get_speech_timestamps(
        audio_tensor, model, sampling_rate=sr,
        min_speech_duration_ms=int(min_duration * 1000),
        max_speech_duration_s=max_duration,
    )

    segments = []
    for ts in speech_timestamps:
        chunk = audio[ts["start"]:ts["end"]]
        # Further split if too long
        chunk_samples = int(max_duration * sr)
        for offset in range(0, len(chunk), chunk_samples):
            subchunk = chunk[offset:offset + chunk_samples]
            if len(subchunk) >= int(min_duration * sr):
                segments.append(subchunk)

    return segments


def process_audio_files(
    audio_dir: Path,
    manifest_path: Path,
    passages_jsonl: Path,
    output_dir: Path,
    config: dict,
) -> dict:
    """Process all downloaded audio files into speaker-disjoint splits.

    Returns stats dict.
    """
    target_sr = config.get("target_sample_rate", 16000)
    min_dur = config.get("min_duration_sec", 1.0)
    max_dur = config.get("max_duration_sec", 15.0)
    vad_method = config.get("vad_method", "energy")
    do_normalize = config.get("normalize_volume", True)
    target_lufs = config.get("target_lufs", -23)

    output_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    manifest = load_manifest(manifest_path)
    passage_meta = load_passage_metadata(passages_jsonl)

    # Load audio files and their metadata
    processed = []
    errors = 0

    audio_files = sorted(audio_dir.glob("*.mp3"))
    logger.info(f"Found {len(audio_files)} audio files in {audio_dir}")

    for i, audio_path in enumerate(audio_files):
        filename = audio_path.name

        if (i + 1) % 100 == 0 or i == 0:
            logger.info(
                f"  Processing [{i+1}/{len(audio_files)}] "
                f"segments={len(processed)} errors={errors}"
            )

        # Get metadata
        meta = manifest.get(filename, {})
        passage_id = meta.get("passage_id", audio_path.stem.rsplit("_", 1)[-1] if "_" in audio_path.stem else audio_path.stem)
        dialect = meta.get("dialect", "")
        speaker = meta.get("speaker_slug", meta.get("speaker", "unknown"))
        town = meta.get("town_slug", meta.get("town", "unknown"))

        # Try to fill missing metadata from passages JSONL
        if not dialect and passage_id in passage_meta:
            dialect = passage_meta[passage_id].get("dialect_class", "")
        if not speaker or speaker == "unknown":
            if passage_id in passage_meta:
                speaker = passage_meta[passage_id].get("speaker_slug", "unknown")

        if not dialect:
            logger.debug(f"  No dialect for {filename}, skipping")
            continue

        try:
            audio, sr = sf.read(audio_path)

            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Resample
            audio = resample_audio(audio, sr, target_sr)
            sr = target_sr

            # Remove silence
            audio = remove_silence(audio, sr)

            # Normalize
            if do_normalize:
                audio = normalize_volume(audio, target_lufs)

            # VAD split into segments
            segments = vad_split(audio, sr, min_dur, max_dur, vad_method)

            # Save segments
            for j, segment in enumerate(segments):
                seg_filename = f"{audio_path.stem}_{j:03d}.wav"
                seg_path = segments_dir / seg_filename

                duration = len(segment) / sr
                sf.write(seg_path, segment, sr)

                processed.append({
                    "path": str(seg_path),
                    "filename": seg_filename,
                    "source_file": filename,
                    "dialect": dialect,
                    "speaker": speaker,
                    "town": town,
                    "passage_id": passage_id,
                    "duration_sec": round(duration, 2),
                })

        except Exception as e:
            logger.debug(f"  Error processing {filename}: {e}")
            errors += 1

    logger.info(f"  Processed {len(processed)} segments from {len(audio_files)} files ({errors} errors)")

    # ── Speaker-disjoint splits ──
    # Group by speaker, then split speakers 70/15/15
    from collections import defaultdict
    import random

    random.seed(42)

    # Group by town (speaker names from Ahotsak have URL slug "hizlariak" for everyone)
    # Use town as proxy for speaker disjunction — different towns = different speakers
    town_samples = defaultdict(list)
    for sample in processed:
        town_samples[sample["town"]].append(sample)

    towns = list(town_samples.keys())
    random.shuffle(towns)

    n = len(towns)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train_towns = set(towns[:n_train])
    val_towns = set(towns[n_train:n_train + n_val])

    splits = {"train": [], "val": [], "test": []}
    for town in towns:
        if town in train_towns:
            splits["train"].extend(town_samples[town])
        elif town in val_towns:
            splits["val"].extend(town_samples[town])
        else:
            splits["test"].extend(town_samples[town])

    # Save CSV manifests
    for split_name, samples in splits.items():
        csv_path = output_dir / f"{split_name}.csv"
        with open(csv_path, "w") as f:
            f.write("path,filename,dialect,speaker,town,passage_id,duration_sec\n")
            for s in samples:
                f.write(
                    f"{s['path']},{s['filename']},{s['dialect']},"
                    f"{s['speaker']},{s['town']},{s['passage_id']},"
                    f"{s['duration_sec']}\n"
                )
        logger.info(f"  {split_name}: {len(samples)} segments, {len(set(s['town'] for s in samples))} towns")

    # Summary stats
    total_duration = sum(s["duration_sec"] for s in processed) / 3600
    stats = {
        "total_segments": len(processed),
        "total_audio_hours": round(total_duration, 2),
        "train_segments": len(splits["train"]),
        "val_segments": len(splits["val"]),
        "test_segments": len(splits["test"]),
        "num_towns": n,
        "errors": errors,
    }

    # Save stats
    stats_path = output_dir / "preprocessing_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Stats saved → {stats_path}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess speech data")
    subparsers = parser.add_subparsers(dest="command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run preprocessing pipeline")
    run_parser.add_argument("--audio-dir", required=True, type=Path, help="Directory with downloaded audio")
    run_parser.add_argument("--passages", required=True, type=Path, help="JSONL file with passage metadata")
    run_parser.add_argument("--output", required=True, type=Path, help="Output directory")
    run_parser.add_argument("--config", type=Path, help="Config YAML file")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command != "run":
        parser.print_help()
        sys.exit(1)

    # Load config
    config = {}
    if args.config and args.config.exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Find manifest
    manifest_path = args.audio_dir / "manifest.jsonl"

    stats = process_audio_files(
        audio_dir=args.audio_dir,
        manifest_path=manifest_path,
        passages_jsonl=args.passages,
        output_dir=args.output,
        config=config,
    )

    # Print METRIC lines for autoresearch
    print(f"\nMETRIC preprocess_segments={stats['total_segments']}")
    print(f"METRIC preprocess_hours={stats['total_audio_hours']}")
    print(f"METRIC preprocess_train={stats['train_segments']}")
    print(f"METRIC preprocess_val={stats['val_segments']}")
    print(f"METRIC preprocess_test={stats['test_segments']}")
    print(f"METRIC preprocess_towns={stats['num_towns']}")
    print(f"METRIC preprocess_errors={stats['errors']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
