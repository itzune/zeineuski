"""
Ahotsak audio downloader — downloads MP3 files from Ahotsak S3 URLs.

Rate-limited, resumable, with progress tracking.

Usage:
    uv run python -m src.data.audio_downloader \
      --passages data/raw/speech/ahotsak/ahotsak_passages_20260608_213742.jsonl \
      --output data/raw/speech/ahotsak/audio \
      --max-audio 2000 \
      --rate-limit 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

USER_AGENT = (
    "ZeineuskiML/0.1 "
    "(Basque dialect research; xezpeleta@gmail.com; rate-limited downloader)"
)

DEFAULT_RATE_LIMIT = 0.5  # seconds between requests
DEFAULT_TIMEOUT = 60  # seconds for download
DEFAULT_MAX_AUDIO = 0  # 0 = no limit


def extract_audio_url(passage: dict) -> Optional[tuple]:
    """Extract media URL from passage dict.
    
    Returns (url, type) where type is 'mp3' or 'mp4'.
    Prefers MP3 audio, falls back to MP4 video (extract audio later).
    """
    audio_url = passage.get("audio_url", "").strip()
    if audio_url and audio_url.endswith(".mp3"):
        return (audio_url, "mp3")
    
    video_url = passage.get("video_url", "").strip()
    if video_url and video_url.endswith(".mp4"):
        return (video_url, "mp4")
    
    return None


def download_file(url: str, dest: Path, session: requests.Session) -> bool:
    if dest.exists():
        logger.debug(f"  Already exists: {dest.name}")
        return True  # Already downloaded

    try:
        resp = session.get(url, timeout=DEFAULT_TIMEOUT, stream=True)
        resp.raise_for_status()

        # Write to temp file then rename (atomic)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        tmp.rename(dest)
        return True

    except requests.exceptions.RequestException as e:
        logger.warning(f"  Failed: {url} — {e}")
        # Clean up temp file
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        return False


def _convert_mp4_to_mp3(mp4_path: Path, mp3_path: Path) -> bool:
    """Extract audio from MP4 to MP3 using ffmpeg."""
    import subprocess
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp4_path),
             "-vn", "-acodec", "libmp3lame", "-q:a", "2",
             str(mp3_path)],
            capture_output=True,
            timeout=120,
            check=True,
        )
        return mp3_path.exists() and mp3_path.stat().st_size > 0
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning(f"  Failed to convert {mp4_path.name}: {e}")
        return False


def download_ahotsak_audio(
    passages_jsonl: Path,
    output_dir: Path,
    max_audio: int = DEFAULT_MAX_AUDIO,
    rate_limit: float = DEFAULT_RATE_LIMIT,
) -> dict:
    """Download audio files for passages with MP3 URLs.

    Returns stats dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # Load passages
    passages = []
    logger.info(f"Loading passages from {passages_jsonl}...")
    with open(passages_jsonl) as f:
        for line in f:
            if line.strip():
                passages.append(json.loads(line))

    logger.info(f"  {len(passages)} passages loaded")

    # Filter: passages with audio URLs
    eligible = []
    for p in passages:
        result = extract_audio_url(p)
        if result:
            url, media_type = result
            eligible.append((p, url, media_type))

    logger.info(f"  {len(eligible)} passages have audio URLs")
    if max_audio and max_audio < len(eligible):
        eligible = eligible[:max_audio]
        logger.info(f"  Limiting to {max_audio} (--max-audio)")

    # Download
    stats = {
        "total": len(eligible),
        "downloaded": 0,
        "skipped": 0,
        "failed": 0,
        "total_bytes": 0,
        "start_time": time.time(),
    }

    # Track which files we download for reporting
    manifests = []

    for i, (passage, url, media_type) in enumerate(eligible):
        # Build filename from passage metadata
        town = passage.get("town_slug", "unknown")
        passage_id = passage.get("passage_id", f"unk{i:05d}")
        media_ext = ".mp4" if media_type == "mp4" else ".mp3"
        filename = f"{town}_{passage_id}{media_ext}"
        final_filename = f"{town}_{passage_id}.mp3"
        dest = output_dir / filename
        final_dest = output_dir / final_filename

        if i > 0:
            time.sleep(rate_limit)

        if (i + 1) % 50 == 0 or i == 0:
            logger.info(
                f"  [{i + 1}/{len(eligible)}] "
                f"downloaded={stats['downloaded']} "
                f"failed={stats['failed']} "
                f"skipped={stats['skipped']}"
            )

        # Skip if final MP3 already exists
        if final_dest.exists():
            size = final_dest.stat().st_size
            stats["downloaded"] += 1
            stats["total_bytes"] += size
            manifests.append({
                "filename": final_filename,
                "town": passage.get("town_name", town),
                "town_slug": town,
                "speaker": passage.get("speaker_name", "unknown"),
                "speaker_slug": passage.get("speaker_slug", "unknown"),
                "dialect": passage.get("dialect_class", ""),
                "confidence": passage.get("dialect_confidence", ""),
                "passage_id": passage_id,
                "duration": passage.get("duration", ""),
                "size_bytes": size,
            })
            continue

        success = download_file(url, dest, session)

        if success:
            # If MP4, extract audio to MP3
            if media_type == "mp4":
                success = _convert_mp4_to_mp3(dest, final_dest)
                if success:
                    # Remove MP4 to save space
                    dest.unlink()
                dest = final_dest
            
            size = dest.stat().st_size if dest.exists() else 0
            stats["total_bytes"] += size
            stats["downloaded"] += 1

            manifests.append({
                "filename": final_filename,
                "town": passage.get("town_name", town),
                "town_slug": town,
                "speaker": passage.get("speaker_name", "unknown"),
                "speaker_slug": passage.get("speaker_slug", "unknown"),
                "dialect": passage.get("dialect_class", ""),
                "confidence": passage.get("dialect_confidence", ""),
                "passage_id": passage_id,
                "duration": passage.get("duration", ""),
                "size_bytes": size,
            })
        else:
            stats["failed"] += 1

    # Save manifest
    manifest_path = output_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for m in manifests:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    logger.info(f"Manifest saved → {manifest_path}")

    # Summary
    elapsed = time.time() - stats["start_time"]
    logger.info(
        f"Download complete: "
        f"{stats['downloaded']} downloaded, "
        f"{stats['failed']} failed, "
        f"{stats['total_bytes'] / (1024 * 1024):.1f} MB "
        f"in {elapsed:.1f}s"
    )

    return stats


def main():
    parser = argparse.ArgumentParser(description="Download Ahotsak audio files")
    parser.add_argument(
        "--passages",
        required=True,
        type=Path,
        help="JSONL file with scraped passages",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for audio files",
    )
    parser.add_argument(
        "--max-audio",
        type=int,
        default=DEFAULT_MAX_AUDIO,
        help="Max audio files to download (0=all)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=DEFAULT_RATE_LIMIT,
        help="Seconds between requests",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    stats = download_ahotsak_audio(
        passages_jsonl=args.passages.resolve(),
        output_dir=args.output.resolve(),
        max_audio=args.max_audio,
        rate_limit=args.rate_limit,
    )

    # Print METRIC for autoresearch
    print(f"\nMETRIC audio_downloaded={stats['downloaded']}")
    print(f"METRIC audio_failed={stats['failed']}")
    print(f"METRIC audio_total_bytes={stats['total_bytes']}")
    print(f"METRIC audio_total_mb={stats['total_bytes'] / (1024 * 1024):.1f}")

    if stats["downloaded"] == 0:
        print("WARNING: No audio files downloaded", file=sys.stderr)
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
