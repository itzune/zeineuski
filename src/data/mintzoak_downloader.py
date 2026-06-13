"""
Mintzoak audio downloader — extracts audio from Vimeo-hosted passages.

Mintzoak stores audio as Vimeo video embeds. This script uses yt-dlp to
download the best audio track and convert to 16kHz mono WAV.

Requirements:
    yt-dlp  (https://github.com/yt-dlp/yt-dlp)
    ffmpeg  (for audio conversion)

Usage:
    uv run python -m src.data.mintzoak_downloader \
      --passages data/raw/speech/mintzoak/mintzoak_passages_xiberoa.jsonl \
      --output data/raw/speech/mintzoak/audio \
      --max-audio 100 \
      --rate-limit 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_RATE_LIMIT = 1.0  # seconds between downloads
DEFAULT_TIMEOUT = 120  # seconds per yt-dlp call


def resolve_vimeo_url(
    town_slug: str, passage_id: str, timeout: int = 12
) -> Optional[str]:
    """
    Lazily probe a passage page to get the Vimeo URL.

    Returns the vimeo_url if found, None if no Vimeo content for this slot.
    """
    url = f"https://www.mintzoak.eus/eu/{town_slug}/pasarteak/{passage_id}/"
    try:
        r = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "ZeineuskiML/0.1 (Basque dialect research)"},
        )
        if r.status_code != 200:
            return None
        vimeo_match = re.search(r"player\.vimeo\.com/video/(\d+)", r.text)
        if vimeo_match:
            return f"https://player.vimeo.com/video/{vimeo_match.group(1)}"
        return None
    except requests.RequestException:
        return None


def check_ytdlp() -> bool:
    """Verify yt-dlp is installed and working."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            logger.info(f"yt-dlp version: {result.stdout.strip()}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False


def download_audio(
    vimeo_url: str,
    output_path: Path,
    passage_id: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> bool:
    """
    Download audio from Vimeo URL and save as 16kHz mono WAV.

    yt-dlp -f bestaudio -x --audio-format wav --audio-quality 0
           --postprocessor-args "-ar 16000 -ac 1"
           -o <output> <vimeo_url>
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # yt-dlp adds its own extension, so we use a template
    output_template = str(output_path.with_suffix("")) + ".%(ext)s"

    cmd = [
        "yt-dlp",
        "-f",
        "bestaudio",
        "-x",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--postprocessor-args",
        "ffmpeg:-ar 16000 -ac 1",
        "-o",
        output_template,
        "--no-playlist",
        "--no-warnings",
        "--quiet",
        vimeo_url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode != 0:
            # Check if it's a transient error or permanent
            stderr = result.stderr.lower()
            if "private" in stderr or "not found" in stderr:
                logger.debug(f"  [{passage_id}] Video unavailable: {vimeo_url}")
                return False
            if "rate" in stderr or "429" in stderr or "limit" in stderr:
                logger.warning(f"  [{passage_id}] Rate limited, backing off...")
                time.sleep(30)
                return False
            logger.warning(f"  [{passage_id}] yt-dlp error: {result.stderr[:200]}")
            return False

        # yt-dlp outputs <output>.wav
        expected = output_path.with_suffix("").with_suffix(".wav")
        if expected.exists():
            # Rename to our desired name
            expected.rename(output_path)
            size_kb = output_path.stat().st_size / 1024
            logger.debug(f"  [{passage_id}] Downloaded: {size_kb:.0f} KB")
            return True
        else:
            logger.warning(f"  [{passage_id}] Output not found: {expected}")
            return False

    except subprocess.TimeoutExpired:
        logger.warning(f"  [{passage_id}] Timeout: {vimeo_url}")
        return False
    except Exception as e:
        logger.warning(f"  [{passage_id}] Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Mintzoak audio downloader via yt-dlp")
    parser.add_argument(
        "--passages",
        required=True,
        help="JSONL file with passage entries (from mintzoak_scraper)",
    )
    parser.add_argument(
        "--output",
        default="data/raw/speech/mintzoak/audio",
        help="Output directory for audio files",
    )
    parser.add_argument(
        "--max-audio",
        type=int,
        default=0,
        help="Max audio files to download (0=all)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=DEFAULT_RATE_LIMIT,
        help="Seconds between downloads",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip files that already exist (default: True)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Timeout per download in seconds",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    if not check_ytdlp():
        logger.error("yt-dlp not found! Install: uv tool install yt-dlp")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load passages
    passages = []
    with open(args.passages) as f:
        for line in f:
            line = line.strip()
            if line:
                passages.append(json.loads(line))

    if args.max_audio > 0:
        passages = passages[: args.max_audio]

    logger.info(f"Will download up to {len(passages)} audio files to {output_dir}")

    # Generate manifest for downloaded files
    manifest_path = output_dir / "manifest.jsonl"

    success = 0
    skipped = 0
    failed = 0
    start_time = time.time()

    with open(manifest_path, "w") as mf:
        for i, passage in enumerate(passages):
            passage_id = passage.get("passage_id", str(i))
            town = passage.get("town_slug", "unknown")
            vimeo_url = passage.get("vimeo_url")
            if not vimeo_url:
                # Lazy resolve — probe the passage page
                vimeo_url = resolve_vimeo_url(town, passage_id)
                if not vimeo_url:
                    # No Vimeo content for this slot — skip silently
                    skipped += 1
                    continue
                passage["vimeo_url"] = vimeo_url

            # Output filename: <town>_<passage_id>.wav
            filename = f"{town}_{passage_id.replace('/', '_')}.wav"
            filepath = output_dir / filename

            if args.skip_existing and filepath.exists():
                skipped += 1
                mf.write(
                    json.dumps(
                        {
                            **passage,
                            "audio_path": str(filepath),
                            "filename": filename,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            if i > 0:
                wait = args.rate_limit
                time.sleep(wait)

            progress = (i + 1) / len(passages) * 100
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            logger.info(
                f"[{i + 1}/{len(passages)}] ({progress:.0f}%, {rate:.0f}/min) "
                f"{town}/{passage_id}"
            )

            ok = download_audio(vimeo_url, filepath, passage_id, args.timeout)

            if ok:
                success += 1
                mf.write(
                    json.dumps(
                        {
                            **passage,
                            "audio_path": str(filepath),
                            "filename": filename,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            else:
                failed += 1

    elapsed = time.time() - start_time
    logger.info(
        f"Done! {success} downloaded, {skipped} skipped, {failed} failed "
        f"({elapsed / 60:.1f} min)"
    )
    logger.info(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
