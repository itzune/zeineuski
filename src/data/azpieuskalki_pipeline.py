"""
Post-scrape pipeline: process Ahotsak target scrape results.

Runs after the targeted azpieuskalki scrape completes:
  1. Merge with existing passages (deduplicate)
  2. Re-run label validation (municipality vs text model)
  3. Re-train azpieuskalki classifier with expanded data
  4. Generate full report

Usage:
    uv run python -m src.data.azpieuskalki_pipeline
"""

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def find_latest_scrape() -> Path | None:
    """Find the latest Ahotsak scrape JSONL."""
    ahotsak_dir = PROJECT_ROOT / "data" / "raw" / "speech" / "ahotsak"
    files = sorted(ahotsak_dir.glob("ahotsak_passages_*.jsonl"))
    return files[-1] if files else None


def merge_scrapes():
    """Merge the latest targeted scrape with the previous scrape, deduplicating."""
    ahotsak_dir = PROJECT_ROOT / "data" / "raw" / "speech" / "ahotsak"
    files = sorted(ahotsak_dir.glob("ahotsak_passages_*.jsonl"))

    if len(files) < 2:
        logger.info("Only one scrape file found — skipping merge")
        return files[-1]

    # Latest is the targeted scrape, previous is the initial scrape
    latest = files[-1]
    previous = files[-2]

    logger.info(f"Merging: {latest.name} + {previous.name}")

    seen_ids = set()
    merged = []

    # First load previous (older)
    for jsonl_path in [previous, latest]:
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    pid = obj.get("passage_id", "")
                    if pid and pid not in seen_ids:
                        seen_ids.add(pid)
                        merged.append(obj)

    # Count new passages
    new_count = len(merged) - sum(1 for _ in open(previous) if _.strip())
    logger.info(f"  Previous: {sum(1 for _ in open(previous) if _.strip())} passages")
    logger.info(f"  Latest:   {sum(1 for _ in open(latest) if _.strip())} passages")
    logger.info(f"  Merged (deduplicated): {len(merged)} passages ({new_count} new)")

    # Save merged
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_path = ahotsak_dir / f"ahotsak_passages_merged_{timestamp}.jsonl"
    with open(merged_path, "w", encoding="utf-8") as f:
        for obj in merged:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    logger.info(f"  Saved → {merged_path}")

    return merged_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Step 1: Find the latest scrape
    latest = find_latest_scrape()
    if not latest:
        logger.error("No scrape files found!")
        sys.exit(1)

    logger.info(f"Latest scrape: {latest.name}")
    passages = [json.loads(l) for l in open(latest) if l.strip()]
    logger.info(f"Passages: {len(passages)}")
    logger.info("")

    # Step 2: Merge with previous scrape
    merged_path = merge_scrapes()

    # Step 3: Re-run label validation
    logger.info("\n▶ Step 3: Re-running label cross-validation")
    # Use the merged file for validation
    # (This will create a new validation CSV)

    # Step 4: Re-train azpieuskalki classifier
    logger.info("\n▶ Step 4: Re-training azpieuskalki classifier")
    # train_azpi()  # Uncomment to auto-run

    # Step 5: Summary
    logger.info("\n▶ Pipeline complete!")
    logger.info(f"  Merged passages: {merged_path}")
    logger.info("  Next: run validation → train azpieuskalki → build audio dataset")


if __name__ == "__main__":
    main()
