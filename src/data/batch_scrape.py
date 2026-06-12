"""Batch scrape specific towns for azpieuskalki data expansion.

Usage:
    uv run python src/data/batch_scrape.py
"""

import json
import time
from pathlib import Path

from src.data.ahotsak_scraper import (
    scrape_town,
    load_municipality_map,
    passage_to_dict,
    create_session,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AHOTSAK_DIR = PROJECT_ROOT / "data" / "raw" / "speech" / "ahotsak"
PASSAGES_FILE = AHOTSAK_DIR / "ahotsak_passages_20260608_210341.jsonl"
MUNI_CSV = PROJECT_ROOT / "data" / "reference" / "municipality_dialect.csv"

# Towns to scrape grouped by motivation
TOWNS_MENDEBAL_SARTALDEA = [
    "zeberio",
    "orozko",
    "bilbo",
    "igorre",
    "mungia",
    "derio",
    "galdakao",
    "getxo",
]

TOWNS_NAFAR_ERDIGUNEA = [
    "odieta",
    "lantz",
    "txulapain",
    "anue",
    "ezkabarte",
    "olaibar",
    "atez",
]

TOWNS_NAFLAP_SARTALDEA = [
    "arbona",
    "sara",
    "ainhoa",
    "donibane-lohizune",
    "baigorri",
]

ALL_TARGETS = TOWNS_MENDEBAL_SARTALDEA + TOWNS_NAFAR_ERDIGUNEA + TOWNS_NAFLAP_SARTALDEA


def load_existing_passages() -> list[dict]:
    """Load all existing passages."""
    passages = []
    if PASSAGES_FILE.exists():
        with open(PASSAGES_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    passages.append(json.loads(line))
    return passages


def main():
    # Load existing data
    existing = load_existing_passages()
    existing_slugs = set(p["town_slug"] for p in existing)
    dialect_map = load_municipality_map()
    session = create_session()

    print(f"Existing passages: {len(existing)} from {len(existing_slugs)} towns")
    print()

    # Filter to unscraped towns
    to_scrape = [t for t in ALL_TARGETS if t.lower() not in existing_slugs]
    if not to_scrape:
        print("All targets already scraped!")
        return

    # Also include towns that were scraped but with too few passages
    # (We don't know the limit used before, but re-scraping is wasteful)
    print(f"Towns to scrape: {len(to_scrape)}")
    for t in to_scrape:
        print(f"  {t}")
    print()

    new_passages = []
    success = 0
    failed = 0

    for i, town_slug in enumerate(to_scrape):
        print(f"[{i + 1}/{len(to_scrape)}] {town_slug}...")
        try:
            passages = scrape_town(town_slug, dialect_map, session=session)
            if passages:
                new_passages.extend(passages)
                print(f"  ✓ {len(passages)} passages")
                success += 1
            else:
                print("  ○ 0 passages (no transcriptions)")
                success += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        time.sleep(0.5)  # be polite

    if not new_passages:
        print("\nNo new passages scraped!")
        return

    # Merge with existing
    existing_ids = set(p["passage_id"] for p in existing)
    merged = list(existing)
    added = 0
    for p in new_passages:
        p_dict = passage_to_dict(p)
        if p_dict["passage_id"] not in existing_ids:
            merged.append(p_dict)
            added += 1
            existing_ids.add(p_dict["passage_id"])

    # Save merged
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = AHOTSAK_DIR / f"ahotsak_passages_{timestamp}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for p in merged:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Also update the canonical file
    with open(PASSAGES_FILE, "w", encoding="utf-8") as f:
        for p in merged:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 50}")
    print(f"Done: {success} success, {failed} failed")
    print(f"New passages: {added}")
    print(f"Total passages: {len(merged)}")
    print(f"Saved to: {out_path} (and {PASSAGES_FILE})")


if __name__ == "__main__":
    main()
