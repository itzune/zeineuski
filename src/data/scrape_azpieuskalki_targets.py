"""
Targeted Ahotsak scraper for azpieuskalki data expansion.

Scrapes top-3 towns per azpieuskalki (by available transcription count)
to get sufficient data for a 9-class sub-dialect classifier.

Usage:
    uv run python -m src.data.scrape_azpieuskalki_targets

This will run the ahotsak_scraper for each target town with appropriate limits.
"""

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from src.data.azpieuskalki_map import AZPIEUSKALKI_MAP

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MUNI_CSV = PROJECT_ROOT / "data" / "reference" / "municipality_dialect.csv"
TOWN_INDEX = PROJECT_ROOT / "data" / "raw" / "speech" / "ahotsak" / "town_index.json"


def load_targets():
    """Build list of (town_slug, max_passages) for top-3 towns per azpieuskalki."""
    town_region = {}
    with open(MUNI_CSV) as f:
        for row in csv.DictReader(f):
            town_region[row["herria"].lower()] = {
                "region": row["eskualdea"],
                "azpieuskalki": AZPIEUSKALKI_MAP.get(row["eskualdea"]),
            }

    with open(TOWN_INDEX) as f:
        index = json.load(f)

    azpi_towns = defaultdict(list)
    for t in index:
        name = t.get("name", t.get("town_name", "")).lower()
        if name in town_region:
            azpi = town_region[name]["azpieuskalki"]
            tc = t.get("transcription_count", 0)
            if azpi and tc > 0:
                azpi_towns[azpi].append((name, tc))

    targets = []
    for azpi in sorted(azpi_towns, key=lambda a: -sum(x[1] for x in azpi_towns[a])):
        towns = sorted(azpi_towns[azpi], key=lambda x: -x[1])[:3]
        for name, count in towns:
            limit = min(count, 100)
            targets.append((azpi, name, limit))

    return targets


def main():
    targets = load_targets()

    print(f"Target towns to scrape: {len(targets)}")
    print()

    # Group by azpieuskalki for display
    current_azpi = None
    total_passages = 0
    for azpi, name, limit in targets:
        if azpi != current_azpi:
            if current_azpi:
                print()
            current_azpi = azpi
            print(f"  {azpi}:")
        print(f"    {name:30s} (limit: {limit})")
        total_passages += limit

    print(f"\n  Total potential: ~{total_passages} passages")
    print()

    # Confirm
    response = input("Proceed with scraping? (y/n): ")
    if response.lower() != "y":
        print("Aborted.")
        return

    results = []
    for i, (azpi, name, limit) in enumerate(targets):
        print(f"\n[{i + 1}/{len(targets)}] Scraping {name} ({azpi}) — limit: {limit}")

        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "src.data.ahotsak_scraper",
            "scrape",
            "--town",
            name,
            "--limit",
            str(limit),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT)
        )

        if result.returncode == 0:
            print(f"  ✓ {name}: done")
        else:
            print(f"  ✗ {name}: FAILED")
            print(f"    stderr: {result.stderr[:200]}")

        results.append((name, result.returncode == 0))

    # Summary
    success = sum(1 for _, ok in results if ok)
    failed = len(results) - success
    print(f"\n{'=' * 50}")
    print(f"Done: {success} success, {failed} failed")

    if failed:
        print("Failed towns:")
        for name, ok in results:
            if not ok:
                print(f"  - {name}")


if __name__ == "__main__":
    main()
