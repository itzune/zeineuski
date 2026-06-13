"""
Mintzoak.eus passage scraper — global listing + concurrent Vimeo probing.

Strategy:
1. One request to /eu/<any_town>/pasarteak/ gets ALL 7,436 passages globally
2. Concurrently probe each passage page to extract:
   - Town (from Erreferentzia link)
   - Dialect label (mapped from region taxonomy)
   - Vimeo URL (if available)
3. Output only passages with Vimeo content

The listing page doesn't filter by town — the town in the URL is decorative.
Each passage page contains the actual town via the Erreferentzia field.

Usage:
    uv run python -m src.data.mintzoak_scraper \
      --output data/raw/speech/mintzoak/mintzoak_passages.jsonl

    uv run python -m src.data.mintzoak_scraper \
      --regions xiberoa --workers 20 \
      --output data/raw/speech/mintzoak/mintzoak_passages_xiberoa.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://www.mintzoak.eus"
USER_AGENT = (
    "ZeineuskiML/0.1 "
    "(Basque dialect research; xezpeleta@gmail.com; rate-limited downloader)"
)
SESSION = None

REGION_TO_DIALECT = {
    "xiberoa": "souletin",
    "amikuze": "nav-lab",
    "bidaxuneko-lurraldea": "nav-lab",
    "errobi": "nav-lab",
    "errobi-aturri": "nav-lab",
    "euskal-kostaldea-aturri": "nav-lab",
    "garazi-baigorri": "nav-lab",
    "hazparneko-lurraldea": "nav-lab",
    "hego-lapurdi": "nav-lab",
    "iholdi-oztibarre": "nav-lab",
}

ALL_REGIONS = list(REGION_TO_DIALECT.keys())


def _get_regions() -> dict[str, str]:
    """Build town_slug → region_slug mapping from all region pages."""
    town_to_region = {}
    for region in ALL_REGIONS:
        url = f"{BASE_URL}/eu/herriak/?reg={region}"
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
        except requests.RequestException:
            logger.warning(f"Failed to fetch region {region}")
            continue

        skip = {
            "bilatu",
            "karta",
            "herriak",
            "lekukoak",
            "albisteak",
            "kontaktua",
            "grabaketak",
            "info",
            "lekukotasun-sortak",
        }
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a"):
            href = a.get("href", "")
            m = re.match(r"/eu/([a-z0-9-]+)/$", href)
            if m and m.group(1) not in skip:
                town_to_region[m.group(1)] = region

    return town_to_region


def _probe_passage(passage_id: str) -> Optional[dict]:
    """
    Probe a single passage page. Returns dict with town, region, dialect, vimeo_url,
    or None if no Vimeo content.
    """
    # We need to use the town slug from the listing page as a prefix for the URL.
    # The actual passage page is at /eu/<any_town>/pasarteak/<id>/
    # We use eskiula as a placeholder — the server resolves the correct passage
    # regardless of the town in the URL path.
    listing_town = "eskiula"
    url = f"{BASE_URL}/eu/{listing_town}/pasarteak/{passage_id}/"

    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": USER_AGENT})
        if r.status_code != 200:
            return None

        text = r.text

        # Extract Vimeo URL
        vimeo_match = re.search(r"player\.vimeo\.com/video/(\d+)", text)
        if not vimeo_match:
            return None  # No video content for this passage

        vimeo_url = f"https://player.vimeo.com/video/{vimeo_match.group(1)}"

        # Extract Erreferentzia → town slug
        # Format: <a href="/eu/<town>/elkarrizketak/<rec_id>/"><rec_id></a>-<seg>
        ref_match = re.search(
            r"Erreferentzia.*?href=\"(/eu/([a-z0-9-]+)/elkarrizketak/[^\"]+)\"",
            text,
            re.DOTALL,
        )
        town = "unknown"
        if ref_match:
            town = ref_match.group(2)

        return {
            "passage_id": passage_id,
            "town_slug": town,
            "vimeo_url": vimeo_url,
        }

    except requests.RequestException:
        return None


def main():
    parser = argparse.ArgumentParser(description="Mintzoak global passage scraper")
    parser.add_argument(
        "--output",
        default="data/raw/speech/mintzoak/mintzoak_passages.jsonl",
    )
    parser.add_argument(
        "--regions",
        nargs="*",
        choices=ALL_REGIONS,
        help="Only include passages from these regions",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Concurrent workers for probing passage pages",
    )
    parser.add_argument(
        "--max-passages",
        type=int,
        default=0,
        help="Limit total passages to probe (0=all)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Scrape global listing
    listing_town = "eskiula"  # Any town works — listing is global
    logger.info(
        f"Fetching global passage listing (via /eu/{listing_town}/pasarteak/)..."
    )
    listing_url = f"{BASE_URL}/eu/{listing_town}/pasarteak/"
    r = requests.get(listing_url, timeout=30, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()

    passage_ids = list(
        set(re.findall(rf"/eu/{re.escape(listing_town)}/pasarteak/([^/]+)/", r.text))
    )
    logger.info(f"Found {len(passage_ids)} unique passage IDs (global total)")

    if args.max_passages > 0:
        passage_ids = passage_ids[: args.max_passages]

    # Step 2: Build town → region mapping
    logger.info("Building town → region mapping...")
    town_to_region = _get_regions()
    logger.info(
        f"  {len(town_to_region)} towns mapped across {len(ALL_REGIONS)} regions"
    )

    # Step 3: Concurrently probe passage pages
    logger.info(f"Probing {len(passage_ids)} passages with {args.workers} workers...")
    results = []
    with_audio = 0
    without_audio = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_probe_passage, pid): pid for pid in passage_ids}

        for i, future in enumerate(as_completed(futures), 1):
            pid = futures[future]
            try:
                result = future.result()
            except Exception as e:
                logger.warning(f"Error probing {pid}: {e}")
                continue

            if result is None:
                without_audio += 1
                continue

            with_audio += 1
            town = result["town_slug"]
            region = town_to_region.get(town, "unknown")
            dialect = REGION_TO_DIALECT.get(region, "unknown")

            entry = {
                "passage_id": result["passage_id"],
                "town_slug": town,
                "region": region,
                "dialect": dialect,
                "vimeo_url": result["vimeo_url"],
            }
            results.append(entry)

            if i % 50 == 0 or i <= 10:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  [{i}/{len(passage_ids)}] {with_audio} with audio, "
                    f"{without_audio} without ({rate:.1f}/s)"
                )

    elapsed = time.time() - start_time
    logger.info(
        f"Probing done: {with_audio} with Vimeo, {without_audio} without "
        f"({elapsed:.1f}s, {len(passage_ids) / elapsed:.1f}/s)"
    )

    # Step 4: Filter by region if requested
    if args.regions:
        before = len(results)
        region_set = set(args.regions)
        results = [r for r in results if r["region"] in region_set]
        logger.info(
            f"Region filter ({args.regions}): {len(results)}/{before} passages kept"
        )

    # Step 5: Write output
    with open(output_path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Output: {output_path} ({len(results)} passages)")

    # Summary by dialect
    from collections import Counter

    dialect_counts = Counter(r["dialect"] for r in results)
    region_counts = Counter(r["region"] for r in results)
    logger.info(
        "By dialect: "
        + ", ".join(
            f"{d}: {c}" for d, c in sorted(dialect_counts.items(), key=lambda x: -x[1])
        )
    )
    logger.info(
        "By region: "
        + ", ".join(
            f"{r}: {c}" for r, c in sorted(region_counts.items(), key=lambda x: -x[1])
        )
    )


if __name__ == "__main__":
    main()
