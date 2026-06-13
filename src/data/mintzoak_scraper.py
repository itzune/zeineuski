"""
Mintzoak.eus passage scraper — fast listing-page parser.

Strategy: Each town's /eu/<town>/pasarteak/ page renders ALL passage slots
(up to 4604) in a single HTML response. We parse the links to get the full
recording_id→segment mapping without probing individual pages.

Then we lazily verify Vimeo content during download — no need to pre-probe.

Usage:
    uv run python -m src.data.mintzoak_scraper \
      --output data/raw/speech/mintzoak/mintzoak_passages.jsonl

    uv run python -m src.data.mintzoak_scraper \
      --regions xiberoa \
      --output data/raw/speech/mintzoak/mintzoak_passages_xiberoa.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://www.mintzoak.eus"
USER_AGENT = (
    "ZeineuskiML/0.1 "
    "(Basque dialect research; xezpeleta@gmail.com; rate-limited downloader)"
)
REQUEST_DELAY = 0.5  # seconds between towns (one request per town, be nice)

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


def _get_towns_for_region(region: str) -> list[str]:
    """Scrape town slugs for a single Mintzoak region."""
    headers = {"User-Agent": USER_AGENT}
    url = f"{BASE_URL}/eu/herriak/?reg={region}"
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
    r = requests.get(url, timeout=20, headers=headers)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    towns = set()
    for a in soup.find_all("a"):
        href = a.get("href", "")
        m = re.match(r"/eu/([a-z0-9-]+)/$", href)
        if m and m.group(1) not in skip:
            towns.add(m.group(1))
    return sorted(towns)


def scrape_town_passages(town_slug: str) -> list[dict]:
    """
    Parse the single /eu/<town>/pasarteak/ listing page.

    Extracts all passage links: /eu/<town>/pasarteak/<rec_id>-<seg_id>/
    Returns a list of {passage_id, recording_id, segment_id, town_slug}.

    Does NOT probe for Vimeo content — that's done lazily during download.
    """
    headers = {"User-Agent": USER_AGENT}
    url = f"{BASE_URL}/eu/{town_slug}/pasarteak/"

    r = requests.get(url, timeout=30, headers=headers)
    r.raise_for_status()

    # Extract all passage links for this town
    pattern = rf"/eu/{re.escape(town_slug)}/pasarteak/(\d+-\d+)/"
    matches = re.findall(pattern, r.text)

    passages = []
    seen = set()
    for passage_id in matches:
        if passage_id in seen:
            continue
        seen.add(passage_id)

        parts = passage_id.split("-")
        rec_id = int(parts[0])
        seg_id = int(parts[1])

        passages.append(
            {
                "passage_id": passage_id,
                "recording_id": rec_id,
                "segment_id": seg_id,
                "town_slug": town_slug,
            }
        )

    # Sort by recording_id, then segment_id
    passages.sort(key=lambda p: (p["recording_id"], p["segment_id"]))

    return passages


def main():
    parser = argparse.ArgumentParser(description="Mintzoak fast passage scraper")
    parser.add_argument(
        "--output",
        default="data/raw/speech/mintzoak/mintzoak_passages.jsonl",
    )
    parser.add_argument(
        "--regions",
        nargs="*",
        choices=ALL_REGIONS,
        help="Only scrape these regions (default: all)",
    )
    parser.add_argument(
        "--towns",
        nargs="*",
        help="Scrape specific towns (overrides regions)",
    )
    parser.add_argument(
        "--max-towns",
        type=int,
        default=0,
        help="Limit total towns",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build town list
    if args.towns:
        town_entries = [("manual", "unknown", t) for t in args.towns]
    else:
        regions_to_scrape = args.regions if args.regions else ALL_REGIONS
        logger.info(f"Fetching town lists for {len(regions_to_scrape)} region(s)...")
        town_entries = []
        for region in regions_to_scrape:
            towns = _get_towns_for_region(region)
            dialect = REGION_TO_DIALECT[region]
            logger.info(f"  {region}: {len(towns)} towns ({dialect})")
            for town in towns:
                town_entries.append((region, dialect, town))

    if args.max_towns > 0:
        town_entries = town_entries[: args.max_towns]

    logger.info(
        f"Will scrape listing pages for {len(town_entries)} towns (1 request each)"
    )

    total_passages = 0
    start_time = time.time()

    with open(output_path, "w") as f:
        for i, (region, dialect, town_slug) in enumerate(town_entries):
            try:
                passages = scrape_town_passages(town_slug)
                logger.info(
                    f"[{i + 1}/{len(town_entries)}] {town_slug} ({region} → {dialect}): "
                    f"{len(passages)} passage slots"
                )

                for p in passages:
                    entry = {
                        **p,
                        "region": region,
                        "dialect": dialect,
                        "vimeo_url": None,  # filled lazily by downloader
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    total_passages += 1

                if i < len(town_entries) - 1:
                    time.sleep(REQUEST_DELAY)

            except requests.RequestException as e:
                logger.error(f"  {town_slug}: FAILED — {e}")
                continue

    elapsed = time.time() - start_time
    logger.info(
        f"Done! {total_passages} passage slots from {len(town_entries)} towns "
        f"({elapsed:.1f}s)"
    )
    logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
