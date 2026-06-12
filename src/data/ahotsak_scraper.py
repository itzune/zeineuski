"""
Ahotsak.eus scraper for Basque dialect-labeled speech data.

Extracts: audio URLs, manual transcriptions (dialectal, not Batua-normalized),
speaker metadata, and municipality → dialect labels.

Data model:
  herria (town) → hizlariak (speakers) → pasarteak (passages) → transcription + audio

Usage:
    uv run python -m src.data.ahotsak_scraper index           # Index all towns
    uv run python -m src.data.ahotsak_scraper scrape --town bermeo --limit 10
    uv run python -m src.data.ahotsak_scraper scrape --all --limit 5
    uv run python -m src.data.ahotsak_scraper stats           # Show collected stats
"""

from __future__ import annotations

import csv
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

AHOTSAK_BASE = "https://ahotsak.eus"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MUNICIPALITY_CSV = PROJECT_ROOT / "data" / "reference" / "municipality_dialect.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "speech" / "ahotsak"
TRANSCRIPTION_ICON = "/static/img/i_transkri.gif"

USER_AGENT = "ZeineuskiML/0.1 (Basque dialect research; xezpeleta@gmail.com; rate-limited scraper)"

# Be very respectful with rate limits
REQUEST_DELAY = 1.0  # seconds between requests
TIMEOUT = 20

# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class Passage:
    """A single transcribed passage from an interview."""

    passage_id: str  # e.g., brm-001-003
    town_slug: str  # e.g., bermeo
    town_name: str  # e.g., Bermeo
    title: str  # passage title
    transcription: str  # full text in dialectal Basque
    speaker_name: str
    speaker_slug: str
    project: str  # e.g., Bermeoko ahotsak
    interviewer: str  # elkarrizketatzailea
    date: str  # recording date
    duration: str  # e.g., 0:02:53
    reference: str  # e.g., BRM-001/003
    audio_url: str  # S3 URL
    video_url: str  # S3 URL
    topics: list[str] = field(default_factory=list)
    dialect_class: str = ""  # filled from municipality map
    dialect_confidence: str = ""  # high/medium/low
    scraped_at: str = ""


# ── Municipality mapping ──────────────────────────────────────────────────────


def _normalize_town(name: str) -> str:
    """Normalize town name for lookup: lowercase, hyphens→spaces."""
    return name.strip().lower().replace("-", " ")


def load_municipality_map() -> dict[str, tuple[str, str]]:
    """Load municipality→dialect mapping. Returns {herria_normalized: (dialect_class, confidence)}."""
    mapping = {}
    with open(MUNICIPALITY_CSV) as f:
        for row in csv.DictReader(f):
            herria = _normalize_town(row["herria"])
            dialect = row["dialect_class"].strip().lower()
            confidence = row["dialect_confidence"].strip()
            if confidence in ("high", "medium", "low"):
                mapping[herria] = (dialect, confidence)
    return mapping


# ── Session ───────────────────────────────────────────────────────────────────


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


# ── Indexing: discover all towns ──────────────────────────────────────────────


def index_towns(session: Optional[requests.Session] = None) -> list[dict]:
    """Index all towns from the herriak page, including transcription counts.

    Returns list of {slug, name, url, territory, speakers, tapes, passages,
                      transcription_count, audio_count, video_count}.
    """
    if session is None:
        session = create_session()

    logger.info("Indexing towns from herriak page (with transcription counts)...")

    resp = session.get(f"{AHOTSAK_BASE}/herriak/", timeout=TIMEOUT)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    towns = []

    # The herriak page has multiple tables, one per province
    province_map = {
        0: "Araba",  # Tables[0] is Araba
        1: "Bizkaia",  # Tables[1] is Bizkaia
        # Tables[2] is Diaspora (not a province, skip)
        3: "Gipuzkoa",  # Tables[3] is Gipuzkoa
        4: "Lapurdi",  # Tables[4] is Lapurdi
        5: "Nafarroa",  # Tables[5] is Nafarroa
        6: "Nafarroa Beherea",  # Tables[6] is Nafarroa Beherea
        7: "Zuberoa",  # Tables[7] is Zuberoa
    }

    territory_counts = {
        0: "araba",
        1: "bizkaia",
        3: "gipuzkoa",
        4: "lapurdi",
        5: "nafarroa",
        6: "nafarroa_beherea",
        7: "zuberoa",
    }

    for table_idx, table in enumerate(soup.find_all("table", class_="herriak")):
        province = province_map.get(table_idx, f"Unknown-{table_idx}")
        territory = territory_counts.get(table_idx, "unknown")

        # If it's the Diaspora table (idx 2), skip
        if table_idx == 2:
            continue

        for tr in table.find_all("tr")[1:]:  # skip header row
            cells = tr.find_all("td")
            if not cells or len(cells) < 2:
                continue

            link = cells[1].find("a")
            if not link:
                continue

            town_name = link.get_text(strip=True)
            href = link.get("href", "")
            slug = href.rstrip("/").split("/")[-1] if href else town_name.lower()
            url = urljoin(AHOTSAK_BASE, href) if href else f"{AHOTSAK_BASE}/{slug}/"

            # Parse numeric fields
            speakers = _parse_int(cells[2]) if len(cells) > 2 else 0
            tapes = _parse_int(cells[3]) if len(cells) > 3 else 0
            passages = _parse_int(cells[4]) if len(cells) > 4 else 0

            # Extract transcription/audio/video counts from cell[5] img titles
            trans_count = 0
            audio_count = 0
            video_count = 0
            if len(cells) > 5:
                for img in cells[5].find_all("img"):
                    title = img.get("title", "")
                    if title.startswith("Transkripzioak:"):
                        trans_count = _parse_int_from_title(title)
                    elif title.startswith("Audioak:"):
                        audio_count = _parse_int_from_title(title)
                    elif title.startswith("Bideoak:"):
                        video_count = _parse_int_from_title(title)

            towns.append(
                {
                    "slug": slug,
                    "name": town_name,
                    "url": url,
                    "territory": territory,
                    "province": province,
                    "speakers": speakers,
                    "tapes": tapes,
                    "passages": passages,
                    "transcription_count": trans_count,
                    "audio_count": audio_count,
                    "video_count": video_count,
                }
            )

    logger.info(f"  Found {len(towns)} towns from page")

    # Summary
    towns_with_trans = sum(1 for t in towns if t["transcription_count"] > 0)
    total_trans = sum(t["transcription_count"] for t in towns)
    logger.info(f"  Towns with transcriptions: {towns_with_trans}")
    logger.info(f"  Total transcriptions: {total_trans}")

    return towns


def _parse_int(cell) -> int:
    """Parse integer from a BeautifulSoup cell, handling dots as thousand separators."""
    text = cell.get_text(strip=True).replace(".", "")
    try:
        return int(text)
    except (ValueError, TypeError):
        return 0


def _parse_int_from_title(title: str) -> int:
    """Extract integer from title like 'Transkripzioak: 139'."""
    num_part = title.split(":")[-1].strip().replace(".", "")
    try:
        return int(num_part)
    except ValueError:
        return 0


# ── Scraping: speakers ────────────────────────────────────────────────────────


def scrape_speakers(
    town_slug: str, session: Optional[requests.Session] = None
) -> list[dict]:
    """Scrape speakers for a town from the town page's speaker table.

    The town page has a table with columns: #, Hizlaria, Zintak, Pasarteak, [icons].
    The last column has img icons with titles like 'Transkripzioak: 22'.
    Only returns speakers with transcriptions (Transkripzioak > 0).
    """
    if session is None:
        session = create_session()

    town_url = f"{AHOTSAK_BASE}/{town_slug}/"
    time.sleep(REQUEST_DELAY)
    resp = session.get(town_url, timeout=TIMEOUT)

    if resp.status_code != 200:
        logger.warning(f"  Town {town_slug} returned {resp.status_code}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    speakers = []
    seen = set()

    # Find the speaker table (headers: #, Hizlaria, Zintak, Pasarteak, [icons])
    for table in soup.find_all("table"):
        headers = [h.get_text(strip=True) for h in table.find_all("th")]
        if "Hizlaria" not in headers and "Hizlariak" not in str(headers):
            continue

        for tr in table.find_all("tr")[1:]:  # skip header
            link = tr.find("a", href=True)
            if not link:
                continue

            href = link["href"]
            name = link.get_text(strip=True)
            speaker_slug = href.rstrip("/").split("/")[-1]

            if not name or not speaker_slug or speaker_slug in seen:
                continue

            # Extract transcription count from icons in this row
            trans_count = 0
            for img in tr.find_all("img"):
                title = img.get("title", "")
                if title.startswith("Transkripzioak:"):
                    trans_count = _parse_int_from_title(title)

            if trans_count > 0:
                seen.add(speaker_slug)
                url = urljoin(town_url, href)
                if not url.endswith("/"):
                    url += "/"
                speakers.append(
                    {
                        "name": name,
                        "slug": speaker_slug,
                        "url": url,
                        "town_slug": town_slug,
                        "transcription_count": trans_count,
                    }
                )

    return speakers


def scrape_speaker_page(
    speaker: dict, session: Optional[requests.Session] = None
) -> dict:
    """Scrape a speaker page for metadata and passage list."""
    if session is None:
        session = create_session()

    url = speaker["url"]
    if not url.endswith("/"):
        url += "/"

    time.sleep(REQUEST_DELAY)
    resp = session.get(url, timeout=TIMEOUT)

    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}", **speaker}

    soup = BeautifulSoup(resp.text, "html.parser")
    content = soup.find("div", id="maincontainer") or soup.find("div", id="content")
    if not content:
        return {"error": "no content", **speaker}

    # Extract passage list. Since we only visit speakers known to have
    # transcriptions, grab all passages. Filter later by actual presence of
    # Transkripzioa on the passage page.
    passages = []
    tables = content.find_all("table")
    if len(tables) >= 2:
        passage_rows = tables[1].find_all("tr")
        for row in passage_rows:
            link = row.find("a", href=re.compile(r"/pasarteak/"))
            if not link:
                continue

            passage_url = urljoin(AHOTSAK_BASE, link["href"])
            if not passage_url.endswith("/"):
                passage_url += "/"

            title = link.get_text(strip=True)

            # Check for transcription icon — it's in the last cell of the row,
            # not in the same cell as the link.
            has_transcription = False
            cells = row.find_all("td")
            if cells:
                last_cell = cells[-1]  # the icon column
                trans_icon = last_cell.find("img", title="Transkribapena eginda")
                if trans_icon:
                    has_transcription = True

            passage_slug = passage_url.rstrip("/").split("/")[-1]

            passages.append(
                {
                    "slug": passage_slug,
                    "title": title,
                    "url": passage_url,
                    "has_transcription": has_transcription,
                }
            )

    return {
        **speaker,
        "passages": passages,
        "num_passages": len(passages),
        "num_transcribed": sum(1 for p in passages if p["has_transcription"]),
    }


# ── Scraping: passages with transcription ─────────────────────────────────────


def scrape_passage(
    town_slug: str,
    passage_slug: str,
    dialect_map: dict,
    session: Optional[requests.Session] = None,
) -> Optional[Passage]:
    """Scrape a single passage page for transcription and metadata."""
    if session is None:
        session = create_session()

    url = f"{AHOTSAK_BASE}/{town_slug}/pasarteak/{passage_slug}/"
    time.sleep(REQUEST_DELAY)
    resp = session.get(url, timeout=TIMEOUT)

    if resp.status_code != 200:
        logger.debug(f"    {passage_slug}: HTTP {resp.status_code}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    # The content is in div#maincontainer, not div#content
    content = soup.find("div", id="maincontainer") or soup.find("div", id="content")
    if not content:
        return None

    # ── Extract transcription ──
    transcription = ""
    trans_header = None
    for h in content.find_all(["h2", "h3", "h4"]):
        if h.get_text(strip=True) == "Transkripzioa":
            trans_header = h
            break

    if not trans_header:
        # No transcription on this page — skip
        return None

    # Get text after the Transkripzioa header
    trans_parts = []
    next_elem = trans_header.find_next_sibling()
    while next_elem and next_elem.name not in ("h2", "h3", "h4"):
        text = next_elem.get_text(strip=True)
        if text and len(text) > 5:
            trans_parts.append(text)
        next_elem = next_elem.find_next_sibling()

    transcription = "\n".join(trans_parts)
    if not transcription or len(transcription) < 20:
        return None

    # ── Extract metadata ──
    # Title
    title = ""
    h1 = content.find("h1")
    if h1:
        title = h1.get_text(strip=True)

    # Metadata block: Proiektua, Data, Iraupena, Erref, Hizlaria(k), etc.
    metadata = {}
    for strong in content.find_all("strong"):
        key = strong.get_text(strip=True).rstrip(":")
        next_text = ""
        sibling = strong.next_sibling
        while sibling and not (hasattr(sibling, "name") and sibling.name == "strong"):
            if hasattr(sibling, "get_text"):
                next_text += sibling.get_text(strip=True)
            elif isinstance(sibling, str):
                next_text += sibling.strip()
            sibling = sibling.next_sibling
        if next_text:
            metadata[key] = next_text.strip()

    # Extract individual fields
    speaker_name = metadata.get("Hizlaria(k)", "")
    project = metadata.get("Proiektua", "")
    interviewer = metadata.get("Elkarrizketatzailea(k)", "")
    date = metadata.get("Data", "")
    duration = metadata.get("Iraupena", "")
    reference = metadata.get("Erref", passage_slug)
    _coder = metadata.get("Kodifikatzailea", "")

    # Speaker name may also be in a link
    if not speaker_name:
        sp_link = content.find("a", href=re.compile(r"/hizlariak/"))
        if sp_link:
            speaker_name = sp_link.get_text(strip=True)

    speaker_slug = ""
    sp_link = content.find("a", href=re.compile(r"/hizlariak/"))
    if sp_link:
        speaker_slug = sp_link["href"].rstrip("/").split("/")[-1]

    # ── Extract audio/video URLs ──
    audio_url = ""
    video_url = ""

    # Look for MP4 (video) and MP3 (audio) sources
    for src in soup.find_all("source", src=True):
        s = src["src"]
        if s.endswith(".mp4"):
            video_url = s
        elif s.endswith(".mp3"):
            audio_url = s

    # Also check for direct links and JavaScript player configs
    if not video_url:
        for a in content.find_all("a", href=re.compile(r"\.mp[34]")):
            if a["href"].endswith(".mp4"):
                video_url = a["href"]
            elif a["href"].endswith(".mp3"):
                audio_url = a["href"]

    # Also search in script tags (sometimes player URLs are in JS)
    if not video_url and not audio_url:
        for script in soup.find_all("script"):
            if script.string:
                mp4_match = re.search(
                    r'["\'](https?://[^"\']+\.mp4)["\']', script.string
                )
                if mp4_match:
                    video_url = mp4_match.group(1)

    # ── Extract topics ──
    topics = []
    gaia_text = metadata.get("Gaia(k)", "")
    if gaia_text:
        # Topics are separated by commas or semicolons
        topics = [t.strip() for t in re.split(r"[;,]", gaia_text) if t.strip()]

    # ── Map to dialect ──
    town_lower = _normalize_town(town_slug)
    dialect_class, dialect_confidence = dialect_map.get(town_lower, ("", ""))

    # Also try matching by town name from metadata
    actual_town = town_slug
    herria_text = metadata.get("Herria", "")
    if herria_text:
        actual_town = herria_text

    return Passage(
        passage_id=passage_slug,
        town_slug=town_slug,
        town_name=actual_town,
        title=title,
        transcription=transcription,
        speaker_name=speaker_name,
        speaker_slug=speaker_slug,
        project=project,
        interviewer=interviewer,
        date=date,
        duration=duration,
        reference=reference,
        audio_url=audio_url,
        video_url=video_url,
        topics=topics,
        dialect_class=dialect_class,
        dialect_confidence=dialect_confidence,
        scraped_at=datetime.now(timezone.utc).isoformat(),
    )


def passage_to_dict(p: Passage) -> dict:
    return {
        "passage_id": p.passage_id,
        "town_slug": p.town_slug,
        "town_name": p.town_name,
        "title": p.title,
        "transcription": p.transcription,
        "speaker_name": p.speaker_name,
        "speaker_slug": p.speaker_slug,
        "project": p.project,
        "interviewer": p.interviewer,
        "date": p.date,
        "duration": p.duration,
        "reference": p.reference,
        "audio_url": p.audio_url,
        "video_url": p.video_url,
        "topics": p.topics,
        "dialect_class": p.dialect_class,
        "dialect_confidence": p.dialect_confidence,
        "scraped_at": p.scraped_at,
    }


# ── Main scraping orchestration ────────────────────────────────────────────────


def scrape_town(
    town_slug: str,
    dialect_map: dict,
    max_passages: Optional[int] = None,
    session: Optional[requests.Session] = None,
) -> list[Passage]:
    """Scrape all transcribed passages for a town."""
    if session is None:
        session = create_session()

    logger.info(f"Scraping {town_slug}...")

    # Step 1: Get speakers with transcriptions (from town page icons)
    speakers = scrape_speakers(town_slug, session)
    logger.info(f"  {len(speakers)} speakers with transcriptions")

    if not speakers:
        return []

    all_passages: list[Passage] = []

    # Step 2: For each speaker, get their transcribed passage links, then scrape each
    for sp in speakers:
        if max_passages and len(all_passages) >= max_passages:
            break

        sp_data = scrape_speaker_page(sp, session)
        if "error" in sp_data:
            continue

        transcribed = [p for p in sp_data.get("passages", []) if p["has_transcription"]]
        logger.debug(
            f"  Speaker {sp['name']}: {len(transcribed)} transcribed (expected ~{sp.get('transcription_count', '?')})"
        )

        for p_data in transcribed:
            if max_passages and len(all_passages) >= max_passages:
                break

            passage = scrape_passage(town_slug, p_data["slug"], dialect_map, session)
            if passage:
                all_passages.append(passage)

    logger.info(f"  Total passages scraped: {len(all_passages)}")
    return all_passages


def scrape_all_towns(
    max_passages_per_town: int = 20,
    dialect_map: Optional[dict] = None,
    session: Optional[requests.Session] = None,
) -> list[Passage]:
    """Scrape transcribed passages from all towns with dialect mapping.

    Only targets towns that have transcriptions (based on herriak page index).
    """
    if dialect_map is None:
        dialect_map = load_municipality_map()
    if session is None:
        session = create_session()

    towns = index_towns(session)

    # Filter: only towns with transcriptions AND dialect mapping
    viable = [
        t
        for t in towns
        if t["transcription_count"] > 0 and t["slug"].lower() in dialect_map
    ]

    no_trans = [
        t
        for t in towns
        if t["transcription_count"] == 0 and t["slug"].lower() in dialect_map
    ]
    no_mapping = [
        t
        for t in towns
        if t["transcription_count"] > 0 and t["slug"].lower() not in dialect_map
    ]

    logger.info(f"Viable towns (transcriptions + dialect mapping): {len(viable)}")
    logger.info(f"Towns with mapping but no transcriptions: {len(no_trans)}")
    logger.info(f"Towns with transcriptions but no mapping: {len(no_mapping)}")
    if no_mapping:
        logger.info(f"  Unmapped (sample): {[t['name'] for t in no_mapping[:10]]}")

    # Sort by transcription count descending (most data first)
    viable.sort(key=lambda t: t["transcription_count"], reverse=True)

    all_passages: list[Passage] = []
    session = create_session()
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_dir / f"ahotsak_passages_{timestamp}.jsonl"

    for town in viable:
        passages = scrape_town(
            town["slug"],
            dialect_map,
            max_passages=max_passages_per_town,
            session=session,
        )
        all_passages.extend(passages)

        # Incremental save to avoid data loss on timeout
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for p in passages:
                f.write(json.dumps(passage_to_dict(p), ensure_ascii=False) + "\n")

        dialect = dialect_map.get(town["slug"].lower(), ("unknown", ""))[0]
        logger.info(
            f"  [{town['slug']}] ({town['territory']}/{dialect}): "
            f"{len(passages)} passages scraped "
            f"(total: {len(all_passages)}, "
            f"expected: {town['transcription_count']})"
        )

    return all_passages


def scrape_targeted_towns(
    per_town_limits: dict[str, int],
    dialect_map: Optional[dict] = None,
    session: Optional[requests.Session] = None,
) -> list[Passage]:
    """Scrape a specific set of towns with individual limits.

    Args:
        per_town_limits: dict mapping town slug → max passages
    """
    if dialect_map is None:
        dialect_map = load_municipality_map()
    if session is None:
        session = create_session()

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_dir / f"ahotsak_passages_{timestamp}.jsonl"

    all_passages: list[Passage] = []
    session_obj = create_session()

    towns = sorted(per_town_limits.items(), key=lambda x: -x[1])
    logger.info(f"Targeted scrape: {len(towns)} towns")

    for i, (town_slug, limit) in enumerate(towns):
        dialect = dialect_map.get(town_slug.lower(), ("unknown", ""))[0]
        logger.info(f"[{i + 1}/{len(towns)}] {town_slug} ({dialect}) — limit: {limit}")

        passages = scrape_town(
            town_slug, dialect_map, max_passages=limit, session=session_obj
        )
        all_passages.extend(passages)

        # Incremental save
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for p in passages:
                f.write(json.dumps(passage_to_dict(p), ensure_ascii=False) + "\n")

        logger.info(f"  → {len(passages)} passages (total: {len(all_passages)})")

    # Also save TSV
    tsv_path = output_dir / f"ahotsak_transcriptions_{timestamp}.tsv"
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            ["text", "dialect", "town", "speaker", "passage_id", "confidence"]
        )
        for p in all_passages:
            writer.writerow(
                [
                    p.transcription,
                    p.dialect_class,
                    p.town_name or p.town_slug,
                    p.speaker_name,
                    p.passage_id,
                    p.dialect_confidence,
                ]
            )

    logger.info(f"\nFinal JSONL: {jsonl_path} ({len(all_passages)} passages)")
    logger.info(f"Final TSV:   {tsv_path}")

    return all_passages


def save_passages(
    passages: list[Passage],
    output_path: Optional[Path] = None,
):
    """Save scraped passages to JSONL and TSV."""
    output_path = output_path or OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as JSONL (full data)
    jsonl_path = output_path / f"ahotsak_passages_{timestamp}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(passage_to_dict(p), ensure_ascii=False) + "\n")

    # Save as TSV (text + dialect for training)
    tsv_path = output_path / f"ahotsak_transcriptions_{timestamp}.tsv"
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "text",
                "dialect",
                "town",
                "speaker",
                "title",
                "passage_id",
                "confidence",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for p in passages:
            writer.writerow(
                {
                    "text": p.transcription.replace("\n", " "),
                    "dialect": p.dialect_class,
                    "town": p.town_name,
                    "speaker": p.speaker_name,
                    "title": p.title,
                    "passage_id": p.passage_id,
                    "confidence": p.dialect_confidence,
                }
            )

    logger.info(f"Saved {len(passages)} passages → {jsonl_path}")
    logger.info(f"Saved transcriptions → {tsv_path}")


# ── Stats ──────────────────────────────────────────────────────────────────────


def show_stats(passages: list[Passage]):
    """Print statistics about scraped passages."""
    from collections import Counter

    if not passages:
        logger.info("No passages collected yet.")
        return

    dialect_counter = Counter(p.dialect_class for p in passages if p.dialect_class)
    town_counter = Counter(p.town_name for p in passages)
    total_chars = sum(len(p.transcription) for p in passages)
    transcribed_with_audio = sum(1 for p in passages if p.audio_url or p.video_url)

    print("\n=== Ahotsak Scraped Data Stats ===")
    print(f"Total passages: {len(passages)}")
    print(f"Total chars: {total_chars:,}")
    print(f"Avg chars/passage: {total_chars // max(len(passages), 1):,}")
    print(f"With audio/video URL: {transcribed_with_audio}/{len(passages)}")

    print("\nDialect distribution:")
    for dialect, count in dialect_counter.most_common():
        print(f"  {dialect}: {count}")

    print("\nTop towns:")
    for town, count in town_counter.most_common(10):
        print(f"  {town}: {count}")

    print(f"\nUnique towns: {len(town_counter)}")
    print(f"Unique speakers: {len(set(p.speaker_name for p in passages))}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def cmd_index():
    """Index all towns with dialect mapping."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dialect_map = load_municipality_map()
    towns = index_towns()

    mapped = sum(1 for t in towns if t["slug"].lower() in dialect_map)
    logger.info(f"Total towns: {len(towns)}")
    logger.info(f"Towns with dialect mapping: {mapped}")
    logger.info(f"Towns without mapping: {len(towns) - mapped}")

    # Save index
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = OUTPUT_DIR / "town_index.json"
    with open(index_path, "w") as f:
        json.dump(towns, f, ensure_ascii=False, indent=2)
    logger.info(f"Index saved → {index_path}")


def cmd_scrape(
    town: str = "all",
    limit: int = 10,
    output: Optional[str] = None,
    targets_file: Optional[str] = None,
):
    """Scrape passages from one or all towns.

    Args:
        town: Town slug (e.g., 'bermeo') or 'all' for all mapped towns.
        limit: Max passages per town.
        output: Output directory for JSONL/TSV files.
        targets_file: JSON file with {town_slug: max_passages} for targeted scraping.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dialect_map = load_municipality_map()
    output_dir = Path(output) if output else OUTPUT_DIR

    if targets_file:
        # Targeted scrape with per-town limits from a JSON file
        with open(targets_file) as f:
            per_town_limits = json.load(f)
        passages = scrape_targeted_towns(per_town_limits, dialect_map)
        if passages:
            show_stats(passages)
        else:
            logger.warning("No passages found.")
    elif town == "all":
        session = create_session()
        passages = scrape_all_towns(
            max_passages_per_town=limit,
            dialect_map=dialect_map,
            session=session,
        )
        if passages:
            save_passages(passages, output_dir)
            show_stats(passages)
        else:
            logger.warning("No passages found.")
    else:
        session = create_session()
        passages = scrape_town(
            town_slug=town,
            dialect_map=dialect_map,
            max_passages=limit,
            session=session,
        )
        if passages:
            save_passages(passages, output_dir)
            show_stats(passages)
        else:
            logger.warning("No passages found.")


def cmd_stats():
    """Show stats from previously scraped data."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Find the latest JSONL file
    jsonl_files = sorted(OUTPUT_DIR.glob("ahotsak_passages_*.jsonl"))
    if not jsonl_files:
        logger.warning("No scraped data found. Run 'scrape' first.")
        return

    latest = jsonl_files[-1]
    logger.info(f"Loading {latest}...")

    passages = []
    with open(latest) as f:
        for line in f:
            data = json.loads(line)
            p = Passage(
                **{k: v for k, v in data.items() if k in Passage.__dataclass_fields__}
            )
            passages.append(p)

    show_stats(passages)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.data.ahotsak_scraper [index|scrape|stats]\n"
            "  index                    Index all towns\n"
            "  scrape --town X          Scrape one town\n"
            "  scrape --all             Scrape all mapped towns\n"
            "  scrape --targets-file F  Scrape towns from JSON {slug: max_passages}\n"
            "  stats                    Show stats from latest scrape\n"
            "\nOptions for scrape:\n"
            "  --town TOWN              Town slug (default: all)\n"
            "  --all                    Scrape all mapped towns\n"
            "  --limit N                Max passages per town (default: 10)\n"
            "  --targets-file FILE.json JSON file with {town_slug: max_passages}\n"
            "  --output DIR             Output directory\n"
        )
        sys.exit(1)

    cmd = sys.argv[1]
    kwargs: dict = {}
    i = 2
    while i < len(sys.argv):
        if sys.argv[i].startswith("--"):
            key = sys.argv[i][2:].replace("-", "_")
            if key == "all":
                kwargs["town"] = "all"
                i += 1
                continue
            raw = (
                sys.argv[i + 1]
                if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--")
                else "true"
            )
            try:
                val = int(raw)
            except ValueError:
                val = raw
            kwargs[key] = val
            i += (
                2
                if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--")
                else 1
            )
        else:
            i += 1

    if cmd == "index":
        cmd_index()
    elif cmd == "scrape":
        cmd_scrape(**kwargs)
    elif cmd == "stats":
        cmd_stats()
    else:
        print(f"Unknown command: {cmd}")
