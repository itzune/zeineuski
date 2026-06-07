"""
Klasikoak.armiarma.eus scraper for Basque dialect-labeled classical literature.

Pipeline:
1. Parse alfa.htm → all work URLs
2. For each work page, extract Zubitegia author link and author name
3. Deduplicate by Zubitegia URL → fetch birthplace
4. Match birthplace against municipality_dialect.csv → dialect label
5. Download chapter texts from each work
6. Output labeled TSV for training
"""

import csv
import html
import re
import time
import urllib.parse
from collections import defaultdict
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

KLASIKOAK_BASE = "https://klasikoak.armiarma.eus"
MUNICIPALITY_CSV = Path("data/reference/municipality_dialect.csv")
OUTPUT_DIR = Path("data/raw/text/klasikoak")
OUTPUT_TSV = OUTPUT_DIR / "klasikoak_labeled.tsv"

REQUEST_DELAY = 0.4
TIMEOUT = 15


def load_municipality_map() -> dict[str, str]:
    mapping = {}
    with open(MUNICIPALITY_CSV) as f:
        for row in csv.DictReader(f):
            herria = row["herria"].strip().lower()
            dialect = row["dialect_class"].strip().lower()
            if row["dialect_confidence"].strip() in ("high", "medium"):
                mapping[herria] = dialect
    return mapping


def extract_municipality(birthplace: str) -> str:
    return birthplace.split("-")[0].strip().lower()


def map_to_dialect(birthplace: str, muni_map: dict) -> tuple[Optional[str], str]:
    mun = extract_municipality(birthplace)
    if mun in muni_map:
        return muni_map[mun], "high"
    for k, v in muni_map.items():
        if mun in k or k in mun:
            return v, "medium"
    return None, "low"


def clean_text(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def split_sentences(text: str, min_words: int = 5) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip().split()) >= min_words]


def scrape_work(work_url: str, session: requests.Session) -> list[str]:
    """
    Fetch all chapters of a work and return list of sentence strings.
    """
    time.sleep(REQUEST_DELAY)
    try:
        resp = session.get(work_url, timeout=TIMEOUT)
        resp.encoding = "iso-8859-1"
    except Exception:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find chapter links in the sidebar
    base_dir = "/".join(work_url.split("/")[:-1])
    base_name = Path(work_url).stem

    chapter_urls: list[str] = []
    for a in soup.find_all("a"):
        href = str(a.get("href", ""))
        if base_name in href and href.endswith(".htm"):
            full = base_dir + "/" + href
            if full != work_url and full not in chapter_urls:
                chapter_urls.append(full)

    if not chapter_urls:
        # Single-page work
        content = extract_content_td(soup)
        return split_sentences(content) if content else []

    all_sentences = []
    for ch_url in chapter_urls[:5]:  # Limit to first 5 chapters
        time.sleep(REQUEST_DELAY * 0.5)
        try:
            ch_resp = session.get(ch_url, timeout=TIMEOUT)
            ch_resp.encoding = "iso-8859-1"
            ch_soup = BeautifulSoup(ch_resp.text, "html.parser")
            content = extract_content_td(ch_soup)
            if content:
                all_sentences.extend(split_sentences(content))
        except Exception:
            pass

    return all_sentences


def extract_content_td(soup: BeautifulSoup) -> str:
    """Extract text from the main content area (right column)."""
    for td in soup.find_all("td"):
        if td.get("valign") == "top" and len(td.get_text(strip=True)) > 100:
            return clean_text(str(td))
    # Fallback: try any large text block
    for td in soup.find_all("td"):
        text = td.get_text(strip=True)
        if len(text) > 500:
            return clean_text(str(td))
    return ""


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    muni_map = load_municipality_map()
    print(f"Municipality map: {len(muni_map)} entries")

    session = requests.Session()
    session.headers.update({"User-Agent": "Zeineuski-Dialect-Research/0.1 (academic)"})

    # Step 1: Collect all work URLs from alfa.htm
    print("\n[1/5] Collecting work URLs...")
    resp = session.get(f"{KLASIKOAK_BASE}/alfa.htm", timeout=TIMEOUT)
    resp.encoding = "iso-8859-1"
    soup = BeautifulSoup(resp.text, "html.parser")

    work_urls = []
    for a in soup.find_all("a"):
        href = a.get("href", "")
        if href.startswith("idazlanak/") and href.endswith(".htm"):
            full = KLASIKOAK_BASE + "/" + href
            if full not in work_urls:
                work_urls.append(full)
    print(f"  Found {len(work_urls)} unique works")

    # Step 2: Extract Zubitegia links and author names from work pages
    print("\n[2/5] Extracting Zubitegia author links...")
    zubi_map: dict[str, dict] = {}  # zubi_url → {name, birthplace, works}

    for i, url in enumerate(work_urls):
        time.sleep(REQUEST_DELAY)
        try:
            wr = session.get(url, timeout=TIMEOUT)
            wr.encoding = "iso-8859-1"
            ws = BeautifulSoup(wr.text, "html.parser")

            zubi_link = ws.find("a", href=lambda h: h and "zubi" in h)
            if not zubi_link:
                continue

            zubi_url = zubi_link["href"]
            # Ensure absolute URL
            if not zubi_url.startswith("http"):
                zubi_url = urllib.parse.urljoin("http://www.armiarma.eus", zubi_url)

            # Author name from title or from the work page sidebar
            author_name = None
            # Try <b> tag near the top of sidebar
            for b_tag in ws.find_all("b"):
                text = b_tag.get_text(strip=True)
                if len(text) > 3 and len(text) < 60:
                    author_name = text
                    break

            if zubi_url not in zubi_map:
                zubi_map[zubi_url] = {
                    "name": author_name,
                    "birthplace": None,
                    "dialect": None,
                    "confidence": None,
                    "works": [],
                }

            work_title = None
            for b_tag in ws.find_all("b"):
                text = b_tag.get_text(strip=True)
                if len(text) > 2:
                    work_title = text
                    break

            zubi_map[zubi_url]["works"].append(
                {"title": work_title or "unknown", "url": url}
            )

        except Exception:
            pass

        if (i + 1) % 50 == 0:
            print(
                f"  {i + 1}/{len(work_urls)} works, {len(zubi_map)} unique authors found"
            )

    print(f"  Found {len(zubi_map)} unique authors with Zubitegia profiles")

    # Step 3: Fetch birthplaces
    print("\n[3/5] Fetching author birthplaces...")
    for i, (zubi_url, info) in enumerate(zubi_map.items()):
        time.sleep(REQUEST_DELAY)
        try:
            zr = session.get(zubi_url, timeout=TIMEOUT)
            zr.encoding = "iso-8859-1"
            zs = BeautifulSoup(zr.text, "html.parser")

            jaio = zs.find("p", class_="jaioHil4")
            if jaio:
                bp = jaio.get_text(strip=True)
                info["birthplace"] = bp
                dialect, conf = map_to_dialect(bp, muni_map)
                info["dialect"] = dialect
                info["confidence"] = conf
        except Exception:
            pass

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(zubi_map)} authors")

    # Report
    dialect_counts = defaultdict(lambda: {"authors": 0, "works": 0})
    unknown = 0
    for info in zubi_map.values():
        if info["dialect"]:
            dialect_counts[info["dialect"]]["authors"] += 1
            dialect_counts[info["dialect"]]["works"] += len(info["works"])
        else:
            unknown += 1

    print("\n  Author dialect distribution:")
    for d, c in sorted(dialect_counts.items()):
        print(f"    {d}: {c['authors']} authors, {c['works']} works")
    print(f"    unknown: {unknown} authors")

    # Step 4: Scrape texts
    print("\n[4/5] Scraping texts...")
    labeled = []
    total_sentences = 0
    works_done = 0

    for zubi_url, info in zubi_map.items():
        if not info["dialect"]:
            continue

        dialect = info["dialect"]
        conf = info["confidence"]
        author = info["name"] or "unknown"
        birthplace = info["birthplace"] or ""

        for work in info["works"]:
            works_done += 1
            if works_done <= 2:  # Print first few
                print(f"  [{works_done}] {author}: {work['title']} [{dialect}]")

            sentences = scrape_work(work["url"], session)

            for sent in sentences:
                labeled.append(
                    {
                        "text": sent,
                        "dialect": dialect,
                        "author": author,
                        "work": work["title"],
                        "birthplace": birthplace,
                        "confidence": conf,
                    }
                )
            total_sentences += len(sentences)

            if works_done % 30 == 0:
                print(f"  [{works_done}] {total_sentences} sentences scraped so far")
                write_results(labeled, f"{OUTPUT_TSV}.partial")

    # Step 5: Save
    print("\n[5/5] Saving...")
    write_results(labeled, OUTPUT_TSV)

    # Final stats
    final = defaultdict(int)
    for item in labeled:
        final[item["dialect"]] += 1

    print("\n=== Done ===")
    print(f"Works scraped: {works_done}")
    print(f"Labeled sentences: {len(labeled)}")
    print("\nSentence distribution:")
    for d, c in sorted(final.items()):
        print(f"  {d}: {c}")


def write_results(data: list[dict], path: Path):
    fieldnames = ["text", "dialect", "author", "work", "birthplace", "confidence"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(data)
    print(f"  Saved {len(data)} rows → {path}")


if __name__ == "__main__":
    main()
