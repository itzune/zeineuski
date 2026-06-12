#!/bin/bash
# Scrape new mendebal-sartaldea towns added to municipality_dialect.csv.
# These towns are in Ahotsak's authoritative sartaldekoa-m list and have transcriptions.
set -euo pipefail

cd "$(dirname "$0")/.."

TOWNS=(
  "arrankudiaga"
  "arrigorriaga"
  "bedia"
  "berango"
  "dima"
  "erandio"
  "etxebarri"
  "fruiz"
  "leioa"
  "lemoa"
  "lemoiz"
  "lezama"
  "loiu"
  "morga"
  "sondika"
  "sopela"
  "urduliz"
  "zamudio"
  "zaratamo"
)

PASSAGE_LIMIT="${1:-200}"

for town in "${TOWNS[@]}"; do
  echo "=== Scraping $town (limit: $PASSAGE_LIMIT) ==="
  uv run python -m src.data.ahotsak_scraper scrape --town "$town" --limit "$PASSAGE_LIMIT" 2>&1 || echo "  FAILED: $town"
  sleep 2
done

echo "=== Done ==="
