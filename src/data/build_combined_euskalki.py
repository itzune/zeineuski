#!/usr/bin/env python3
"""Build a combined 5-class euskalki training file from Klasikoak + azpieuskalki data.

Strategy: combine Klasikoak literary texts (good for 3-class XNLI) with
azpieuskalki Ahotsak oral texts (covers souletin/navarrese oral features).
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEXT_DIR = PROJECT_ROOT / "data" / "processed" / "text"

EUSK_LABELS = ["western", "central", "navarrese", "nav-lab", "souletin"]

# Max samples per class from azpieuskalki to avoid drowning Klasikoak
MAX_AZP_SAMPLES = {
    "western": 999999,
    "central": 999999,
    "navarrese": 999999,
    "nav-lab": 999999,
    "souletin": 999999,
}


def main():
    random.seed(42)

    # Load Klasikoak 6-class, filter to 5-class (exclude batua)
    klasikoa_lines = {label: [] for label in EUSK_LABELS}
    with open(TEXT_DIR / "train_6class.txt") as f:
        for line in f:
            for label in EUSK_LABELS:
                if f"__label__{label}" in line:
                    klasikoa_lines[label].append(line.strip())
                    break

    # Load azpieuskalki oral data
    azp_lines = {label: [] for label in EUSK_LABELS}
    with open(TEXT_DIR / "train_euskalki_5class.txt") as f:
        for line in f:
            for label in EUSK_LABELS:
                if f"__label__{label}" in line:
                    azp_lines[label].append(line.strip())
                    break

    # Combine: all Klasikoak + capped azpieuskalki oral
    combined = []
    counts = Counter()

    for label in EUSK_LABELS:
        # All Klasikoak
        for line in klasikoa_lines[label]:
            combined.append(line)
            counts[label] += 1

        # Capped azpieuskalki oral
        oral_pool = azp_lines[label]
        random.shuffle(oral_pool)
        max_oral = MAX_AZP_SAMPLES[label]
        oral_added = 0
        for line in oral_pool:
            # Skip duplicates (exact match)
            if line in klasikoa_lines[label]:
                continue
            combined.append(line)
            counts[label] += 1
            oral_added += 1
            if oral_added >= max_oral:
                break

    # Save
    output = TEXT_DIR / "train_euskalki_combined.txt"
    random.shuffle(combined)
    with open(output, "w") as f:
        for line in combined:
            f.write(line + "\n")

    print(f"Combined training data: {len(combined)} lines → {output}")
    for label in EUSK_LABELS:
        k_count = len(klasikoa_lines[label])
        a_count = min(len(azp_lines[label]), MAX_AZP_SAMPLES[label])
        print(f"  {label}: {counts[label]} (Klasikoak={k_count} + Ahotsak={a_count})")


if __name__ == "__main__":
    main()
