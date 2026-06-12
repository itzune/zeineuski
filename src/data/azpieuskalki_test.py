"""
Test harness: per-dialect azpieuskalki submodel architecture comparison.

Compares:
  1. Flat model: 1 fastText model on all azpieuskalki classes
  2. Per-dialect submodels: one model per dialect, own classes only
  3. Per-dialect with crossovers: submodels include transition-zone azpieuskalkiak

Uses whatever data is available NOW (partial scrape + initial 289 passages).
Does NOT wait for the full scrape to finish.

Usage:
    uv run python -m src.data.azpieuskalki_test
"""

import json
import csv
import logging
import random
import tempfile
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import fasttext

# Patch numpy 2.x compatibility
import fasttext.FastText as ft_mod


def _patch_fasttext():
    source = open(ft_mod.__file__).read()
    if "np.array(probs, copy=False)" in source:
        source = source.replace("np.array(probs, copy=False)", "np.asarray(probs)")
        exec(source, ft_mod.__dict__)


_patch_fasttext()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Transition zones ────────────────────────────────────────────────────
TRANSITION_CROSSOVERS = {
    "ipar-goi-nafarrera": ["beterri"],
    "debagoiena": ["goierri"],
    "goierri": ["debagoiena"],
    "debabarrena": ["beterri", "sortaldekoa"],
    "beterri": ["debabarrena"],
    "sortaldekoa": ["debabarrena"],
    "sartaldeko-naf-lap": ["sortaldeko-naf-lap"],
    "sortaldeko-naf-lap": ["sartaldeko-naf-lap", "zuberera"],
    "ipar-goi-nafarrera": ["hego-goi-nafarrera", "beterri"],
    "hego-goi-nafarrera": ["ipar-goi-nafarrera"],
}

# ── Data loading ─────────────────────────────────────────────────────────


def load_all_passages() -> list[dict]:
    """Load ALL available Ahotsak passages: initial scrape + partial targeted scrape."""
    ahotsak_dir = PROJECT_ROOT / "data" / "raw" / "speech" / "ahotsak"
    jsonls = sorted(ahotsak_dir.glob("ahotsak_passages_*.jsonl"))
    if not jsonls:
        raise FileNotFoundError(f"No JSONLs in {ahotsak_dir}")

    seen = set()
    passages = []
    for jl in jsonls:
        with open(jl) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                pid = obj.get("passage_id", "")
                if pid and pid not in seen:
                    seen.add(pid)
                    passages.append(obj)

    logger.info(f"Loaded {len(passages)} passages from {len(jsonls)} JSONL file(s)")
    return passages


def load_town_map() -> dict[str, tuple]:
    """herria → (eskualdea, dialect_class)."""
    csv_path = PROJECT_ROOT / "data" / "reference" / "municipality_dialect.csv"
    town_map = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            town = row["herria"].strip().lower()
            esk = row.get("eskualdea", "").strip()
            dialect = row.get("dialect_class", "").strip()
            if town and esk:
                town_map[town] = (esk, dialect)
    return town_map


def resolve_azpieuskalki(
    herria: str, town_map: dict, azpi_map: dict
) -> tuple[str, str]:
    """Resolve town → (azpieuskalki, dialect). Handles both slug and name."""
    h = herria.lower().strip()

    # Direct lowercase match
    info = town_map.get(h)
    if not info:
        # Try: lowercase town from CSV in slug / slug in CSV town
        for csv_town, v in town_map.items():
            csv_l = csv_town.lower()
            if h == csv_l or h in csv_l or csv_l in h:
                info = v
                break
    if not info:
        # Last resort: check town_name field variants
        for csv_town, v in town_map.items():
            if csv_town.lower().replace("  ", " ") == h.replace("-", " ").replace(
                "  ", " "
            ):
                info = v
                break
    if not info:
        return "??", "??"
    esk, dialect = info
    azpi = azpi_map.get(esk, "??")
    return azpi, dialect


# ── Sentence extraction ──────────────────────────────────────────────────


def extract_sentences(
    passages: list[dict], azpi_label: str = None
) -> list[tuple[str, str]]:
    """Extract sentences from passages, optionally filtered to a specific azpieuskalki.

    Returns list of (sentence, azpieuskalki_label).
    """
    from src.data.azpieuskalki_map import AZPIEUSKALKI_MAP

    town_map = load_town_map()
    sentences = []

    for p in passages:
        herria = p.get("town_slug", "") or p.get("herria", "")
        azpi, dialect = resolve_azpieuskalki(herria, town_map, AZPIEUSKALKI_MAP)
        if azpi_label and azpi != azpi_label:
            continue
        if azpi == "??":
            continue

        text = p.get("transcription", "")
        if not text:
            continue

        # Split into sentences (naive: split on line breaks, filter short)
        for line in text.split("\n"):
            line = line.strip()
            if len(line) < 15:
                continue
            # Quick dedup of speaker tags, timestamps
            if line.startswith("[") or line.startswith("("):
                continue
            sentences.append((line, azpi))

    return sentences


# ── fastText helpers ─────────────────────────────────────────────────────


def train_fasttext(
    sentences: list[tuple[str, str]],
    model_path: str,
    lr: float = 0.5,
    epochs: int = 50,
    dim: int = 100,
) -> tuple:
    """Train a fastText model. Returns (model, train_lines, class_counts)."""
    if len(sentences) < 10:
        return None, 0, {}

    # Count classes
    class_counts = Counter(label for _, label in sentences)
    n_classes = len(class_counts)
    if n_classes < 2:
        return None, len(sentences), class_counts

    # Write training file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        train_path = f.name
        for text, label in sentences:
            f.write(f"__label__{label} {text}\n")

    try:
        model = fasttext.train_supervised(
            input=train_path,
            lr=lr,
            epoch=epochs,
            dim=dim,
            wordNgrams=2,
            minn=3,
            maxn=6,
            loss="softmax",
            verbose=0,
        )
    finally:
        Path(train_path).unlink(missing_ok=True)

    return model, len(sentences), class_counts


def evaluate_model(model, sentences: list[tuple[str, str]]) -> dict:
    """Evaluate a fastText model on held-out sentences. Returns metrics dict."""
    if model is None or len(sentences) < 10:
        return {"accuracy": 0, "n": 0, "per_class": {}}

    # Write test file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        test_path = f.name
        for text, label in sentences:
            # Strip newlines (fastText error: "predict processes one line at a time")
            clean = text.replace("\n", " ").replace("\r", " ").strip()
            f.write(f"__label__{label} {clean}\n")

    try:
        result = model.test(test_path)
    finally:
        Path(test_path).unlink(missing_ok=True)

    # result is (N, precision, recall)
    accuracy = result[1]

    # Per-class via predict
    correct = Counter()
    total = Counter()
    for text, label in sentences:
        clean = text.replace("\n", " ").replace("\r", " ").strip()
        pred_label, _ = model.predict(clean, k=1)
        pred = pred_label[0].replace("__label__", "")
        total[label] += 1
        if pred == label:
            correct[label] += 1

    per_class = {}
    for cls in sorted(set(list(total.keys()))):
        per_class[cls] = {
            "accuracy": correct[cls] / max(total[cls], 1),
            "n": total[cls],
        }

    return {"accuracy": accuracy, "n": sum(total.values()), "per_class": per_class}


# ── Main experiment ──────────────────────────────────────────────────────


def run_experiment():
    from src.data.azpieuskalki_map import AZPIEUSKALKI_MAP

    town_map = load_town_map()
    passages = load_all_passages()

    # Count azpieuskalki distribution
    azpi_counter = Counter()
    dialect_azpis = defaultdict(Counter)
    for p in passages:
        herria = p.get("town_slug", "") or p.get("herria", "")
        azpi, dialect = resolve_azpieuskalki(herria, town_map, AZPIEUSKALKI_MAP)
        if azpi != "??":
            azpi_counter[azpi] += 1
            dialect_azpis[dialect][azpi] += 1

    print("=" * 70)
    print("  AZPIEUSKALKI SUBMODEL ARCHITECTURE TEST")
    print(
        f"  Data: {len(passages)} passages ({azpi_counter.total()} with azpieuskalki)"
    )
    print("=" * 70)
    print()
    print("Azpieuskalki distribution:")
    for azpi, count in azpi_counter.most_common():
        n_sentences = count * 10  # estimate
        print(f"  {azpi:25s} {count:4d} passages (~{n_sentences:5d} sents)")
    print()

    # ── Extract all sentences ────────────────────────────────────────────
    all_sentences = extract_sentences(passages)
    logger.info(f"Extracted {len(all_sentences)} sentences total")

    if len(all_sentences) < 100:
        logger.error("Not enough sentences — need more data. Try when scrape finishes.")
        return

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_sentences)
    split = int(len(all_sentences) * 0.85)
    train_sents = all_sentences[:split]
    test_sents = all_sentences[split:]

    # ── EXPERIMENT 1: Flat model ─────────────────────────────────────────
    print("─" * 70)
    print("EXPERIMENT 1: Flat model (all azpieuskalki classes)")
    train_classes = Counter(l for _, l in train_sents)
    test_classes = Counter(l for _, l in test_sents)
    print(f"  Train: {len(train_sents)} sentences, {len(train_classes)} classes")
    print(f"  Test:  {len(test_sents)} sentences, {len(test_classes)} classes")
    print(f"  Per class: {dict(test_classes.most_common())}")

    model_flat, n_train_flat, _ = train_fasttext(train_sents, "/tmp/azpi_flat.bin")
    if model_flat:
        result_flat = evaluate_model(model_flat, test_sents)
        print(f"  Accuracy: {result_flat['accuracy'] * 100:.2f}%")
        print("  Per class:")
        for cls, m in sorted(result_flat["per_class"].items()):
            print(f"    {cls:25s} {m['accuracy'] * 100:5.1f}% (n={m['n']})")
        print()
    else:
        print("  SKIPPED: not enough classes")
        result_flat = None

    # ── EXPERIMENT 2: Per-dialect submodels (own classes only) ────────────
    print("─" * 70)
    print("EXPERIMENT 2: Per-dialect submodels (own azpieuskalki only)")

    dialect_models = {}
    for dialect in ["western", "central", "navarrese", "nav-lab"]:
        own_azpis = set(dialect_azpis[dialect].keys())
        if len(own_azpis) < 2:
            print(f"  {dialect}: {len(own_azpis)} classes → skip")
            continue

        # Train on all sentences from this dialect's azpieuskalki classes
        dialect_train = [(s, l) for s, l in train_sents if l in own_azpis]
        dialect_test = [(s, l) for s, l in test_sents if l in own_azpis]

        if len(dialect_train) < 50 or len(dialect_test) < 10:
            print(
                f"  {dialect}: train={len(dialect_train)}, test={len(dialect_test)} → skip"
            )
            continue

        model, n_train, class_counts = train_fasttext(
            dialect_train, f"/tmp/azpi_{dialect}.bin"
        )
        if model is None:
            print(f"  {dialect}: couldn't train → skip")
            continue

        result = evaluate_model(model, dialect_test)
        dialect_models[dialect] = (model, result)

        print(
            f"  {dialect}: {len(class_counts)} classes, {n_train} train, {len(dialect_test)} test"
        )
        print(f"    Accuracy: {result['accuracy'] * 100:.2f}%")
        for cls, m in sorted(result["per_class"].items()):
            print(f"      {cls:25s} {m['accuracy'] * 100:5.1f}% (n={m['n']})")
    print()

    # ── EXPERIMENT 3: Per-dialect submodels WITH crossovers ───────────────
    print("─" * 70)
    print("EXPERIMENT 3: Per-dialect submodels with transition-zone crossovers")

    dialect_models_xover = {}
    for dialect in ["western", "central", "navarrese", "nav-lab"]:
        own_azpis = set(dialect_azpis[dialect].keys())
        if len(own_azpis) < 2:
            print(f"  {dialect}: {len(own_azpis)} classes → skip")
            continue

        # Add crossover azpieuskalkiak
        all_azpis = set(own_azpis)
        for azpi in own_azpis:
            for adj in TRANSITION_CROSSOVERS.get(azpi, []):
                # Only add if we have data for it (from any dialect)
                if azpi_counter.get(adj, 0) > 0:
                    all_azpis.add(adj)

        print(
            f"  {dialect}: {len(own_azpis)} own + {len(all_azpis - own_azpis)} crossover = {len(all_azpis)} classes"
        )

        dialect_train = [(s, l) for s, l in train_sents if l in all_azpis]
        dialect_test = [(s, l) for s, l in test_sents if l in all_azpis]

        if len(dialect_train) < 50 or len(dialect_test) < 10:
            print(f"    train={len(dialect_train)}, test={len(dialect_test)} → skip")
            continue

        model, n_train, class_counts = train_fasttext(
            dialect_train, f"/tmp/azpi_x_{dialect}.bin"
        )
        if model is None:
            print("    couldn't train → skip")
            continue

        result = evaluate_model(model, dialect_test)
        dialect_models_xover[dialect] = (model, result)

        for cls in sorted(all_azpis):
            marker = "← own" if cls in own_azpis else "← crossover"
            m = result["per_class"].get(cls, {"accuracy": 0, "n": 0})
            print(f"      {cls:25s} {m['accuracy'] * 100:5.1f}% (n={m['n']}) {marker}")
        print(f"    OVERALL: {result['accuracy'] * 100:.2f}%")
    print()

    # ── Summary ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    if result_flat:
        print(f"  Flat 10-class:         {result_flat['accuracy'] * 100:5.1f}%")
    for dialect, (_, result) in dialect_models.items():
        print(f"  {dialect:12s} (own only):   {result['accuracy'] * 100:5.1f}%")
    for dialect, (_, result) in dialect_models_xover.items():
        print(f"  {dialect:12s} (xover):     {result['accuracy'] * 100:5.1f}%")
    print()

    # Per-dialect macro average
    if dialect_models:
        avg_own = sum(r["accuracy"] for _, r in dialect_models.values()) / len(
            dialect_models
        )
        print(f"  Per-dialect macro avg (own):    {avg_own * 100:.1f}%")
    if dialect_models_xover:
        avg_xover = sum(r["accuracy"] for _, r in dialect_models_xover.values()) / len(
            dialect_models_xover
        )
        print(f"  Per-dialect macro avg (xover):  {avg_xover * 100:.1f}%")


if __name__ == "__main__":
    run_experiment()
