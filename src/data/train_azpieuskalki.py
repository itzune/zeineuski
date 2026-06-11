"""
Train a 3rd-tier azpieuskalki (sub-dialect) classifier using Ahotsak transcriptions.

This adds a hierarchical layer:
  Tier 1: batua vs dialectal (binary)
  Tier 2: dialect (5-class: western/central/navarrese/nav-lab/souletin) — existing
  Tier 3: azpieuskalki (11-class) — NEW, trained on Ahotsak spoken data

Data sources:
  - Ahotsak spoken transcriptions (primary, ~35K sentences across 9 azpieuskalkiak)
  - SU AZIA Zuberotarra corpus (external, ~6.7K zuberera sentences from suazia.com)
    See: docs/data_sources/suazia_zuberotarra.md

Usage:
    uv run python -m src.data.train_azpieuskalki prepare   # Format data
    uv run python -m src.data.train_azpieuskalki train     # Train model
    uv run python -m src.data.train_azpieuskalki evaluate  # Evaluate
    uv run python -m src.data.train_azpieuskalki all       # Full pipeline
"""

from __future__ import annotations

import csv
import json
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEXT_DIR = PROJECT_ROOT / "data" / "processed" / "text"
MODELS_DIR = PROJECT_ROOT / "models"
AHOTSAK_DIR = PROJECT_ROOT / "data" / "raw" / "speech" / "ahotsak"
MUNI_CSV = PROJECT_ROOT / "data" / "reference" / "municipality_dialect.csv"
VALIDATION_DIR = PROJECT_ROOT / "data" / "processed" / "speech"

# External Zuberotarra corpus — see docs/data_sources/suazia_zuberotarra.md
SUAZIA_CORPUS_PATH = (
    PROJECT_ROOT / "data" / "raw" / "text" / "suazia" / "suazia_train_clean.txt"
)
SUAZIA_DOCS_PATH = PROJECT_ROOT / "docs" / "data_sources" / "suazia_zuberotarra.md"

from src.data.azpieuskalki_map import (
    AZPIEUSKALKI_MAP,
    AZPIEUSKALKI_NAMES,
    AHOTSAK_TO_OUR_LABEL,
)


# ── External corpus injection ────────────────────────────────────────────────


def inject_suazia_zuberera(azpi_sentences: dict[str, list[str]]) -> int:
    """Inject SU AZIA Zuberotarra corpus into azpieuskalki training data.

    The SU AZIA corpus provides ~6.7K written Zuberotarra sentences
    (pastoral scripts + blog articles) scraped from suazia.com via
    Wayback Machine. This addresses the severe under-representation
    of zuberera in the Ahotsak-only training set (750 sentences vs
    13K for mendebal-sortaldea).

    See docs/data_sources/suazia_zuberotarra.md for corpus details.

    Returns: number of sentences injected.
    """
    if not SUAZIA_CORPUS_PATH.exists():
        logger.warning(f"SU AZIA corpus not found at {SUAZIA_CORPUS_PATH}")
        logger.warning("  Run: python nongoeuskara/build/scrape_suazia.py")
        return 0

    lines = SUAZIA_CORPUS_PATH.read_text(encoding="utf-8").split("\n")

    count = 0
    for line in lines:
        line = line.strip()
        if not line or not line.startswith("__label__zuberera "):
            continue
        # Extract text after the label prefix
        text = line[len("__label__zuberera ") :].strip()
        if len(text) < 15:
            continue
        azpi_sentences["zuberera"].append(text)
        count += 1

    logger.info(
        f"SU AZIA zuberera corpus: injected {count} sentences "
        f"(from {SUAZIA_CORPUS_PATH.name})"
    )
    logger.info(
        f"  Note: this is written Zuberotarra (pastoral/literary), "
        f"not spoken. See {SUAZIA_DOCS_PATH}"
    )
    return count


# ── Data loading ──────────────────────────────────────────────────────────────


def normalize_town_name(name: str) -> str:
    """Normalize town names for matching: lowercase, strip, handle hyphens/spaces."""
    return name.strip().lower().replace("-", " ")


def load_town_mappings() -> dict[str, dict]:
    """Load town → dialect + region → azpieuskalki mappings.

    Uses Ahotsak authoritative labels from the CSV 'notes' field when available
    ('ahotsak:<azpieuskalki>'), falling back to region-based AZPIEUSKALKI_MAP.
    """
    town_map = {}
    with open(MUNI_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            town = normalize_town_name(row["herria"])
            region = row["eskualdea"]
            note = row.get("notes", "").strip()

            # First try: Ahotsak authoritative label from notes field
            azpi = None
            if note.startswith("ahotsak:"):
                ahotsak_label = note.split(":", 1)[1]
                azpi = AHOTSAK_TO_OUR_LABEL.get(ahotsak_label)

            # Fallback: region-based mapping
            if azpi is None:
                azpi = AZPIEUSKALKI_MAP.get(region)

            town_map[town] = {
                "dialect": row["dialect_class"],
                "dialect_confidence": row["dialect_confidence"],
                "region": region,
                "azpieuskalki": azpi,
            }
    return town_map


def load_passages(jsonl_path: Optional[Path] = None) -> list[dict]:
    """Load scraped Ahotsak passages."""
    if jsonl_path is None:
        jsonl_files = sorted(AHOTSAK_DIR.glob("ahotsak_passages_*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No passage JSONL in {AHOTSAK_DIR}")
        path = jsonl_files[-1]
    else:
        path = jsonl_path
    logger.info(f"Loading passages from {path.name}")

    passages = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                passages.append(json.loads(line))
    logger.info(f"  {len(passages)} passages loaded")
    return passages


def load_validation_agreements() -> set[str]:
    """Load passage IDs that had agreement between municipality label and text model."""
    csvs = sorted(VALIDATION_DIR.glob("ahotsak_label_validation_*.csv"))
    if not csvs:
        logger.warning("No validation CSV found — using all passages")
        return set()

    agreed = set()
    with open(csvs[-1], encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["outcome"].startswith("agreement") or row["outcome"] == "ambiguous":
                agreed.add(row["passage_id"])
    logger.info(f"  {len(agreed)} passages with agreement from validation")
    return agreed


# ── Sentence cleaning ─────────────────────────────────────────────────────────


def clean_sentence(text: str) -> str | None:
    """Clean a sentence for training. Returns None if unusable."""
    # Remove speaker tags
    text = re.sub(r"-\s*\w+\s*:", " ", text)
    text = re.sub(r"^\s*-\s*", "", text, flags=re.MULTILINE)
    # Remove metadata
    text = re.sub(r"Egilea\(k\):.*$", "", text)
    # Remove bracketed annotations
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Quality filters
    if len(text) < 20:
        return None
    if len(text) > 300:
        return None
    # Must have at least some alphabetic chars
    if not any(c.isalpha() for c in text):
        return None

    return text


def split_sentences(text: str) -> list[str]:
    """Split transcription into sentences, with dialectal punctuation handling."""
    sentences = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        for part in parts:
            cleaned = clean_sentence(part)
            if cleaned:
                sentences.append(cleaned)
    return sentences


# ── Data preparation ──────────────────────────────────────────────────────────


def prepare_azpieuskalki_data(
    min_samples: int = 15,
    test_split: float = 0.15,
    seed: int = 42,
    jsonl_path: Optional[Path] = None,
    validate: bool = True,
    oversample_factor: int | None = None,
) -> dict:
    """Build azpieuskalki-labeled training data from Ahotsak passages.

    Returns dict with train_path, test_path, label_distribution, etc.
    """
    random.seed(seed)

    town_map = load_town_mappings()
    passages = load_passages(jsonl_path)
    validation_agreed = load_validation_agreements() if validate else set()

    # Filter: only use passages WHERE text model agreed with municipality label
    # (higher confidence in the town → dialect → azpieuskalki chain).
    # Skip validation when we have authoritative geographic labels (e.g., new Nafarroa mappings).
    if validate and validation_agreed:
        filtered = [p for p in passages if p["passage_id"] in validation_agreed]
    else:
        filtered = passages
    logger.info(
        f"  Using {len(filtered)}/{len(passages)} passages (validation filter: {validate})"
    )

    # Build azpieuskalki-labeled sentences
    azpi_sentences: dict[str, list[str]] = defaultdict(list)
    town_participation = Counter()

    for p in filtered:
        town = normalize_town_name(p.get("town_name") or p.get("town_slug", ""))
        if town not in town_map or not town_map[town].get("azpieuskalki"):
            continue

        azpi = town_map[town]["azpieuskalki"]
        text = p.get("transcription", "")

        sentences = split_sentences(text)
        azpi_sentences[azpi].extend(sentences)
        town_participation[town] += len(sentences)

    # ── Inject external Zuberotarra corpus (SU AZIA) ──
    # Zuberera has very little spoken data in Ahotsak (~750 sentences).
    # The SU AZIA corpus adds ~6.7K written Zuberotarra sentences to
    # significantly improve coverage of this minority subdialect.
    injected = inject_suazia_zuberera(azpi_sentences)
    if injected > 0:
        logger.info(
            f"  Zuberera: {len(azpi_sentences.get('zuberera', []))} total "
            f"({injected} from SU AZIA)"
        )

    # Filter: only keep azpieuskalkiak with enough samples
    active_azpies = {}
    for azpi, sents in azpi_sentences.items():
        if len(sents) >= min_samples:
            active_azpies[azpi] = sents
        else:
            logger.info(
                f"  Dropping {azpi}: only {len(sents)} sentences (min {min_samples})"
            )

    # Log distribution
    logger.info("\nAzpieuskalki distribution:")
    total = 0
    for azpi in sorted(active_azpies, key=lambda a: -len(active_azpies[a])):
        name = AZPIEUSKALKI_NAMES.get(azpi, azpi)
        count = len(active_azpies[azpi])
        logger.info(f"  {azpi:25s} {name:55s} {count:5d} sentences")
        total += count
    logger.info(f"  {'─' * 85}")
    logger.info(
        f"  Total: {total} sentences across {len(active_azpies)} azpieuskalkiak"
    )

    # Train/test split (stratified by azpieuskalki)
    train_lines = []
    test_lines = []

    for azpi, sents in active_azpies.items():
        random.shuffle(sents)
        n_test = max(1, int(len(sents) * test_split))
        test_sents = sents[:n_test]
        train_sents = sents[n_test:]

        for s in train_sents:
            train_lines.append(f"__label__{azpi} {s}")
        for s in test_sents:
            test_lines.append(f"__label__{azpi} {s}")

    random.shuffle(train_lines)
    random.shuffle(test_lines)

    # Class balancing: oversample minority classes in training set
    if oversample_factor is not None:
        # Count per class in training set
        train_counts = Counter()
        for line in train_lines:
            label = line.split()[0].replace("__label__", "")
            train_counts[label] += 1

        max_count = max(train_counts.values())
        target_count = max(max_count // oversample_factor, 100)

        logger.info(
            f"\nClass balancing (target: {target_count}, factor: {oversample_factor}x):"
        )

        balanced_train = []
        for azpi in active_azpies:
            class_lines = [l for l in train_lines if l.startswith(f"__label__{azpi} ")]
            count = len(class_lines)
            if count < target_count:
                # Oversample
                repeat = (target_count // count) + 1
                oversampled = (class_lines * repeat)[:target_count]
                balanced_train.extend(oversampled)
                logger.info(
                    f"  {azpi:25s}: {count:5d} → {target_count} (oversampled {repeat}x)"
                )
            else:
                # Keep original (or downsample)
                balanced_train.extend(class_lines[:target_count])
                logger.info(f"  {azpi:25s}: {count:5d} → {target_count}")

        random.shuffle(balanced_train)
        train_lines = balanced_train

    # Save
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = TEXT_DIR / "train_azpieuskalki.txt"
    test_path = TEXT_DIR / "test_azpieuskalki.txt"

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(test_path, "w", encoding="utf-8") as f:
        f.write("\n".join(test_lines) + "\n")

    logger.info(f"\nTrain: {len(train_lines)} sentences → {train_path}")
    logger.info(f"Test:  {len(test_lines)} sentences → {test_path}")

    return {
        "train_path": train_path,
        "test_path": test_path,
        "train_lines": len(train_lines),
        "test_lines": len(test_lines),
        "azpieuskalkiak": list(active_azpies.keys()),
        "n_azpies": len(active_azpies),
    }


# ── Training ──────────────────────────────────────────────────────────────────


def train_model(
    train_path: Path,
    autotune_duration: int = 60,
    dim: int = 200,
    epoch: int = 25,
    lr: float = 0.5,
    word_ngrams: int = 2,
    loss: str = "ns",
    min_count: int = 1,
    minn: int = 0,
    maxn: int = 0,
) -> Path:
    """Train a fastText azpieuskalki classifier.

    Args:
        loss: Loss function. 'ns' = negative sampling (default, good for large output spaces),
              'hs' = hierarchical softmax (better for imbalanced classes),
              'ova' = one-vs-all (binary cross-entropy per class, can help with imbalance).
    """
    import fasttext.FastText as ft_mod

    source = open(ft_mod.__file__).read()
    source = source.replace("np.array(probs, copy=False)", "np.asarray(probs)")
    exec(source, ft_mod.__dict__)

    import fasttext

    output_path = MODELS_DIR / "azpieuskalki.bin"

    logger.info(
        f"Training azpieuskalki model (dim={dim}, epoch={epoch}, lr={lr}, wordNgrams={word_ngrams})..."
    )

    if autotune_duration > 0:
        logger.info(f"Autotuning for {autotune_duration}s...")
        model = fasttext.train_supervised(
            str(train_path),
            autotuneDuration=autotune_duration,
            dim=dim,
            wordNgrams=word_ngrams,
            loss=loss,
            minCount=min_count,
            minn=minn,
            maxn=maxn,
            bucket=200000,
            thread=8,
            verbose=2,
        )
    else:
        model = fasttext.train_supervised(
            str(train_path),
            dim=dim,
            epoch=epoch,
            lr=lr,
            wordNgrams=word_ngrams,
            loss=loss,
            minCount=min_count,
            minn=minn,
            maxn=maxn,
            bucket=200000,
            thread=8,
            verbose=2,
        )

    # Quantize (skip for small models — can be done separately)
    # model.quantize(str(train_path))

    MODEL_DIR = MODELS_DIR
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    logger.info(f"Model saved → {output_path}")

    return output_path


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate_model(model_path: Path, test_path: Path) -> dict:
    """Evaluate azpieuskalki model on test set."""
    import fasttext.FastText as ft_mod

    source = open(ft_mod.__file__).read()
    source = source.replace("np.array(probs, copy=False)", "np.asarray(probs)")
    exec(source, ft_mod.__dict__)

    import fasttext

    model = fasttext.load_model(str(model_path))

    # fastText built-in test
    samples, precision, recall = model.test(str(test_path), k=1)

    # Per-class
    y_true, y_pred = [], []
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("__label__"):
                continue
            true_label = line.split()[0].replace("__label__", "")
            text = " ".join(line.split()[1:])
            labels, probs = model.predict(text.strip(), k=1)
            pred_label = labels[0].replace("__label__", "")
            y_true.append(true_label)
            y_pred.append(pred_label)

    from sklearn.metrics import classification_report

    classes = sorted(set(y_true) | set(y_pred))
    report = classification_report(
        y_true, y_pred, labels=classes, zero_division=0, output_dict=True
    )

    # Per-class accuracy
    per_class = {}
    for cls in classes:
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        total = sum(1 for t in y_true if t == cls)
        per_class[cls] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0,
        }

    return {
        "accuracy": precision,
        "samples": samples,
        "per_class": per_class,
        "classification_report": report,
        "classes": classes,
    }


def print_evaluation(result: dict):
    """Pretty-print evaluation results."""
    print(f"\n{'=' * 60}")
    print("Azpieuskalki Classifier Evaluation")
    print(f"{'=' * 60}")
    print(
        f"Overall accuracy: {result['accuracy']:.4f} ({result['accuracy'] * 100:.2f}%)"
    )
    print(f"Test samples:     {result['samples']}")

    print("\nPer-class accuracy:")
    print(
        f"  {'Azpieuskalki':25s} {'Name':50s} {'Correct':>7s} {'Total':>6s} {'Acc':>7s}"
    )
    print(f"  {'─' * 25} {'─' * 50} {'─' * 7} {'─' * 6} {'─' * 7}")

    for cls in sorted(
        result["per_class"], key=lambda c: -result["per_class"][c]["total"]
    ):
        info = result["per_class"][cls]
        name = AZPIEUSKALKI_NAMES.get(cls, cls)
        name = name[:48] + ".." if len(name) > 50 else name
        print(
            f"  {cls:25s} {name:50s} {info['correct']:7d} {info['total']:6d} {info['accuracy']:6.2%}"
        )

    # Random baseline
    n_classes = len(result["classes"])
    random_baseline = 1.0 / n_classes
    print(f"\nRandom baseline: {random_baseline:.2%} ({n_classes} classes)")
    print(f"Improvement:     {result['accuracy'] / random_baseline:.1f}x random")


# ── Hierarchical inference demo ───────────────────────────────────────────────


def demo_hierarchical(texts: list[str]):
    """Demonstrate the 3-tier hierarchical inference pipeline."""
    import fasttext.FastText as ft_mod

    source = open(ft_mod.__file__).read()
    source = source.replace("np.array(probs, copy=False)", "np.asarray(probs)")
    exec(source, ft_mod.__dict__)

    import fasttext

    dialect_model = fasttext.load_model(str(MODELS_DIR / "hier_dialect_final.bin"))
    azpi_model = fasttext.load_model(str(MODELS_DIR / "azpieuskalki.bin"))

    # Tier 1: Binary batua/dialectal (simplified — we skip since all our data is dialectal)
    # Tier 2: Dialect
    # Tier 3: Azpieuskalki

    for text in texts:
        clean = text.strip().replace("\n", " ")
        d_labels, d_probs = dialect_model.predict(clean, k=2)
        a_labels, a_probs = azpi_model.predict(clean, k=3)

        dialect = d_labels[0].replace("__label__", "")
        d_conf = float(d_probs[0])

        print(f"\nText: {clean[:100]}...")
        print(f"  Tier 2 (dialect):   {dialect} ({d_conf:.2f})")
        print("  Tier 3 (azpi top-3):")
        for l, p in zip(a_labels, a_probs):
            label = l.replace("__label__", "")
            name = AZPIEUSKALKI_NAMES.get(label, label)
            print(f"    {label:25s} → {name:50s} ({float(p):.3f})")


# ── CLI ───────────────────────────────────────────────────────────────────────


def cmd_prepare(validate: bool = True):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    prepare_azpieuskalki_data(min_samples=5, validate=validate)


def cmd_train(validate: bool = True):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = prepare_azpieuskalki_data(min_samples=5, validate=validate)
    model_path = train_model(result["train_path"])
    logger.info(f"\nTraining complete → {model_path}")


def cmd_evaluate(validate: bool = True):
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Find test set
    test_path = TEXT_DIR / "test_azpieuskalki.txt"
    if not test_path.exists():
        logger.info("No test set found. Running prepare + train first.")
        result = prepare_azpieuskalki_data(min_samples=5, validate=validate)
        test_path = result["test_path"]

    model_path = MODELS_DIR / "azpieuskalki.bin"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    eval_result = evaluate_model(model_path, test_path)
    print_evaluation(eval_result)


def cmd_all(validate: bool = True):
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=" * 60)
    logger.info("PHASE 1: Data preparation")
    logger.info("=" * 60)
    prep = prepare_azpieuskalki_data(min_samples=5, validate=validate)

    logger.info(f"\n{'=' * 60}")
    logger.info("PHASE 2: Training")
    logger.info("=" * 60)
    model_path = train_model(prep["train_path"])

    logger.info(f"\n{'=' * 60}")
    logger.info("PHASE 3: Evaluation")
    logger.info("=" * 60)
    eval_result = evaluate_model(model_path, prep["test_path"])
    print_evaluation(eval_result)

    logger.info(f"\n{'=' * 60}")
    logger.info("PHASE 4: Hierarchical inference demo")
    logger.info("=" * 60)

    # Load some test examples for demo
    test_examples = []
    with open(prep["test_path"]) as f:
        for line in f:
            text = " ".join(line.strip().split()[1:])
            if text and len(text) > 30:
                test_examples.append(text)
            if len(test_examples) >= 5:
                break

    demo_hierarchical(test_examples)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or sys.argv[1] not in ("prepare", "train", "evaluate", "all"):
        print(
            "Usage: python -m src.data.train_azpieuskalki [prepare|train|evaluate|all] [--no-validate]"
        )
        sys.exit(1)

    validate = "--no-validate" not in sys.argv

    cmds = {
        "prepare": lambda: cmd_prepare(validate=validate),
        "train": lambda: cmd_train(validate=validate),
        "evaluate": lambda: cmd_evaluate(validate=validate),
        "all": lambda: cmd_all(validate=validate),
    }
    cmds[sys.argv[1]]()
