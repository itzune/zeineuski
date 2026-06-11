"""
Task 3.5.3 — Incorporate Ahotsak transcriptions into text classifier training.

Takes validated Ahotsak passages, cleans them, formats as fastText training data,
then trains a new model variant and evaluates against XNLI test sets.

Usage:
    uv run python -m src.data.ahotsak_text_training prepare   # Clean + format Ahotsak data
    uv run python -m src.data.ahotsak_text_training train     # Train model with Ahotsak
    uv run python -m src.data.ahotsak_text_training evaluate  # Compare with/without Ahotsak
    uv run python -m src.data.ahotsak_text_training all       # Run everything
"""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "speech"
TEXT_DIR = PROJECT_ROOT / "data" / "processed" / "text"
MODELS_DIR = PROJECT_ROOT / "models"
XNLI_DIR = PROJECT_ROOT / "data" / "raw" / "text" / "xnli_dialectal"

# Input
VALIDATION_CSV = PROCESSED_DIR  # find latest

# Output
AHOTSAK_TRAIN = TEXT_DIR / "train_ahotsak.txt"
AHOTSAK_SPLIT = TEXT_DIR / "train_ahotsak_sentences.txt"
TRAIN_COMBINED = TEXT_DIR / "train_with_ahotsak.txt"
MODEL_WITH_AHOTSAK = MODELS_DIR / "fasttext_dialect_with_ahotsak.bin"
MODEL_WITH_AHOTSAK_QUANT = MODELS_DIR / "fasttext_dialect_with_ahotsak.ftz"

# Baseline
TRAIN_HYBRID = TEXT_DIR / "train_hybrid.txt"
MODEL_BASELINE = MODELS_DIR / "hier_dialect_final.bin"

# ── Sentence splitting for dialectal Basque ───────────────────────────────────


def split_dialectal_sentences(text: str) -> list[str]:
    """Split dialectal Basque text into sentences.

    Handles dialectal punctuation (may be irregular) and speaker tags.
    """
    # Remove speaker tags like "- Pauli:" or "- Mari:"
    text = re.sub(r"-\s*\w+\s*:", " ", text)
    text = re.sub(r"^\s*-\s*", "", text, flags=re.MULTILINE)

    # Remove metadata footer like "Egilea(k):..."
    text = re.sub(r"Egilea\(k\):.*$", "", text)

    # Split on sentence boundaries
    # Basque uses ., ?, !, ... but also dialectal forms may have irregular breaks
    # Primary: split on newlines first, then on sentence punctuation
    sentences = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Split on sentence-ending punctuation
        parts = re.split(r"(?<=[.!?])\s+", line)
        for part in parts:
            part = part.strip()
            if len(part) >= 15 and any(c.isalpha() for c in part):
                sentences.append(part)

    return sentences


def clean_transcription(text: str) -> str:
    """Clean a transcription for training."""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove common artifacts
    text = re.sub(r"\[.*?\]", "", text)  # bracketed annotations
    text = re.sub(r"\(.*?\)", " ", text)  # parenthetical (but keep 4-digit years)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Data preparation ──────────────────────────────────────────────────────────


def find_latest_validation() -> Path:
    """Find the latest validation CSV."""
    csvs = sorted(PROCESSED_DIR.glob("ahotsak_label_validation_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No validation CSV in {PROCESSED_DIR}")
    return csvs[-1]


def prepare_ahotsak_data() -> tuple[Path, int, int]:
    """Load validated Ahotsak passages, clean, split, and format as fastText data.

    Returns: (output_path, num_passages, num_sentences)
    """
    csv_path = find_latest_validation()
    logger.info(f"Loading validation results from {csv_path.name}")

    # Load passages with agreement or medium agreement
    passages = []
    rejected = Counter()

    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            outcome = row.get("outcome", "")
            muni_confidence = row.get("municipality_confidence", "")
            label = row.get("municipality_label", "")

            # Include: agreement (any confidence) and ambiguous (low confidence text model)
            # Exclude: flag_mismatch (model disagrees with high confidence)
            if outcome.startswith("agreement") or outcome == "ambiguous":
                text = row.get("transcription", "")
                if text and label:
                    passages.append((label, muni_confidence, text))
            else:
                rejected[outcome] += 1

    logger.info(f"  Using {len(passages)} passages")
    logger.info(f"  Rejected: {dict(rejected)}")

    # Split into sentences
    total_sentences = 0
    label_counts = Counter()

    # Save as: one sentence per line in fastText format
    ahotsak_lines = []
    sentence_lines = []

    for label, confidence, text in passages:
        cleaned = clean_transcription(text)
        if len(cleaned) < 20:
            continue

        # For fastText training: keep full text as single entry
        # Line format: __label__western text here...
        ahotsak_lines.append(f"__label__{label} {cleaned}")

        # For sentence-level analysis
        sentences = split_dialectal_sentences(text)
        for sent in sentences:
            sent = clean_transcription(sent)
            if len(sent) >= 15:
                sentence_lines.append(f"__label__{label} {sent}")
                label_counts[label] += 1
                total_sentences += 1

    # Save passage-level
    AHOTSAK_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    with open(AHOTSAK_TRAIN, "w", encoding="utf-8") as f:
        for line in ahotsak_lines:
            f.write(line + "\n")

    # Save sentence-level
    with open(AHOTSAK_SPLIT, "w", encoding="utf-8") as f:
        for line in sentence_lines:
            f.write(line + "\n")

    logger.info(f"  Saved {len(ahotsak_lines)} passages → {AHOTSAK_TRAIN}")
    logger.info(f"  Saved {total_sentences} sentences → {AHOTSAK_SPLIT}")
    logger.info(f"  Label distribution: {dict(label_counts.most_common())}")

    return AHOTSAK_TRAIN, len(ahotsak_lines), total_sentences


# ── Training ──────────────────────────────────────────────────────────────────


def train_with_ahotsak(
    train_path: Path,
    use_sentences: bool = False,
    autotune_duration: int = 300,
    dim: int = 300,
    epoch: int = 50,
    lr: float = 0.5,
    word_ngrams: int = 3,
) -> Path:
    """Train a new fastText model combining existing data + Ahotsak.

    Args:
        train_path: Path to Ahotsak data (passage-level or sentence-level)
        use_sentences: If True, use sentence-level Ahotsak; if False, passage-level
        autotune_duration: Seconds for autotune (0 to skip)
    """
    import numpy as np

    # Patch numpy 2.x compatibility
    import fasttext.FastText as ft_mod
    source = open(ft_mod.__file__).read()
    source = source.replace("np.array(probs, copy=False)", "np.asarray(probs)")
    exec(source, ft_mod.__dict__)

    import fasttext

    # Step 1: Combine existing training data with Ahotsak
    ahotsak_source = AHOTSAK_SPLIT if use_sentences else AHOTSAK_TRAIN

    # Count lines
    with open(TRAIN_HYBRID) as f:
        hybrid_lines = sum(1 for _ in f)
    with open(ahotsak_source) as f:
        ahotsak_lines = sum(1 for _ in f)

    logger.info(f"Existing data: {hybrid_lines:,} lines")
    logger.info(f"Ahotsak data:  {ahotsak_lines:,} lines ({'sentences' if use_sentences else 'passages'})")

    # Combine
    TRAIN_COMBINED.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_COMBINED, "w", encoding="utf-8") as out:
        with open(TRAIN_HYBRID) as f:
            shutil.copyfileobj(f, out)
        with open(ahotsak_source) as f:
            shutil.copyfileobj(f, out)

    total_lines = hybrid_lines + ahotsak_lines
    logger.info(f"Combined: {total_lines:,} lines → {TRAIN_COMBINED}")

    # Step 2: Train
    logger.info(f"Training fastText (dim={dim}, epoch={epoch}, lr={lr}, wordNgrams={word_ngrams})...")

    model = fasttext.train_supervised(
        str(TRAIN_COMBINED),
        dim=dim,
        epoch=epoch,
        lr=lr,
        wordNgrams=word_ngrams,
        minCount=2,
        bucket=200000,
        thread=8,
        verbose=2,
    )

    # Step 3: Autotune (optional)
    if autotune_duration > 0:
        logger.info(f"Autotuning for {autotune_duration}s...")
        model = fasttext.train_supervised(
            str(TRAIN_COMBINED),
            autotuneDuration=autotune_duration,
            dim=dim,
            wordNgrams=word_ngrams,
            minCount=2,
            bucket=200000,
            thread=8,
            verbose=2,
        )

    # Step 4: Quantize
    logger.info("Quantizing model...")
    model.quantize(str(TRAIN_COMBINED))

    # Step 5: Save
    MODEL_WITH_AHOTSAK.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_WITH_AHOTSAK))

    # Also save quantized version
    q_path = str(MODEL_WITH_AHOTSAK_QUANT)
    # model already quantized above, save_model saves qat version
    # Actually, let's save separately
    model.save_model(str(MODEL_WITH_AHOTSAK_QUANT))

    logger.info(f"Model saved → {MODEL_WITH_AHOTSAK}")

    # Print label counts in training data
    labels = Counter()
    with open(TRAIN_COMBINED) as f:
        for line in f:
            line = line.strip()
            if line and line.startswith("__label__"):
                label = line.split()[0].replace("__label__", "")
                labels[label] += 1
    logger.info(f"Training data distribution: {dict(labels.most_common())}")

    return MODEL_WITH_AHOTSAK


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate_model(model_path: Path, test_path: Path) -> dict:
    """Evaluate a fastText model on a test set."""
    import numpy as np

    import fasttext.FastText as ft_mod
    source = open(ft_mod.__file__).read()
    source = source.replace("np.array(probs, copy=False)", "np.asarray(probs)")
    exec(source, ft_mod.__dict__)

    import fasttext

    model = fasttext.load_model(str(model_path))

    # fastText test
    result = model.test(str(test_path), k=1)
    samples, precision, recall = result

    # Detailed evaluation
    y_true = []
    y_pred = []
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("__label__"):
                continue
            true_label = line.split()[0].replace("__label__", "")
            text = " ".join(line.split()[1:])
            # predict returns list of labels and numpy array of probs
            labels, probs = model.predict(text.strip(), k=1)
            pred_label = labels[0].replace("__label__", "")
            y_true.append(true_label)
            y_pred.append(pred_label)

    # Per-class accuracy
    from sklearn.metrics import classification_report, confusion_matrix

    classes = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=classes, zero_division=0, output_dict=True)

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    return {
        "samples": samples,
        "accuracy": precision,  # fastText reports precision as accuracy for k=1
        "precision": precision,
        "recall": recall,
        "y_true": y_true,
        "y_pred": y_pred,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }


def load_xnli_test() -> Path:
    """Find the XNLI test file."""
    # Check common locations
    candidates = [
        TEXT_DIR / "test_6class.txt",
        TEXT_DIR / "test_expanded_3class.txt",
        TEXT_DIR / "test.txt",
        XNLI_DIR / "test_6class.txt",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"No XNLI test file found. Checked: {candidates}")


def compare_models():
    """Compare baseline model vs model trained with Ahotsak data."""
    import numpy as np

    test_path = load_xnli_test()

    print("\n" + "=" * 60)
    print("COMPARING: Baseline vs Baseline + Ahotsak")
    print("=" * 60)

    results = {}
    for name, path in [
        ("Baseline (hybrid)", MODEL_BASELINE),
        ("Baseline + Ahotsak", MODEL_WITH_AHOTSAK),
    ]:
        if not path.exists():
            print(f"\n⚠ Model not found: {path}")
            continue

        print(f"\n{'─' * 40}")
        print(f"Model: {name} ({path.name})")
        print(f"Test set: {test_path.name}")

        result = evaluate_model(path, test_path)

        print(f"Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
        print(f"Samples: {result['samples']}")

        # Per-class
        cr = result["classification_report"]
        print("\nPer-class metrics:")
        for cls in result["classes"]:
            if cls in cr and isinstance(cr[cls], dict):
                metrics = cr[cls]
                print(f"  {cls:12s}: P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1-score']:.3f}  support={metrics['support']:.0f}")

        # Confusion matrix
        print("\nConfusion matrix:")
        header = "         " + "".join(f"{c:>10s}" for c in result["classes"])
        print(header)
        for i, cls in enumerate(result["classes"]):
            row = "".join(f"{result['confusion_matrix'][i][j]:>10d}" for j in range(len(result["classes"])))
            print(f"  {cls:7s} {row}")

        results[name] = result

    # Comparison summary
    if len(results) == 2:
        base_acc = results["Baseline (hybrid)"]["accuracy"]
        ahotsak_acc = results["Baseline + Ahotsak"]["accuracy"]
        delta = ahotsak_acc - base_acc
        print(f"\n{'=' * 40}")
        print(f"Accuracy delta: {delta:+.4f} ({delta*100:+.2f}%)")
        if delta > 0:
            print("✓ Ahotsak data improves accuracy!")
        elif delta < 0:
            print("✗ Ahotsak data degrades accuracy")
        else:
            print("= No change")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────


def cmd_prepare():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    prepare_ahotsak_data()


def cmd_train():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ahotsak_path, n_passages, n_sentences = prepare_ahotsak_data()
    train_with_ahotsak(ahotsak_path, use_sentences=True)


def cmd_evaluate():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    compare_models()


def cmd_all():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Step 1: Prepare Ahotsak data")
    ahotsak_path, n_passages, n_sentences = prepare_ahotsak_data()

    logger.info(f"\nStep 2: Train model with Ahotsak data")
    train_with_ahotsak(ahotsak_path, use_sentences=True)

    logger.info(f"\nStep 3: Compare models")
    compare_models()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("prepare", "train", "evaluate", "all"):
        print("Usage: python -m src.data.ahotsak_text_training [prepare|train|evaluate|all]")
        sys.exit(1)

    cmd = sys.argv[1]
    {
        "prepare": cmd_prepare,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "all": cmd_all,
    }[cmd]()
