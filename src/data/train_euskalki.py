"""Train a 5-class euskalki (dialect) model from azpieuskalki training data.

Maps 12 azpieuskalki sub-dialect labels → 5 euskalki labels:
  western, central, navarrese, nav-lab, souletin

Previously the euskalki model was only trained on 3 classes (western, central,
nav-lab) — souletin and navarrese had zero training examples. This caused
souletin texts to be misclassified as 'central' and navarrese as random.

Usage:
    uv run python -m src.data.train_euskalki train    # Train full model
    uv run python -m src.data.train_euskalki evaluate  # Evaluate all models
    uv run python -m src.data.train_euskalki all       # Train + evaluate
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import fasttext

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEXT_DIR = PROJECT_ROOT / "data" / "processed" / "text"
MODELS_DIR = PROJECT_ROOT / "models"

# Input: azpieuskalki training data (78,655 lines, 12 classes)
AZP_TRAIN = TEXT_DIR / "train_azpieuskalki.txt"
AZP_TEST = TEXT_DIR / "test_azpieuskalki.txt"

# Output
EUSK_TRAIN = TEXT_DIR / "train_euskalki_5class.txt"
EUSK_TEST = TEXT_DIR / "test_euskalki_5class.txt"
EUSK_MODEL = MODELS_DIR / "euskalki_5class.bin"

# ── Mapping ───────────────────────────────────────────────────────────────────

AZP_TO_EUSK = {
    "mendebal-sartaldea": "western",
    "mendebal-sortaldea": "western",
    "erdialde-sartaldea": "central",
    "erdialde-sortaldea": "central",
    "nafar-ipar-sartaldea": "navarrese",
    "nafar-erdigunea": "navarrese",
    "nafar-hego-sartaldea": "navarrese",
    "nafar-sortaldea": "navarrese",
    "ekialde-nafarra": "navarrese",
    "naflap-sartaldea": "nav-lab",
    "naflap-sortaldea": "nav-lab",
    "zuberera": "souletin",
}

EUSK_LABEL_ORDER = ["western", "central", "navarrese", "nav-lab", "souletin"]

# ── Data preparation ──────────────────────────────────────────────────────────


def map_file(input_path: Path, output_path: Path) -> tuple[int, Counter]:
    """Map azpieuskalki labels → euskalki labels and save."""
    lines = 0
    counts: Counter[str] = Counter()
    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            # Match __label__<azpieuskalki> in the line
            new_line = line
            for azp, eusk in AZP_TO_EUSK.items():
                old_label = f"__label__{azp}"
                new_label = f"__label__{eusk}"
                if old_label in new_line:
                    new_line = new_line.replace(old_label, new_label)
                    counts[eusk] += 1
                    break
            f_out.write(new_line + "\n")
            lines += 1
    return lines, counts


def prepare_data() -> tuple[int, Counter, int, Counter]:
    """Prepare euskalki train and test files from azpieuskalki data."""
    print("Preparing 5-class euskalki data from azpieuskalki...")

    if not AZP_TRAIN.exists():
        raise FileNotFoundError(
            f"{AZP_TRAIN} not found. Run azpieuskalki pipeline first."
        )
    if not AZP_TEST.exists():
        raise FileNotFoundError(
            f"{AZP_TEST} not found. Run azpieuskalki pipeline first."
        )

    train_lines, train_counts = map_file(AZP_TRAIN, EUSK_TRAIN)
    test_lines, test_counts = map_file(AZP_TEST, EUSK_TEST)

    print(f"  Train: {train_lines} lines → {EUSK_TRAIN}")
    for label in EUSK_LABEL_ORDER:
        if label in train_counts:
            print(f"    {label}: {train_counts[label]}")

    print(f"  Test:  {test_lines} lines → {EUSK_TEST}")
    for label in EUSK_LABEL_ORDER:
        if label in test_counts:
            print(f"    {label}: {test_counts[label]}")

    return train_lines, train_counts, test_lines, test_counts


# ── Training ──────────────────────────────────────────────────────────────────


def train_model(
    output_path: Path | None = None,
    dim: int = 200,
    epoch: int = 75,
    lr: float = 0.2,
    word_ngrams: int = 2,
    minn: int = 2,
    maxn: int = 6,
    min_count: int = 1,
    loss: str = "ns",
    bucket: int = 500000,
    seed: int = 42,
    thread: int = 4,
) -> Path:
    """Train 5-class euskalki fastText model."""
    if output_path is None:
        output_path = EUSK_MODEL

    if not EUSK_TRAIN.exists():
        prepare_data()

    print(f"\nTraining {output_path.name}...")
    print(f"  dim={dim}, epoch={epoch}, lr={lr}, wordNgrams={word_ngrams}")
    print(f"  minn={minn}, maxn={maxn}, minCount={min_count}, loss={loss}")
    print(f"  bucket={bucket}, seed={seed}, thread={thread}")

    t0 = time.time()
    model = fasttext.train_supervised(
        input=str(EUSK_TRAIN),
        dim=dim,
        epoch=epoch,
        lr=lr,
        wordNgrams=word_ngrams,
        minn=minn,
        maxn=maxn,
        minCount=min_count,
        loss=loss,
        bucket=bucket,
        seed=seed,
        thread=thread,
    )

    model.save_model(str(output_path))
    elapsed = time.time() - t0
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB) in {elapsed:.0f}s")
    return output_path


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate_model(model_path: Path, test_path: Path, label_order: list[str]) -> dict:
    """Evaluate a model and return comprehensive metrics."""
    model = fasttext.load_model(str(model_path))

    # Collect predictions
    y_true: list[str] = []
    y_pred: list[str] = []

    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Extract true label
            true_label = None
            for label in label_order:
                if f"__label__{label}" in line:
                    true_label = label
                    break
            if true_label is None:
                continue

            # Extract text (remove all __label__* prefixes)
            text = re.sub(r"__label__\S+\s*", "", line).strip()
            if not text:
                continue

            pred = model.predict(text, k=1)
            pred_label = pred[0][0].replace("__label__", "")

            y_true.append(true_label)
            y_pred.append(pred_label)

    # Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0.0

    # Per-class metrics
    class_metrics: dict[str, dict] = {}
    for label in label_order:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    # Confusion matrix
    confusion: dict[str, dict[str, int]] = {}
    for true_label in label_order:
        confusion[true_label] = {pred_label: 0 for pred_label in label_order}
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1

    # Weighted and macro F1
    total_support = sum(m["support"] for m in class_metrics.values())
    weighted_f1 = (
        sum(m["f1"] * m["support"] for m in class_metrics.values()) / total_support
        if total_support > 0
        else 0.0
    )
    macro_f1 = (
        sum(m["f1"] for m in class_metrics.values()) / len(label_order)
        if label_order
        else 0.0
    )

    return {
        "model": model_path.name,
        "test": test_path.name,
        "samples": len(y_true),
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "class_metrics": class_metrics,
        "confusion": confusion,
    }


def print_report(result: dict, label_order: list[str]) -> None:
    """Print evaluation report."""
    print(f"\n{'=' * 70}")
    print(f"Model: {result['model']}  |  Test: {result['test']}")
    print(f"{'=' * 70}")
    print(f"Samples: {result['samples']}")
    print(f"Accuracy:     {result['accuracy']:.4f} ({result['accuracy'] * 100:.2f}%)")
    print(f"Weighted F1:  {result['weighted_f1']:.4f}")
    print(f"Macro F1:     {result['macro_f1']:.4f}")

    print(
        f"\n{'Class':<14s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}"
    )
    print("-" * 56)
    for label in label_order:
        m = result["class_metrics"].get(label, {})
        print(
            f"{label:<14s} {m.get('precision', 0):>10.4f} "
            f"{m.get('recall', 0):>10.4f} {m.get('f1', 0):>10.4f} "
            f"{m.get('support', 0):>10d}"
        )

    # Confusion matrix
    print("\nConfusion matrix:")
    header = " " * 14 + "".join(f"{lb:>10s}" for lb in label_order)
    print(header)
    for true_label in label_order:
        row = "".join(
            f"{result['confusion'].get(true_label, {}).get(pred_label, 0):>10d}"
            for pred_label in label_order
        )
        print(f"{true_label:<14s}{row}")

    # Highlight problem areas
    print("\nMisclassification analysis:")
    for true_label in label_order:
        conf = result["confusion"].get(true_label, {})
        top_err = sorted(
            ((pl, cnt) for pl, cnt in conf.items() if pl != true_label and cnt > 0),
            key=lambda x: -x[1],
        )[:2]
        if top_err:
            total_true = sum(conf.values())
            err_str = ", ".join(
                f"→ {pl} ({cnt}/{total_true}={cnt / total_true:.1%})"
                for pl, cnt in top_err
            )
            print(f"  {true_label}: {err_str}")


# ── Comparison ────────────────────────────────────────────────────────────────


def compare_with_old(
    new_result: dict, old_model_path: Path, test_path: Path, label_order: list[str]
) -> None:
    """Compare new 5-class model vs old 3-class model on 5-class test."""
    if not old_model_path.exists():
        print(f"\nSkipping comparison: {old_model_path} not found")
        return

    old_result = evaluate_model(old_model_path, test_path, label_order)

    print(f"\n{'=' * 70}")
    print("COMPARISON: 3-class vs 5-class model on 5-class test")
    print(f"{'=' * 70}")

    for name, res in [("3-class (old)", old_result), ("5-class (new)", new_result)]:
        print(f"\n{name}:")
        print(
            f"  Accuracy={res['accuracy']:.4f}  Weighted F1={res['weighted_f1']:.4f}  Macro F1={res['macro_f1']:.4f}"
        )
        for label in label_order:
            m = res["class_metrics"].get(label, {})
            print(
                f"  {label:<14s} F1={m.get('f1', 0):.4f}  P={m.get('precision', 0):.4f}  R={m.get('recall', 0):.4f}"
            )

    delta_acc = new_result["accuracy"] - old_result["accuracy"]
    delta_wf1 = new_result["weighted_f1"] - old_result["weighted_f1"]
    print(f"\nDelta: accuracy={delta_acc:+.4f}  weighted F1={delta_wf1:+.4f}")
    if delta_acc > 0:
        print("✓ 5-class model improves accuracy!")
    else:
        print("✗ 5-class model degrades accuracy")
    print("  Note: old model had 0 train samples for souletin and navarrese.")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/evaluate 5-class euskalki model"
    )
    parser.add_argument(
        "command",
        choices=["prepare", "train", "evaluate", "all"],
        help="Action to run",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=75,
        help="Training epochs (default: 75)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=200,
        help="Vector dimension (default: 200)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.2,
        help="Learning rate (default: 0.2)",
    )
    parser.add_argument(
        "--thread",
        type=int,
        default=4,
        help="Number of training threads (default: 4; 0=all CPUs)",
    )
    args = parser.parse_args()

    cmd = args.command

    if cmd in ("prepare", "all"):
        prepare_data()

    if cmd in ("train", "all"):
        train_model(
            dim=args.dim,
            epoch=args.epoch,
            lr=args.lr,
            thread=args.thread,
        )

    if cmd in ("evaluate", "all"):
        if not EUSK_MODEL.exists():
            print(f"Error: model not found at {EUSK_MODEL}")
            print("Run 'train' first.")
            sys.exit(1)

        # Evaluate on 5-class test set
        print("\n" + "=" * 70)
        print("EVALUATING: 5-class euskalki model")
        print("=" * 70)

        new_result = evaluate_model(EUSK_MODEL, EUSK_TEST, EUSK_LABEL_ORDER)
        print_report(new_result, EUSK_LABEL_ORDER)

        # Compare with old model
        old_model = MODELS_DIR / "hier_dialect_best.bin"
        compare_with_old(new_result, old_model, EUSK_TEST, EUSK_LABEL_ORDER)

        # Also test the specific sentence that was misclassified
        print(f"\n{'=' * 70}")
        print("SANITY CHECK: 'Bizitza ez da nihurendako aisa'")
        print(f"{'=' * 70}")
        test_sentence = "Bizitza ez da nihurendako aisa"

        # Test with new model
        new_m = fasttext.load_model(str(EUSK_MODEL))
        new_pred = new_m.predict(test_sentence, k=3)
        print("New 5-class model:")
        for label, conf in zip(new_pred[0], new_pred[1]):
            name = label.replace("__label__", "")
            print(f"  {name}: {conf:.4f}")

        # Test with old model
        if old_model.exists():
            old_m = fasttext.load_model(str(old_model))
            old_pred = old_m.predict(test_sentence, k=3)
            print("Old 3-class model:")
            for label, conf in zip(old_pred[0], old_pred[1]):
                name = label.replace("__label__", "")
                print(f"  {name}: {conf:.4f}")


if __name__ == "__main__":
    main()
