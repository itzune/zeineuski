#!/usr/bin/env python3
"""Evaluate all hierarchical model variants on XNLI 3-class and 6-class test sets."""

# NumPy 2.0 monkey-patch for fastText (must run before importing fasttext)
import numpy as np  # noqa: E402

_np_array = np.array  # noqa: E402


def _patched_array(obj, copy=None, **kwargs):  # noqa: E302
    return np.asarray(obj, **kwargs)


np.array = _patched_array  # noqa: E402

import fasttext  # noqa: E402
import os  # noqa: E402
from collections import defaultdict  # noqa: E402

MODEL_DIR = "models"
DATA_DIR = "data/processed/text"

VARIANTS = ["final", "quantized", "compact", "tiny", "web"]

DIALECT_NAMES = {
    "western": "Mendebaldekoa",
    "central": "Erdialdekoa",
    "navarrese": "Nafarrera",
    "nav-lab": "Nafar-lapurtera",
    "souletin": "Zuberera",
    "batua": "Batua",
}

DIALECT_TO_3CLASS = {
    "western": "western",
    "central": "central",
    "navarrese": "nav-lab",
    "nav-lab": "nav-lab",
    "souletin": "nav-lab",
    "batua": "batua",
}


def load_data_6class(path):
    """Load fastText format: __label__X text"""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            label = parts[0].replace("__label__", "")
            text = parts[1] if len(parts) > 1 else ""
            data.append((text, label))
    return data


def load_data_3class(path):
    """Load XNLI format: __label__X text"""
    return load_data_6class(path)  # Same format


def eval_hierarchical(binary_model_path, dialect_model_path, test_data, n_classes=6):
    """Evaluate hierarchical classifier.

    Step 1: binary model (batua vs dialectal)
    Step 2: dialect model (5-class euskalkiak) for dialectal predictions
    """
    binary = fasttext.load_model(binary_model_path)
    dialect = fasttext.load_model(dialect_model_path)

    correct = 0
    total = 0
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})

    for text, true_label in test_data:
        total += 1
        per_class[true_label]["total"] += 1

        # Step 1: binary
        bin_pred = binary.predict(text, k=1)
        bin_label = bin_pred[0][0].replace("__label__", "")

        if bin_label == "batua":
            pred_label = "batua"
        else:
            # Step 2: dialect
            dial_pred = dialect.predict(text, k=1)
            pred_label = dial_pred[0][0].replace("__label__", "")

        # Map to 3-class: model labels → XNLI labels
        if n_classes == 3:
            three_map = {
                "batua": "batua",
                "western": "western",
                "central": "central",
                "nav-lab": "nav",
                "navarrese": "nav",
                "souletin": "nav",
            }
            true_3 = three_map.get(true_label, true_label)
            pred_3 = three_map.get(pred_label, pred_label)
            if pred_3 == true_3:
                correct += 1
                per_class[true_label]["correct"] += 1
        else:
            if pred_label == true_label:
                correct += 1
                per_class[true_label]["correct"] += 1

    acc = correct / total if total > 0 else 0
    return acc, correct, total, per_class


def get_model_size(binary_path, dialect_path):
    return os.path.getsize(binary_path) + os.path.getsize(dialect_path)


def format_size(bytes):
    mb = bytes / (1024 * 1024)
    if mb >= 1000:
        return f"{mb / 1024:.1f}GB"
    return f"{mb:.0f}MB"


def eval_single_model(model_path, test_data):
    """Evaluate a single model (non-hierarchical)."""
    model = fasttext.load_model(model_path)
    correct = 0
    total = 0
    for text, true_label in test_data:
        total += 1
        pred = model.predict(text, k=1)
        pred_label = pred[0][0].replace("__label__", "")
        if pred_label == true_label:
            correct += 1
    return correct / total if total > 0 else 0, correct, total


def main():
    # Load test data
    test_6class = load_data_6class(f"{DATA_DIR}/val_6class.txt")
    test_3class = load_data_3class(f"{DATA_DIR}/test_expanded_3class.txt")

    print(f"Test 6-class: {len(test_6class)} samples")
    print(f"Test 3-class (XNLI): {len(test_3class)} samples")
    print()

    # Header
    print(
        f"{'Variant':<12} {'Size':>8} {'XNLI 3cl':>10} {'Test 6cl':>10} {'Batua F1':>10} | {'W':>7} {'C':>7} {'NL':>7} {'N':>7} {'S':>7}"
    )
    print("-" * 110)

    results = []
    for variant in VARIANTS:
        binary_path = f"{MODEL_DIR}/hier_binary_{variant}.bin"
        dialect_path = f"{MODEL_DIR}/hier_dialect_{variant}.bin"

        if not (os.path.exists(binary_path) and os.path.exists(dialect_path)):
            print(f"{variant:<12} {'N/A':>8} {'—':>10} {'—':>10} {'—':>10}")
            continue

        size = get_model_size(binary_path, dialect_path)

        # Evaluate on 3-class XNLI test
        xnli_acc, _, _, xnli_per_class = eval_hierarchical(
            binary_path, dialect_path, test_3class, n_classes=3
        )

        # Evaluate on 6-class test
        test6_acc, test6_correct, test6_total, per_class = eval_hierarchical(
            binary_path, dialect_path, test_6class, n_classes=6
        )

        # Per-class F1 for batua
        batua_tp = per_class["batua"]["correct"]
        batua_total = per_class["batua"]["total"]
        batua_pred_total = sum(
            1
            for _, label in test_6class
            if label != "batua"  # approximated
        )
        # Simpler: use accuracy as proxy since per-class metrics on 6-class need
        # counting predictions too. Let's compute properly.
        batua_precision = batua_tp / max(batua_pred_total, 1)
        batua_recall = batua_tp / max(batua_total, 1)
        batua_f1 = (
            2 * batua_precision * batua_recall / (batua_precision + batua_recall)
            if (batua_precision + batua_recall) > 0
            else 0
        )

        # Actually, let's compute proper per-class F1 using sklearn-style
        # Collect predictions for 6-class
        binary = fasttext.load_model(binary_path)
        dialect = fasttext.load_model(dialect_path)
        y_true = []
        y_pred = []
        for text, true_label in test_6class:
            y_true.append(true_label)
            bin_pred = binary.predict(text, k=1)
            bin_label = bin_pred[0][0].replace("__label__", "")
            if bin_label == "batua":
                y_pred.append("batua")
            else:
                dial_pred = dialect.predict(text, k=1)
                y_pred.append(dial_pred[0][0].replace("__label__", ""))

        # Per-class F1
        classes = sorted(set(y_true))
        per_class_f1 = {}
        for cls in classes:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            per_class_f1[cls] = f1

        batua_f1 = per_class_f1.get("batua", 0)

        # Per-class F1 for display
        w_f1 = per_class_f1.get("western", 0)
        c_f1 = per_class_f1.get("central", 0)
        nl_f1 = per_class_f1.get("nav-lab", 0)
        n_f1 = per_class_f1.get("navarrese", 0)
        s_f1 = per_class_f1.get("souletin", 0)

        print(
            f"{variant:<12} {format_size(size):>8} {xnli_acc * 100:>9.2f}% {test6_acc * 100:>9.2f}% {batua_f1:>9.3f} "
            f"| {w_f1:.3f} {c_f1:.3f} {nl_f1:.3f} {n_f1:.3f} {s_f1:.3f}"
        )

        results.append(
            {
                "variant": variant,
                "size_mb": size / (1024 * 1024),
                "xnli_3class": xnli_acc,
                "test_6class": test6_acc,
                "batua_f1": batua_f1,
                "per_class_f1": per_class_f1,
            }
        )

    print()
    print(
        "Per-class F1 legend: W=Western, C=Central, NL=Nav-Lab, N=Navarrese, S=Souletin"
    )

    return results


if __name__ == "__main__":
    results = main()
