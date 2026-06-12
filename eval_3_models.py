#!/usr/bin/env python3
"""Evaluate 3 azpieuskalki models on all 12 labels with per-label F1 scores.

Models:
  1. azpieuskalki.bin         — baseline (dim=200, bucket=200000, default fastText)
  2. azpieuskalki_q.bin       — quantized version of baseline
  3. azpieuskalki_b50000.bin  — with bucket=50000 (compact)

Test set: data/processed/text/test_azpieuskalki.txt
"""

import os
import sys
from collections import defaultdict

import numpy as np

_np_array = np.array


def _patched_array(obj, copy=None, **kwargs):
    return np.asarray(obj, **kwargs)


np.array = _patched_array

# noqa — fastText monkey-patch must run before import

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "data"))
from azpieuskalki_map import AZPIEUSKALKI_NAMES  # noqa: E402

import fasttext  # noqa: E402

MODEL_DIR = "models"
TEST_PATH = "data/processed/text/test_azpieuskalki.txt"

ALL_LABELS = [
    "mendebal-sartaldea",
    "mendebal-sortaldea",
    "erdialde-sartaldea",
    "erdialde-sortaldea",
    "nafar-ipar-sartaldea",
    "nafar-erdigunea",
    "nafar-hego-sartaldea",
    "nafar-sortaldea",
    "naflap-sartaldea",
    "naflap-sortaldea",
    "zuberera",
    "ekialde-nafarra",
]

MODELS = ["azpieuskalki", "azpieuskalki_q", "azpieuskalki_b50000"]


def load_test_data(path):
    """Load test data, returning list of (text, label) tuples."""
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            label = parts[0].replace("__label__", "")
            text = parts[1] if len(parts) > 1 else ""
            data.append((text, label))
    return data


def evaluate_model(model_path, test_data):
    """Evaluate a model and return per-label and overall metrics."""
    model = fasttext.load_model(model_path)

    y_true, y_pred = [], []
    for text, true_label in test_data:
        labels, probs = model.predict(text.strip(), k=1)
        pred_label = labels[0].replace("__label__", "")
        y_true.append(true_label)
        y_pred.append(pred_label)

    # Per-label metrics
    per_label = {}
    for label in ALL_LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0
        recall = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        true_count = sum(1 for t in y_true if t == label)

        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "true_count": true_count,
        }

    # Overall metrics
    total_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    overall_accuracy = total_correct / len(y_true) if y_true else 0

    # Macro F1
    macro_f1 = sum(per_label[l]["f1"] for l in ALL_LABELS) / len(ALL_LABELS)

    # Weighted F1
    total_samples = len(y_true)
    weighted_f1 = sum(
        per_label[l]["f1"] * per_label[l]["true_count"] / total_samples
        for l in ALL_LABELS
    )

    # Bottom-5 mean and min
    sorted_by_f1 = sorted(per_label.items(), key=lambda x: x[1]["f1"])
    bottom5 = sorted_by_f1[:5]
    bottom5_mean_f1 = sum(v["f1"] for _, v in bottom5) / 5
    bottom5_min_f1 = bottom5[0][1]["f1"]

    return {
        "overall_accuracy": overall_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "bottom5_mean_f1": bottom5_mean_f1,
        "bottom5_min_f1": bottom5_min_f1,
        "per_label": per_label,
    }


def format_size(path):
    size = os.path.getsize(path)
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f}MB"
    return f"{size / 1024:.0f}KB"


def main():
    test_data = load_test_data(TEST_PATH)
    print(f"Test set: {len(test_data)} samples from {TEST_PATH}\n")

    # Show label distribution in test set
    label_counts = defaultdict(int)
    for _, label in test_data:
        label_counts[label] += 1
    print("Label distribution in test set:")
    for l in ALL_LABELS:
        name = AZPIEUSKALKI_NAMES.get(l, l)
        print(f"  {l:25s} {name:55s} {label_counts.get(l, 0):5d}")
    print()

    results = {}
    for model_name in MODELS:
        model_path = os.path.join(MODEL_DIR, f"{model_name}.bin")
        if not os.path.exists(model_path):
            print(f"[SKIP] {model_name}: not found at {model_path}")
            continue

        print(f"Evaluating {model_name} ({format_size(model_path)})...")
        res = evaluate_model(model_path, test_data)
        results[model_name] = res

        print(f"  Overall accuracy: {res['overall_accuracy']:.4f}")
        print(f"  Weighted F1:      {res['weighted_f1']:.4f}")
        print(f"  Macro F1:         {res['macro_f1']:.4f}")
        print(f"  Bottom-5 mean F1: {res['bottom5_mean_f1']:.4f}")
        print(f"  Bottom-5 min F1:  {res['bottom5_min_f1']:.4f}")
        print()

    if len(results) < 2:
        print("Need at least 2 models for comparison.")
        return

    # ── Comparison table: per-label F1 ──
    print("=" * 120)
    print("Per-label F1 comparison across models")
    print("=" * 120)

    header_parts = [f"{'Label':28s}"]
    for model_name in MODELS:
        header_parts.append(f"{model_name:>12s}")
    header_parts.append(f"{'Best':>10s}")
    header_parts.append(f"{'#Test':>7s}")
    print("".join(header_parts))
    print("-" * 120)

    for label in ALL_LABELS:
        parts = []
        f1s = []
        for model_name in MODELS:
            if model_name in results:
                f1 = results[model_name]["per_label"][label]["f1"]
                f1s.append(f1)
                parts.append(f"  {f1:.3f}  ")
            else:
                parts.append(f"  {'—':>6}  ")

        best_f1 = max(f1s) if f1s else 0
        parts.append(f"  {best_f1:.3f}")
        parts.append(f"  {label_counts.get(label, 0):5d}")

        label_str = f"{label:28s}"
        print(label_str + "".join(parts))

    print("-" * 120)

    # Summary rows
    for metric_name, metric_key in [
        ("OVERALL (macro F1)", "macro_f1"),
        ("OVERALL (weighted F1)", "weighted_f1"),
        ("OVERALL (accuracy)", "overall_accuracy"),
    ]:
        summary_parts = [f"{metric_name:28s}"]
        best_vals = []
        for model_name in MODELS:
            if model_name in results:
                val = results[model_name][metric_key]
                best_vals.append(val)
                summary_parts.append(f"  {val:.3f}  ")
            else:
                summary_parts.append(f"  {'—':>6}  ")
        summary_parts.append(f"  {max(best_vals) if best_vals else 0:.3f}")
        summary_parts.append(f"  {len(test_data):5d}")
        print("".join(summary_parts))

    print()

    # Best model breakdown
    best_model = max(results, key=lambda m: results[m]["weighted_f1"])
    print(
        f"Best model: {best_model} (weighted F1 = {results[best_model]['weighted_f1']:.4f})"
    )
    print()
    print(
        f"{'Label':28s} {'Precision':>10s} {'Recall':>9s} {'F1':>8s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'Support':>8s}"
    )
    print("-" * 100)

    for label in ALL_LABELS:
        metrics = results[best_model]["per_label"][label]
        print(
            f"{label:28s} {metrics['precision']:>9.3f}  {metrics['recall']:>8.3f} {metrics['f1']:>8.3f} "
            f"{metrics['tp']:>5d}  {metrics['fp']:>5d}  {metrics['fn']:>5d}  {metrics['true_count']:>7d}"
        )


if __name__ == "__main__":
    main()
