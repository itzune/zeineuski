#!/usr/bin/env python3
"""Evaluate all hierarchical model variants on clean test sets only.

Clean test sets (0% train overlap, verified):
  - test_expanded_3class.txt  (2505 samples, 3-class XNLI: western/central/nav-lab)
  - test_6class.txt           (4005 samples, 4-class: batua/western/central/nav-lab)
  - val_6class_clean.txt      (1482 samples, 4-class: batua/western/central/nav-lab)

val_6class.txt has 68.4% overlap with train → NOT USED.
navarrese/souletin have no clean test data available.
"""

# NumPy 2.0 monkey-patch for fastText (must run before importing fasttext)
import numpy as np  # noqa: E402

_np_array = np.array  # noqa: E402


def _patched_array(obj, copy=None, **kwargs):  # noqa: E302
    return np.asarray(obj, **kwargs)


np.array = _patched_array  # noqa: E402

import fasttext  # noqa: E402
import os  # noqa: E402

MODEL_DIR = "models"
DATA_DIR = "data/processed/text"

VARIANTS = ["final", "quantized", "compact", "tiny", "web"]

THREE_MAP = {
    "batua": "batua",
    "western": "western",
    "central": "central",
    "nav-lab": "nav",
    "navarrese": "nav",
    "souletin": "nav",
}


def load_data(path):
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


def eval_hierarchical(binary_model_path, dialect_model_path, test_data, n_classes=None):
    """Evaluate hierarchical classifier. n_classes=None uses raw labels."""
    binary = fasttext.load_model(binary_model_path)
    dialect = fasttext.load_model(dialect_model_path)

    y_true, y_pred = [], []

    for text, true_label in test_data:
        # Step 1: binary
        bin_pred = binary.predict(text, k=1)
        bin_label = bin_pred[0][0].replace("__label__", "")

        if bin_label == "batua":
            pred_label = "batua"
        else:
            dial_pred = dialect.predict(text, k=1)
            pred_label = dial_pred[0][0].replace("__label__", "")

        y_true.append(true_label)
        y_pred.append(pred_label)

    # Map to 3-class if needed
    if n_classes == 3:
        y_true_mapped = [THREE_MAP.get(label, label) for label in y_true]
        y_pred_mapped = [THREE_MAP.get(label, label) for label in y_pred]
    else:
        y_true_mapped = y_true
        y_pred_mapped = y_pred

    # Accuracy
    correct = sum(1 for t, p in zip(y_true_mapped, y_pred_mapped) if t == p)
    acc = correct / len(y_true_mapped) if y_true_mapped else 0

    # Per-class F1
    classes = sorted(set(y_true_mapped))
    per_class_f1 = {}
    for cls in classes:
        tp = sum(
            1 for t, p in zip(y_true_mapped, y_pred_mapped) if t == cls and p == cls
        )
        fp = sum(
            1 for t, p in zip(y_true_mapped, y_pred_mapped) if t != cls and p == cls
        )
        fn = sum(
            1 for t, p in zip(y_true_mapped, y_pred_mapped) if t == cls and p != cls
        )
        precision = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0
        recall = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        per_class_f1[cls] = f1

    return acc, correct, len(y_true_mapped), per_class_f1


def get_model_size(binary_path, dialect_path):
    return os.path.getsize(binary_path) + os.path.getsize(dialect_path)


def format_size(bytes_val):
    mb = bytes_val / (1024 * 1024)
    if mb >= 1000:
        return f"{mb / 1024:.1f}GB"
    return f"{mb:.0f}MB"


def main():
    test_3class = load_data(f"{DATA_DIR}/test_expanded_3class.txt")
    test_4class = load_data(f"{DATA_DIR}/test_6class.txt")

    print(f"XNLI 3-class test: {len(test_3class)} samples (0% train overlap)")
    print(f"Test 6-class (4):  {len(test_4class)} samples (0% train overlap)")
    print("(navarrese/souletin: no clean test data available)")
    print()

    print(
        f"{'Variant':<12} {'Size':>8} {'XNLI 3cl':>10} {'Test (4cl)':>11} {'Batua F1':>9}",
        end="",
    )
    class_order = sorted(set(label for _, label in test_4class))
    for c in class_order:
        print(f" {c[:4]:>5}", end="")
    print()
    print("-" * 90)

    for variant in VARIANTS:
        binary_path = f"{MODEL_DIR}/hier_binary_{variant}.bin"
        dialect_path = f"{MODEL_DIR}/hier_dialect_{variant}.bin"

        if not (os.path.exists(binary_path) and os.path.exists(dialect_path)):
            print(f"{variant:<12} {'N/A':>8}")
            continue

        size = get_model_size(binary_path, dialect_path)

        # XNLI 3-class
        xnli_acc, _, _, xnli_f1 = eval_hierarchical(
            binary_path, dialect_path, test_3class, n_classes=3
        )

        # Test 4-class
        test4_acc, _, _, test4_f1 = eval_hierarchical(
            binary_path, dialect_path, test_4class
        )

        print(
            f"{variant:<12} {format_size(size):>8} {xnli_acc * 100:>9.2f}% {test4_acc * 100:>10.2f}% {test4_f1.get('batua', 0):>8.3f}",
            end="",
        )
        for c in class_order:
            if c in test4_f1:
                print(f" {test4_f1[c]:.3f}", end="")
            else:
                print(f" {'—':>5}", end="")
        print()

    print()
    print(
        "Legend: batua=Batua, west=Mendebaldekoa, cent=Erdialdekoa, nav-=Nafar-lapurtera"
    )


if __name__ == "__main__":
    main()
