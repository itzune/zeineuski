#!/bin/bash
set -euo pipefail
# Autoresearch measure script: train azpieuskalki and report F1 metrics.
# Reads hyperparams from env vars (defaults to baseline if unset).

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Hyperparameters (set via env, default to baseline) ──
LOSS="${LOSS:-ns}"
DIM="${DIM:-200}"
EPOCH="${EPOCH:-75}"
LR="${LR:-0.2}"
WORDNGRAMS="${WORDNGRAMS:-2}"
MINN="${MINN:-2}"
MAXN="${MAXN:-6}"
MINCOUNT="${MINCOUNT:-1}"
OVERSAMPLE_FACTOR="${OVERSAMPLE_FACTOR:-}"     # empty = no oversampling
AUTOTUNE="${AUTOTUNE:-0}"                       # 0 = disabled

python3 << PYEOF
import fasttext.FastText as ft_mod
source = open(ft_mod.__file__).read()
source = source.replace("np.array(probs, copy=False)", "np.asarray(probs)")
exec(source, ft_mod.__dict__)
import fasttext
import sys
import os
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import f1_score

# ── Config from env ──
loss = os.environ.get("LOSS", "ns")
dim = int(os.environ.get("DIM", "200"))
epoch = int(os.environ.get("EPOCH", "75"))
lr = float(os.environ.get("LR", "0.2"))
word_ngrams = int(os.environ.get("WORDNGRAMS", "2"))
minn = int(os.environ.get("MINN", "2"))
maxn = int(os.environ.get("MAXN", "6"))
min_count = int(os.environ.get("MINCOUNT", "1"))
oversample_factor = os.environ.get("OVERSAMPLE_FACTOR", "")
autotune = int(os.environ.get("AUTOTUNE", "0"))

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "."))
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "text" / "train_azpieuskalki.txt"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "text" / "test_azpieuskalki.txt"
MODEL_PATH = PROJECT_ROOT / "models" / "azpieuskalki.bin"

# ── Data preparation (via train_azpieuskalki module) ──
import logging
logging.basicConfig(level=logging.WARNING)

# Add src to path
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data"))
from train_azpieuskalki import prepare_azpieuskalki_data, train_model

# Prepare data (no validation filter — we use all passages)
osf_raw = oversample_factor.strip()
osf = int(osf_raw) if osf_raw else None
prep = prepare_azpieuskalki_data(min_samples=5, validate=False, oversample_factor=osf)

# ── Training ──
import time
t0 = time.time()

model = fasttext.train_supervised(
    str(prep["train_path"]),
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
    verbose=0,
)
model.save_model(str(MODEL_PATH))
train_time = time.time() - t0

# ── Evaluation ──
class_names = sorted(os.listdir(str(PROJECT_ROOT / "data" / "processed" / "text")))

# Collect predictions
y_true = []
y_pred = []
with open(prep["test_path"]) as f:
    for line in f:
        line = line.strip()
        if not line or not line.startswith("__label__"):
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        true_label = parts[0].replace("__label__", "")
        text = parts[1]
        labels, probs = model.predict(text.strip(), k=1)
        pred_label = labels[0].replace("__label__", "")
        y_true.append(true_label)
        y_pred.append(pred_label)

# Per-class F1
classes = sorted(set(y_true) | set(y_pred))
per_class_f1 = {}
for cls in classes:
    f1 = f1_score(
        [1 if t == cls else 0 for t in y_true],
        [1 if p == cls else 0 for p in y_pred],
        zero_division=0
    )
    per_class_f1[cls] = f1

# Aggregate metrics
macro_f1 = sum(per_class_f1.values()) / len(per_class_f1) if per_class_f1 else 0

# Weighted F1
from sklearn.metrics import f1_score as wf1
weighted_f1 = wf1(y_true, y_pred, average='weighted', zero_division=0)

# Bottom-5 mean/min F1
f1_sorted = sorted(per_class_f1.values())
bottom5 = f1_sorted[:5] if len(f1_sorted) >= 5 else f1_sorted
bottom5_mean_f1 = sum(bottom5) / len(bottom5) if bottom5 else 0
bottom5_min_f1 = min(bottom5) if bottom5 else 0

# Overall accuracy
correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
accuracy = correct / len(y_true) if y_true else 0

# ── Output ──
print(f"METRIC weighted_f1={weighted_f1:.6f}")
print(f"METRIC macro_f1={macro_f1:.6f}")
print(f"METRIC bottom5_mean_f1={bottom5_mean_f1:.6f}")
print(f"METRIC bottom5_min_f1={bottom5_min_f1:.6f}")
print(f"METRIC overall_accuracy={accuracy:.6f}")
print(f"METRIC train_time={train_time:.1f}")

# Per-class for debugging
for cls in sorted(per_class_f1.keys()):
    print(f"DEBUG {cls}={per_class_f1[cls]:.4f}")

print(f"DEBUG n_classes={len(classes)}")
print(f"DEBUG total_test={len(y_true)}")
PYEOF
