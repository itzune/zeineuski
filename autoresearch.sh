#!/bin/bash
set -euo pipefail

# Autoresearch: fastText Basque Dialect Classification Optimization
# Uses Python fastText API (CLI not available).

TRAIN_FILE="data/processed/text/train_hybrid.txt"
VAL_FILE="data/processed/text/val_hybrid.txt"
XNLI_TEST_FILE="data/processed/text/test_expanded_3class.txt"
MODEL_FILE="/tmp/fasttext_ar_model.bin"

MINN=${MINN:-3}
MAXN=${MAXN:-6}
WORDNGRAMS=${WORDNGRAMS:-2}
DIM=${DIM:-100}
LR=${LR:-0.1}
EPOCH=${EPOCH:-25}
LOSS=${LOSS:-softmax}
BUCKET=${BUCKET:-2000000}
MINCOUNT=${MINCOUNT:-1}

uv run python3 -c "
import fasttext
import time
import os

start = time.time()

model = fasttext.train_supervised(
    input='$TRAIN_FILE',
    lr=$LR,
    dim=$DIM,
    epoch=$EPOCH,
    wordNgrams=$WORDNGRAMS,
    minn=$MINN,
    maxn=$MAXN,
    loss='$LOSS',
    bucket=$BUCKET,
    minCount=$MINCOUNT,
    verbose=0,
    thread=4,
)

model.save_model('$MODEL_FILE')
train_time = time.time() - start

# XNLI test (primary metric)
result = model.test('$XNLI_TEST_FILE')
xnli_p1 = result[1]
xnli_r1 = result[2]

# XNLI per-class F1
labels_xnli = model.test_label('$XNLI_TEST_FILE', k=1, threshold=0.0)
xnli_west_f1 = labels_xnli.get('__label__western', {}).get('f1score', 0) or 0
xnli_cent_f1 = labels_xnli.get('__label__central', {}).get('f1score', 0) or 0
xnli_nlab_f1 = labels_xnli.get('__label__nav-lab', {}).get('f1score', 0) or 0

# Val
val_result = model.test('$VAL_FILE')
val_p1 = val_result[1]
labels_val = model.test_label('$VAL_FILE', k=1, threshold=0.0)
val_f1_sum = 0
val_f1_n = 0
for l in ['__label__western', '__label__central', '__label__navarrese', '__label__nav-lab', '__label__souletin']:
    f1 = labels_val.get(l, {}).get('f1score', 0)
    if f1 is not None and f1 == f1:  # skip NaN
        val_f1_sum += f1
        val_f1_n += 1
val_macro_f1 = val_f1_sum / val_f1_n if val_f1_n > 0 else 0

# Output
print(f'METRIC xnli_acc={xnli_p1*100:.2f}')
print(f'METRIC val_acc={val_p1*100:.2f}')
print(f'METRIC val_f1={val_macro_f1:.4f}')
print(f'METRIC train_time_s={train_time:.1f}')
print(f'METRIC xnli_western_f1={xnli_west_f1:.4f}')
print(f'METRIC xnli_central_f1={xnli_cent_f1:.4f}')
print(f'METRIC xnli_navlab_f1={xnli_nlab_f1:.4f}')
print(f'PARAMS minn=$MINN maxn=$MAXN wordNgrams=$WORDNGRAMS dim=$DIM lr=$LR epoch=$EPOCH loss=$LOSS')
print(f'TRAIN {train_time:.1f}s')
print(f'XNLI P@1={xnli_p1*100:.2f}% (W={xnli_west_f1:.4f} C={xnli_cent_f1:.4f} NL={xnli_nlab_f1:.4f})')
print(f'VAL   P@1={val_p1*100:.2f}% F1={val_macro_f1:.4f}')
" 2>&1
