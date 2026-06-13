#!/bin/bash
set -euo pipefail
# Autoresearch measure.sh — 5-class euskalki model benchmark

# Build combined training data (Klasikoak + azpieuskalki)
python3 src/data/build_combined_euskalki.py 2>&1

# Train
python3 -c "
import fasttext, time, os, sys
sys.path.insert(0, '.')
from src.data.train_euskalki import EUSK_MODEL

params = {
    'dim': int('${DIM:-100}'),
    'epoch': int('${EPOCH:-75}'),
    'lr': float('${LR:-0.2}'),
    'wordNgrams': int('${WORD_NGRAMS:-2}'),
    'minn': int('${MINN:-2}'),
    'maxn': int('${MAXN:-6}'),
    'minCount': int('${MIN_COUNT:-1}'),
    'loss': 'ns',
    'bucket': int('${BUCKET:-500000}'),
    'seed': int('${SEED:-42}'),
    'thread': int('${THREAD:-4}'),
}
t0 = time.time()
model = fasttext.train_supervised(input='data/processed/text/train_euskalki_combined.txt', **params)
model.save_model(str(EUSK_MODEL))
elapsed = time.time() - t0
size_mb = os.path.getsize(EUSK_MODEL) / (1024*1024)
print(f'MODEL_SIZE_MB={size_mb:.1f}')
print(f'TRAIN_TIME_S={elapsed:.0f}')
" 2>&1

# Evaluate
python3 src/data/train_euskalki.py evaluate 2>&1
