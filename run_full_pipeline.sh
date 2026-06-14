#!/bin/bash
set -euo pipefail
export PATH="/root/.local/bin:$PATH"
cd /root/zeineuski
PYTHON=/root/zeineuski/.venv/bin/python

TS=$(date -u +%Y%m%d_%H%M%S)
LOG="/tmp/pipeline_${TS}.log"
echo "=== Full pipeline started at $(date -u) ===" | tee "$LOG"

# ── Step 1: Wait for train to finish ──
TRAIN_PID=$(pgrep -f "whisper_did extract.*train" | head -1)
if [ -n "$TRAIN_PID" ]; then
    echo "Waiting for train extraction (PID $TRAIN_PID)..." | tee -a "$LOG"
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        sleep 300
        LAST=$(tail -1 /tmp/mintzoak_extract_train.log 2>/dev/null || true)
        echo "  $(date +%H:%M) $LAST" | tee -a "$LOG"
    done
    echo "Train extraction finished at $(date -u)" | tee -a "$LOG"
else
    echo "Train extraction not running — checking if output exists" | tee -a "$LOG"
fi

# Verify train output
TRAIN_OUT="models/speech/whisper_merged_train_emb.pkl"
if [ ! -f "$TRAIN_OUT" ]; then
    echo "ERROR: $TRAIN_OUT not found! Aborting." | tee -a "$LOG"
    exit 1
fi
TRAIN_SIZE=$(du -h "$TRAIN_OUT" | cut -f1)
echo "Train embeddings: $TRAIN_SIZE" | tee -a "$LOG"

# ── Step 2: Extract val ──
echo "" | tee -a "$LOG"
echo "=== Step 2: val extraction ===" | tee -a "$LOG"
$PYTHON -u -m src.models.speech.whisper_did extract \
  --manifest data/processed/speech/merged/val.csv \
  --output models/speech/whisper_merged_val_emb.pkl \
  --device cuda 2>&1 | tee -a "$LOG"
VAL_SIZE=$(du -h models/speech/whisper_merged_val_emb.pkl | cut -f1)
echo "Val embeddings: $VAL_SIZE" | tee -a "$LOG"

# ── Step 3: Extract test ──
echo "" | tee -a "$LOG"
echo "=== Step 3: test extraction ===" | tee -a "$LOG"
$PYTHON -u -m src.models.speech.whisper_did extract \
  --manifest data/processed/speech/merged/test.csv \
  --output models/speech/whisper_merged_test_emb.pkl \
  --device cuda 2>&1 | tee -a "$LOG"
TEST_SIZE=$(du -h models/speech/whisper_merged_test_emb.pkl | cut -f1)
echo "Test embeddings: $TEST_SIZE" | tee -a "$LOG"

# ── Step 4: Train MLP ──
echo "" | tee -a "$LOG"
echo "=== Step 4: Training MLP ===" | tee -a "$LOG"
$PYTHON -u -m src.models.speech.whisper_did train \
  --train-emb models/speech/whisper_merged_train_emb.pkl \
  --val-emb models/speech/whisper_merged_val_emb.pkl \
  --test-emb models/speech/whisper_merged_test_emb.pkl \
  --output models/speech/whisper_dialect_merged \
  --config configs/speech/whisper.yaml \
  --device cuda 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Pipeline complete at $(date -u) ===" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"

# ── Print results summary ──
echo "" | tee -a "$LOG"
echo "=== RESULTS ===" | tee -a "$LOG"
grep "METRIC\|Accuracy\|Macro F1\|per-class" "$LOG" | tail -20 | tee -a "$LOG"
