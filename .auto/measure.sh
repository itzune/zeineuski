#!/bin/bash
set -euo pipefail
# Autoresearch measure.sh — Speech DID pipeline benchmark
# Phases: download audio → preprocess → ECAPA-TDNN train → evaluate
#
# Outputs METRIC lines for autoresearch to parse.

START_TIME=$(date +%s)

# ── Phase 1: Download Audio ───────────────────────────────────────────────────

echo "--- Phase 1: Downloading audio from Ahotsak S3 ---"
PHASE1_START=$(date +%s)

uv run python -m src.data.audio_downloader \
  --passages data/raw/speech/ahotsak/ahotsak_passages_20260608_213742.jsonl \
  --output data/raw/speech/ahotsak/audio \
  --max-audio 0 \
  --rate-limit 0.5

PHASE1_END=$(date +%s)
DOWNLOAD_TIME=$((PHASE1_END - PHASE1_START))

AUDIO_FILES=$(find data/raw/speech/ahotsak/audio -type f | wc -l)
echo "Downloaded ${AUDIO_FILES} audio files in ${DOWNLOAD_TIME}s"

# ── Phase 2: Preprocess Audio ─────────────────────────────────────────────────

echo "--- Phase 2: Preprocessing audio ---"
PHASE2_START=$(date +%s)

uv run python -m src.data.speech_preprocessing run \
  --audio-dir data/raw/speech/ahotsak/audio \
  --passages data/raw/speech/ahotsak/ahotsak_passages_20260608_213742.jsonl \
  --output data/processed/speech/ahotsak \
  --config configs/speech/preprocessing.yaml

PHASE2_END=$(date +%s)
PREPROC_TIME=$((PHASE2_END - PHASE2_START))

# Count processed samples
TOTAL_SAMPLES=$(wc -l < data/processed/speech/ahotsak/train.csv 2>/dev/null || echo 0)
TRAIN_SAMPLES=$(cat data/processed/speech/ahotsak/train.csv 2>/dev/null | wc -l || echo 0)

echo "Preprocessed ${TOTAL_SAMPLES} samples in ${PREPROC_TIME}s"

# ── Phase 3: Train ECAPA-TDNN ─────────────────────────────────────────────────

echo "--- Phase 3: Training ECAPA-TDNN classifier ---"
PHASE3_START=$(date +%s)

uv run python -m src.models.speech.ecapa_tdnn train \
  --train-manifest data/processed/speech/ahotsak/train.csv \
  --val-manifest data/processed/speech/ahotsak/val.csv \
  --test-manifest data/processed/speech/ahotsak/test.csv \
  --output models/speech/ecapa_dialect \
  --config configs/speech/ecapa.yaml \
  --embedding-only

PHASE3_END=$(date +%s)
TRAIN_TIME=$((PHASE3_END - PHASE3_START))

echo "Training completed in ${TRAIN_TIME}s"

# ── Phase 4: Evaluate ─────────────────────────────────────────────────────────

echo "--- Phase 4: Evaluating ---"
PHASE4_START=$(date +%s)

EVAL_OUTPUT=$(uv run python -m src.models.speech.ecapa_tdnn evaluate \
  --model models/speech/ecapa_dialect \
  --test-manifest data/processed/speech/ahotsak/test.csv \
  --output-stats 2>&1)

PHASE4_END=$(date +%s)
EVAL_TIME=$((PHASE4_END - PHASE4_START))

# Parse metrics from eval output
ACCURACY=$(echo "$EVAL_OUTPUT" | grep "^ACCURACY:" | awk '{print $2}')
MACRO_F1=$(echo "$EVAL_OUTPUT" | grep "^MACRO_F1:" | awk '{print $2}')
NUM_CLASSES=$(echo "$EVAL_OUTPUT" | grep "^NUM_CLASSES:" | awk '{print $2}')

# ── Report Metrics ─────────────────────────────────────────────────────────────

TOTAL_TIME=$((PHASE4_END - START_TIME))

MODEL_SIZE=$(du -sb models/speech/ecapa_dialect 2>/dev/null | awk '{print $1}' || echo 0)
MODEL_SIZE_MB=$((MODEL_SIZE / 1048576))

AUDIO_HOURS=$(echo "$EVAL_OUTPUT" | grep "^AUDIO_HOURS:" | awk '{print $2}' || echo 0)

echo ""
echo "METRIC accuracy=${ACCURACY:-0}"
echo "METRIC macro_f1=${MACRO_F1:-0}"
echo "METRIC num_classes=${NUM_CLASSES:-0}"
echo "METRIC total_time_s=${TOTAL_TIME}"
echo "METRIC download_time_s=${DOWNLOAD_TIME}"
echo "METRIC preprocess_time_s=${PREPROC_TIME}"
echo "METRIC train_time_s=${TRAIN_TIME}"
echo "METRIC eval_time_s=${EVAL_TIME}"
echo "METRIC model_size_mb=${MODEL_SIZE_MB}"
echo "METRIC audio_files=${AUDIO_FILES}"
echo "METRIC audio_hours=${AUDIO_HOURS}"
echo "METRIC train_samples=${TRAIN_SAMPLES}"
