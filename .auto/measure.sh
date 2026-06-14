#!/bin/bash
set -euo pipefail
# Autoresearch — Whisper encoder dialect classifier benchmark (GPU: root@10.2.121.210)
# Merged dataset: Ahotsak (36K) + Mintzoak (160K) = 197K segments

GPU_HOST="root@10.2.121.210"
GPU_DIR="/root/zeineuski"

# ── Sync code changes to GPU ──────────────────────────────────────────────────
rsync -az -e "ssh" \
  /home/xezpeleta/Dev/itzune/zeineuski/src/ \
  "${GPU_HOST}:${GPU_DIR}/src/"

rsync -az -e "ssh" \
  /home/xezpeleta/Dev/itzune/zeineuski/configs/ \
  "${GPU_HOST}:${GPU_DIR}/configs/"
  2>&1 | tail -1

# ── Train + evaluate on GPU ───────────────────────────────────────────────────
ssh "${GPU_HOST}" "export PATH=\$HOME/.local/bin:\$PATH && cd ${GPU_DIR} && uv run python -m src.models.speech.whisper_did train \
  --train-emb ${GPU_DIR}/models/speech/whisper_merged_train_emb.pkl \
  --val-emb ${GPU_DIR}/models/speech/whisper_merged_val_emb.pkl \
  --test-emb ${GPU_DIR}/models/speech/whisper_merged_test_emb.pkl \
  --output ${GPU_DIR}/models/speech/whisper_dialect_merged \
  --config ${GPU_DIR}/configs/speech/whisper.yaml \
  --device cuda" 2>&1 | grep -vE 'FutureWarning|^Note:|warnings'
