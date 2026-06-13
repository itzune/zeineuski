#!/bin/bash
set -euo pipefail
# Autoresearch — speech dialect model benchmark (GPU: root@10.2.121.210)

GPU_HOST="root@10.2.121.210"
GPU_DIR="/root/zeineuski"

# ── Sync code changes to GPU ──────────────────────────────────────────────────
rsync -az -e "ssh" \
  /home/xezpeleta/Dev/itzune/zeineuski/src/ \
  "${GPU_HOST}:${GPU_DIR}/src/" \
  2>&1 | tail -1

# ── Train + evaluate on GPU ───────────────────────────────────────────────────
# Preprocessing is already done (data/processed/speech/ahotsak_full/ exists)
ssh "${GPU_HOST}" "cd ${GPU_DIR} && uv run python -m src.models.speech.ecapa_tdnn train \
  --train-manifest ${GPU_DIR}/data/processed/speech/ahotsak_full/train.csv \
  --val-manifest ${GPU_DIR}/data/processed/speech/ahotsak_full/val.csv \
  --test-manifest ${GPU_DIR}/data/processed/speech/ahotsak_full/test.csv \
  --output ${GPU_DIR}/models/speech/ecapa_dialect \
  --config ${GPU_DIR}/configs/speech/ecapa.yaml \
  --embedding-only \
  --device cuda \
  --output-stats" 2>&1 | grep -vE 'FutureWarning|^Note:|warnings'
