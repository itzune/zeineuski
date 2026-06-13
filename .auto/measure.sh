#!/bin/bash
set -euo pipefail
# Autoresearch — Whisper encoder dialect classifier benchmark (GPU: root@10.2.121.210)

GPU_HOST="root@10.2.121.210"
GPU_DIR="/root/zeineuski"

# ── Sync code changes to GPU ──────────────────────────────────────────────────
rsync -az -e "ssh" \
  /home/xezpeleta/Dev/itzune/zeineuski/src/ \
  /home/xezpeleta/Dev/itzune/zeineuski/configs/ \
  "${GPU_HOST}:${GPU_DIR}/" \
  2>&1 | tail -1

# ── Train + evaluate on GPU ───────────────────────────────────────────────────
ssh "${GPU_HOST}" "export PATH=\$HOME/.local/bin:\$PATH && cd ${GPU_DIR} && uv run python -m src.models.speech.whisper_did train \
  --train-emb ${GPU_DIR}/models/speech/whisper_train_emb3d.pkl \
  --val-emb ${GPU_DIR}/models/speech/whisper_val_emb3d.pkl \
  --test-emb ${GPU_DIR}/models/speech/whisper_test_emb3d.pkl \
  --output ${GPU_DIR}/models/speech/whisper_dialect \
  --config ${GPU_DIR}/configs/speech/whisper.yaml \
  --device cuda" 2>&1 | grep -vE 'FutureWarning|^Note:|warnings'
