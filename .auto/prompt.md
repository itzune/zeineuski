# Autoresearch: Speech Model Baseline (ECAPA-TDNN)

## Objective
Build the first speech-based dialect identification model for Basque using Ahotsak.eus scraped audio. Start with a simple but effective approach: extract speaker embeddings with ECAPA-TDNN → train SVM classifier on top. Target: >60% accuracy on a 5-class held-out test set.

## Metrics
- **Primary**: accuracy (%, higher is better) — dialect classification on test set
- **Secondary**: macro_f1, per_dialect_f1, train_time_s, model_size_mb, audio_hours, num_classes

## How to Run
`./.auto/measure.sh` — runs the full pipeline: download → preprocess → train → evaluate.

IMPORTANT: For fast iterations during optimization, use a subset of data:
```bash
# Subset pipeline (fast, for iteration):
uv run python -m src.data.speech_preprocessing run \
  --audio-dir data/raw/speech/ahotsak/audio_subset \
  --passages data/raw/speech/ahotsak/ahotsak_passages_20260608_213742.jsonl \
  --output data/processed/speech/ahotsak_subset \
  --config configs/speech/preprocessing.yaml

uv run python -m src.models.speech.ecapa_tdnn train \
  --train-manifest data/processed/speech/ahotsak_subset/train.csv \
  --val-manifest data/processed/speech/ahotsak_subset/val.csv \
  --test-manifest data/processed/speech/ahotsak_subset/test.csv \
  --output models/speech/ecapa_dialect \
  --config configs/speech/ecapa.yaml \
  --embedding-only --device cpu --output-stats
```

Full pipeline:
```bash
./.auto/measure.sh
```

## Files in Scope
- `src/data/audio_downloader.py` — Download MP3s/MP4s from Ahotsak S3, convert MP4→MP3
- `src/data/speech_preprocessing.py` — VAD, resample, normalize, town-disjoint splits
- `src/models/speech/ecapa_tdnn.py` — ECAPA embedding extraction + SVM classifier
- `configs/speech/preprocessing.yaml` — VAD method (energy), sample rate, durations
- `configs/speech/ecapa.yaml` — SVM kernel, C, gamma, classifier type
- `src/data/ahotsak_scraper.py` — Reference only (used for passage metadata)

## Off Limits
- `src/data/text_*.py` — Text pipeline, unrelated
- `src/models/text/` — Text models
- `src/cli.py`, `src/inference.py` — CLI
- `src/evaluation/` — Upstream evaluation

## Constraints
- Speaker/town-disjoint splits: no town in both train and test
- Model must train on CPU (no GPU available)
- Python 3.11+, uv-managed dependencies
- Output models to `models/speech/ecapa_dialect/`
- Download rate-limited (0.3s between requests, 0.05s for MP4 conversion)
- VAD method: energy-based (Silero hangs on trust prompt)

## What's Been Tried

### Run 1 — ECAPA-TDNN embedding + SVM (90 files, 3 dialects, town-disjoint)
- **Data**: 90 files (30/dialect × 3: western, central, navarrese), 1690 VAD segments, 3.1 audio hours
- **Splits**: train=1347 (11 towns), val=97 (2 towns), test=246 (4 towns)
- **Result**: 43.50% accuracy, macro F1=0.297
  - central: F1=0.34 (64 test samples)
  - navarrese: F1=0.55 (182 test samples)
  - western: F1=0.00 (0 test samples — all in train/val due to town split)
- **Issue**: Town-disjoint splits unbalanced — western had 0 test samples
- **Next**: More data, better split strategy, hyperparameter tuning

### Infrastructure built
- Audio downloader: 1,587/2,430 files downloaded (ongoing background process)
- Dialect coverage so far: western=734, central=563, navarrese=113, (nav-lab and souletin pending)
- All MP4→MP3 conversion working via ffmpeg
- Energy-based VAD working (Silero untrusted repo prompt blocks it)
- ECAPA-TDNN model cached at `models/speech/speechbrain_cache/`
