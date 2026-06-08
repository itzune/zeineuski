# Autoresearch: Speech Model Baseline (ECAPA-TDNN + fastText)

## Objective
Build the first speech-based dialect identification model for Basque using Ahotsak.eus scraped audio. Start with a simple but effective approach: extract speaker embeddings with ECAPA-TDNN → train a classifier on top. The goal is >60% accuracy on a 5-class (or 3-class) held-out test set.

Phase 1: Download audio, preprocess (VAD, resample), organize into speaker-disjoint train/val/test splits.
Phase 2: Extract ECAPA-TDNN embeddings, train classifier, evaluate.

## Metrics
- **Primary**: accuracy (%, higher is better) — 5-class dialect classification on test set
- **Secondary**: per_dialect_f1, train_time_s, model_size_mb, audio_hours

## How to Run
`./.auto/measure.sh` — runs the full pipeline: download → preprocess → train → evaluate.

## Files in Scope
- `src/data/speech_loader.py` — Load audio, apply preprocessing, create speaker-disjoint splits
- `src/data/speech_preprocessing.py` — VAD, resampling, normalization
- `src/data/audio_downloader.py` — NEW: Download MP3s from Ahotsak S3 URLs
- `src/models/speech/ecapa_tdnn.py` — ECAPA-TDNN embedding extraction + classifier training
- `configs/speech/preprocessing.yaml` — Audio preprocessing config
- `configs/speech/ecapa.yaml` — ECAPA-TDNN training config

## Off Limits
- `src/data/ahotsak_scraper.py` — Already works, don't touch
- `src/data/text_*.py` — Text pipeline, unrelated
- `src/models/text/` — Text models
- `src/cli.py`, `src/inference.py` — CLI and inference

## Constraints
- Audio downloads must be rate-limited (1 req/s, respect Ahotsak's S3)
- Speaker-disjoint splits: no speaker in both train and test
- Model must train on consumer hardware (CPU or single GPU)
- Python 3.11+, uv-managed dependencies
- Output models to `models/speech/`

## What's Been Tried
*Nothing yet — this is the first speech session.*

### Current state
- 2,508 Ahotsak passages scraped (2,430 with audio URLs)
- Dialect distribution: western=1019, navarrese=618, central=583, nav-lab=230, souletin=37
- 168-town municipality→dialect mapping available
- Audio URLs point to DreamObjects S3 (ahotsbiltegia-1.s3.us-east-005.dream.io)
- Empty stub files exist for all speech model modules
