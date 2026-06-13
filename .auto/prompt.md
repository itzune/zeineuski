# Autoresearch: Whisper Encoder Basque Dialect Classification

## Objective
Maximize 5-class Basque dialect classification accuracy using a frozen Whisper
large-v3-eu encoder + MLP classifier. Target: >60% overall accuracy on town-disjoint
splits. The embeddings are pre-extracted (no re-extraction needed between runs).

## Metrics
- **Primary**: accuracy (higher is better)
- **Secondary**: macro_f1, per-class F1 — tradeoff monitors (especially nav-lab, souletin)

## How to Run
`./.auto/measure.sh` — syncs code to GPU, trains MLP on pre-extracted embeddings, evaluates.

## Files in Scope
- `src/models/speech/whisper_did.py` — **THE file**. Contains `MLPClassifier` and `train_mlp()`.
- `configs/speech/whisper.yaml` — hyperparameters: hidden_dim, dropout, lr, epochs, batch_size
- `.auto/measure.sh` — benchmark script

## Off Limits
- `models/speech/whisper_*_emb.pkl` — embeddings are fixed (pre-extracted from 36K audio segments)
- `data/` directories — training/test data is fixed
- The Whisper encoder weights — frozen, cannot be fine-tuned in this setup

## Constraints
- Train MLP only — no architecture changes to Whisper encoder
- Training must complete within 5 minutes
- Must use the fixed town-disjoint train/val/test splits

## Hyperparameter Space
- **hidden_dim**: 128, 256, 512, 1024 — MLP hidden layers
- **dropout**: 0.1, 0.3, 0.5, 0.7
- **epochs**: 30, 50, 100, 200
- **lr**: 1e-4, 3e-4, 1e-3, 3e-3
- **batch_size**: 32, 64, 128, 256
- **weight_decay**: 1e-5, 1e-4, 1e-3
- **MLP depth**: 1 layer vs 2 layers vs 3 layers
- **Class weights**: none vs balanced vs custom

## Baseline
- Accuracy: 59.06%
- Western: 76%, Navarrese: 47%, Central: 40%
- Souletin: 3%, Nav-lab: 2%

## What's Been Tried
- Initial run: hidden_dim=512, dropout=0.3, epochs=50, lr=1e-3, batch=64, 2-layer MLP — 59.06%
