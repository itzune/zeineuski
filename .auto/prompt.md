# Autoresearch: Whisper Encoder Basque Dialect Classification (Merged Dataset)

## Objective
Maximize 5-class Basque dialect classification macro F1 using a frozen Whisper
large-v3-eu encoder + MLP classifier on the merged Ahotsak+Mintzoak dataset
(197K segments, 259h). The primary bottleneck is class imbalance: nav-lab has
89K train samples vs 5K central (18:1 ratio).

## Metrics
- **Primary**: macro F1 (higher is better) — balances minority and majority classes
- **Secondary**: accuracy, per-class F1 (western, central, navarrese, nav-lab, souletin)

## Baseline (merged dataset, current config)
- Accuracy: 73.89%
- Macro F1: 0.4327
- Nav-lab: 0.86 | Souletin: 0.28 | Western: 0.58 | Central: 0.35 | Navarrese: 0.10

## Target
- Recover western (>0.70) and navarrese (>0.30) F1 without cratering nav-lab
- Target macro F1: ≥0.50

## How to Run
`./.auto/measure.sh` — syncs code to GPU, trains MLP on pre-extracted embeddings, evaluates.

## Files in Scope
- `src/models/speech/whisper_did.py` — MLPClassifier, train_mlp(), FocalLoss
- `configs/speech/whisper.yaml` — hyperparameters
- `.auto/measure.sh` — benchmark script

## Off Limits
- `models/speech/whisper_merged_*_emb.pkl` — 197K embeddings, pre-extracted, fixed
- `data/` directories — training/test data is fixed
- Whisper encoder weights — frozen

## Constraints
- Training must complete within 5 minutes on GPU
- Town-disjoint splits (no town in both train and test)

## Hyperparameter Space
- **hidden_dim**: 256, 512, 1024
- **dropout**: 0.2, 0.3, 0.5
- **epochs**: 50, 100, 150
- **lr**: 1e-4, 5e-4, 1e-3, 5e-3
- **batch_size**: 32, 64, 128
- **focal_gamma**: 1.0, 2.0, 3.0, 4.0
- **Class weights**: none vs balanced vs compute from sklearn
- **Balanced subsampling**: downsample nav-lab to 5K/10K/20K

## What's Been Tried (combined dataset)
- Baseline: focal gamma=2.0, lr=5e-4, hidden_dim=512, dropout=0.3, batch=64, epochs=100
  → 0.433 macro F1, 73.89% acc. Nav-lab dominates (0.86), navarrese collapses (0.10)

## Key Insight
The 18:1 nav-lab:central imbalance causes the model to default to nav-lab for
ambiguous samples. Navarrese (phonetically intermediate between central and
nav-lab) is the biggest casualty. Solutions: balanced subsampling, higher focal
gamma, or class weights.
