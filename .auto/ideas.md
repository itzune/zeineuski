# Whisper Encoder Basque Dialect Classification — Results Log

Session: 2026-06-13 · 14 runs (4 keep, 10 discard) · Best accuracy=62.15%, Best macro F1=0.362

## What worked
- **Frozen Whisper large-v3-eu encoder → 2-layer MLP** — 59.6% accuracy on 5-class town-disjoint splits. The encoder captures phonetic/prosodic features that preserve dialectal patterns before the decoder normalizes to batua. This is a +10.1pp improvement over ECAPA-TDNN frozen embeddings (49.5%).
- **mean+std+max pooling (3840-dim)** — Captures temporal distribution of encoder features. Gives richer representation than mean-only (~1280-dim). Helps minority classes: nav-lab F1 improved 4× (0.01→0.048) though still low.
- **Focal loss (gamma=2.0)** — Down-weights easy (western) examples and focuses gradient on minority classes. Best macro F1 of 0.362 (+3.4% vs baseline). Central +2.7pp, navarrese +2.7pp, souletin +2.5pp.
- **Lower learning rate (5e-4 vs 1e-3)** — +2pp accuracy (62.15% vs 60.12%). Navarrese jumps to 51.1% F1, souletin to 10.7%. Stochastic variance ±2pp across runs.
- **Prefixed embeddings** — Extracting and caching embeddings once (~25 min for 36K segments) enables fast experimentation (30s per training run).

## What didn't work
- **Class weights** — Accuracy metric punishes minority improvement. No real gain when using accuracy as primary metric.
- **3-layer MLP** — Overfits. More capacity just amplifies majority class (western +2.9pp, central -0.9pp). 2 layers is the sweet spot.
- **hidden_dim=1024** — Overfits. 512 works best.
- **High dropout (0.5)** — Kills minority class representations. Dropout=0.3 is optimal.
- **Focal gamma=3.0** — Too aggressive. Central and souletin drop. Gamma=2.0 is optimal.
- **Lower lr (3e-4) + heavy regularization** — No improvement, slightly worse.

## Best config per metric
### Accuracy-optimized: 59.64%
```
pooling: mean_std_max, hidden_dim: 512, num_layers: 2
class_weights: true, dropout: 0.3
batch_size: 64, lr: 1e-3, epochs: 100
```
Western 78.5%, Navarrese 44.5%, Central 38.2%

### Macro F1-optimized: 0.362
```
pooling: mean_std_max, hidden_dim: 512, num_layers: 2
loss: focal, focal_gamma: 2.0, dropout: 0.3
```
Central 41.9%, Navarrese 47.4%, Souletin 13.3%

## Remaining bottlenecks (5-class)
- **nav-lab (0-1% F1)**: 2,353 train samples but town-disjoint split makes it the hardest. The dialect is very close to navarrese — effectively a 4.5-class problem.
- **souletin (8-13% F1)**: Only 348 train samples, 37 source towns. Data scarcity is the bottleneck.
- **central (38-42% F1)**: 658 source towns but samples vary wildly in recording quality.

## Quick experiments (post-finalize, 2026-06-13)
- **Exp 37: 5-seed ensemble** → 59.90% acc, 0.3436 macro F1 — No improvement, just noise averaging
- **Exp 38a: LR sweep 5e-4** → **62.15% acc, 0.3606 macro F1** ✅ Best accuracy (+2pp). Navarrese +5.4pp, souletin +3.6pp
- **Exp 38b: LR sweep 1e-4** → 58.28% — Undertrained
- **Exp 38c: LR sweep 5e-3** → 59.98% — Too aggressive
- **Exp 39: Wider MLP (1024 dim)** → 59.31% — Overfits western, drags down minority classes
- **Whisper fine-tuning test** (100/class, 3 unfrozen layers) → Failed — NaN loss with fp16, 21% acc with fp32. 500 samples not enough for 39M params
- **LR=5e-4 is now default** in configs/speech/whisper.yaml

## Promising untested ideas
- **Attention pooling during training** (instead of frozen mean_std_max) — requires re-extraction but the ADI-20 paper showed this works
- **3-class reduction** (western/central/navarrese group) — should surpass 65-70% accuracy
- **Whisper encoder fine-tuning** (unfreeze last 2-3 encoder layers) — could learn dialect-specific phonetic features
- **wav2vec2 XLSR models** — cross-lingual pretraining might give better phonetic granularity
- **Class-specific thresholds** — calibrated decision boundaries for imbalanced data
- **Multi-seed averaging** — train 5 MLPs with different seeds, ensemble predict. Should reduce stochastic variance (noise floor ~1-2pp).
