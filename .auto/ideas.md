# Whisper Encoder Basque Dialect Classification — Results Log

## Session: 2026-06-14 · Merged Ahotsak+Mintzoak dataset · 14 experiments · 3 keep, 11 discard

### Final results

| | Ahotsak only | +Mintzoak (raw) | +Mintzoak (balanced 10K) |
|---|---|---|---|
| Accuracy | 62.15% | 73.89% | **70.52%** |
| Macro F1 | 0.361 | 0.433 | **0.510** |
| Nav-Lab | 0.02 | 0.86 | 0.82 |
| Western | 0.79 | 0.58 | 0.69 |
| Central | 0.38 | 0.35 | 0.33 |
| Souletin | 0.11 | 0.28 | 0.40 |
| Navarrese | 0.51 | 0.10 | 0.31 |

### Best config
```yaml
loss: crossentropy
balanced_subsample: 10000   # 50K total balanced dataset
hidden_dim: 512
dropout: 0.3
num_layers: 2
learning_rate: 5e-4
epochs: 100
batch_size: 64
```

### What worked
- **Balanced subsampling (10K/class)**: +17.8% macro F1 over imbalanced merged dataset. Single most impactful change.
- **CE without focal**: With balanced data, crossentropy alone is optimal. Focal loss + balanced data = redundant regularizers.
- **2-layer MLP, hidden_dim=512**: Still the sweet spot even with balanced data.
- **lr=5e-4**: Consistent across both imbalanced and balanced setups.

### What didn't work
- Focal gamma=3.0/4.0 with balanced data: flat to slightly worse
- Class weights: worse than balanced subsampling alone
- hidden_dim=1024: overfits even on balanced data
- num_layers=3: identical to 2-layer
- dropout=0.5: less capacity hurts central
- 5K subsample: too aggressive, central collapses
- 20K subsample: tilts back toward nav-lab
- 150 epochs: converged at 100

### Key insight
Balanced subsampling (10K/class) is the silver bullet. It solves the 18:1
nav-lab:central imbalance without complex loss functions. Training is 2.5× faster
(90s vs 4 min) and results are stronger (0.510 vs 0.433 macro F1).
