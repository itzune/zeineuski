# Whisper Encoder Basque Dialect Classification — Results Log

Session: 2026-06-14 · Merged Ahotsak+Mintzoak dataset · 1 run baseline

## Baseline (merged dataset)
- **Accuracy**: 73.89% | **Macro F1**: 0.4327
- nav-lab: 0.86 | souletin: 0.28 | western: 0.58 | central: 0.35 | navarrese: 0.10
- Config: focal gamma=2.0, lr=5e-4, hidden_dim=512, dropout=0.3, batch=64, epochs=100

## Optimization targets
1. **Balanced subsampling** — Downsample nav-lab from 89K to 10K → 50K balanced dataset
   - Expected: recovers western (>0.70) and navarrese (>0.30), nav-lab stays >0.60
2. **Higher focal gamma** — γ=3.0 or 4.0 penalizes nav-lab harder without losing data
3. **Class weights** — sklearn compute_weights, may complement focal loss
4. **Lower LR** — Current 5e-4 may be too aggressive with 89K nav-lab g'radient domination
5. **More dropout** — 0.5 may regularize nav-lab overfitting

## What worked (Ahotsak-only, historical)
- Focal loss (γ=2.0) over crossentropy: +3.4% macro F1
- LR=5e-4 over 1e-3: +2pp accuracy
- mean_std_max pooling over mean-only: +4× nav-lab F1
- 2-layer MLP: sweet spot (1-layer weak, 3-layer overfits)

## Dead ends (Ahotsak-only, historical)
- ECAPA-TDNN: 49.5% — speaker identity, not dialect
- XLSR wav2vec2: 29.6% — BP1 fine-tuning erased dialect variation
- ASR pipeline: 42% — decoder normalizes to batua
- Whisper encoder fine-tuning: 21% — 39M params, NaN with fp16
- 5-seed ensemble: no benefit, just noise averaging
