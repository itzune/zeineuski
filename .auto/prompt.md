# Autoresearch: Azpieuskalki 12-class F1 Optimization

## Objective
Maximize the overall macro-averaged F1 score of the 12-class azpieuskalki (sub-dialect) fastText classifier. The current baseline is 82.38% accuracy with severe class imbalance in per-class F1: top-5 classes are 80-95% F1 but bottom-5 are 55-68% F1. The training data is fixed (51,837 sentences from Ahotsak oral archive + SU AZIA Zuberotarra corpus). All optimization is through fastText hyperparameters and data preparation strategies.

## Metrics
- **Primary**: weighted_f1 (higher is better) — weighted-average F1 across all 12 classes
- **Secondary**: macro_f1, bottom5_mean_f1, bottom5_min_f1, overall_accuracy — tradeoff monitors

## How to Run
`./.auto/measure.sh` — re-prepares data, trains model, evaluates, outputs `METRIC` lines.

## Files in Scope
- `src/data/train_azpieuskalki.py` — **THE file**. Contains `prepare_azpieuskalki_data()` (data loading, sentence cleaning, ovsersampling) and `train_model()` (fastText config). All hyperparameter changes happen here.
- `.auto/measure.sh` — benchmark script, may be updated to capture more signals

## Off Limits
- `data/` directories — training/test data is fixed, do NOT modify passages, CSV, or text files
- `models/` — the output model file is temporary, always written to `models/azpieuskalki.bin`
- Any file outside this project — no new deps, no system-level changes

## Constraints
- fastText only — no architecture changes, no PyTorch, no sklearn classifiers
- Training must complete within 5 minutes (single run)
- Must produce a valid `azpieuskalki.bin` model file
- The `loss` parameter supports: `ns` (negative sampling), `hs` (hierarchical softmax), `ova` (one-vs-all)
- The `oversample_factor` parameter: None (disabled), or int divisor for target count per class
- Existing optimal baseline: dim=200, lr=0.2, epoch=75, wordNgrams=2, minn=2, maxn=6, loss=ns, minCount=1, NO oversampling, NO autotune

## Hyperparameter Space

Key knobs to explore:
- **loss**: `ns` (current), `hs` (hierarchical softmax, good for structure), `ova` (one-vs-all binary CE, theoretically best for imbalanced multi-class)
- **oversample_factor**: None, 2, 4, 8 — controls `target = max(max_class_count // factor, 100)`. Lower factor = closer to balanced.
- **wordNgrams**: 2 (current), 3 — more context for dialectal phrases
- **minn/maxn**: (2,6) current, (2,8), (3,7), (3,8) — char n-gram coverage for Basque morphology
- **dim**: 200 (current), 300 — model capacity
- **epoch**: 75 (current), 50, 100, 150 — training duration
- **lr**: 0.2 (current), 0.1, 0.3, 0.5 — learning rate

## What's Been Tried
_(will be updated as experiments run)_
