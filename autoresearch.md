# Autoresearch: fastText Basque Dialect Classification Optimization

## Objective
Optimize fastText hyperparameters for Basque dialect classification to maximize XNLI test accuracy while maintaining or improving in-domain (Klasikoak) validation accuracy. The best current model achieves 96.0% XNLI accuracy with the hybrid training set (17,955 sentences, 5 classes).

## Metrics
- **Primary**: xnli_acc (%, higher is better) — XNLI dialectal test accuracy (3-class remapped)
- **Secondary**: val_acc, val_f1, train_time_s, xnli_western_f1, xnli_central_f1, xnli_navlab_f1 — Klasikoak val accuracy/macro-F1, training wall time, per-class XNLI F1

## How to Run
`./autoresearch.sh` — trains a fastText model, evaluates on both Klasikoak validation and XNLI test, outputs METRIC lines.

## Optimization Dimensions
Based on fastText literature and our domain knowledge (character n-grams capture Basque morphological dialect markers):

1. **Character n-gram range** (minn, maxn): Smaller grams capture morphology (suffixes like -gaz, -det), larger grams capture lexical items. Basque suffixes are 2-4 chars.
2. **Word n-grams** (wordNgrams): Capture multi-word dialectal expressions.
3. **Learning rate and epochs**: Tradeoff between convergence and overfitting.
4. **Model capacity** (dim): 100 works well but maybe 200 helps with 5 classes.
5. **Loss function**: softmax vs hierarchical softmax vs one-vs-all.
6. **Subsampling**: Reduce dominance of nav-lab class (52% of training data).
7. **Bucket size**: Affects training speed and subword granularity.

## Files in Scope
- `autoresearch.sh` — training + evaluation script (the only file we modify)
- `autoresearch.md` — this file (update "What's Been Tried" as we go)

## Off Limits
- `data/processed/text/` — training/evaluation data must not change
- `src/` — source code not relevant, this is pure hyperparameter search
- `models/` — final models saved by the script; intermediate models use temp paths

## Constraints
- Training must complete within 120s (conversational pace)
- Must evaluate on both XNLI test (2,505 sentences) and Klasikoak val (4,489 sentences)
- Must support 5-class output (western, central, navarrese, nav-lab, souletin)
- XNLI test uses 3-class remap: navarrese→central, souletin→nav-lab

## EuskanolDS Test Results (2026-06-07)

The best model (lr=0.2, epoch=75, XNLI=96.85%) was tested on HiTZ/EuskanolDS `gold` split — 927 real-world Spanish-Basque code-switched tweets. No ground-truth dialect labels exist for this dataset.

### Findings

- **No Navarrese or Souletin predictions.** The dataset is Hegoalde-only (Spanish + Basque), so the absence of Iparralde dialects is expected and correct.
- **Only 5.7% of predictions exceed 0.85 confidence.** The model is honest about uncertainty on code-switched text.
- **51% of samples have confidence below 0.5.** Spanish tokens dilute the dialect signal — the model sees characters it never encountered in training (monolingual Basque classical literature).
- **Many samples are standard Batua**, not dialectal Basque. Since Batua is absent from the 5-class training data, the model has no correct label for these and spreads probability across classes — a reasonable failure mode given the taxonomy.

### Conclusion

The model cannot reliably label this dataset. Two fundamental mismatches explain the poor performance: (1) the training data is monolingual Basque literature, not code-switched social media; and (2) the 5-class taxonomy lacks Batua, forcing the model to assign dialect labels to standard Basque text. Adding Batua as a 6th class and training on informal/social-media text would address both issues. The per-sample predictions are saved at `data/processed/text/euskanol_gold_predictions.jsonl` for manual review.

## Optimization Experiments (17 runs)

### Baseline
minn=3, maxn=6, wordNgrams=2, dim=100, lr=0.1, epoch=25, loss=softmax → XNLI=96.09%, Val=97.75%, F1=0.9756

### Char n-gram range sweeps
- **minn=2, maxn=8:** XNLI=94.89% (-1.2%). Wider range hurts generalization.
- **minn=2, maxn=5:** XNLI=94.89%. Losing 6-grams drops useful signal.
- **minn=2, maxn=6:** XNLI=95.37%. Including bigrams adds noise.
- **minn=3, maxn=7:** XNLI=96.77%. Best Western F1 (0.9772) but not overall.
- **minn=4, maxn=6:** XNLI=96.81%. minn=3 is optimal — 3-grams capture Basque suffixes (gaz, det, zea).

### Learning rate & epochs
- **lr=0.05:** XNLI=95.21% (-0.88%). Too slow.
- **lr=0.2, epoch=25:** XNLI=96.49% (+0.40%). Best single change.
- **lr=0.2, epoch=50:** XNLI=96.77%. Continued improvement.
- **lr=0.2, epoch=75:** XNLI=96.85%. **Best overall.** Western=0.9766, Central=0.9600, Nav-Lab=0.9687.
- **lr=0.2, epoch=100:** XNLI=96.81%. Diminishing returns.
- **lr=0.3, epoch=25:** XNLI=96.57%. Too aggressive.
- **lr=0.15, epoch=100:** XNLI=96.81%. No advantage.

### Other dimensions
- **dim=200:** XNLI=96.01%. Not worth extra training time.
- **wordNgrams=3:** XNLI=96.77%. Trigrams add no gain.
- **hs (hierarchical softmax):** XNLI=96.25%. Softmax wins for 5-class.
- **minCount=5, bucket=4M:** No effect.

### Best configuration
`minn=3, maxn=6, wordNgrams=2, dim=100, lr=0.2, epoch=75, loss=softmax` → XNLI=96.85% (+0.76%), train=13.5s

Central remains hardest class — shares transition-zone features with Western.
