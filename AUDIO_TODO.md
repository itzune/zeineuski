# Audio Pipeline — Pending Improvements

Three strategies to further improve the Whisper encoder + MLP dialect classifier.

**Current best: macro F1 0.5342** (5-class, merged Ahotsak+Mintzoak, 10K balanced + navarrese augmentation ×3).

## Results Summary

| Strategy | Macro F1 | vs baseline | Dataset | Status |
|---|---|---|---|---|
| **Baseline (mean_std_max, 768dim)** | **0.5193** | — | Full merged | ✅ Done |
| **Strategy 3: Navarrese augmentation** | **0.5342** | **+1.5pp** | Full merged | ✅ **New best** |
| Strategy 1: Attention pooling | 0.4765 | −4.3pp | Full merged | ❌ Rejected |
| Strategy 2: Audio+Text fusion | 0.6175 | +9.8pp | Ahotsak subset (21%) | ⚠️ Partial |

### Per-class breakdown (best model vs baseline)

| Class | Baseline | +Augmentation | Δ |
|---|---|---|---|
| **Macro F1** | 0.5193 | **0.5342** | **+1.5pp** |
| Central | 0.3918 | 0.3424 | −4.9pp |
| Nav-Lab | 0.8174 | 0.8348 | +1.7pp |
| Navarrese | 0.3210 | **0.3760** | **+5.5pp** |
| Souletin | 0.3929 | 0.4197 | +2.7pp |
| Western | 0.6733 | 0.6984 | +2.5pp |

---

## 1. Attention Pooling over Time Dimension ❌

**Expected impact:** High. **Actual:** Negative (−4.3pp).

Modified `WhisperEncoder.extract()` to return frame-level embeddings (seq_len × 1280) instead of mean-pooled. Implemented `AttentionPooling(nn.Module)` with a 2-layer bottleneck (1280→256→1), softmax-weighted sum over 8 temporal segments.

**Re-extraction:** ~2h on L40 (197K segments). Stores (8, 1280) tensors per sample (~4.7GB for train).

**Result:** ❌ Macro F1 0.4765 (vs 0.5193 baseline). Nav-lab +3.2pp but western −18.8pp. Attention learns to focus on nav-lab-like segments, collapsing western discrimination. Fixed 8-segment boundaries are too coarse.

**Files:** `src/models/speech/whisper_did.py` (AttentionPooling class, extraction modes, training adapters).

---

## 2. Audio + Text Model Fusion ⚠️ Partial

**Expected impact:** High (when transcriptions available). **Actual:** +32pp on Ahotsak subset, but only 21% data coverage.

Late-fusion MLP that concatenates Whisper audio embeddings (1280-dim) with fastText logits (5-dim) from transcriptions. Implemented as `FusionMLP` with separate projection heads:

```
Audio embed (1280) → Linear(768) → ReLU → Dropout
Text logits (5) → Linear(192) → ReLU
                  → Concat → Linear(384) → Linear(5) → Dialect
```

**Approach:**
1. Loaded Ahotsak passages JSONL with transcriptions (2,508 passages)
2. Mapped audio segment filenames → passage_id via regex parsing
3. Ran fastText `euskalki_5class.bin` on each transcription → 5-class probability vector
4. Concatenated audio embeddings + text logits → 1285-dim feature vectors
5. Trained FusionMLP (hidden_dim=512, dropout=0.3, lr=5e-4, 100 epochs)

**Result on Ahotsak subset (7.6K test samples):**
- Audio-only baseline: **macro F1 0.296**
- Audio+Text fusion: **macro F1 0.618** (+32.2pp!)
- Per-class: Central 0.33→0.66 (+33pp), Navarrese 0.39→0.76 (+37pp), Western 0.72→0.99 (+27pp), Souletin 0.02→0.67 (+65pp)

**Bottleneck:** Only 21% of the merged dataset (Ahotsak side) has transcriptions. Mintzoak passages (89K nav-lab, 10K souletin) have no text. Tested on the Ahotsak-only subset, but overall merged performance limited by missing Mintzoak transcriptions.

**Next step:** Run Whisper ASR on Mintzoak segments to get transcriptions (est. 2-3h for 160K segments on L40), then re-fuse on the full dataset. Expected full merged macro F1: 0.65–0.70.

**Files:** `scripts/fusion_train.py`, `models/euskalki_5class.bin` (uploaded to server).

---

## 3. Navarrese Data Augmentation ✅

**Expected impact:** Medium. **Actual:** +1.5pp macro F1, +5.5pp navarrese. **New best model.**

Embedding-level augmentation (noise injection + dropout + scaling) on the 9K navarrese train samples, tripling them to 27K. No audio re-processing needed — works directly on pre-extracted Whisper embeddings.

**Augmentation pipeline:**
- Gaussian noise at SNR 20-30 dB
- Random dimension dropout (5-15% zeroed)
- Random scaling (×0.85 to ×1.15)
- 1-2 transforms applied per copy, seeded with RandomState(42)

**Added to `whisper_did.py`:**
- `augment_embeddings()` function (embedding-level transforms)
- `augment` CLI subcommand: `--embeddings`, `--output`, `--labels`, `--factor`

**Training:** Same config as baseline (hidden_dim=768, dropout=0.3, lr=5e-4, batch_size=64, epochs=100, balanced_subsample=10000). Train time ~70s.

**Result:** ✅ Macro F1 0.5342 (+1.5pp). Navarrese +5.5pp (0.32→0.38). Central −4.9pp (model de-prioritizes smallest class after navarrese expansion). Souletin +2.7pp, Western +2.5pp.

**Model saved:** `models/speech/whisper_dialect_aug/classifier.pkl`

---

## Decision Framework (Updated)

| Strategy | Macro F1 gain | Coverage | Effort | Verdict |
|---|---|---|---|---|
| Navarrese augmentation | +1.5pp | Full dataset | Low (1min augment + 70s train) | ✅ **Deploy** |
| Audio+text fusion | +32pp | 21% only (Ahotsak) | Medium (needs ASR for rest) | ⚠️ Add ASR |
| Attention pooling | −4.3pp | Full dataset | High (2h re-extract) | ❌ Reject |

**Recommended next step:** Run Whisper ASR on Mintzoak segments (160K, est. 2-3h on L40) to get transcriptions, then retrain audio+text fusion on the full merged dataset. Expected macro F1: 0.65–0.70.

---

## Other Completed Tasks

### External Video Analysis ("Ahots Hariak")
Analyzed 30-min Kanaldude documentary (225 segments) using the baseline model:
- 78.7% nav-lab, 12.4% central, 8.0% souletin, 0.9% western
- Results at `data/processed/speech/external_analysis/results.json`
- Analysis script: `scripts/analyze_external_video.sh`

---

## Deployment Notes

### Hardware for inference

| Resource | Minimum | Recommended |
|---|---|---|
| GPU VRAM | ~4 GB | 6+ GB |
| CPU RAM | 8 GB | 16 GB |
| Disk | ~5.8 GB (Whisper model) | 8+ GB |
| GPU required? | No — CPU works but 8-10× slower |

- Whisper large-v3-eu encoder: 1.5B params, ~5.8 GB on disk (fp32), ~2.9 GB in memory (fp16)
- MLP classifier: 5 MB (negligible)
- Inference speed on L40: ~26 segments/second. On consumer GPU (RTX 3060): ~15-20/s. On CPU: ~2-3/s

### WASM / browser deployment

**Not feasible with the current architecture.** The Whisper large-v3-eu encoder is 1.5B parameters (~6 GB on disk), while browser WASM memory caps at 4 GB (realistically much less on mobile). Even if it could load, inference would take minutes per segment on CPU/WASM rather than the milliseconds needed for a usable UX.

**The path to WASM deployment:**
1. First, improve the teacher model — current best is macro F1 0.534. Fusion (strategy 2) needs Mintzoak ASR transcriptions to reach 0.65-0.70.
2. Once the teacher is good enough, use knowledge distillation: run the teacher on all 197K segments to get soft labels, then train a tiny student model (CNN-based, 10-50 MB, e.g. MobileNet or ECAPA-TDNN-lite) to mimic the teacher's predictions from raw audio.
3. Convert the student to ONNX → WASM-friendly format. At 10-50 MB and sub-100ms inference, browser/phone deployment becomes practical.

**Bottom line:** The current pipeline proves the approach works (0.534 macro F1 with frozen Whisper + MLP), but deployment on edge devices requires distillation into a much smaller model, and that's only worth doing once the teacher crosses a reasonable quality threshold (≥0.65 macro F1).

---

## Conclusions

### What worked

1. **Mean_std_max pooling over frozen Whisper encoder** — simple but effective. The 3840-dim concatenated statistics (mean, std, max of encoder hidden states) capture enough phonetic variation to discriminate 5 Basque dialects at 0.519 macro F1 with just a 2-layer 768-dim MLP.

2. **Subsampling to 10K/class** — the single most impactful change (+17.8pp macro F1). The 18:1 class imbalance (89K nav-lab vs 5K central) was the primary bottleneck. Simple balanced subsampling solved it without needing focal loss or class weights.

3. **Embedding-level augmentation** — tripling navarrese via noise/dropout/scaling on the 1280-dim embeddings (no audio re-processing) improved navarrese from 0.32 → 0.38 (+5.5pp) and pushed overall macro F1 to 0.534. The tradeoff was central dropping from 0.39 → 0.34, suggesting the model re-allocates capacity away from the smallest class when navarrese expands.

4. **Audio+text fusion* (partial)** — on the Ahotsak subset where transcriptions exist, concatenating fastText 5-class logits with Whisper embeddings improved macro F1 from 0.296 → 0.618 (+32pp). The text model captures lexical markers (erran/esan, bertze/beste) that the audio model misses. This is the most promising direction but is currently blocked by missing Mintzoak transcriptions.

### What didn't work

1. **Attention pooling** — learning frame-level attention weights over 8 temporal segments hurt overall performance (−4.3pp macro F1). The attention mechanism collapsed toward nav-lab-like segments, destroying western discrimination (−18.8pp). Fixed segment boundaries are too coarse; the model can't learn which specific time steps are dialect-bearing per class.

2. **Focal loss + balanced data** — redundant when subsampling already handles imbalance. Label smoothing, mixup augmentation, SGD optimizer, and hidden_dim=1024 all performed worse than the simple mean_std_max + 768-dim + CE setup.

3. **Unbalanced training** — the model defaults to nav-lab (82% F1) while navarrese gets 10% and central 35%. Without subsampling, the imbalance dominates everything else.

### Remaining bottlenecks

1. **Navarrese (37.6% F1)** — still the hardest non-lab dialect. Phonetically intermediate between central and nav-lab, with only 9K training samples (smallest after subsampling). Augmentation helped (+5.5pp) but more is needed — possibly targeted data collection or ASR-based text features.

2. **Central (34.2% F1)** — largest geographic spread (187 towns, diverse sub-varieties) and inconsistent recording quality. Dropped −4.9pp after navarrese augmentation (model de-prioritized the smallest training class). Fusion with text features would directly help here since central has strong lexical markers.

3. **Mintzoak transcriptions** — the 160K Iparralde segments (89K nav-lab, 10K souletin) have no text. Running Whisper ASR on them (est. 2-3h on L40) would unlock the full fusion pipeline and is the single highest-impact next step.

### Next step

Run Whisper ASR on Mintzoak segments → retrain audio+text fusion on the full merged dataset. Expected macro F1: 0.65-0.70. This would make knowledge distillation into a WASM-deployable student model worth pursuing.
