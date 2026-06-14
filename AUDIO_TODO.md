# Audio Pipeline — Pending Improvements

Three strategies to further improve the Whisper encoder + MLP dialect classifier
(current best: macro F1 0.519, 5-class, merged Ahotsak+Mintzoak, 10K balanced).

## 1. Attention Pooling over Time Dimension

**Expected impact:** High. Proven in ADI-20 Arabic dialect paper (arxiv 2511.10070).

Instead of mean+std+max pooling (3840-dim static vector), use a learnable
attention layer that weights each encoder time step by its dialect relevance.
This lets the model focus on dialect-bearing temporal segments (stressed
syllables, vowel length transitions, specific phoneme realizations) rather than
averaging the entire utterance.

**Why it helps navarrese/central:** These phonetically intermediate dialects
differ from nav-lab/western in specific time-localized features (e.g., vowel
nasalization, intonation contours). Attention can zoom in on those segments
while mean pooling dilutes them across silence and shared phonemes.

**Task:**
- [x] Modify `WhisperEncoder.extract()` to return frame-level embeddings (seq_len × 1280) instead of mean-pooled
- [x] Implement `AttentionPooling(nn.Module)` with a 2-layer Q/K/V or simple weighted sum
- [x] Re-extract all 197K embeddings (~2h on L40, done)
- [x] Retrain MLP on attention-pooled representations
- [x] Tune attention hidden dim and number of heads

**Result:** ❌ Negative. Macro F1 0.4765 (vs 0.5193 baseline).
Attention pooling with 8 fixed temporal segments hurts overall performance:
  - Nav-lab +3.2pp (0.82 → 0.85) but western −18.8pp (0.67 → 0.48)
  - Attention learns to focus on nav-lab-like segments, collapsing western discrimination
  - Fixed 8-segment boundaries are too coarse; the model can't learn *which* segments
    are dialect-bearing for each class separately

**Next idea:** try multi-head attention with learned segment boundaries, or
pooling over variable-length segments based on energy/VAD. Or accept that
mean_std_max (3840-dim) already captures sufficient temporal statistics.

**Tradeoff:** Re-extraction cost (~2h). Training time unchanged (~90s).

---

## 2. Audio + Text Model Fusion

**Expected impact:** Medium-High. Combines complementary signal types.

The fastText text model achieves 82.5% weighted F1 on 12-class azpieuskalki
using lexical-morphological features (dialectal word choice, case endings, verb
forms). The audio model captures phonetic features (pronunciation, prosody).
Humans use both to ID dialects — the models should too.

**Approach:** Late-fusion MLP that concatenates audio logits (or embeddings) from
the Whisper pipeline with text logits from the fastText model, then trains a
small fusion layer.

**Why it helps:** Navarrese shares phonetics with nav-lab but has distinct
lexical markers (erran vs esan, bertze vs beste, guti vs gutxi). The text model
can separate these while the audio model handles pronunciation differences.

**Task:**
- [ ] Transcribe all test audio via Whisper ASR (or use pre-transcribed data)
- [ ] Run fastText text model on transcriptions → text logits/embeddings
- [ ] Implement `AudioTextFusion(nn.Module)` with concatenation + 1-2 MLP layers
- [ ] Train on combined (audio_embedding, text_logits) pairs
- [ ] Tune fusion weight (learned or fixed)

**Tradeoff:** ASR bottleneck — Whisper decoder accuracy is ~42% on dialectal
speech (batua normalization strips dialect markers). Can mitigate by using
pre-existing Ahotsak/Mintzoak transcriptions where available, or accepting
noisy ASR as a soft signal.

---

## 3. Navarrese Data Augmentation

**Expected impact:** Low-Medium. Directly targets the bottleneck class.

Navarrese has only 9,072 train segments — even with balanced 10K subsampling,
it's the hardest class (macro F1 0.32). Augmentation can double or triple this
by creating synthetic variants of existing samples.

**Techniques:**
- **SpecAugment:** Time masking (mask 10-15% of time steps), frequency masking (mask 10-15% of mel bins)
- **Pitch shift:** ±50 cents (preserves dialect identity, varies speaker range)
- **Speed perturbation:** 0.9× and 1.1× (adds tempo variation)
- **Background noise:** Mix in 5-10dB of random Ahotsak/Mintzoak background audio

All augmentations use `torchaudio` transforms applied to the raw waveform before
re-extraction.

**Task:**
- [ ] Implement augmentation pipeline in `src/models/speech/whisper_did.py`
- [ ] Augment navarrese samples 2× (9K → 18K), optionally central 1.5× (5K → 7.5K)
- [ ] Re-extract embeddings for augmented samples (est. ~30 min on L40)
- [ ] Retrain MLP with augmented + balanced dataset
- [ ] Verify dialect signal isn't destroyed by aggressive augmentation (validate F1 on unaugmented test)

**Tradeoff:** Re-extraction for ~15K new samples (~30 min). Risk of destroying
subtle dialectal phonetic features with too-aggressive augmentation — need
ablation on augmentation strength.

---

## Decision Framework

| Strategy | Central gain | Navarrese gain | Effort | Risk |
|---|---|---|---|---|
| Attention pooling | Medium (+3-5pp) | Medium (+3-5pp) | High (7h re-extract) | Low |
| Audio+text fusion | Low (+1-2pp) | Medium (+3-5pp) | Medium (ASR + code) | Medium (ASR noise) |
| Navarrese augmentation | None | Medium (+3-5pp) | Low (30min re-extract) | Low-Medium |

**Recommended order:** Attention pooling → Navarrese augmentation → Audio+text fusion.
Start with the highest-impact, lowest-risk option. Augmentation is a quick win
for navarrese while attention pooling runs. Fusion is the long-term play.
