# Zeineuski: Basque Dialect Identification

## Original Intent

Build and train a model capable of distinguishing between different Basque dialects (euskalkiak) — starting with text, then extending to speech.

---

## Refined Prompt

Design and implement a **fine-grained dialect identification (DID) system for Basque (Euskara)** that classifies input text or speech into one of the major Basque dialect groups (Bizkaiera, Gipuzkera, Nafarrera, Lapurtera, Zuberera, and standard Batua), with the possibility of multi-label classification where a sample may belong to more than one dialect.

The project is split into two phases:

### Phase 1 — Text-Based Dialect Identification
Build a system that, given a written Basque sentence or short paragraph, predicts which dialect(s) it belongs to. The system must handle:
- Closely related varieties with significant lexical, morphological, and orthographic overlap.
- Informal, non-standardized spelling (social media posts, user-generated content).
- Short inputs (single sentences, not long documents).
- The possibility that a given text is valid in multiple dialects.

### Phase 2 — Speech-Based Dialect Identification
Extend the system to spoken Basque. Given an audio utterance, predict the speaker's dialect. The system must handle:
- Speaker variability (age, gender, recording conditions).
- Short utterances (a few seconds).
- Streaming / real-time scenarios as a stretch goal.

### Scientific framing
This is a **Dialect Identification (DID)** problem, a subfield of Language Identification (LID), specifically the **fine-grained, closely-related variety** case. It is analogous to work done on:
- Arabic dialect identification (NADI shared tasks)
- Frisian dialectal ASR
- Chinese dialect/subdialect identification
- Cousin-language identification (Czech/Slovak, Spanish/Catalan)

---

## Scope

| Dimension          | In Scope                                                            | Out of Scope (for now)                                                   |
| ------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Languages          | Basque dialects (Bizkaiera, Gipuzkera, Nafarrera, Lapurtera, Zuberera, Batua) | Other languages or unrelated dialect families.                           |
| Modalities         | Text (Phase 1), Speech (Phase 2)                                    | Multimodal fusion, video, prosody-only.                                  |
| Output             | Single-label or multi-label dialect classification per sample.      | Dialect strength scoring (level of dialectness), dialect-to-Batua translation. |
| Deployment         | Offline inference; streaming optional for Phase 2.                  | Production-scale serving, mobile inference (unless trivially achievable). |
| Data collection    | Gathering, cleaning, and labeling existing Basque corpora.          | Creating new spoken corpora from scratch (synthetic data augmentation is in scope). |

---

## Functional Requirements

### FR1 — Dialect Classification (Text)
- Accept raw Basque text input (sentence or paragraph).
- Return a predicted dialect label (primary), optionally with confidence scores per dialect.
- Support both single-label (argmax) and multi-label (thresholded) output modes.

### FR2 — Dialect Classification (Speech)
- Accept audio input (WAV/MP3, 16 kHz mono recommended).
- Return a predicted dialect label with confidence scores.
- Handle utterances as short as 2–5 seconds.

### FR3 — Model Training Pipeline
- Scripted, reproducible training pipeline (Python, config-driven).
- Support for multiple model architectures: traditional (fastText + n-grams), transformer-based (fine-tuned multilingual BERT / XLM-R / Basque BERT), and tokenizer-based (UniLID-style).
- For speech: support fine-tuning of pre-trained SSL models (XLSR, Whisper) with optional LoRA/QLoRA.

### FR4 — Evaluation Framework
- Metrics: accuracy, macro F1, per-class precision/recall, confusion matrix.
- Stratified train/val/test split, respecting dialect distribution and speaker non-overlap (for speech).
- Baseline comparison: fastText char n-grams for text, ECAPA-TDNN for speech.

### FR5 — Data Handling
- Support for multiple input formats: CSV, JSON, TXT with dialect labels.
- Handling of dialect-labeled and unlabeled data (semi-supervised / self-training stretch goal).
- Data augmentation: orthographic perturbation (CHALIS-inspired), back-translation, voice conversion (for speech).

---

## Non-Functional Requirements

### NFR1 — Resource Efficiency
- Must be trainable on a single consumer GPU (≤24 GB VRAM) or free-tier cloud GPU.
- Inference should run on CPU for text, optionally GPU for speech.
- Training time should not exceed a few hours for text, a few days for speech.

### NFR2 — Reproducibility
- All experiments versioned (random seeds, hyperparameters, data splits).
- Code and configuration under version control.
- Pretrained model weights and datasets documented and preserved.

### NFR3 — Modularity
- Text and speech pipelines separated but sharing evaluation tooling.
- Model architecture pluggable (swap classifiers without rewriting data pipeline).

### NFR4 — Documentation
- README with setup instructions, data format specification, and usage examples.
- Inline documentation for key functions.
- A datasheet or data card documenting the corpus used (provenance, dialects, size, collection method).

---

## Constraints & Assumptions

### Constraints
1. **No large existing labeled Basque dialect corpus.** Data must be assembled from public sources (local media, parliamentary records, social media, Euskaltzaindia/Elhuyar resources) and manually or heuristically labeled.
2. **Limited compute budget.** Consumer-grade hardware or free-tier cloud GPU only.
3. **No pre-trained Basque dialect model exists.** All models are built from scratch or fine-tuned from general-purpose multilingual models.
4. **Basque dialects form a continuum**, not discrete categories — some sentences may genuinely belong to multiple dialects.

### Assumptions Made
1. **Dialect labels are available or inferrable.** _Assumption_: we can obtain region-labeled text from sources like local Basque newspapers (Berria → Gipuzkera, Goiena → Gipuzkera Bizkaiera mix, etc.), parliamentary transcriptions, or existing academic corpora.
2. **Basque Batua is treated as a separate "dialect" class.** _Assumption_: Standard Basque is the target for mixed/neutral text, and distinguishing it from vernacular dialects is valuable.
3. **Target dialect set is the five main euskalkiak + Batua.** _Assumption_: Zuazoa's classification (Bizkaiera, Gipuzkera, Nafarrera, Lapurtera — Nafarrera-Lapurtera, Zuberera) is used. Sub-dialects (e.g., Goierri vs Beterri within Gipuzkera) are out of initial scope.
4. **Multi-label framing is desirable but not mandatory for v1.** _Assumption_: Single-label classification is the baseline; multi-label is explored if data allows.
5. **Speech data is even scarcer than text.** _Assumption_: Phase 2 may require synthesizing speech from dialect-labeled text or using voice conversion to augment data, following the Arabic ADI community's approach.

---

## Success Criteria

| Criterion                     | Target (Text)                          | Target (Speech)                    |
| ----------------------------- | -------------------------------------- | ---------------------------------- |
| Accuracy (single-label)       | > 75% on a balanced test set           | > 65% on a balanced test set       |
| Macro F1 (multi-label)        | > 0.55                                | N/A (stretch)                      |
| Training data efficiency       | Works with ≤ 500 labeled samples/dialect | Works with ≤ 100 utterances/dialect |
| Outperforms baseline           | Beats fastText char n-grams            | Beats ECAPA-TDNN                  |
| Confusion matrix              | Most errors between neighboring dialects (e.g., Gipuzkera ↔ Nafarrera), not random | Same pattern expected             |
| Deployment                    | Inference < 100ms per sentence on CPU  | Inference < real-time (1×) on GPU |

---

## Open Questions

1. **Data sourcing**: Which existing Basque corpora have dialect metadata? Are there already labeled datasets from UPV/EHU, Elhuyar, or IXA group? Is social media scraping (Twitter/X, Mastodon) viable for Basque dialect data?
2. **Dialect taxonomy**: Which classification scheme should be the target? Zuazoa's 5-dialect model? A finer sub-dialect level? Should Batua be a separate class or handled as "neutral"?
3. **Evaluation data**: How to obtain gold-standard test data? Could we recruit native speakers from each dialect region to annotate a held-out test set?
4. **Orthographic normalization**: Should input be normalized to a canonical spelling, or should the model learn to be robust to spelling variation (as CHALIS recommends)?
5. **Multi-label ground truth**: How to determine whether a sentence is truly valid in multiple dialects? Expert annotation needed.
6. **Speech data availability**: Is there any existing Basque dialect speech corpus? If not, what's the plan — record speakers, scrape regional radio, or synthesize?
7. **Licensing**: What are the licensing constraints on Basque media content used for training? Is CC-BY or similar available?

---

## Key References from arXiv (discovered in research)

1. **UniLID** (2026) — Tokenizer-based language ID, extremely data-efficient (5 samples/class), excels at fine-grained dialect ID. [arXiv:2602.17655](https://arxiv.org/abs/2602.17655)
2. **CHALIS** (2026) — Benchmark for language ID in difficult scenarios (cousin languages + orthographic noise). [arXiv:2606.06088](https://arxiv.org/abs/2606.06088)
3. **CTC-DID** (2026) — CTC-based dialect ID for speech, outperforms Whisper + ECAPA-TDNN on low-resource settings. [arXiv:2601.12199](https://arxiv.org/abs/2601.12199)
4. **Frisian Dialectal ASR** (2025) — Low-resource European language with dialects, SSL fine-tuning + auxiliary LID. [arXiv:2502.04883](https://arxiv.org/abs/2502.04883)
5. **Multi-Label Arabic DI** (2026) — Curriculum learning + pseudo-labeling for multi-label dialect ID. [arXiv:2602.12937](https://arxiv.org/abs/2602.12937)
6. **Memory-Efficient Fine-Tuning for DID** (2024) — LoRA/MEFT for speech dialect ID with reduced GPU requirements. [arXiv:2512.02074](https://arxiv.org/abs/2512.02074)

## Related Shared Tasks & Benchmarks

- **NADI** (Nuanced Arabic Dialect Identification) — Annual shared task since 2018, multi-label DID, dialect-to-MSA translation.
- **VarDial** (Workshop on NLP for Similar Languages, Varieties, and Dialects) — Covers low-resource and closely related language identification.
- **FLEURS** — Multilingual speech benchmark, includes some language variety distinctions.
