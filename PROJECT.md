# Zeineuski: Basque Dialect Identification

## Project Title

**Zeineuski** — Fine-Grained Dialect Identification for Basque (Euskara)

---

## Executive Summary

Zeineuski is a two-phase machine learning project to build the first open-source dialect identification (DID) system for Basque. Phase 1 targets **text-based classification** across 6 dialect categories (5 regional euskalkiak + Batua), leveraging data-efficient methods like UniLID and fine-tuned Latxa LLM. Phase 2 extends this to **speech**, using pre-trained SSL models (Whisper/XLSR) fine-tuned on Ahotsak.eus and Mintzoak.eus oral archives. The project addresses a gap: no existing model can distinguish Basque dialects automatically, and existing LID systems perform poorly on closely related varieties. Key differentiator: multi-label support to handle the dialect continuum.

---

## Objective

Build and release a system that, given a Basque text or speech sample, predicts which dialect(s) it belongs to, using between 3 classes (Western, Central, Navarrese-Lapurdian) and 6 classes (+ Navarrese, Souletin, Batua) depending on data availability. The primary metric is >75% accuracy (text) and >65% accuracy (speech) on balanced test sets.

---

## Target Users / Stakeholders

| User/Stakeholder | Need |
|---|---|
| **Basque linguists & dialectologists** | Automated classification of large corpora; dialect distribution analysis. |
| **Cultural heritage orgs** (Badihardugu, EKE) | Enrichment of oral archives with dialect metadata. |
| **NLP researchers** (HiTZ/Ixa, Elhuyar) | A fine-grained LID component for downstream Basque NLP pipelines (ASR, MT). |
| **Basque media & content platforms** | Content filtering/tagging by dialect, improved accessibility. |
| **Open-source community** | Reusable models and datasets for low-resource dialect NLP. |

---

## Scope

### In Scope

- Text dialect classification: 6 classes (Zuazo's 5 + Batua), fallback to 3 classes if data insufficient.
- Speech dialect classification: same target classes, initially 3-class grouping.
- Data collection & curation: assembling, cleaning, and labeling text and speech corpora from publicly available sources. Labeling strategy combines multiple approaches: explicit metadata, geo-proxy (author origin, interview/recording location → municipality → dialect), lexical markers, and human annotation for high-quality evaluation sets.
- Training pipeline: reproducible, config-driven, supporting multiple architectures.
- Evaluation framework: accuracy, macro F1, confusion matrix; held-out test sets.
- Model release: pretrained weights on Hugging Face Hub under permissive license.
- Data augmentation for low-resource dialects (orthographic noise, LLM-based style transfer, voice conversion).

### Out of Scope

- Dialect-to-Batua translation (separate project).
- Dialect strength / level-of-dialectness scoring.
- Production API deployment (inference scripts only).
- Mobile/edge inference optimization.
- Video or multimodal data.
- Sub-dialect classification (e.g., Goierri vs Beterri within Gipuzkera).

---

## Proposed Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| **Language** | Python 3.11+ | Ecosystem for NLP/ML; Hugging Face, PyTorch, fastText support. |
| **Package manager** | `uv` (Astral) | Fast, modern Python packaging; project bootstrap via `uv init`. |
| **Deep learning** | PyTorch 2.x + Hugging Face Transformers | Standard stack; supports Latxa, XLM-R, Whisper, XLSR. |
| **Text baseline** | fastText + UniLID | fastText is the strongest classical LID baseline at character n-gram level; UniLID is the SOTA for data-efficient fine-grained DID. |
| **Text fine-tuned** | Latxa (7B) via LoRA/QLoRA | State-of-the-art Basque LLM from HiTZ; LoRA keeps training feasible on 24 GB GPU. |
| **Speech fine-tuned** | Whisper (medium/large-v3) + XLSR | Whisper supports Basque; XLSR is SSL-pretrained multilingual wav2vec 2.0. CTC-DID architecture as alternative path. |
| **Parameter efficiency** | LoRA / QLoRA / MEFT | Required for training 7B+ models on consumer GPU. Memory-Efficient Fine-Tuning reduces VRAM up to 73%. |
| **Experiment tracking** | MLflow or Weights & Biases (free tier) | Reproduce all runs; hyperparameter search. |
| **Version control** | Git + DVC | Code + data versioning; DVC for large audio files. |
| **Evaluation** | scikit-learn, seqeval | Standard classification metrics; multi-label support. |
| **Data augmentation** | NLPAug, custom orthographic perturbations | CHALIS-inspired noise; LLM-based dialectal style transfer via Latxa. |
| **Audio processing** | librosa, torchaudio, ffmpeg | Resampling, VAD, feature extraction. |
| **Model hosting** | Hugging Face Hub | Release pretrained weights, tokenizers, and dataset cards. |

**Bootstrapping:** `uv init zeineuski`, then `uv add torch transformers datasets fasttext evaluate scikit-learn librosa soundfile jiwer wandb`. DVC added via `uv add --dev dvc`.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│                                                                  │
│  Text Corpora              Speech Corpora                        │
│  ┌──────────────┐         ┌──────────────────────┐              │
│  │ XNLI dialectal│         │ Ahotsak.eus (7k+)    │              │
│  │ splits ✓      │         │ Mintzoak.eus (1.2k+),│              │
│  │ BasPhyCowest  │         │ Common Voice (eu),   │              │
│  │ Social media  │         │ Parliament (1.4k h)  │              │
│  └──────────────┘         └──────────────────────┘              │
│          │                          │                            │
│          ▼                          ▼                            │
│  ┌──────────────┐         ┌──────────────────────┐              │
│  │ Preprocessing│         │ Audio preprocessing   │              │
│  │ - cleaning   │         │ - 16 kHz resample     │              │
│  │ - dedup      │         │ - VAD / chunking      │              │
│  │ - aug        │         │ - augment (VC, noise) │              │
│  └──────────────┘         └──────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL LAYER                                 │
│                                                                  │
│  Phase 1 (Text)                 Phase 2 (Speech)                 │
│  ┌────────────────┐            ┌───────────────────┐            │
│  │ fastText        │            │ ECAPA-TDNN         │            │
│  │ (char n-gram    │            │ (speaker embed.    │            │
│  │  baseline)      │            │  baseline)         │            │
│  ├────────────────┤            ├───────────────────┤            │
│  │ UniLID          │            │ Whisper fine-tuned │            │
│  │ (tokenizer-     │            │ (encoder + classif.│            │
│  │  based DID)     │            │  head)             │            │
│  ├────────────────┤            ├───────────────────┤            │
│  │ Latxa + LoRA    │            │ XLSR + LoRA        │            │
│  │ (Basque LLM     │            │ (wav2vec 2.0       │            │
│  │  fine-tuned)    │            │  fine-tuned)       │            │
│  └────────────────┘            ├───────────────────┤            │
│                                │ CTC-DID             │            │
│                                │ (streaming-capable) │            │
│                                └───────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION LAYER                              │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Metrics: accuracy, macro F1, per-class P/R, confusion     │   │
│  │ Stratified k-fold CV; speaker-disjoint splits (speech)    │   │
│  │ Test sets: XNLI dialectal (text), held-out Ahotsak (sp.)  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                                  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Inference scripts (CLI + Python API)                      │   │
│  │ Hugging Face Model Hub release                            │   │
│  │ Dataset cards + datasheets                                │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Text pipeline uses three-tier architecture**: fastText (classical baseline), UniLID (tokenizer-based SOTA for few-shot DID), Latxa+LoRA (deep fine-tuned for maximum accuracy). This tiered approach mirrors the NADI shared task strategy and provides graceful degradation if compute is limited.

2. **Speech pipeline uses two encoders**: Whisper (transformer-based, strong multilingual representations; supports Basque) and XLSR (wav2vec 2.0-based, proven for Frisian dialect task). CTC-DID is the streaming-capable alternative.

3. **Multi-label via thresholding**: The model outputs per-class probabilities; threshold is tuned on validation data using F1 optimization. Not a hard classification boundary.

4. **Data scarcity mitigation**: LLM-based style transfer (Latxa: Batua → dialect) for text augmentation; voice conversion for speech augmentation (validated for Arabic ADI).

---

## Milestones / Phases

### Phase 0 — Foundation ✅ (2026-06-06)
| Task | Owner | Deliverable | Status |
|---|---|---|---|
| Set up repo, env (`uv`), CI, experiment tracking | AI | Git repo with `pyproject.toml`, README | ✅ |
| Inventory and download all known text resources | AI | Dataset registry with sizes, dialects, licenses | ✅ |
| Build municipality→dialect mapping table | AI | `data/reference/municipality_dialect.csv` (101 towns) | ✅ |
| Scrape Klasikoak.armiarma.eus for dialect data | AI | 24,924 labeled sentences across 5 dialects | ✅ |

### Phase 1 — Text DID Baseline (2026-06-06 to 2026-06-07)
| Task | Owner | Deliverable | Status |
|---|---|---|---|
| Train fastText baseline (char n-grams 3–6) | AI | fastText model + evaluation report | ✅ |
| Train fastText on XNLI only (835/dialect, 3-class) | AI | 93.6% accuracy on XNLI test | ✅ |
| Train fastText on Klasikoak only (19.9k sents, 5-class) | AI | 98.2% val acc, 33.9% cross-domain | ✅ |
| Train fastText hybrid (XNLI + Klasikoak, 5-class) | AI | **97.8% val, 96.0% XNLI test** | ✅ |
| Implement XLM-R fine-tuning (encoder + classifier) | AI | XLM-R DID model (87.8% XNLI test) | ✅ |
| Set up GPU server (NVIDIA L40, 48 GB VRAM) | AI | SSH at 10.2.121.210, /opt/zeineuski/ | ✅ |

### Phase 1 — Text DID Advanced (Weeks 6–8)
| Task | Owner | Deliverable |
|---|---|---|
| Fine-tune Latxa 7B with QLoRA on dialect data | TBD | Latxa-dialect LoRA adapter |
| Implement multi-label output (threshold tuning) | TBD | Multi-label evaluation pipeline |
| Data augmentation: Latxa-based Batua→dialect transfer | TBD | Augmented dataset + ablation study |
| Final Phase 1 model; publish on Hugging Face | TBD | `hitz-zentroa/zeineuski-text-v1` |

### Phase 2 — Speech DID Baseline (Weeks 9–11)
| Task | Owner | Deliverable |
|---|---|---|
| Inventory and preprocess speech corpora | TBD | Audio dataset registry; chunked/transcribed files |
| Train ECAPA-TDNN baseline | TBD | ECAPA-TDNN model + evaluation |
| Fine-tune Whisper medium/large-v3 | TBD | Whisper-DID model |
| Fine-tune XLSR + classification head | TBD | XLSR-DID model |
| Evaluate all speech baselines | TBD | Comparison report |

### Phase 2 — Speech DID Advanced (Weeks 12–14)
| Task | Owner | Deliverable |
|---|---|---|
| Implement CTC-DID for streaming scenario | TBD | CTC-DID model |
| Voice conversion augmentation for low-resource dialects | TBD | Augmented speech dataset + ablation |
| Multi-task training: ASR + DID (Frisian approach) | TBD | Jointly trained model |
| Final Phase 2 model; publish on Hugging Face | TBD | `hitz-zentroa/zeineuski-speech-v1` |

### Phase 3 — Polish & Release (Weeks 15–16)
| Task | Owner | Deliverable |
|---|---|---|
| Unified inference CLI (`zeineuski predict --text/--speech`) | TBD | `src/cli.py` |
| Final evaluation across all models | TBD | Leaderboard-style report |
| Dataset cards, model cards, datasheets | TBD | Hugging Face documentation |
| Write-up / paper draft | TBD | arXiv preprint or workshop submission |

---

## Deliverables

| # | Deliverable | Format | Target |
|---|---|---|---|
| D1 | Text dialect identification model (fastText baseline) | `.bin` + Hugging Face | Week 3 |
| D2 | Text dialect identification model (UniLID) | Hugging Face model repo | Week 4 |
| D3 | Text dialect identification model (Latxa + LoRA adapter) | Hugging Face adapter | Week 7 |
| D4 | Curated Basque dialect text dataset | Hugging Face dataset | Week 8 |
| D5 | Speech dialect identification model (Whisper fine-tuned) | Hugging Face model repo | Week 10 |
| D6 | Speech dialect identification model (XLSR fine-tuned) | Hugging Face model repo | Week 11 |
| D7 | Curated Basque dialect speech dataset | Hugging Face dataset | Week 12 |
| D8 | Unified inference script (CLI + Python API) | `src/cli.py` | Week 15 |
| D9 | Evaluation report (all models, all metrics) | Markdown/LaTeX | Week 16 |
| D10 | Dataset and model cards | Hugging Face Hub | Week 16 |

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Insufficient labeled data for Zuberera and Navarrese** | High | High | 3-class fallback (Western/Central/Navarrese-Lapurdian); LLM-based style transfer augmentation; prioritize data collection for these dialects. Mintzoak.eus partially covers Northern dialects. |
| **Latxa 7B too large for 24 GB GPU even with QLoRA** | Medium | Medium | Use Latxa 1B variant if available; fall back to XLM-R encoder-only fine-tuning; explore 4-bit quantization (QLoRA). |
| **Ahotsak.eus audio quality degrades performance** | High | Medium | Apply denoising preprocessing; use Common Voice + Parliament as cleaner baseline; Whisper is robust to moderate noise. |
| **Licensing blocks use of Ahotsak.eus data** | Medium | High | Engage Badihardugu Association early; fall back to Mozilla Common Voice + Parliament + Mintzoak for fully open-licensed pipeline. |
| **Dialect continuum makes hard classification boundaries artificial** | Medium | Low | Multi-label output is built from the start; confusion between neighboring dialects is expected and acceptable. |
| **No native speaker annotators available for evaluation** | Medium | High | Partner with HiTZ/Ixa or UPV/EHU linguistics department; use existing XNLI dialectal splits as gold standard for 3-class evaluation. |
| **Whisper not supporting Basque well enough** | Low | Medium | Whisper large-v3 supports Basque; XLSR is the fallback (proven for Basque in prior work). Verify Basque WER on Common Voice before committing. |

---

## Success Metrics

| Metric | Phase 1 Target (Text) | Phase 2 Target (Speech) | Measurement Method |
|---|---|---|---|
| Accuracy (single-label, 3-class) | > 85% | > 75% | Held-out test set (XNLI dialectal for text; Ahotsak for speech) |
| Accuracy (single-label, 6-class) | > 75% | > 65% | Same, with full 6-class annotation |
| Macro F1 (multi-label, 3-class) | > 0.60 | N/A (v2) | Threshold-tuned on validation |
| Outperforms baseline | > fastText char n-grams | > ECAPA-TDNN | Direct comparison on same test split |
| Training time | < 8 hours on 1× RTX 4090 | < 48 hours on 1× RTX 4090 | Wall-clock measurement |
| Inference latency | < 100 ms per sentence (CPU) | < real-time on GPU | Benchmark script |
| Data efficiency | Works with ≤ 500 samples/dialect | Works with ≤ 100 utterances/dialect | Ablation: subsample training data |
| Reproducibility | All runs logged with seed/hparams | All runs logged with seed/hparams | MLflow/WandB dashboard |

---

## Key Decisions Recorded

1. **Dialect taxonomy:** Zuazo's 5-dialect model + Batua (6 classes). 3-class grouping (Western, Central, Navarrese-Lapurdian) as fallback. Sub-dialects excluded from v1.
2. **Basque BERT replaced by Latxa:** The project will use Latxa (HiTZ) as the Basque-specific LM backbone, not an older BERT variant. Latxa is based on Llama and benefits from continuous pre-training on EusCrawl.
3. **Multi-label from the start:** The model will output per-class probabilities with a tuned threshold, not just argmax. Full multi-label ground truth annotation deferred to v2.
4. **No orthographic normalization at inference:** Following CHALIS findings, the model will be trained to handle variation, not normalize it away. Training augmentation will include diacritic stripping and informal spelling variants.
5. **Phase 2 speech data strategy:** Ahotsak.eus and Mintzoak.eus are the primary dialect-labeled speech sources. Mozilla Common Voice and Parliament corpus provide cleaner Batua baselines. Voice conversion augmentation is budgeted as a risk mitigation for Zuberera.
6. **Package management:** `uv` by Astral for dependency and environment management (modern, fast, lockfile-based). Replaces pip/poetry.
7. **Model hosting:** All released models go to Hugging Face Hub under the `hitz-zentroa` organization (or equivalent, pending permission) or a new `zeineuski` organization.
8. **Labeling strategy: quality over quantity.** The primary data labeling philosophy is to prefer few high-confidence dialect labels over many noisy ones. A layered geo-proxy approach is used: (a) municipality-to-dialect mapping table as the foundational artifact; (b) author-origin inference for classical Basque literature (Klasikoak.eus, Axular, Mogel, etc.); (c) recording/interview location for oral archive transcriptions (Ahotsak.eus, Mintzoak.eus); (d) social media user location; (e) lexical markers as a secondary signal. Only `high` and `medium` confidence labels are used for training; `low`-confidence samples are stored but excluded. Human annotation is reserved for a small, carefully curated evaluation set.
9. **XNLI dialectal data obtained from hitz-zentroa/Catalog-of-Basque-Dialects:** The repo at [github.com/hitz-zentroa/Catalog-of-Basque-Dialects](https://github.com/hitz-zentroa/Catalog-of-Basque-Dialects) contains the full XNLIvar parallel dataset: 5,010 test sentences and 621 native sentences translated into 3 dialects (Western, Central, Navarrese-Lapurdian) by native speakers. This is a gold-standard parallel evaluation set. Downloaded and stored at `data/raw/text/xnli_dialectal/`.
10. **ikerHerrero/Basque_Dialects_Classification as prior art:** A RoBERTa-based model (ixa-ehu/roberta-eus-cc100-base-cased) fine-tuned for 5-dialect classification exists on Hugging Face. F1=0.6846 on evaluation. The model is available but the training dataset is not published. This serves as a baseline comparison point and validates that the 5-dialect classification problem is tractable with existing Basque NLP models.
11. **Klasikoak.armiarma.eus is a viable dialect-labeled data source:** The site hosts 467 classical Basque literary works from pre-Batua authors. Author birthplaces are extracted from Literaturaren Zubitegia (`zubi/egileak/` pages, `jaioHil4` class) and mapped to dialect via the municipality table. 66+ authors have birthplaces with known dialect. The scraper is at `src/data/klasikoak_scraper.py`.
12. **fastText character n-grams are extremely effective for Basque dialect classification:** The char-level n-gram model captures orthographic, morphological, and lexical dialect markers directly from surface form (e.g., -gaz sociative for Western, x- word-initial for Nav-Lab). With the hybrid dataset (Klasikoak + XNLI train, 18k sentences), fastText achieves 97.8% on in-domain validation and 96.0% cross-domain on XNLI test — significantly outperforming XLM-R (87.8%) on the same test set.
13. **Classical literature vs. modern text requires domain mixing:** A model trained only on Klasikoak (16th-19th century literature) collapses to Nav-Lab predictions on modern XNLI sentences (33.9% cross-domain). Mixing Klasikoak with XNLI train data (hybrid training) resolves this: 96.0% XNLI accuracy while maintaining 97.8% in-domain.
14. **GPU server details:** NVIDIA L40 with 48 GB VRAM at `10.2.121.210`, Python 3.11.15 via uv, PyTorch 2.11.0+cu128, transformers 5.10.2. Project synced to `/opt/zeineuski/`. Previously running llama.cpp (Gemma 4 12B), stopped to free GPU for training.
15. **transformers v5 API changes:** `evaluation_strategy` → `eval_strategy` kwarg, Trainer no longer accepts `tokenizer=` kwarg. These were fixed in `src/models/text/train_xlmr.py`.
16. **fastText numpy 2.0 bug:** `predict()` method crashes with `ValueError: Unable to avoid copy while creating an array`. The `test()` and `test_label()` evaluation methods work correctly. This is a known fastText issue with NumPy 2.x.

## Execution Results Log

### 2026-06-06: MVP fastText on XNLI dialectal
- **Data:** XNLI test set split 50/50 → 835/dialect train, 835/dialect test (3 classes: Western, Central, Nav-Lab)
- **Config:** char ngrams 3–6, word ngrams 2, dim=100, lr=0.1, 25 epochs
- **Results:** Accuracy 93.6%, Macro F1 0.936
  - Western: P=0.946, R=0.978, F1=0.962
  - Central: P=0.938, R=0.879, F1=0.907
  - Nav-Lab: P=0.924, R=0.953, F1=0.938
- **Model:** `models/fasttext_dialect.bin` (769 MB), quantized `models/fasttext_dialect.ftz` (99 MB, 93.1% acc)

### 2026-06-06: XLM-R baseline on GPU server
- **Setup:** `FacebookAI/xlm-roberta-base`, batch=32, lr=5e-6, 10 epochs, early stopping patience=4
- **Results:** Accuracy 87.8%, Macro F1 0.878
  - Western: F1=0.910, Central: F1=0.843, Nav-Lab: F1=0.882
- **Training time:** ~108s on NVIDIA L40 (48 GB)
- **Conclusion:** fastText (93.6%) significantly outperforms XLM-R (87.8%) on this small dataset

### 2026-06-07: Klasikoak.armiarma.eus scraping
- **Scraper:** `src/data/klasikoak_scraper.py` — parses alfa.htm, extracts Zubitegia author profiles, maps birthplace → dialect
- **Results:** 24,924 labeled sentences across 5 dialects from 66+ authors
  - Nav-Lab: 13,105 (52.6%, 22 authors)
  - Central: 4,006 (16.1%, 13 authors)
  - Western: 3,887 (15.6%, 12 authors)
  - Navarrese: 2,446 (9.8%, 3 authors)
  - Souletin: 1,480 (5.9%, 6 authors)
- **Output:** `data/raw/text/klasikoak/klasikoak_labeled.tsv`

### 2026-06-07: Expanded fastText — Klasikoak-only (5-class)
- **Data:** 19,939 train / 4,985 val (80/20 split, stratified)
- **Results:** Val accuracy 98.2%, Macro F1 0.980
  - Western: F1=0.977, Central: F1=0.967, Navarrese: F1=0.997, Nav-Lab: F1=0.985, Souletin: F1=0.972
- **Cross-domain:** Only 33.9% on XNLI test (trained on classical literature, tested on modern translated sentences)

### 2026-06-07: Hybrid fastText (Klasikoak + XNLI, 5-class)
- **Data:** 17,955 train / 4,489 val (combined Klasikoak + XNLI train, stratified split)
- **Results:**
  - In-domain (Klasikoak val): Accuracy 97.8%, Macro F1 0.976
  - Cross-domain (XNLI test, 3-class remap): Accuracy **96.0%**
    - Western: P=0.972, R=0.970, F1=0.971
    - Central: P=0.965, R=0.935, F1=0.950
    - Nav-Lab: P=0.944, R=0.976, F1=0.960
- **Model:** `models/fasttext_dialect_hybrid.bin`
- **Best model so far.** Beats XNLI-only (93.6%) and supports 5-class output.

### Current Best Results Summary
| Model | Train Data | XNLI Test Acc | Klasikoak Val Acc | Classes |
|---|---|---|---|---|
| fastText (XNLI-only) | 2,505 | 93.6% | — | 3 |
| XLM-R (XNLI-only) | 2,505 | 87.8% | — | 3 |
| fastText (Klasikoak-only) | 19,939 | 33.9% | 98.2% | 5 |
| **fastText (Hybrid)** | **17,955** | **96.0%** | **97.8%** | **5** |
