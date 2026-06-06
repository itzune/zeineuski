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
│  │ splits        │         │ Mintzoak.eus (1.2k+),│              │
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

### Phase 0 — Foundation (Weeks 1–2)
| Task | Owner | Deliverable |
|---|---|---|
| Set up repo, env (`uv`), CI, experiment tracking | TBD | Git repo with `pyproject.toml`, README |
| Inventory and download all known text resources | TBD | Dataset registry with sizes, dialects, licenses |
| Build text preprocessing pipeline (clean, dedup, format) | TBD | `src/data/text_preprocessing.py` |
| Annotate 200 sentences per dialect for dev/test | TBD | `data/annotated/dev.csv`, `data/annotated/test.csv` |

### Phase 1 — Text DID Baseline (Weeks 3–5)
| Task | Owner | Deliverable |
|---|---|---|
| Train fastText baseline (char n-grams 3–6) | TBD | fastText model + evaluation report |
| Implement and train UniLID | TBD | UniLID model (Hugging Face-compatible) |
| Implement XLM-R fine-tuning (encoder + classifier) | TBD | XLM-R DID model |
| Evaluate all baselines; select best architecture | TBD | Comparison report (accuracy, F1, confusion matrix) |

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
