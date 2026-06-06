# Zeineuski: Basque Dialect Identification

## Original Intent

Build and train a model capable of distinguishing between different Basque dialects (euskalkiak) — starting with text, then extending to speech.

---

## Refined Prompt

Design and implement a **fine-grained dialect identification (DID) system for Basque (Euskara)** that classifies input text or speech into one of the major Basque dialect groups (Bizkaiera/Western, Gipuzkera/Central, Nafarrera/Navarrese, Nafarrera-Lapurtera/Navarrese-Labourdin, Zuberera/Souletin, and standard Batua), with the possibility of multi-label classification where a sample may belong to more than one dialect.

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
- Cousin-language identification (Czech/Slovak, Spanish/Catalan, Portuguese/Galician)

---

## Scope

| Dimension          | In Scope                                                            | Out of Scope (for now)                                                   |
| ------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Languages          | Basque dialects (Bizkaiera, Gipuzkera, Nafarrera, Nafarrera-Lapurtera, Zuberera, Batua) | Other languages or unrelated dialect families.                           |
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
- Support for multiple model architectures: traditional (fastText + n-grams), transformer-based (fine-tuned multilingual BERT / XLM-R / Latxa), and tokenizer-based (UniLID-style).
- For speech: support fine-tuning of pre-trained SSL models (XLSR, Whisper) with optional LoRA/QLoRA.

### FR4 — Evaluation Framework
- Metrics: accuracy, macro F1, per-class precision/recall, confusion matrix.
- Stratified train/val/test split, respecting dialect distribution and speaker non-overlap (for speech).
- Baseline comparison: fastText char n-grams for text, ECAPA-TDNN for speech.

### FR5 — Data Handling
- Support for multiple input formats: CSV, JSON, TXT with dialect labels.
- Handling of dialect-labeled and unlabeled data (semi-supervised / self-training stretch goal).
- Data augmentation: orthographic perturbation (CHALIS-inspired), LLM-based standard-to-dialect style transfer, voice conversion (for speech).

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
1. **No large existing labeled Basque dialect corpus.** Data must be assembled from public sources and manually or heuristically labeled. The dominant presence of Standard Basque (*Batua*) in most formal digital media (newspapers, parliament) is an additional limiting factor.
2. **Limited compute budget.** Consumer-grade hardware or free-tier cloud GPU only.
3. **No pre-trained Basque dialect model exists.** All models are built from scratch or fine-tuned from general-purpose multilingual models (e.g., XLM-R, mBERT) or Basque-specific LLMs (e.g., Latxa).
4. **Basque dialects form a continuum**, not discrete categories — some sentences may genuinely belong to multiple dialects. Transition zones between dialects are documented by Zuazo.

### Assumptions Made
1. **Dialect labels are available or inferrable.** _Assumption_: we can obtain region-labeled text from oral archives (Ahotsak.eus), manually adapted NLI datasets (XNLI dialectal splits), and social media data. Note: major Basque newspapers like *Berria* publish primarily in Batua, making them less useful for dialectal training data.
2. **Basque Batua is treated as a separate "dialect" class.** _Assumption_: Standard Basque is the target for mixed/neutral text, and distinguishing it from vernacular dialects is valuable.
3. **Target dialect set follows Zuazo's 5+1 classification.** _Assumption_: Zuazo's five-dialect model (Western/Bizkaiera, Central/Gipuzkera, Navarrese, Navarrese-Labourdin, Souletin/Zuberera) + Batua is used. Note that Zuazo's taxonomy supersedes the older 8-dialect Bonaparte classification. Sub-dialects (e.g., Goierri vs Beterri within Gipuzkera) are out of initial scope.
4. **Multi-label framing is desirable but not mandatory for v1.** _Assumption_: Single-label classification is the baseline; multi-label is explored if data allows.
5. **Speech data is even scarcer than text.** _Assumption_: Phase 2 will rely on Ahotsak.eus oral archive, Mintzoak.eus (Northern Basque Country), Mozilla Common Voice (Basque), and the Basque Parliament Speech Corpus. Synthesis via TTS voice conversion may be required to augment dialectal data.

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

## Open Questions — Partial Answers

The following questions were originally open; research has provided partial or full answers for each.

### 1. Data sourcing ✅ (partially resolved)
**Known resources:**
- **Ahotsak.eus** — Oral archive with 7,000+ interviews across all Basque dialect regions. Publicly accessible. The primary ground-truth source for dialectal speech and transcriptions.
- **XNLI Dialectal Splits** (HiTZ, 2026) — XNLI test set manually adapted into Western, Central, and Navarrese-Lapurdian dialects. A gold-standard evaluation set.
- **BasPhyCowest** (HiTZ, 2026) — Physical commonsense reasoning dataset adapted into Western Basque via LLM + manual validation.
- **Basque Social Media Corpus** — Twitter/X data used in prior research for informal dialectal variation.
- **"A Catalog of Basque Dialectal Resources"** (Bengoetxea, Gonzalez-Dios, Agerri, 2026) — Comprehensive inventory of existing online collections and adapted datasets. **This is the primary entry point for resource discovery.**
- **Mozilla Common Voice (Basque)** — Hundreds of hours of validated speech; partially sourced from EITB sentences.
- **Basque Parliament Speech Corpus** — 1,400+ hours (2013–2022), transcribed; predominantly Batua but useful for baseline.
- **Gaitu-Data (HiTZ/Aholab)** — Repository for Basque speech models and datasets.

**Remaining gap:** No large, fully labeled, multi-dialect text corpus exists. Most newspaper content (including *Berria*) is in Batua. Social media scraping remains viable for informal dialectal content.

### 2. Dialect taxonomy ✅ (resolved)
Zuazo's modern 5-dialect classification is the standard reference for NLP work:
1. **Western** (Bizkaiera/Biscayan)
2. **Central** (Gipuzkera/Gipuzkoan)
3. **Navarrese** (Nafarrera)
4. **Navarrese-Labourdin** (Nafarrera-Lapurtera)
5. **Souletin** (Zuberera/Zuberoan)
6. **Batua** (Standard Basque) — treated as a 6th class

This supersedes the older 8-dialect Bonaparte classification. Zuazo also identifies interdialect and intradialect transition zones (e.g., Debagoiena in Gipuzkoa, bordering Bizkaia, shows mixed Gipuzkera/Bizkaiera features). The Roncalese dialect is now extinct and not included.

**Confirmed geographic scopes (from Zuazo's official site, [euskalkiak.eus](http://euskalkiak.eus)):**
- **Western (Bizkaiera):** Bizkaia, Deba ibar (most of it), and Arabako Aramaio. Some Western features extend further (Burunda, Urolaldea).
- **Central (Gipuzkera):** Most of Gipuzkoa. Increasingly influencing western Navarre (Araitz-Betelu, Larraun, Basaburua, Imotz).
- **Navarrese:** Most of Navarre.
- **Navarrese-Labourdin:** Lapurdi, Nafarroa Beherea, Luzaide valley, and a few villages in western Zuberoa (Domintxaine-Berroeta, Etxarri, Arüe-Ithorrotze-Olhaibi, Lohitzüne-Oihergi, Pagola).
- **Souletin (Zuberera):** Most of Zuberoa, plus the Bearnese village of **Eskiula** (outside the Basque Country proper).

**Confirmed transition zones (do not use for `high`-confidence training labels):**
| Zone | Location | Between |
|---|---|---|
| Deba ibar (north) | Elgoibar, Mendaro, Mutriku | Western ↔ Central |
| Burunda | Sakana, Navarre west | Western ↔ Central ↔ Navarrese |
| Gipuzkoa NE | Errenteria, Lezo, Oiartzun, Hondarribia, Irun | Central ↔ Navarrese ↔ Nav-Lab |
| Navarre west | Araitz, Larraun, Basaburua, Imotz | Central ↔ Navarrese |
| Baztan + Urdazubi-Zugarramurdi | Navarre north | Navarrese ↔ Nav-Lab |
| Aezkoa | Navarre NE | Navarrese ↔ Nav-Lab |
| Amikuze | Nafarroa Beherea | Nav-Lab ↔ Souletin |

**Recommendation:** Use the 3-class grouping (Western, Central, Navarrese-Lapurdian) as a fallback if data is insufficient for 5+1 class classification, matching the XNLI dialectal split schema.

### 3. Evaluation data ✅ (partially resolved)
- The **XNLI dialectal splits** provide a ready gold-standard evaluation set for 3 dialect groups.
- For all 5+1 dialect classes, annotation by native speakers will still be required.
- **Recommendation:** Use XNLI dialectal splits as the primary evaluation benchmark for Phase 1, and supplement with custom annotation for dialects not yet covered (especially Zuberera and Navarrese).

### 4. Orthographic normalization ⚠️ (open, with guidance)
CHALIS research recommends that models be trained to be **robust to orthographic variation** rather than relying on normalization. This aligns with the use of diacritic-stripped and noisy input in the CHALIS benchmark. For Basque, this is particularly relevant for social media data.

**Recommendation:** Do not normalize input during inference. Instead, augment training data with orthographic noise (diacritic removal, informal spelling variants). This is the approach validated by CHALIS.

### 5. Multi-label ground truth ⚠️ (open)
No existing automated solution. Expert annotation or inter-annotator agreement protocols are still required. The HiTZ group's manually adapted datasets provide some parallel signal but do not address the multi-label case directly.

**Recommendation:** Defer multi-label to Phase 1 v2. Train single-label first and use model confidence distributions to identify candidate multi-label samples for human annotation.

### 6. Speech data availability ✅ (partially resolved)
- **Ahotsak.eus** — 7,000+ dialect-labeled interviews across all Basque dialect regions; primary ground-truth source. Access requires research collaboration or direct web scraping (check ToS).
- **Mintzoak.eus** — Oral memory portal for **Northern Basque Country (Ipar Euskal Herria)**, managed by the Basque Cultural Institute (EKE). Contains 1,204 recordings, 1,007 speakers, 7,436 segments, and 6,297 audio/video files. Particularly valuable for **Lapurtera, Nafarrera-Lapurtera, and Zuberera** — the Northern dialects most underrepresented in other corpora. Check licensing via the site's *Erabilpen baldintzak* (usage terms). [mintzoak.eus](https://www.mintzoak.eus/eu/)
- **Mozilla Common Voice (Basque)** — Open-source, validated; limited dialectal metadata.
- **Basque Parliament Speech Corpus** — 1,400+ hours, primarily Batua.
- **OpenSLR SLR76** — Multi-speaker Basque speech corpus.
- **Aholab (UPV/EHU)** — Research group for Basque speech technology; potential collaboration.

**Remaining gap:** No publicly available, fully dialect-labeled speech corpus. Mintzoak partially fills the Northern dialect gap, but Southern dialects (Bizkaiera, Gipuzkera) are better covered by Ahotsak. Synthesis via LLM-based TTS or voice conversion may still be needed for Zuberera and Navarrese specifically.

### 7. Licensing ⚠️ (open, requires due diligence)
- Ahotsak.eus is publicly accessible but licensing for ML training must be confirmed with Badihardugu Association.
- EITB raw audio is **not freely available** for training due to copyright — confirmed. Research collaborations with Aholab/HiTZ are the standard path.
- Mozilla Common Voice is CC0 licensed — freely usable.
- XNLI dialectal splits: licensing follows the original XNLI license (check MultiNLI/XNLI terms).
- Basque Parliament Speech Corpus: publicly available; licensing should be confirmed.

**Recommendation:** Use Mozilla Common Voice and Parliament corpus as the baseline, and seek a formal research agreement with HiTZ/Aholab for access to Ahotsak data.

---

## Key References

### Core NLP/DID Methods
1. **UniLID** (2026) — Tokenizer-based language ID using UnigramLM; data-efficient (≥5 samples/class); supports incremental addition of new dialects without retraining; competitive with fastText and GlotLID-M on fine-grained dialect ID. [arXiv:2602.17655](https://arxiv.org/abs/2602.17655) — *Authors: Clara Meister, Ahmetcan Yavuz, Pietro Lesci, Tiago Pimentel.*
2. **CHALIS** (2026) — Benchmark for LID in difficult scenarios: cousin languages (Czech/Slovak, Spanish/Catalan, Portuguese/Galician, Danish/Norwegian) and orthographic noise (diacritic removal, transliteration, homoglyph attacks, internet slang). Available on Hugging Face: `michal-tichy/CHALIS`. [arXiv:2606.06088](https://arxiv.org/abs/2606.06088) — *Authors: Michal Tichý, Jindřich Libovický.*
3. **CTC-DID** (2026) — CTC-based dialect ID for Arabic speech; treats dialect ID as a limited-vocabulary ASR task; inherently streaming-capable; outperforms baselines on short utterances and ADI tasks. [arXiv:2601.12199](https://arxiv.org/abs/2601.12199)
4. **Frisian Dialectal ASR** (2025) — SSL fine-tuning (ICASSP 2025) for low-resource European language with dialects (Clay, Wood, South Frisian); auxiliary LID task during fine-tuning; key finding: ASR gaps between standard and dialectal speech are significant and dialect-collection methodology matters. [arXiv:2502.04883](https://arxiv.org/abs/2502.04883)
5. **Multi-Label Arabic DI** (2026) — Curriculum learning + pseudo-labeling for multi-label dialect ID. [arXiv:2602.12937](https://arxiv.org/abs/2602.12937)
6. **Memory-Efficient Fine-Tuning for DID** (2024) — LoRA/MEFT for speech dialect ID with reduced GPU requirements. [arXiv:2512.02074](https://arxiv.org/abs/2512.02074)

### Basque-Specific Resources
7. **"A Catalog of Basque Dialectal Resources"** (2026) — Bengoetxea, Gonzalez-Dios, Agerri (HiTZ/Ixa). Comprehensive inventory of online dialectal collections (news, radio, social media) and standard-to-dialectal adaptations (XNLI, BasPhyCowest). **Primary reference for data sourcing.** [IXA/HiTZ](https://www.ixa.eus/)
8. **Latxa LLM** (HiTZ) — Basque-specific LLM family (7B–70B) based on Llama 2/3.1, continually pre-trained on EusCrawl. Available on Hugging Face: `hitz-zentroa`. A strong candidate for fine-tuning as a dialect classifier backbone.
9. **BASYQUE** (Ixa Group) — Tool for analyzing syntactic variation in Basque, targeting North-Eastern varieties.
10. **Ahotsak.eus** — Oral heritage archive with 7,000+ interviews across all Basque dialect regions; primary ground-truth source for dialect-labeled speech.
11. **Mintzoak.eus** — Oral memory portal for Northern Basque Country (Ipar Euskal Herria), managed by EKE (Basque Cultural Institute). 1,204 recordings, 1,007 speakers, 7,436 segments, 6,297 audio/video files; focuses on Northern dialects (Lapurtera, Nafarrera-Lapurtera, Zuberera). [mintzoak.eus](https://www.mintzoak.eus/eu/)
12. **Basque Parliament Speech Corpus** — 1,400+ hours (2013–2022), bilingual (Basque/Spanish), transcribed; primarily Batua but large-scale and openly available.
13. **euskalkiak.eus** — Koldo Zuazo's official Basque dialect website (CC BY 4.0, 2015). An essential dialectological reference containing: (a) per-dialect linguistic feature descriptions (phonology, morphology, syntax, lexicon) for all 5 dialects — directly usable for building the lexical/morphological marker lexicon for labeling (Task 1.2); (b) detailed transition zone documentation (Burunda, Araitz-Betelu, Deba Ibar, Baztan-Zugarramurdi, Aezkoa, Amikuze); (c) a general dialect map PDF ([mapa_orokorra.pdf](http://euskalkiak.eus/img/mapa_orokorra.pdf)) — primary reference for the municipality→dialect mapping table (Task 1.0); (d) annotated dialect speech video samples (`ikus-entzunezkoak`) organized by dialect — potentially usable as labeled speech data; (e) comprehensive dialectological bibliography per dialect. [euskalkiak.eus](http://euskalkiak.eus/)

## Related Shared Tasks & Benchmarks

- **NADI** (Nuanced Arabic Dialect Identification) — Annual shared task since 2018, multi-label DID, dialect-to-MSA translation.
- **VarDial** (Workshop on NLP for Similar Languages, Varieties, and Dialects) — Covers low-resource and closely related language identification.
- **FLEURS** — Multilingual speech benchmark, includes some language variety distinctions.

---

## Improvement Suggestions

1. **Goiena classification error (original doc):** The original document listed "Goiena → Gipuzkera Bizkaiera mix". This is partially correct — Goiena covers the **Debagoiena** region of Gipuzkoa (not Bizkaia), where the spoken Basque exhibits transitional Gipuzkera/Bizkaiera features. Goiena publishes primarily in Batua, not in local dialect, which limits its direct utility for dialect training data.

2. **Zuazoa → Zuazo spelling:** The document consistently spells the linguist's name as "Zuazoa" — the correct spelling is **Koldo Zuazo** (no trailing 'a', which in Basque would be the definite article suffix). References should be corrected accordingly.

3. **Basque BERT reference is outdated:** The FR3 mention of "Basque BERT" should be updated to include **Latxa** (HiTZ, 2024–2026), which is the current state-of-the-art Basque-specific LLM and a more suitable fine-tuning candidate than older encoder-only models.

4. **Dialect label for "Lapurtera":** The original scope table lists "Lapurtera" as a standalone dialect, but under Zuazo's classification it is merged into **Nafarrera-Lapurtera** (Navarrese-Labourdin). The refined prompt correctly uses "Nafarrera-Lapurtera" in most places, but the scope table header used "Lapurtera" separately — this should be made consistent.

5. **LLM-based dialect adaptation as data strategy:** Given the scarcity of dialectal data, the HiTZ group's approach of using LLMs (Latxa) for **standard-to-dialect style transfer** is now a validated strategy (BasPhyCowest). This should be considered as a first-class data augmentation method, not just a stretch goal.

6. **CTC-DID is Arabic-specific:** The reference to CTC-DID in the context of Basque speech should note that the method was validated on Arabic (ADI), and adaptation to Basque would require retraining — the architecture is transferable but not the model weights.
