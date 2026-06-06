# Zeineuski — Implementation Plan

## Plan Overview

This document defines the complete execution plan for **Zeineuski**, a fine-grained dialect identification (DID) system for Basque (Euskara), covering both text and speech modalities. The plan is structured as a sequence of epics, each containing multiple tasks with concrete objectives, actions, outputs, and validation checks.

**Total estimated duration:** 16 weeks (see PROJECT.md milestones).
**Team:** 1–2 developers with Python/ML experience (configurable per-task assignment).
**Compute:** 1× GPU with ≥24 GB VRAM (RTX 4090 or equivalent cloud instance).
**Primary language:** Python 3.11+
**Package manager:** `uv` (Astral)

---

## Assumptions

| # | Assumption | Impact if False |
|---|---|---|
| A1 | XNLI dialectal splits are available and contain ≥500 labeled sentences per dialect group (at least for the 3-class grouping). | Must bootstrap annotations from scratch; significantly delays Phase 1. **Resolved:** ✓ Obtained from `hitz-zentroa/Catalog-of-Basque-Dialects` (5,010 test sentences × 3 dialects + 621 native sentences × 3 dialects). Stored at `data/raw/text/xnli_dialectal/`. |
| A2 | Ahotsak.eus and Mintzoak.eus audio is accessible (downloadable or via research agreement) within the first 8 weeks. | Phase 2 speech pipeline limited to Common Voice + Parliament; dialect diversity severely reduced. |
| A3 | Latxa 7B can be fine-tuned with QLoRA on a single 24 GB GPU. | Fall back to XLM-R encoder-only; accuracy ceiling lowers. |
| A4 | At least 1 native Basque speaker is available for annotation quality checks and evaluation data validation (hours/week, not full-time). | Evaluation quality suffers; rely solely on existing labeled datasets (XNLI splits). |
| A5 | Hugging Face Hub token and organization access (`zeineuski` or personal) is set up by end of Phase 0. | Delays model release; models stored locally until resolved. |
| A6 | fastText Python bindings (`fasttext` PyPI package) install cleanly on the target system. | Use `fasttext-wheel` or fall back to scikit-learn character n-gram pipeline. |
| A7 | Existing BasPhyCowest and XNLI dialectal splits are licensed for derivative use (training + redistribution). | Must negotiate or avoid distributing those datasets; train with them but release only model weights. |
| A8 | A municipality-to-dialect mapping table can be constructed from Zuazo's atlas and Euskaltzaindia data, covering the main Basque Country villages at sufficient resolution. | Geo-proxy labeling (author origin, interview location) is unreliable for transition zones; confidence tier `medium` must be downgraded to `low`. |
| A9 | Works of classical Basque authors can be obtained in digital text form (Project Gutenberg Basque, Klasikoak.eus, or similar). | Author-origin geo-proxy strategy limited to social media and archival transcriptions. |

---

## Work Breakdown Structure

### Epic 0: Foundation & Infrastructure
### Epic 1: Text Data Pipeline
### Epic 2: Text Models (Baselines)
### Epic 3: Text Models (Advanced)
### Epic 4: Speech Data Pipeline
### Epic 5: Speech Models (Baselines)
### Epic 6: Speech Models (Advanced)
### Epic 7: Evaluation, Integration & Release

---

## Epic 0: Foundation & Infrastructure

### Task 0.1 — Repository Bootstrap

**Objective:** Create the project skeleton with modern Python tooling, Git, and CI.

**Concrete actions:**
1. Create project directory and initialize `uv`:
   ```bash
   cd /home/xezpeleta/Dev/itzune/zeineuski
   uv init --python 3.11
   ```
2. Configure `pyproject.toml` with project metadata:
   - name: `zeineuski`
   - description: "Fine-grained dialect identification for Basque (Euskara)"
   - authors: `[{name: "Zeineuski Team"}]`
   - readme: `README.md`
3. Add core dependencies:
   ```bash
   uv add torch transformers datasets accelerate peft bitsandbytes
   uv add fasttext evaluate scikit-learn
   uv add wandb
   uv add --dev pytest pytest-cov ruff mypy pre-commit
   uv add --dev dvc dvc-s3  # or dvc-gs for cloud storage
   ```
4. Create directory structure:
   ```
   zeineuski/
   ├── src/
   │   ├── __init__.py
   │   ├── data/
   │   │   ├── __init__.py
   │   │   ├── text_loader.py
   │   │   ├── text_preprocessing.py
   │   │   ├── speech_loader.py
   │   │   └── speech_preprocessing.py
   │   ├── models/
   │   │   ├── __init__.py
   │   │   ├── text/
   │   │   │   ├── __init__.py
   │   │   │   ├── fasttext_classifier.py
   │   │   │   ├── unilid_classifier.py
   │   │   │   ├── xlmr_classifier.py
   │   │   │   └── latxa_classifier.py
   │   │   └── speech/
   │   │       ├── __init__.py
   │   │       ├── ecapa_tdnn.py
   │   │       ├── whisper_did.py
   │   │       ├── xlsr_did.py
   │   │       └── ctc_did.py
   │   ├── evaluation/
   │   │   ├── __init__.py
   │   │   ├── metrics.py
   │   │   └── reporter.py
   │   ├── augmentation/
   │   │   ├── __init__.py
   │   │   ├── text_aug.py
   │   │   └── speech_aug.py
   │   └── cli.py
   ├── configs/
   │   ├── text/
   │   │   ├── fasttext.yaml
   │   │   ├── unilid.yaml
   │   │   ├── xlmr.yaml
   │   │   └── latxa.yaml
   │   └── speech/
   │       ├── ecapa.yaml
   │       ├── whisper.yaml
   │       ├── xlsr.yaml
   │       └── ctc.yaml
   ├── tests/
   │   ├── __init__.py
   │   ├── test_data/
   │   ├── test_models/
   │   └── test_evaluation/
   ├── data/             # gitignored, tracked by DVC
   │   ├── raw/
   │   │   ├── text/
   │   │   └── speech/
   │   ├── processed/
   │   └── annotated/
   ├── models/           # gitignored, tracked by DVC
   ├── notebooks/
   ├── .gitignore
   ├── .dvc/
   ├── .pre-commit-config.yaml
   ├── pyproject.toml
   └── README.md
   ```
5. Set up pre-commit hooks: ruff (lint + format), mypy (type check), trailing-whitespace, end-of-file-fixer.
6. Configure `.gitignore`:
   ```
   __pycache__/
   *.pyc
   .venv/
   data/
   models/
   *.pth
   *.bin
   wandb/
   .env
   *.egg-info/
   dist/
   ```
7. Write initial `README.md` with project description, setup instructions, and quickstart.

**Expected output:** Working project with `uv sync` succeeding, all directories created, pre-commit configured.

**Validation:**
- `uv sync` installs all dependencies without errors.
- `uv run pytest` runs (0 tests, passing).
- `uv run ruff check src/` passes on empty files.
- `git status` shows clean structure.

---

### Task 0.2 — Experiment Tracking Setup

**Objective:** Configure Weights & Biases for experiment logging.

**Concrete actions:**
1. Create Weights & Biases project `zeineuski` (free tier).
2. Create `src/utils/tracking.py`:
   ```python
   import wandb
   import yaml
   from pathlib import Path

   def init_run(config_path: str, run_name: str, tags: list[str] | None = None):
       config = yaml.safe_load(Path(config_path).read_text())
       wandb.init(project="zeineuski", name=run_name, config=config, tags=tags or [])

   def log_metrics(metrics: dict, step: int | None = None):
       wandb.log(metrics, step=step)

   def log_confusion_matrix(cm, class_names: list[str], title: str = "Confusion Matrix"):
       wandb.log({title: wandb.plot.confusion_matrix(
           probs=None, y_true=[], preds=[], class_names=class_names)})
       # Use wandb.sklearn.plot_confusion_matrix if available
   ```
3. Create `src/utils/config.py` with a `load_config()` helper using YAML.
4. Verify connection: run a one-line script that creates a WandB run and logs a dummy metric.

**Expected output:** `src/utils/tracking.py`, `src/utils/config.py`, WandB project created.

**Validation:** A test run appears in WandB dashboard with logged metrics.

---

### Task 0.3 — CI Pipeline

**Objective:** Set up GitHub Actions for linting, type checking, and testing on push.

**Concrete actions:**
1. Create `.github/workflows/ci.yaml`:
   ```yaml
   name: CI
   on: [push, pull_request]
   jobs:
     lint-and-test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with:
             python-version: "3.11"
         - name: Install uv
           run: pip install uv
         - name: Install dependencies
           run: uv sync
         - name: Lint
           run: uv run ruff check src/ tests/
         - name: Type check
           run: uv run mypy src/ --ignore-missing-imports
         - name: Test
           run: uv run pytest tests/ -v --cov=src --cov-report=term
   ```
2. Push to trigger CI and verify all steps pass.

**Expected output:** `.github/workflows/ci.yaml`, green CI badge in README.

**Validation:** Push triggers CI; all steps pass (linter, type checker, tests).

---

## Epic 1: Text Data Pipeline

### Task 1.0 — Municipality-to-Dialect Mapping Table

**Objective:** Build a structured lookup table mapping Basque Country municipalities to dialect class. This is the foundational artifact for geo-proxy labeling (Task 1.2) and must be completed before any location-based label inference.

**Rationale:** Author birthplace, interview location (Ahotsak/Mintzoak), and social media user location are all geo-signals. Without a systematic geo→dialect mapping, these signals cannot be reliably converted to dialect labels.

**Concrete actions:**
1. Create `data/reference/municipality_dialect.csv` with schema:
   ```csv
   herria,probintzia,eskualdea,dialect_class,dialect_confidence,notes
   Bermeo,Bizkaia,Busturialdea,western,high,
   Oñati,Gipuzkoa,Debagoiena,central,medium,"transition zone; Gipuzkera/Bizkaiera contact"
   Sara,Lapurdi,,nav-lab,high,"Axular's birthplace; strong Lapurdian"
   Atharratze,Zuberoa,,souletin,high,
   Elizondo,Nafarroa,,navarrese,high,
   ```
2. Primary sources for the mapping:
   - **Zuazo's dialect atlas** ("Euskalkiak", 2010/2014) — includes geographic boundaries at municipality level.
   - **euskalkiak.eus** (Koldo Zuazo, CC BY 4.0) — official online companion to the atlas. Download the general dialect map: [mapa_orokorra.pdf](http://euskalkiak.eus/img/mapa_orokorra.pdf). Transition zones are documented in the [sailkapenak](http://euskalkiak.eus/sailkapenak.php) section.
   - **Euskaltzaindia's euskalkiak map** — official dialect zoning, publicly available.
   - **Ahotsak.eus metadata** — already maps recordings to municipalities; cross-reference for validation.
   - **Mintzoak.eus `herriak` section** — same, for Northern Basque Country villages.
3. Classify each municipality with one of: `western`, `central`, `navarrese`, `nav-lab`, `souletin`, `batua` (for purely urban/standard contexts).
4. Assign `dialect_confidence`:
   - `high`: well within a single dialect zone (e.g., Bermeo → Bizkaiera).
   - `medium`: near a dialect border but predominantly one dialect.
   - `low`: in a known transition zone (Debagoiena, Bidasoa valley, Baztan) — do not use for training data without additional signal.
5. Document all transition zones explicitly in a `notes` field.
6. Target coverage: ≥200 municipalities (sufficient for all major Basque population centers).

**Expected output:** `data/reference/municipality_dialect.csv`, `data/reference/transition_zones.md`.

**Validation:**
- All 7 Basque historical territories have ≥10 municipalities represented.
- All 5 dialect classes + Batua have ≥20 `high`-confidence municipalities.
- Transition zones are explicitly flagged (not assigned a confident label).
- A native speaker or dialectologist validates a sample of 20 assignments.

---

### Task 1.1 — Inventory & Download Text Resources

**Objective:** Gather all known Basque dialect text resources into a single registry.

**Status: IN PROGRESS.** Partially completed on 2026-06-06.

**Concrete actions:**

1. **✓ Done — Create `data/raw/text/` directory.**
2. **✓ Done — Identify and download resources (in priority order):**

   | Source | Dialect Coverage | Format | Access | Status |
   |---|---|---|---|---|
   | XNLI dialectal splits (HiTZ) | Western, Central, Navarrese-Lapurdian | TSV | `hitz-zentroa/Catalog-of-Basque-Dialects` | ✓ Downloaded (5,010+621 per dialect) |
   | HiTZ/xnli-eu (Batua baseline) | Batua | Arrow/JSONL | Hugging Face `HiTZ/xnli-eu` | ✓ Downloaded (400k total) |
   | Basqueparl (Parliament) | Batua | Arrow | Hugging Face `HiTZ/basqueparl` | ✓ Sampled 50k from 342k |
   | Wikipedia Basque | Batua | Arrow | Hugging Face `wikimedia/wikipedia` | ✓ Sampled 50k from 416k |
   | ikerHerrero/Basque_Dialects_Classification | 5 dialects | Model | Hugging Face | ✓ Identified, model exists (F1=0.68), training data not published |
   | BasPhyCowest (HiTZ) | Western | JSON/TSV | Request from HiTZ | ☐ Pending — not yet public |
   | Klasikoak.eus / classical Basque literature | All dialects (pre-Batua authors) | HTML/TXT | Public domain | ☐ Reachable, needs scraping + author→dialect mapping |
   | Basque social media corpus (Twitter/X) | All (informal) | JSONL | Academic Twitter API or HiTZ corpus | ☐ Pending |

3. **✓ Done — Create `data/raw/text/registry.csv`** with all sources documented.
4. If a source is not immediately available, file a data access request and mark it as `pending`.

**Expected output:** `data/raw/text/registry.csv` with all known sources documented.

**Validation:** At least 3 sources downloaded; registry has ≥5 rows; each downloaded file exists on disk.

---

### Task 1.2 — Dialect Label Inference (Geo-Proxy & Lexical Strategies)

**Objective:** For all text sources without explicit dialect labels, apply a multi-strategy geo-proxy and lexical labeling approach. **Priority: few but high-quality labels over many noisy ones.** Only samples with `high` or `medium` confidence are used for training; `low`-confidence samples are stored but excluded from training by default.

**Labeling strategies (applied in priority order):**

| Strategy | Signal | Applicable Sources | Typical Confidence |
|---|---|---|---|
| **Explicit label** | XNLI/BasPhyCowest metadata | HiTZ datasets | gold |
| **Author origin** | Known birthplace of author → municipality → dialect (via Task 1.0 table) | Classical literature (Axular, Mogel, Dechepare...), archival texts | high (if author origin is `high` in municipality table) |
| **Interview location** | Recording location in Ahotsak/Mintzoak metadata → municipality → dialect | Oral archive transcriptions | high/medium |
| **User location** | Declared location in social media profile → municipality → dialect | Twitter/X | medium |
| **Lexical signal** | Dialect-specific morphological/lexical markers in text | Any source | medium (if multiple markers agree) |
| **User bio keywords** | Self-identification in bio ("bizkaitarra", "gipuzkoarra"...) | Twitter/X | medium |

**Concrete actions:**
1. **Prerequisite:** Task 1.0 must be complete. Load the municipality→dialect mapping from `data/reference/municipality_dialect.csv`.
2. Implement `src/data/text_labeling.py` with:
   - `label_by_municipality(municipality: str, mapping_df: pd.DataFrame) -> tuple[str | None, str]`: Look up the municipality in the mapping table; return `(dialect_class, confidence)`.
   - `label_by_author_origin(author_name: str, author_db: dict) -> tuple[str | None, str]`: Map author to origin municipality, then to dialect. Maintain `data/reference/basque_authors.csv` with columns `author, municipality, era, notes`.
     ```csv
     author,municipality,dialect_class,era,notes
     Pedro Agerre Axular,Sara,nav-lab,17th c.,"born Sara (Lapurdi); classic Lapurdian prose"
     Juan Antonio Mogel,Eibar,western,18th c.,"wrote in Bizkaiera despite living in Gipuzkoa — verify individually"
     Bernard Dechepare,Garazi,nav-lab,16th c.,"first printed Basque book"
     Frai Bartolome,Muxika,western,18th c.,"Bizkaia; Bizkaiera"
     Domingo Agirre,Ondarroa,western,19th–20th c.,""
     Kirikiño,Bilbo,western,early 20th c.,""
     Txomin Agirre,Ondarroa,western,19th c.,""
     Resurreccion Maria Azkue,Lekeitio,western,19th–20th c.,"also wrote grammars"
     Jean Etxepare,Banka,navarrese,early 20th c.,"Navarro-Labourdin variety"
     ```
   - `label_by_location(user_location: str, mapping_df: pd.DataFrame) -> tuple[str | None, str]`: Fuzzy-match user-declared location to municipality table.
   - `label_by_keyword_lexicon(text: str, dialect_lexicons: dict) -> tuple[str | None, str]`: Match known dialect-specific words. Build a lexicon of 50–100 morphological/lexical markers per dialect. **Primary source:** the [ezaugarriak](http://euskalkiak.eus/ezaugarriak.php) section of euskalkiak.eus (CC BY 4.0) provides ready-to-use feature lists per dialect:
     - **Western:** -gaz (sociative), -au- vowel pattern, dot/dau verb forms, -i bordering -x-, lexicon: *domeka, bariku, armozu, berba, gura, itxi, gitxi*
     - **Central:** -det/-degu verb forms, [x] word-initial j, *al* in yes/no questions, lexicon: *aitona, apreta, esnatu, iritsi, kilker, beta, triku*
     - **Navarrese:** -s instrumental (burus, eskus), strong stress on penultimate syllable, syncope (ekarri→kárri), lexicon: *aunitz, ugalde, arroitu, goatze, dermio, orantz*
     - **Nav-Lab:** x- word-initial (xori, ximino), nehor 'inor' pronouns, -ño diminutive, lexicon: *sehi, gako, pairatu, xingar, elgar, altxagarri, ahantzi*
     - **Souletin:** ü vowel (ezagün, lagün), -ai- for -au- (gáiza←gauza), bokal sudurkariak, lexicon: *ürhentü, bedatse, haboro, amiñi, mithil, ükhen, ardu*
   Only return a label if ≥3 markers agree.
   - `combine_signals(signals: list[tuple[str | None, str]]) -> tuple[str | None, str]`: Aggregate multiple signals using a priority order. If two `high`-confidence signals agree → `high`; if they disagree → `low`.
3. Apply all applicable strategies to each corpus. Store labeled output as `data/processed/text/<source>_labeled.jsonl`:
   ```json
   {"text": "...", "dialect": "western", "confidence": "high", "source": "klasikoak", "labeling_method": "author_origin", "author": "Domingo Agirre", "author_municipality": "Ondarroa"}
   ```
4. Use `confidence` tiers for training:
   - `gold` / `high` → training + evaluation candidate
   - `medium` → training only (not test set)
   - `low` → stored for potential future use; excluded from training

**Expected output:** `src/data/text_labeling.py`, `data/reference/basque_authors.csv`, labeled JSONL files per source.

**Validation:**
- Each dialect class has ≥100 `high`-confidence samples after all strategies are applied.
- Spot-check 20 `high`-confidence samples per dialect manually; ≥90% accuracy expected.
- Author origin labels are double-checked against known literary sources (e.g., Auñamendi encyclopedia).
- No municipality in a transition zone is assigned `high` confidence.

---

### Task 1.3 — Text Preprocessing Pipeline

**Objective:** Build a reusable preprocessing module that cleans, normalizes, deduplicates, and formats text for model training.

**Concrete actions:**
1. Implement `src/data/text_preprocessing.py` with these functions:

   | Function | Purpose |
   |---|---|
   | `clean_text(text: str) -> str` | Remove URLs, @mentions, hashtag symbols (keep text), extra whitespace, HTML entities. Optionally remove emoji. |
   | `normalize_basque(text: str, keep_diacritics: bool = True) -> str` | Normalize Basque-specific characters: ñ → n (if stripping), replace inconsistent apostrophe usage (ʻ vs ' vs `), normalize ñ/Ñ. By default keep diacritics (CHALIS recommendation). |
   | `deduplicate(texts: list[str], threshold: float = 0.95) -> list[str]` | Near-dedup using MinHash or Jaccard similarity on character n-grams. Remove exact duplicates first. |
   | `filter_length(texts: list[str], min_chars: int = 20, max_chars: int = 2000) -> list[str]` | Filter by character length. |
   | `filter_language(texts: list[str], lid_model=None) -> list[str]` | Run fastText LID; keep only texts classified as Basque (`eu`). |
   | `prepare_dataset(texts: list[str], labels: list[str], output_path: str) -> None` | Write to JSONL/CSV in Hugging Face datasets-compatible format. |

2. Create `src/data/text_loader.py` with:
   - `load_labeled_data(sources: list[str]) -> Dataset`: Load from registry, apply preprocessing, return Hugging Face `Dataset`.
   - `create_splits(dataset: Dataset, stratify_by: str = "dialect", test_size: float = 0.15, val_size: float = 0.10) -> DatasetDict`: Stratified train/val/test split.

3. Create a default preprocessing config at `configs/text/preprocessing.yaml`:
   ```yaml
   min_chars: 20
   max_chars: 2000
   keep_diacritics: true
   dedup_threshold: 0.95
   language_filter: true
   language_confidence_threshold: 0.7
   ```

**Expected output:** `src/data/text_preprocessing.py`, `src/data/text_loader.py`, `configs/text/preprocessing.yaml`.

**Validation:**
- Unit test: `pytest tests/test_data/test_preprocessing.py` with 10 hand-crafted inputs.
- Pipeline runs end-to-end on XNLI splits: `uv run python -m src.data.text_loader --source xnli_dialectal --output data/processed/text/xnli_clean/`.
- Output dataset has correct column names (`text`, `label`), no empty texts, label distribution logged.

---

### Task 1.4 — Manual Annotation of Dev/Test Set

**Objective:** Create a small, high-quality manually annotated dataset for reliable evaluation. **Quality over quantity is the guiding principle here.** A smaller set with trustworthy labels is strictly better than a larger set with noisy ones, especially for evaluation.

**Prioritization strategy for sample selection:**
1. Prefer samples already assigned `high` confidence by geo-proxy (Task 1.2) — use them as candidates, not final gold labels.
2. Prioritize dialectally distinctive sentences (i.e., those with morphological/lexical markers clearly placing them in one dialect).
3. Include some genuinely ambiguous samples to populate the multi-label test cases.
4. Ensure geographic diversity within each dialect class (e.g., multiple sub-regions of Bizkaia for Western).

**Concrete actions:**
1. Sample 100–150 sentences per dialect (fewer, better quality) from `high`-confidence geo-proxy outputs and oral archive transcriptions.
2. Create annotation spreadsheet (Google Sheets or CSV) with columns: `id, text, source, geo_proxy_label, geo_proxy_confidence, dialect_annotated, annotator_notes, is_ambiguous, valid_dialects`.
3. Recruit 1–2 native Basque speakers (from HiTZ/Ixa network or UPV/EHU linguistics). Ideally one speaker per dialect region.
4. Annotation workflow:
   - Annotator reads the sentence and selects dialect label (or "Batua" if standard, or "ambiguous" if unclear).
   - If ambiguous, annotator lists all valid dialects (multi-label ground truth).
   - Each sentence annotated by at least 1 person; 30% double-annotated for IAA (higher than default because dataset is small).
   - Annotator also rates dialect confidence on a 3-point scale: clear / possible / unclear.
5. Compute inter-annotator agreement (Cohen's kappa). If κ < 0.6, revise annotation guidelines and re-annotate the disputed subset.
6. **Discard** samples where annotators disagree and cannot resolve — do not force a label. These are stored as `ambiguous` and used only for multi-label analysis.
7. Convert to `data/annotated/text/dev.jsonl` and `data/annotated/text/test.jsonl`.

**Expected output:** `data/annotated/text/dev.jsonl` (≈150–180 total), `data/annotated/text/test.jsonl` (≈200–250 total). Smaller than originally planned but higher quality.

**Validation:**
- Each dialect has ≥15 annotated sentences in dev and ≥25 in test.
- IAA κ ≥ 0.6 on 30% double-annotated subset.
- At least 1 native speaker has reviewed the full set.
- No sample in the test set has `confidence = low` from the annotator.

---

## Epic 2: Text Models (Baselines)

### Task 2.1 — fastText Baseline

**Objective:** Train a character n-gram fastText classifier as the classical baseline.

**Concrete actions:**
1. Implement `src/models/text/fasttext_classifier.py`:
   ```python
   import fasttext

   def train_fasttext(
       train_path: str,
       model_output: str,
       lr: float = 0.1,
       dim: int = 100,
       epoch: int = 25,
       word_ngrams: int = 2,
       minn: int = 3,    # min char n-gram
       maxn: int = 6,    # max char n-gram
       loss: str = "softmax",
   ) -> fasttext.FastText._FastText:
       model = fasttext.train_supervised(
           input=train_path,
           lr=lr,
           dim=dim,
           epoch=epoch,
           wordNgrams=word_ngrams,
           minn=minn,
           maxn=maxn,
           loss=loss,
           verbose=2,
       )
       model.save_model(model_output)
       return model
   ```
2. Create config `configs/text/fasttext.yaml`:
   ```yaml
   model: fasttext
   char_ngram_range: [3, 6]
   word_ngrams: 2
   dim: 100
   lr: 0.1
   epoch: 25
   loss: softmax
   ```
3. Convert training data to fastText format (`__label__western text here\n`) in `src/data/text_loader.py`.
4. Run training: `uv run python -m src.models.text.fasttext_classifier --config configs/text/fasttext.yaml`.
5. Evaluate on annotated test set; log metrics to WandB.

**Hyperparameter sweep:** Grid search over `minn` ∈ {2,3,4}, `maxn` ∈ {5,6,7}, `lr` ∈ {0.05, 0.1, 0.2}. Log all to WandB and select best by validation macro F1.

**Expected output:** `models/fasttext_dialect.bin`, evaluation metrics logged to WandB.

**Validation:**
- Model loads and predicts correctly: `model.predict("kaixo zer moduz", k=3)` returns dialect labels with probabilities.
- Accuracy on dev set ≥ 70% (3-class). If below, review data quality before proceeding to next task.
- Confusion matrix shows expected pattern (neighboring dialects confused most).

---

### Task 2.2 — UniLID Implementation

**Objective:** Implement the UniLID tokenizer-based dialect identification method (Meister et al., 2026).

**Concrete actions:**
1. Study the UniLID paper (arXiv:2602.17655). The core idea: learn dialect-conditional unigram distributions over a shared SentencePiece/Unigram vocabulary; at inference, segment the input with each dialect's unigram model and pick the dialect that maximizes the probability.
2. Implement `src/models/text/unilid_classifier.py`:
   ```python
   from tokenizers import Tokenizer, models
   from tokenizers.trainers import UnigramTrainer
   import sentencepiece as spm

   class UniLIDClassifier:
       def __init__(self, vocab_size: int = 8000):
           self.vocab_size = vocab_size
           self.dialect_models: dict[str, spm.SentencePieceProcessor] = {}

       def train(
           self,
           dialect_texts: dict[str, list[str]],
           model_dir: str,
       ):
           """Train one Unigram tokenizer per dialect, all sharing the same vocabulary."""
           # 1. Collect all texts to build shared vocabulary
           all_texts = []
           for texts in dialect_texts.values():
               all_texts.extend(texts)

           # 2. Train shared vocabulary using SentencePiece Unigram
           # Write all texts to temp file
           # Train shared tokenizer

           # 3. For each dialect, estimate language-conditional probabilities
           # using EM over the shared vocabulary

           # Save each dialect model
           pass

       def predict(self, text: str) -> dict[str, float]:
           """Score text under each dialect model; return probabilities."""
           pass
   ```
3. **Note:** The full UniLID implementation requires replicating the UnigramLM estimation procedure from the paper. This is non-trivial. As a pragmatic fallback, use the `tokenizers` library from Hugging Face to train one Unigram tokenizer per dialect and score via perplexity. This approximates UniLID but is simpler.
4. Create config `configs/text/unilid.yaml`.
5. Train and evaluate following the same protocol as fastText.

**Pragmatic alternative if UniLID implementation proves too complex:**
Use fastText with language-specific subword embeddings (already covered in Task 2.1) + add a Hugging Face `AutoModelForSequenceClassification` with XLM-R as a stronger multilingual baseline (moved up from Phase 1 Advanced if needed).

**Expected output:** `models/unilid_dialect/` with per-dialect tokenizer files and scoring script.

**Validation:**
- Per-dialect perplexity scores differ meaningfully (dialect model scores its own text higher).
- Accuracy ≥ fastText baseline. If not, document why and keep fastText as the primary classical baseline.

---

### Task 2.3 — XLM-R Fine-Tuning Baseline

**Objective:** Fine-tune XLM-RoBERTa as a strong multilingual transformer baseline.

**Concrete actions:**
1. Implement `src/models/text/xlmr_classifier.py`:
   ```python
   from transformers import (
       AutoTokenizer, AutoModelForSequenceClassification,
       Trainer, TrainingArguments, DataCollatorWithPadding
   )
   from datasets import Dataset, DatasetDict
   import numpy as np
   from sklearn.metrics import accuracy_score, f1_score

   def train_xlmr(
       dataset: DatasetDict,
       model_name: str = "FacebookAI/xlm-roberta-base",
       num_labels: int = 6,
       output_dir: str = "models/xlmr_dialect",
       batch_size: int = 16,
       lr: float = 2e-5,
       epochs: int = 5,
       warmup_ratio: float = 0.1,
       weight_decay: float = 0.01,
   ):
       tokenizer = AutoTokenizer.from_pretrained(model_name)
       model = AutoModelForSequenceClassification.from_pretrained(
           model_name, num_labels=num_labels
       )

       def tokenize(examples):
           return tokenizer(examples["text"], truncation=True, max_length=256)

       tokenized = dataset.map(tokenize, batched=True)

       training_args = TrainingArguments(
           output_dir=output_dir,
           per_device_train_batch_size=batch_size,
           per_device_eval_batch_size=batch_size,
           learning_rate=lr,
           num_train_epochs=epochs,
           warmup_ratio=warmup_ratio,
           weight_decay=weight_decay,
           evaluation_strategy="epoch",
           save_strategy="epoch",
           load_best_model_at_end=True,
           metric_for_best_model="eval_macro_f1",
           logging_steps=50,
           report_to="wandb",
       )

       def compute_metrics(eval_pred):
           logits, labels = eval_pred
           preds = np.argmax(logits, axis=-1)
           return {
               "accuracy": accuracy_score(labels, preds),
               "macro_f1": f1_score(labels, preds, average="macro"),
           }

       trainer = Trainer(
           model=model,
           args=training_args,
           train_dataset=tokenized["train"],
           eval_dataset=tokenized["validation"],
           data_collator=DataCollatorWithPadding(tokenizer),
           compute_metrics=compute_metrics,
       )
       trainer.train()
       trainer.save_model(output_dir)
       tokenizer.save_pretrained(output_dir)
       return trainer
   ```
2. Create config `configs/text/xlmr.yaml`.
3. Run training first on 3-class, then on 6-class if data allows.
4. Generate confusion matrix and per-class F1 report.

**Expected output:** `models/xlmr_dialect/`, training metrics in WandB.

**Validation:**
- 3-class accuracy > 80% on XNLI dialectal test set.
- Training converges (loss decreases monotonically, no NaN).
- Inference time < 50ms per sentence on CPU.

---

### Task 2.4 — Baselines Comparison & Selection

**Objective:** Compare all three baselines (fastText, UniLID, XLM-R) and select the best architecture to carry forward.

**Concrete actions:**
1. Create `src/evaluation/reporter.py`:
   - `compare_models(model_metrics: dict[str, dict]) -> pd.DataFrame`: Side-by-side metric table.
   - `plot_confusion_matrices(model_cms: dict[str, np.ndarray], class_names: list[str]) -> None`: Grid of confusion matrices.
2. Run evaluation on the annotated test set (Task 1.4) for all three models.
3. Generate comparison report with these columns:

   | Model | 3-Class Acc | 6-Class Acc | Macro F1 | Train Time | Inference (ms) | Model Size |
   |---|---|---|---|---|---|---|
   | fastText | — | — | — | — | — | — |
   | UniLID | — | — | — | — | — | — |
   | XLM-R | — | — | — | — | — | — |

4. Select winner based on: accuracy first, then data efficiency, then inference speed. Document selection rationale.
5. Log comparison to WandB as a report.

**Expected output:** Comparison report in WandB, decision documented in `docs/baseline_comparison.md`.

**Validation:** All 3 models evaluated on same test split with same preprocessing.

---

## Epic 3: Text Models (Advanced)

### Task 3.1 — Latxa Fine-Tuning with QLoRA

**Objective:** Fine-tune Latxa 7B (Basque-specific LLM) with QLoRA for dialect classification.

**Concrete actions:**
1. Implement `src/models/text/latxa_classifier.py`:
   ```python
   import torch
   from transformers import (
       AutoTokenizer, AutoModelForCausalLM,
       BitsAndBytesConfig, TrainingArguments, Trainer
   )
   from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
   from datasets import DatasetDict

   def train_latxa(
       dataset: DatasetDict,
       model_name: str = "hitz-zentroa/Latxa-7b-v1.1",
       output_dir: str = "models/latxa_dialect",
       lora_r: int = 16,
       lora_alpha: int = 32,
       lora_dropout: float = 0.05,
       batch_size: int = 4,
       lr: float = 2e-4,
       epochs: int = 3,
       max_length: int = 256,
   ):
       # 4-bit quantization config
       bnb_config = BitsAndBytesConfig(
           load_in_4bit=True,
           bnb_4bit_compute_dtype=torch.bfloat16,
           bnb_4bit_use_double_quant=True,
           bnb_4bit_quant_type="nf4",
       )

       model = AutoModelForCausalLM.from_pretrained(
           model_name,
           quantization_config=bnb_config,
           device_map="auto",
           torch_dtype=torch.bfloat16,
       )
       tokenizer = AutoTokenizer.from_pretrained(model_name)
       tokenizer.pad_token = tokenizer.eos_token

       model = prepare_model_for_kbit_training(model)

       lora_config = LoraConfig(
           r=lora_r,
           lora_alpha=lora_alpha,
           target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
           lora_dropout=lora_dropout,
           bias="none",
           task_type="CAUSAL_LM",
       )
       model = get_peft_model(model, lora_config)

       # Format input: "Dialektua: {dialect}\nTestua: {text}\n---"
       # For classification, use causal LM with label in prompt
       # Alternative: add a classification head (modify model forward)

       # ... training code similar to Task 2.3 but with causal LM formatting

       model.save_pretrained(output_dir)
       tokenizer.save_pretrained(output_dir)
   ```
2. **Critical design choice:** Latxa is a causal LM, not an encoder. For classification, two options:
   - **Option A (recommended):** Use prompt-based classification. Format: `"Sailkatu euskalki honen arabera: {text}\nEuskalkia:"` and extract the next-token probability over dialect labels. This is few-shot compatible but less accurate than fine-tuning a classification head.
   - **Option B:** Add a classification head by extracting the last hidden state of the final token and passing through a linear layer. This requires modifying the model's forward pass but gives best accuracy.
   - **Decision:** Implement Option A first (simpler, no model modification). If accuracy is insufficient (< fastText), implement Option B.
3. Create config `configs/text/latxa.yaml`.
4. Monitor GPU memory during training. If OOM, reduce `batch_size` to 1–2, increase gradient accumulation, reduce `max_length` to 128, and/or reduce `lora_r` to 8.
5. Train first on 3-class, then 6-class if data allows.
6. Save LoRA adapter only (not full model weights) to save disk space.

**Expected output:** `models/latxa_dialect/` (LoRA adapter + tokenizer), training log in WandB.

**Validation:**
- OOM does not occur with the specified config.
- 3-class accuracy > 85% (should beat XLM-R).
- LoRA adapter is ≤ 50 MB (easy to distribute).

---

### Task 3.2 — Data Augmentation: Latxa-Based Dialect Style Transfer

**Objective:** Use Latxa to generate synthetic dialectal text from Batua, addressing data scarcity for Zuberera and Navarrese.

**Concrete actions:**
1. Implement `src/augmentation/text_aug.py`:
   ```python
   def generate_dialectal_paraphrase(
       text: str,
       target_dialect: str,
       model,
       tokenizer,
       num_return: int = 5,
   ) -> list[str]:
       """Use Latxa to paraphrase Batua text into a target dialect."""
       prompt = (
           f"Idatzi testu hau {DIALECT_NAMES[target_dialect]} euskalkian, "
           f"esanahi bera mantenduz, baina lexiko, morfologia eta ortografia "
           f"aldatuz:\n\nTestua: {text}\n\n{DIALECT_NAMES[target_dialect]} euskalkian:"
       )
       # Generate with temperature 0.8, top_p=0.9
       # Filter outputs: check they differ from input (>20% char-level edit distance)
       # Filter outputs: run fastText LID to verify they're still Basque
       pass

   def create_augmented_dataset(
       source_dataset,
       target_dialect: str,
       num_augment: int = 500,
   ) -> Dataset:
       """Generate augmented samples for a target dialect."""
       pass
   ```
2. Generate 500 augmented samples for each low-resource dialect (Zuberera, Navarrese, Navarrese-Labourdin).
3. Validate quality: have a native speaker review 50 generated samples per dialect. Rate fluency (1–5) and dialect authenticity (1–5). Discard if mean rating < 3.5.
4. Run an ablation study: train model with and without augmented data; measure accuracy delta.
5. Only ship augmented data if ablation shows improvement ≥2% accuracy on the target dialect.

**Expected output:** `data/augmented/text/dialect_paraphrases.jsonl`, ablation report.

**Validation:**
- Generated samples are syntactically valid Basque (fastText confidence > 0.9).
- Ablation shows positive or neutral impact on test accuracy.
- If ablation shows negative impact, discard augmentation and document as a failed experiment.

---

### Task 3.3 — Multi-Label Output & Threshold Tuning

**Objective:** Implement multi-label classification via probability thresholding.

**Concrete actions:**
1. Modify model inference to return per-class probabilities (softmax) instead of argmax.
2. Implement `src/evaluation/multilabel.py`:
   ```python
   def find_optimal_thresholds(
       probs: np.ndarray,     # shape: [n_samples, n_classes]
       y_true: np.ndarray,    # shape: [n_samples, n_classes] multi-hot
       metric: str = "macro_f1",
   ) -> dict[str, float]:
       """Per-class threshold optimization using validation set."""
       # Grid search or Bayesian optimization over per-class thresholds
       pass

   def predict_multilabel(
       probs: np.ndarray,
       thresholds: dict[str, float],
   ) -> np.ndarray:
       """Apply per-class thresholds to get multi-label predictions."""
       pass
   ```
3. Use the annotated dev set (Task 1.4) which includes multi-label annotations for ambiguous sentences.
4. Optimize thresholds on dev; evaluate on test.
5. Report: multi-label accuracy, macro F1, Hamming loss, exact match ratio.

**Expected output:** `src/evaluation/multilabel.py`, threshold config file, multi-label metrics in WandB.

**Validation:**
- Multi-label macro F1 > 0.55 (per success criteria).
- Threshold values are interpretable (e.g., `western: 0.4`, `souletin: 0.3`).

---

### Task 3.4 — Text Model Finalization & Release

**Objective:** Select final Phase 1 model, package it, and release on Hugging Face.

**Concrete actions:**
1. Select best model from all text experiments (fastText, UniLID, XLM-R, Latxa) based on test set metrics.
2. Create Hugging Face model card using template from `src/utils/model_card_template.md`:
   ```markdown
   ---
   language: eu
   tags: [basque, dialect-identification, euskara, euskalkiak]
   datasets: [xnli-dialectal, basphycowest]
   metrics: [accuracy, f1]
   ---
   # Zeineuski — Basque Dialect Identification (Text)
   ...
   ```
3. Push model to Hugging Face Hub:
   ```bash
   huggingface-cli upload zeineuski/zeineuski-text-v1 models/latxa_dialect/ --private
   ```
4. Create dataset card for the curated dialect corpus and push as `zeineuski/zeineuski-text-corpus`.
5. Write inference example in README.

**Expected output:** Model on Hugging Face Hub, dataset on Hugging Face Hub, updated README.

**Validation:**
- `model = AutoModel.from_pretrained("zeineuski/zeineuski-text-v1")` works.
- Inference example produces correct output for a known test sentence.

---

## Epic 4: Speech Data Pipeline

### Task 4.1 — Inventory & Download Speech Resources

**Objective:** Gather all known Basque dialect speech resources.

**Concrete actions:**
1. Create `data/raw/speech/` directory.
2. Identify and access resources:

   | Source | Dialect Coverage | Hours (est.) | Access Method |
   |---|---|---|---|
   | Ahotsak.eus | All 5 dialects + sub-dialects | 700+ hours (7k interviews) | Contact Badihardugu Association for research access |
   | Mintzoak.eus | Navarrese-Labourdin, Souletin | 50+ hours (1.2k recordings) | Check usage terms at mintzoak.eus/eu/erabilpen-baldintzak/ |
   | **euskalkiak.eus** (ikus-entzunezkoak) | All 5 dialects | Small (video clips per dialect) | **CC BY 4.0** — freely usable; scrape or contact koldo.zuazo@gmail.com |
   | Mozilla Common Voice (Basque) | Primarily Batua, some regional | 100+ hours validated | `datasets.load_dataset("mozilla-foundation/common_voice_17_0", "eu")` |
   | Basque Parliament Speech Corpus | Batua (90%+), some dialectal | 1,400+ hours | Publicly available; check with HiTZ/Aholab |
   | OpenSLR SLR76 | Batua (multi-speaker) | ~10 hours | `wget https://www.openslr.org/resources/76/` |

3. Create `data/raw/speech/registry.csv` with columns: `source, dialect, num_utterances, total_hours, format, license, download_status`.
4. For Ahotsak.eus: investigate scraping feasibility (respect robots.txt and ToS). If scraping, estimate time and proceed only if < 1 week of effort. Otherwise, file formal research access request to Badihardugu.
5. For Mintzoak.eus: check the site's usage terms (`erabilpen baldintzak`). If downloadable, scrape or request bulk access from EKE (Basque Cultural Institute).

**Expected output:** `data/raw/speech/registry.csv`, downloaded audio files for at least 2 sources.

**Validation:**
- At least Common Voice + Parliament downloaded.
- Ahotsak access plan documented (research request filed or scraping code written).
- Each dialect has some audio in the registry (even if from Ahotsak only).

---

### Task 4.2 — Audio Preprocessing Pipeline

**Objective:** Build a preprocessing module that resamples, chunks, performs VAD, and formats audio for training.

**Concrete actions:**
1. Implement `src/data/speech_preprocessing.py`:

   | Function | Purpose |
   |---|---|
   | `resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray` | Resample to 16 kHz mono using librosa. |
   | `vad_split(audio: np.ndarray, sr: int, min_duration: float = 1.0, max_duration: float = 15.0) -> list[np.ndarray]` | Voice activity detection using Silero VAD or energy-based threshold; split into utterances. |
   | `remove_silence(audio: np.ndarray, sr: int, threshold_db: float = -40) -> np.ndarray` | Trim leading/trailing silence. |
   | `normalize_volume(audio: np.ndarray, target_db: float = -23) -> np.ndarray` | LUFS or RMS normalization. |
   | `extract_features(audio: np.ndarray, sr: int, feature_type: str = "mfcc") -> np.ndarray` | For ECAPA-TDNN baseline: extract MFCCs. |
   | `prepare_whisper_dataset(audio_paths: list[str], labels: list[str], output_path: str) -> None` | Create Whisper-compatible dataset (audio array + label). |

2. Implement `src/data/speech_loader.py`:
   - `load_audio_dataset(sources: list[str], max_duration: float = 15.0) -> Dataset`: Load from registry, preprocess, return Hugging Face `Dataset` with columns `audio` (sampling_rate + array) and `label`.
   - `create_speaker_disjoint_splits(dataset, test_size=0.15, val_size=0.10) -> DatasetDict`: Ensure no speaker appears in both train and test.

3. Create config `configs/speech/preprocessing.yaml`:
   ```yaml
   target_sample_rate: 16000
   min_duration_sec: 1.0
   max_duration_sec: 15.0
   vad_method: silero     # or "energy"
   normalize_volume: true
   target_lufs: -23
   ```

**Expected output:** `src/data/speech_preprocessing.py`, `src/data/speech_loader.py`, `configs/speech/preprocessing.yaml`.

**Validation:**
- Pipeline runs on a sample of 100 Common Voice utterances.
- Output dataset has correct schema (`audio`, `label`).
- VAD splits result in ≥1 valid chunk per long utterance.
- Speaker-disjoint split: no speaker ID appears in both train and test.

---

## Epic 5: Speech Models (Baselines)

### Task 5.1 — ECAPA-TDNN Baseline

**Objective:** Train an ECAPA-TDNN speaker embedding model adapted for dialect classification.

**Concrete actions:**
1. Install `speechbrain`:
   ```bash
   uv add speechbrain
   ```
2. Implement `src/models/speech/ecapa_tdnn.py` using SpeechBrain's ECAPA-TDNN:
   ```python
   from speechbrain.inference.speaker import EncoderClassifier

   def train_ecapa_tdnn(
       train_manifest: str,
       val_manifest: str,
       output_dir: str,
       num_classes: int = 6,
   ):
       """Fine-tune ECAPA-TDNN for dialect classification."""
       # SpeechBrain recipes expect CSV manifests:
       # ID,duration,wav_path,label
       # Use SpeechBrain's ECAPA recipe from voxceleb and adapt
       pass
   ```
3. Or, use a simpler approach: extract ECAPA-TDNN embeddings and train an SVM/MLP classifier on top (simpler, faster, comparable performance for DID tasks).
4. Create config `configs/speech/ecapa.yaml`.
5. Train and evaluate on speaker-disjoint splits.

**Expected output:** `models/ecapa_dialect/`, evaluation report.

**Validation:**
- Accuracy > 60% (3-class) on held-out test set.
- SpeechBrain imports and runs without errors.

---

### Task 5.2 — Whisper Fine-Tuning for Dialect ID

**Objective:** Fine-tune OpenAI Whisper (medium or large-v3) as a dialect classifier.

**Concrete actions:**
1. Implement `src/models/speech/whisper_did.py`:
   ```python
   from transformers import (
       WhisperFeatureExtractor, WhisperProcessor,
       WhisperForConditionalGeneration, Trainer, TrainingArguments
   )

   class WhisperDialectClassifier:
       def __init__(self, model_name: str = "openai/whisper-medium"):
           self.processor = WhisperProcessor.from_pretrained(model_name)
           self.model = WhisperForConditionalGeneration.from_pretrained(model_name)

       def freeze_encoder(self):
           """Freeze Whisper encoder, train only classification head."""
           for param in self.model.model.encoder.parameters():
               param.requires_grad = False

       def add_classification_head(self, num_classes: int):
           """Replace the decoder with a classification head."""
           hidden_size = self.model.config.d_model
           self.classifier = torch.nn.Sequential(
               torch.nn.Linear(hidden_size, hidden_size // 2),
               torch.nn.ReLU(),
               torch.nn.Dropout(0.1),
               torch.nn.Linear(hidden_size // 2, num_classes),
           )

       def forward(self, input_features, **kwargs):
           encoder_outputs = self.model.model.encoder(input_features)
           # Mean pool over time dimension
           pooled = encoder_outputs.last_hidden_state.mean(dim=1)
           logits = self.classifier(pooled)
           return logits
   ```
2. **Alternative approach (simpler):** Use the Whisper encoder output directly as fixed features and train a light classifier (SVM/MLP). This avoids modifying the Whisper forward pass but may underperform fine-tuning.
   - **Decision:** Start with frozen encoder + MLP classifier (fast training, low GPU). If accuracy < ECAPA-TDNN, switch to full fine-tuning.
3. Create config `configs/speech/whisper.yaml`.
4. Handle variable-length audio: pad/truncate to 30 seconds for Whisper.
5. Train and evaluate.

**Expected output:** `models/whisper_did/`, evaluation report.

**Validation:**
- 3-class accuracy > 65% on held-out test set.
- Training completes within 24 hours on single GPU.

---

### Task 5.3 — XLSR Fine-Tuning for Dialect ID

**Objective:** Fine-tune XLSR (wav2vec 2.0 multilingual) following the Frisian dialect paper's approach.

**Concrete actions:**
1. Implement `src/models/speech/xlsr_did.py`:
   ```python
   from transformers import (
       Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification,
       Trainer, TrainingArguments
   )

   def train_xlsr(
       dataset,
       model_name: str = "facebook/wav2vec2-xls-r-300m",
       num_labels: int = 6,
       output_dir: str = "models/xlsr_dialect",
       batch_size: int = 8,
       lr: float = 3e-5,
       epochs: int = 10,
   ):
       feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
       model = Wav2Vec2ForSequenceClassification.from_pretrained(
           model_name, num_labels=num_labels
       )
       # Freeze CNN feature extractor (first few layers)
       model.freeze_feature_extractor()

       # ... standard Trainer setup
   ```
2. **Multi-task option (Frisian approach):** Add an auxiliary CTC head for ASR to improve representations. This requires transcriptions for training data. If transcriptions available (Common Voice, Parliament), implement multi-task training with weighted loss: `L = L_did + λ * L_ctc`.
3. Create config `configs/speech/xlsr.yaml`.
4. Train on 3-class first, then 6-class.

**Expected output:** `models/xlsr_did/`, optionally with multi-task ASR+DID head.

**Validation:**
- 3-class accuracy > 70%.
- Training converges within 48 hours.

---

### Task 5.4 — Speech Baselines Comparison

**Objective:** Compare ECAPA-TDNN, Whisper, and XLSR; select best.

**Concrete actions:**
1. Evaluate all 3 models on the same speaker-disjoint test set.
2. Generate comparison table (same format as Task 2.4).
3. Select winner based on accuracy, training time, and model size.
4. Document in `docs/speech_baseline_comparison.md`.

**Expected output:** Comparison report, WandB report.

**Validation:** All 3 evaluated on identical test split.

---

## Epic 6: Speech Models (Advanced)

### Task 6.1 — CTC-DID Implementation

**Objective:** Implement CTC-based dialect ID for streaming-capable inference (Farooq & Saz, 2026).

**Concrete actions:**
1. Implement `src/models/speech/ctc_did.py`:
   - Frame dialect ID as a limited-vocabulary ASR task where dialect tags are treated as output labels.
   - Use the SSL encoder (XLSR or Whisper encoder) as feature extractor.
   - CTC head outputs probabilities over dialect labels.
   - During training, repeat dialect label to match encoder output length (using the Language-Agnostic Heuristic from the paper).
2. Reference implementation approach from arXiv:2601.12199.
3. Train and compare against non-CTC models.
4. Test streaming mode: chunk audio, run inference on each chunk, aggregate predictions.

**Expected output:** `models/ctc_did/`, streaming inference demo script.

**Validation:**
- Accuracy within 5% of best non-CTC model.
- Streaming mode produces stable predictions across chunks (variance < 0.1 in predicted probability).

---

### Task 6.2 — Voice Conversion Augmentation

**Objective:** Augment low-resource dialect speech using voice conversion.

**Concrete actions:**
1. Research available voice conversion tools: YourTTS, FreeVC, or Hugging Face VC models.
2. Implement `src/augmentation/speech_aug.py`:
   ```python
   def convert_speaker_to_dialect(
       audio: np.ndarray,
       source_speaker: str,
       target_dialect_speaker: str,
       vc_model,
   ) -> np.ndarray:
       """Convert speech from source speaker to sound like target dialect speaker."""
       # This preserves linguistic content but changes speaker characteristics
       # Not true dialect conversion — augments speaker diversity
       pass

   def add_background_noise(audio: np.ndarray, noise_path: str, snr_db: float) -> np.ndarray:
       """Mix audio with background noise at specified SNR."""
       pass

   def speed_perturb(audio: np.ndarray, sr: int, factor: float = 0.9) -> np.ndarray:
       """Slight speed perturbation for augmentation."""
       pass
   ```
3. Primary augmentation strategy: speed perturbation (0.9×, 1.0×, 1.1×), noise injection, and SpecAugment (frequency/time masking). These are standard for speech and do not require VC.
4. VC-based augmentation (if implemented): apply only to Zuberera where data is most scarce.
5. Run ablation: train with and without augmentation.

**Expected output:** `src/augmentation/speech_aug.py`, augmented dataset, ablation report.

**Validation:** Augmentation improves Zuberera per-class recall by ≥2%.

---

### Task 6.3 — Multi-Task ASR + DID

**Objective:** Implement joint ASR and dialect ID training (Frisian approach, arXiv:2502.04883).

**Concrete actions:**
1. Modify XLSR or Whisper model to have two heads: CTC for ASR + classification for DID.
2. Weighted loss: `total_loss = did_loss + 0.3 * ctc_loss`.
3. Requires transcriptions for training data. Use Common Voice + Parliament (both have transcriptions).
4. Train and evaluate; compare against DID-only training.
5. **Decision gate:** Only implement if transcriptions are available for at least 50% of training utterances. Otherwise skip and document.

**Expected output:** Multi-task model, comparison report vs. DID-only.

**Validation:** Multi-task model has ≥ same DID accuracy as DID-only model; ASR WER on Common Voice test set < 30%.

---

### Task 6.4 — Speech Model Finalization & Release

**Objective:** Package and release the best speech model.

**Concrete actions:**
1. Select best model from all speech experiments.
2. Create Hugging Face model card.
3. Push to Hub: `zeineuski/zeineuski-speech-v1`.
4. Create dataset card for curated speech corpus.
5. Document known limitations (e.g., "performs worst on Zuberera due to data scarcity").

**Expected output:** Speech model on Hugging Face Hub.

**Validation:**
- Loading and inference works: `model = AutoModel.from_pretrained("zeineuski/zeineuski-speech-v1")`.
- Inference example documented.

---

## Epic 7: Evaluation, Integration & Release

### Task 7.1 — Unified CLI

**Objective:** Build a single command-line interface for both text and speech inference.

**Concrete actions:**
1. Implement `src/cli.py` using `click` or `argparse`:
   ```bash
   # Text inference
   uv run zeineuski predict --text "Gaur goizean goiz jaiki naiz"
   # → {"dialect": "central", "probabilities": {"western": 0.05, "central": 0.82, ...}}

   # Speech inference
   uv run zeineuski predict --speech audio.wav
   # → {"dialect": "western", "probabilities": {...}}

   # Batch mode
   uv run zeineuski predict --text-file input.txt --output results.jsonl

   # Multi-label mode
   uv run zeineuski predict --text "..." --multilabel --threshold 0.3
   ```
2. Implement `src/inference.py`:
   - `load_model(modality: str, model_path: str) -> Any`: Load text or speech model.
   - `predict_text(text: str, model, multilabel: bool = False) -> dict`: Run text inference.
   - `predict_speech(audio_path: str, model) -> dict`: Run speech inference.
3. Add `[project.scripts]` to `pyproject.toml`:
   ```toml
   [project.scripts]
   zeineuski = "src.cli:main"
   ```

**Expected output:** `src/cli.py`, `src/inference.py`, working CLI.

**Validation:**
- `uv run zeineuski predict --text "kaixo"` returns valid JSON.
- `uv run zeineuski predict --speech tests/fixtures/sample.wav` returns valid JSON.
- Batch mode processes 100 sentences in < 5 seconds.

---

### Task 7.2 — Final Evaluation Report

**Objective:** Produce a comprehensive evaluation comparing all models and documenting findings.

**Concrete actions:**
1. Create `docs/evaluation_report.md`:
   - Executive summary: best model, key metrics, main findings.
   - Per-model detailed results (accuracy, F1, confusion matrix, per-class metrics).
   - Ablation studies (data augmentation impact, multi-label vs single-label, 3-class vs 6-class).
   - Error analysis: 20 most-confused sample pairs, patterns observed.
   - Comparison to published baselines (fastText, ECAPA-TDNN).
   - Known limitations and future work.
2. Generate all figures: confusion matrix heatmaps, per-class F1 bar charts, training curves.
3. Log final report to WandB.

**Expected output:** `docs/evaluation_report.md`, figures in `docs/figures/`.

**Validation:** Report includes all models, all metrics, error analysis.

---

### Task 7.3 — Dataset & Model Cards

**Objective:** Create responsible AI documentation for datasets and models.

**Concrete actions:**
1. Create dataset cards for:
   - `zeineuski-text-corpus`: sources, dialects, sizes, preprocessing, known biases, licensing.
   - `zeineuski-speech-corpus`: same for speech.
2. Create model cards for:
   - `zeineuski-text-v1`: architecture, training data, metrics, limitations, intended use.
   - `zeineuski-speech-v1`: same for speech.
3. Use Hugging Face model card template. Include:
   - Dataset composition (dialect distribution, speaker demographics if known).
   - Evaluation results (all metrics).
   - Bias assessment: which dialects/demographics perform worst.
   - Out-of-scope uses (not for speaker identification, not for content moderation).
4. Push cards to Hugging Face Hub alongside models.

**Expected output:** Dataset cards, model cards on Hugging Face Hub.

**Validation:** Cards follow Hugging Face template; all fields completed; bias section substantive.

---

### Task 7.4 — Write-Up / Paper Draft

**Objective:** Document the project as a reproducible research paper.

**Concrete actions:**
1. Write initial draft in LaTeX or Markdown (target: ACL/VarDial workshop format).
2. Structure:
   - Abstract
   - Introduction (Basque dialect identification gap)
   - Related Work (Arabic ADI, Frisian, CHALIS, UniLID)
   - Data (sources, statistics, preprocessing)
   - Method (model architectures, training setup)
   - Experiments (baselines, advanced models, ablation)
   - Results & Analysis
   - Conclusion & Future Work
3. Include all figures and tables from evaluation report.
4. Release on arXiv as preprint.
5. Submit to VarDial 2027 or similar workshop.

**Expected output:** Paper draft in `docs/paper/`, arXiv submission.

**Validation:** Draft has all sections; results are reproducible from configs/checkpoints.

---

## Task Dependencies

```
Phase 0 (Foundation)
  0.1 Repo Bootstrap
  0.2 Experiment Tracking
  0.3 CI Pipeline
      │
Phase 1 (Text Data)          Phase 2 (Text Baselines)
  1.1 Data Inventory ──────────► 2.1 fastText
  1.2 Dialect Labeling ────────► 2.2 UniLID
  1.3 Text Preprocessing ──────► 2.3 XLM-R
  1.4 Manual Annotation ───────► 2.4 Baselines Comparison
                                      │
Phase 3 (Text Advanced)               │
  3.1 Latxa Fine-Tuning ◄─────────────┘
  3.2 Data Augmentation
  3.3 Multi-Label Tuning ──────┐
  3.4 Text Finalization ◄──────┘
      │
Phase 4 (Speech Data)          Phase 5 (Speech Baselines)
  4.1 Speech Inventory ─────────► 5.1 ECAPA-TDNN
  4.2 Audio Preprocessing ──────► 5.2 Whisper
                                  5.3 XLSR
                                  5.4 Speech Baselines Comparison
                                      │
Phase 6 (Speech Advanced)              │
  6.1 CTC-DID ◄────────────────────────┘
  6.2 Voice Conversion Aug.
  6.3 Multi-Task ASR+DID
  6.4 Speech Finalization ──────┐
                                │
Phase 7 (Release)               │
  7.1 Unified CLI ◄─────────────┴── 3.4 + 6.4
  7.2 Final Evaluation Report
  7.3 Dataset & Model Cards
  7.4 Write-Up
```

---

## Testing Strategy

### Unit Tests
| Module | What to Test | Framework |
|---|---|---|
| `src/data/text_preprocessing.py` | `clean_text`, `normalize_basque`, `deduplicate`, `filter_length`, `filter_language` | `pytest` |
| `src/data/text_loader.py` | `load_labeled_data`, `create_splits` | `pytest` |
| `src/data/speech_preprocessing.py` | `resample_audio`, `vad_split`, `normalize_volume` | `pytest` |
| `src/data/speech_loader.py` | `load_audio_dataset`, `create_speaker_disjoint_splits` | `pytest` |
| `src/models/text/*.py` | Model loads, forward pass returns expected shape, predict returns valid output | `pytest` |
| `src/models/speech/*.py` | Same for speech models | `pytest` |
| `src/evaluation/metrics.py` | `compute_metrics` returns expected dict shape | `pytest` |
| `src/evaluation/multilabel.py` | `find_optimal_thresholds`, `predict_multilabel` | `pytest` |
| `src/augmentation/text_aug.py` | Augmented text is valid Basque, differs from input | `pytest` |
| `src/augmentation/speech_aug.py` | Augmented audio has correct shape, duration | `pytest` |
| `src/cli.py` | CLI commands return 0 exit code, valid JSON output | `pytest` |

### Integration Tests
| Scenario | What to Verify |
|---|---|
| Text pipeline end-to-end | Raw text → preprocessed → model prediction → JSON output |
| Speech pipeline end-to-end | Raw audio → preprocessed → model prediction → JSON output |
| Data augmentation integration | Augmented data → training → evaluation (ablation) |
| Hugging Face Hub upload | `push_to_hub` succeeds; model loads back correctly |
| Multi-label pipeline | Text → probabilities → threshold → multi-label output |

### End-to-End Tests
| Scenario | What to Verify |
|---|---|
| Full text training run (fastText) | Completes in < 1 hour; model saves; evaluation metrics exist |
| Full speech training run (Whisper, small dataset) | Completes in < 12 hours; model saves; evaluation metrics exist |
| CLI predict on known examples | Output matches expected dialect for 5 hand-picked ground-truth examples |
| Reproducibility | Two runs with same seed produce identical metrics |

---

## Rollout Strategy

### Internal Milestones
1. **Week 2:** Text data pipeline functional; development environment working.
2. **Week 5:** All text baselines trained and compared.
3. **Week 8:** Phase 1 text model released on Hugging Face.
4. **Week 11:** All speech baselines trained and compared.
5. **Week 14:** Phase 2 speech model released on Hugging Face.
6. **Week 16:** Final release with CLI, paper, complete documentation.

### Release Checklist
- [ ] All models pushed to Hugging Face Hub with model cards.
- [ ] All datasets pushed with dataset cards.
- [ ] CLI works: `uv run zeineuski predict --text "test sentence"`.
- [ ] README has quickstart that runs in < 5 commands.
- [ ] CI is green on `main`.
- [ ] All tests pass (`uv run pytest`).
- [ ] Paper draft submitted to arXiv.
- [ ] WandB project is public (or shared with collaborators).
- [ ] Documentation includes known limitations and failure modes.

---

## Monitoring & Observability

### During Training
- **WandB dashboard** for all runs: loss curves, learning rate, GPU utilization, gradient norms.
- **Alert** if loss plateaus for > 3 epochs → early stopping or LR reduction.
- **Alert** if GPU memory > 90% sustained → reduce batch size.
- **Log** per-epoch metrics to WandB: accuracy, F1, confusion matrix.

### During Inference
- **Log** prediction latency per sample (p50, p95, p99).
- **Log** prediction confidence distribution (histogram).
- **Flag** predictions with confidence < 0.5 for manual review.

### Data Drift
- **Monitor** dialect distribution in input data over time (if deployed).
- **Monitor** average text length and character distribution.
- Alert if input distribution shifts significantly from training distribution (KL divergence > threshold).

---

## Risks, Unknowns, and Fallbacks

| Risk | Fallback |
|---|---|
| **UniLID implementation too complex** | Skip; use XLM-R as primary neural baseline. Document that UniLID was attempted and abandoned. |
| **Latxa 7B OOM even with QLoRA** | Use `hitz-zentroa/Latxa-1.4b` or `Latxa-2.7b` smaller variants; if unavailable, fall back to XLM-R. |
| **Ahotsak.eus data not accessible** | Use Common Voice + Parliament + Mintzoak only. Accuracy target lowered to 55% 3-class for speech. Document gap. |
| **Zuberera data insufficient (< 50 samples)** | Drop Zuberera from 6-class; rename to 5-class (Navarrese-Labourdin absorbs any Zuberera samples). |
| **Whisper does not support Basque well** | Verified: Whisper large-v3 supports Basque. If WER on Common Voice > 30%, abandon Whisper and use only XLSR. |
| **Multi-label annotations not obtained** | Ship single-label only; document multi-label as future work. |
| **VC augmentation degrades quality** | Drop VC; use only SpecAugment + speed perturbation + noise injection. |
| **CTC-DID underperforms baseline** | Skip CTC-DID; document as attempted but not beneficial for Basque. |

---

## Definition of Done

A task is **done** when:

1. **Code** is committed to `main` (or merged via PR).
2. **Tests pass**: `uv run pytest tests/ -v` is green.
3. **Linting passes**: `uv run ruff check src/` is clean.
4. **Training completed** (if ML task): model weights saved, metrics logged to WandB.
5. **Documentation updated**: relevant sections of README or docs/ reflect the change.
6. **Config committed**: the YAML config used for the experiment is in `configs/`.
7. **Review** (if applicable): another developer has reviewed the PR.

The **project** is done when all tasks in Phase 7 are complete and the release checklist is fully ticked.

---

## Appendix: Quick Reference

### Dialect Label Mapping
```python
# 6-class variant (Zuazo's 5 + Batua)
# NOTE: navarrese and nav-lab are separate classes here — do NOT share the same index.
DIALECT_LABELS_6CLASS = {
    "western": 0,    # Bizkaiera
    "central": 1,    # Gipuzkera
    "navarrese": 2,  # Nafarrera
    "nav-lab": 3,    # Nafarrera-Lapurtera (Navarrese-Labourdin)
    "souletin": 4,   # Zuberera
    "batua": 5,      # Standard Basque
}
# 3-class variant (fallback if data insufficient for 6-class)
DIALECT_LABELS_3CLASS = {
    "western": 0,
    "central": 1,
    "nav-lab": 2,    # groups Navarrese + Navarrese-Labourdin + Souletin
}
# Alias for backwards compatibility — use 6-class by default
DIALECT_LABELS = DIALECT_LABELS_6CLASS
```

> **Note (bug fix):** An earlier version of this file incorrectly assigned both `navarrese` and `nav-lab` to index `2` within the 6-class dict. This has been corrected — they are now distinct indices (2 and 3). The 3-class grouping still maps all Northern varieties to index `2`.

### Minimum Viable Data Requirements
| Dialect | Text (sentences) | Speech (utterances) |
|---|---|---|
| Western (Bizkaiera) | 500 | 100 |
| Central (Gipuzkera) | 500 | 100 |
| Navarrese-Labourdin | 300 | 80 |
| Navarrese | 200 | 50 |
| Souletin (Zuberera) | 100 | 30 |
| Batua | 1000 | 500 |
| **Total minimum** | **2,600** | **860** |

If any dialect falls below the minimum, merge into the 3-class grouping and document the gap.
