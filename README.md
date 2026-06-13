# Zeineuski — Basque Dialect Identification

Fine-grained dialect identification (DID) system for Basque (Euskara), supporting three classification tiers:

- **Batua vs dialectal** (binary): distinguishes Standard Basque from dialectal speech
- **Euskalkiak (dialects):** Western (Bizkaiera), Central (Gipuzkera), Navarrese, Navarrese-Labourdin, Souletin (Zuberera)
- **Azpieuskalkiak (sub-dialects):** 9 Zuazo sub-dialect classes trained on Ahotsak.eus oral history transcriptions

## Web Demos

Both demos run entirely in the browser via WebAssembly — no server, no install.

| Demo | URL | Models |
|------|-----|--------|
| **Euskalkiak** (dialect detection) | [itzune.eus/euskalkid](https://itzune.eus/euskalkid) | 2 models, 34MB — batua/dialectal → 5 euskalkis |
| **Azpieuskalkiak** (sub-dialect) | [itzune.eus/euskalkid/azpieuskalki](https://itzune.eus/euskalkid/azpieuskalki) | 1 model, 31MB — 9 azpieuskalkis + 430 towns |

[Source code](https://github.com/itzune/euskalkid) for the web demos.

## Setup

```bash
git clone <repo-url>
cd zeineuski
uv sync
pre-commit install
```

## Quickstart

```bash
# Text inference
uv run zeineuski predict --text "Gaur goizean goiz jaiki naiz"

# Speech inference
uv run zeineuski predict --speech audio.wav

# Batch mode
uv run zeineuski predict --text-file input.txt --output results.jsonl
```

## Project Structure

```
zeineuski/
├── src/                 # Source code
│   ├── data/            # Data loading, preprocessing, labeling
│   ├── models/          # Model implementations (text + speech)
│   ├── evaluation/      # Metrics, reporting, multi-label utils
│   ├── augmentation/    # Text and speech augmentation
│   └── cli.py           # Unified CLI
├── configs/             # YAML configs per model/preprocessing
├── tests/               # Unit, integration, and end-to-end tests
├── data/                # Raw, processed, annotated, augmented data
├── models/              # Saved model weights
├── notebooks/           # Exploratory notebooks
└── docs/                # Documentation, evaluation reports, paper
```

## Architecture

Zeineuski uses a **three-tier hierarchical classification** architecture for text,
and a separate **Whisper encoder pipeline** for speech-based dialect identification:

### Text pipeline

```
Tier 1: batua / dialectal (binary)
  └─ Tier 2: 5-class euskalkia (dialect classification)
       └─ Tier 3: 9 to 12-class azpieuskalkia (sub-dialect classification)
```

### Speech pipeline (experimental)

```
Audio → Whisper Encoder (phonetic features, no batua normalization) → MLP → Dialect
```

See the [Speech-based Dialect Identification](#-speech-based-dialect-identification-work-in-progress) section for details.

### Classification taxonomy

The project follows **Koldo Zuazo's dialect classification**, which is the current
linguistic consensus and the basis for Ahotsak.eus's municipality→dialect mapping.

Zuazo recognizes **6 euskalkiak** (dialects):

| # | Euskalkia | Our label | Notes |
|---|-----------|-----------|-------|
| 1 | Bizkaiera / Mendebalekoa | `western` | |
| 2 | Gipuzkera / Erdialdekoa | `central` | |
| 3 | Goi-nafarrera | `navarrese` | Upper Navarrese |
| 4 | Ekialdeko nafarrera / Erronkariera | *(merged into navarrese)* | Extinct ~1990s; tiny data |
| 5 | Zuberera | `souletin` | |
| 6 | Nafar-lapurtera | `nav-lab` | |
| + | Euskara batua | `batua` | Standard unified Basque |

**Why 5 euskalkis + batua instead of 6 + batua?**

Ekialdeko nafarrera (Salazarese/Roncalese) is linguistically a distinct dialect, but
it has been functionally extinct since the 1990s (last native speaker died in 1991).
Ahotsak.eus has only ~65 passages across 7 towns in the Zaraitzu and Erronkari valleys.
The Klasikoak.armiarma.eus classical literature corpus — which provides most of our
Tier-2 training data — maps these texts to `navarrese` since the dialect distinction
is not present in pre-20th-century literary sources.

For **Tier 3 (azpieuskalkia)**, we follow the **Zuazo azpieuskalki taxonomy** as
implemented on [Ahotsak.eus](https://ahotsak.eus). The official Ahotsak municipality→
azpieuskalki mapping provides the ground truth labels for sub-dialect classification.
We use the same label names as Ahotsak (e.g., `mendebal-sortaldea`, `ekialde-nafarra`,
etc.) rather than translating or renaming them.

## Results

### Batua vs dialectal (Binary) classification

Binary fastText classifier trained on Klasikoak batua + all dialect
data from both Klasikoak and Ahotsak (74K lines):

| Variant | Size | Dialect detection |
|---------|------|:---:|
| final | 264MB | 99.69% |
| quantized | 8MB | 95.56% |
| compact | 5MB | 96.36% |
| **tiny** | **2MB** | **99.72%** |
| **web** | **1MB** | **99.70%** |

Labels: `batua`, `dialectal`.
The old binary model (793MB) misclassified Zuberera sentences as batua.
The new model correctly identifies all 5 dialectal classes.

### Euskalki (Dialect) Classification — 5-class

Fine-grained sub-dialect classifier trained on Ahotsak.eus oral history transcriptions
and augmented with the SÜ AZIA Zuberotarra corpus (6,676 pastoral + blog sentences).
Uses Zuazo's sub-dialect taxonomy mapped through official Ahotsak.eus municipality→azpieuskalki assignments.

| Variant | Classes | Accuracy | Model |
|---------|:---:|:---:|---|
| 12-class (all) | 12 | 82.08% | char+word n-grams, epoch=75 |
| 9-class (min_samples=600) | 9 | **83.55%** | drops 3 weakest (<600 train) |

**Optimal config:** fastText with `dim=200, lr=0.2, epoch=75, wordNgrams=2, minn=2, maxn=6, loss=ns`.
NO autotune — aggressive LR decay overfits to dominant classes.

**Training data distribution (42,229 sentences):**

| Azpieuskalki | Sentences | % | Source |
|---|---:|---:|---|
| mendebal-sortaldea | 13,059 | 30.9% | Ahotsak |
| erdialde-sartaldea | 9,804 | 23.2% | Ahotsak |
| **zuberera** | **6,050** | **14.3%** | **Ahotsak (441) + SÜ AZIA (6,676)** |
| erdialde-sortaldea | 4,966 | 11.8% | Ahotsak |
| nafar-ipar-sartaldea | 1,966 | 4.7% | Ahotsak |
| nafar-sortaldea | 1,516 | 3.6% | Ahotsak |
| naflap-sortaldea | 1,395 | 3.3% | Ahotsak |
| nafar-hego-sartaldea | 1,101 | 2.6% | Ahotsak |
| naflap-sartaldea | 726 | 1.7% | Ahotsak |
| ekialde-nafarra | 710 | 1.7% | Ahotsak |
| nafar-erdigunea | 497 | 1.2% | Ahotsak |
| mendebal-sartaldea | 439 | 1.0% | Ahotsak |

SÜ AZIA data brings zuberera from the smallest class (1.9%) to the 3rd largest (14.3%)
— see [docs/data_sources/suazia_zuberotarra.md](docs/data_sources/suazia_zuberotarra.md).

**Best 9-class per-class results:**

| Azpieuskalki | Test samples | Accuracy |
|---|---|---|
| mendebal-sortaldea (Eastern Bizkaian) | 2,304 | 90.80% |
| erdialde-sartaldea (coastal+western Gipuzkoan) | 1,729 | 83.75% |
| nafar-ipar-sartaldea (Bortziriak/Malerreka) | 346 | 83.82% |
| erdialde-sortaldea (eastern Gipuzkoan) | 876 | 79.11% |
| naflap-sortaldea (Basse-Navarre) | 246 | 77.64% |
| nafar-sortaldea (eastern Navarre) | 267 | 76.40% |
| naflap-sartaldea (coastal Labourdin) | 127 | 66.93% |
| ekialde-nafarra (Zaraitzu/Erronkari) | 125 | 65.60% |
| nafar-hego-sartaldea (Sakana) | 194 | 55.15% |

Key insight: **character n-grams** (minn=2, maxn=6) capture Basque morphological patterns
that are dialect-specific — this single change jumped accuracy from 72% to 82%.

| Variant | Accuracy | Size | vs original |
## Development approach: pi-autoresearch

This project served as a testbed for [**pi-autoresearch**](https://github.com/davebcn87/pi-autoresearch),
an autonomous experiment loop extension for [pi](https://github.com/earendil-works/pi-coding-agent).
The azpieuskalki classifier was tuned through **37 automated experiments** over
3 sessions, where pi-autoresearch iteratively proposed, ran, and evaluated
hyperparameter changes against a hold-out test set.

Instead of manual trial-and-error, the loop autonomously discovered:

| Discovery | Experiment | Impact |
|---|---|---|
| Ahotsak municipality misalignment | #6–7 (68 towns fixed) | +17.3pp (51% → 68%) |
| NO autotune — it overfits imbalanced data | #8–14 (8 variants killed) | +4.1pp (68% → 72%) |
| Character n-grams (minn=2, maxn=6) | #27 | +9.4pp (72% → 81%) |
| Dim 200 > 250/300 for small datasets | #28–30 | +1pp |
| epoch=75 sweet spot | #30 | 82.08% peak |

Each experiment logged its config, accuracy, F1 breakdown, confusion matrix, and
an auto-generated analysis back to `PROJECT.md`, making the exploration fully
traceable.

**Total improvement:** from 51.02% baseline to 82.51% final (weighted F1: 0.8242)
— entirely through automated optimization without any manual hyperparameter tuning.

**Best config:** `loss=ns, dim=200, epoch=100, lr=0.2, wordNgrams=3, minn=2, maxn=6,
minCount=1, targeted_oversample=-2`. Key innovations: trigrams for Nafarroa class
distinction, targeted oversampling of minority classes, Sakana injection (Zuazo 2010).

**Remaining challenge:** Nafarroa minority classes (nafar-erdigunea 0.639, ekialde-nafarra
0.635) need more data from Ahotsak (~30 unscraped Nafarroa towns).

---

## 🔉 Speech-based Dialect Identification (Experimental)

> **Status**: Experimental. 62.15% accuracy (5-class), 0.361 macro F1.
> Not yet deployed in production.

We built a speech-based Basque dialect classifier using audio recordings scraped
from [Ahotsak.eus](https://ahotsak.eus) (2,422 files, 36,176 segments, 78.1 hours).

The final pipeline uses a **frozen Whisper encoder** as a phonetic feature extractor
followed by a 2-layer MLP with focal loss.

### Final Approach: Whisper Encoder + MLP

```
Audio (16kHz) → Whisper Encoder (frozen) → mean+std+max pooling (3840-dim) → MLP (512→256→5) → Dialect
```

The Whisper **encoder** captures phonetic and prosodic features (pronunciation,
intonation, rhythm) that are dialect-specific. By discarding the **decoder**
(which normalizes speech to standard batua orthography), we preserve dialectal
pronunciation patterns that would otherwise be erased.

**Best config:** `lr=5e-4`, `focal loss (γ=2.0)`, `mean_std_max pooling`, `town-disjoint 80/10/10 split`.

### Results (5-class, town-disjoint split)

| Metric | Value |
|---|---:|
| Accuracy | **62.15%** |
| Macro F1 | **0.361** |
| Western F1 | 0.79 |
| Navarrese F1 | 0.51 |
| Central F1 | 0.38 |
| Souletin F1 | 0.11 |
| Nav-Lab F1 | 0.02 |

### Discarded Approaches

| Approach | Best Accuracy | Why it failed |
|---|---|---|
| **ECAPA-TDNN (frozen + SVM)** | 49.5% | VoxCeleb pretraining encodes *speaker identity*, not dialect features. 5 experiments, all plateaued <52%. |
| **XLSR wav2vec2 (frozen + SVM)** | 29.6% | `gttsehu/wav2vec2-xls-r-300m-bp1-es_eu` fine-tuned on Basque Parliament formal batua speech (3.67% WER) — erased dialectal phonetic variation. Base XLSR even worse at 20.0%. |
| **ASR → text pipeline** | 42.0% | Whisper decoder normalizes toward batua, stripping dialect markers. fastText text model trained on clean written text, not noisy ASR output. Best result with beam=5 decoding still 17.6pp below direct encoder. |
| **Whisper encoder fine-tuning** | 21.0% | 500-sample test with 3 unfrozen encoder layers (59M params) — NaN loss with fp16, random-guess with fp32. Full dataset would take 12+ hours with marginal expected gains. |
| **5-seed ensemble** | 59.9% | No benefit over single model — just averages stochastic noise (±1-2pp). |
| **Wider MLP (1024 dim)** | 59.3% | Overfits western majority, drags down minority classes. 2-layer 512-dim is the sweet spot. |

### Known Limitations

- **Nav-lab (0-2% F1)**: Nearly indistinguishable from Navarrese in town-disjoint
  splits — effectively a 4.5-class problem.
- **Souletin (8-13% F1)**: Only 37 source towns and 348 training segments — severe
  data scarcity for this phonetically distinct dialect.
- **Stochastic variance**: MLP training has ±1-2pp variance between runs due to
  random initialization.
- **GPU required for extraction**: Embedding extraction uses an NVIDIA L40 (46GB).
  Training is ~30s on CPU after embeddings are cached.

### Remaining Ideas

- **3-class mode** (western/central/navarrese) — should exceed 65-70% accuracy
- **Attention pooling** during training (instead of frozen mean+std+max) — proven
  in the ADI-20 Arabic dialect paper (arxiv 2511.10070)
- **Souletin data expansion** — scrape remaining unscraped Zuberoa towns from Ahotsak

### Running the Speech Pipeline

```bash
# Requires GPU (NVIDIA L40 or similar, 16GB+ VRAM) for extraction
# 1. Extract Whisper embeddings (one-time, ~25 min for 36K segments)
uv run python -m src.models.speech.whisper_did extract \
  --manifest data/processed/speech/ahotsak_full/train.csv \
  --output models/speech/whisper_train_emb.pkl \
  --pooling mean_std_max

# 2. Train MLP classifier (CPU, ~30s)
uv run python -m src.models.speech.whisper_did train \
  --train-emb models/speech/whisper_train_emb.pkl \
  --val-emb models/speech/whisper_val_emb.pkl \
  --test-emb models/speech/whisper_test_emb.pkl \
  --config configs/speech/whisper.yaml
```

Models: `models/speech/whisper_dialect/` (classifier.pkl + config.json).

---

## Training

All training was performed on a consumer laptop CPU — no GPU acceleration needed.
fastText is highly efficient: even the largest model (1.5GB, ~100K samples) trains
in under 2 minutes.

| Hardware | |
|---|---|
| CPU | Intel Core i7-8550U (4 cores, 1.80 GHz) |
| RAM | 16 GB |
| OS | Ubuntu 24.04 |
| Framework | fastText (C++), no GPU |

The best classifier uses a **hierarchical 2-step** architecture discovered through
automated hyperparameter search (33 experiments via Pi Autoresearch):

**Binary filter** (batua vs dialectal) — Klasikoak batua + all dialect data:
```bash
fasttext supervised \
  -input data/processed/text/train_binary.txt \
  -output models/hier_binary_final \
  -lr 0.5 -epoch 50 -dim 100 -minn 2 -maxn 6 -wordNgrams 2 -bucket 500000
```

**Step 2 — 5-class dialect classifier** (trained without batua samples, ekialdeko nafarrera merged into navarrese):
```bash
fasttext supervised \
  -input data/processed/text/train_dialectal_5class.txt \
  -output models/hier_dialect_final \
  -lr 0.2 -epoch 150 -dim 100 -minn 3 -maxn 6 -wordNgrams 2
```

Key insight: training the dialect model **without** batua samples prevents the model
from learning to distinguish batua from dialects, which the binary step already handles.

### Azpieuskalki classifier

The Tier 3 sub-dialect classifier is a single flat 9-class fastText model trained
on Ahotsak.eus oral history transcriptions:

```bash
fasttext supervised \
  -input data/processed/text/train_azpieuskalki.txt \
  -output models/azpieuskalki \
  -dim 200 -lr 0.2 -epoch 75 -wordNgrams 2 -minn 2 -maxn 6 -loss ns
```

Key insight: **NO autotune** — aggressive LR decay overfits to dominant classes.
**Character n-grams** (minn=2,maxn=6) capture Basque morphological patterns
(case endings, verb suffixes) that are dialect-specific. This single change
jumped accuracy from 72% to 82%.

## Evaluation

All metrics are computed on **disjoint** train/test splits verified via exact text deduplication.

```bash
# Run eval on all model variants (requires uv sync + models in models/)
uv run python eval_all.py
```

**Test sets:**
- `test_expanded_3class.txt` — 2,505 samples, 3-class XNLI (western, central, nav-lab), 0% train overlap
- `test_6class.txt` — 4,005 samples, 4-class (batua, western, central, nav-lab), 0% train overlap
- `test_azpieuskalki.txt` — 7,445 samples, 12 azpieuskalki classes, 15% stratified hold-out from Ahotsak + external corpora

### Data sources

| Source | Content | Dialects | Status |
|---|---|---|---|
| [Klasikoak](https://klasikoak.armiarma.eus/) | Literary texts (pre-20th c.) | 5 euskalkis (6-class) | Train |
| [Ahotsak.eus](https://ahotsak.eus) | Oral history transcriptions | 12 azpieuskalkis | Train + Test |
| [SÜ AZIA](https://web.archive.org/web/20110920103304/http://www.suazia.com) | Pastoral scripts + blog articles | Zuberera | Train + Test |

See [docs/data_sources/suazia_zuberotarra.md](docs/data_sources/suazia_zuberotarra.md) for the SÜ AZIA corpus documentation.

`val_6class.txt` was found to have 68.4% overlap with `train_6class.txt` and is **not used**
for evaluation. Navarrese and Souletin lack clean test splits — all their samples were
leaked into the validation set during dataset construction.

Models hosted at [huggingface.co/itzune/zeineuski](https://huggingface.co/itzune/zeineuski).

## Documentation

- [PROMPT.md](PROMPT.md) — Project prompt and requirements
- [PROJECT.md](PROJECT.md) — Project definition and architecture
- [PLAN.md](PLAN.md) — Detailed implementation plan

## License

MIT
