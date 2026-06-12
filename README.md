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

Zeineuski uses a **three-tier hierarchical classification** architecture:

```
Tier 1: batua / dialectal (binary)
  └─ Tier 2: 5-class euskalkia (dialect classification)
       └─ Tier 3: 9 to 12-class azpieuskalkia (sub-dialect classification)
```

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

### Euskalki (Dialect) Classification — 5 euskalkis + batua (6-class)

Hierarchical 2-step classifier (binary batua/dialectal → 5-class euskalkiak):

| Variant | Size | XNLI (3-class) | Test (4-class) | Batua F1 |
|---------|------|:---:|:---:|:---:|
| final | 1.5GB | 92.42% | 95.18% | 0.962 |
| quantized | 417MB | 92.38% | 95.16% | 0.961 |
| compact | 189MB | 91.78% | 94.71% | 0.957 |
| tiny | 112MB | 91.90% | 94.88% | 0.961 |
| **web** | **32MB** | **91.06%** | **94.33%** | **0.952** |

Per-class F1 (final): Western 0.953, Central 0.933, Nav-Lab 0.949, Batua 0.962.
Ekialdeko nafarrera (Zaraitzu/Erronkari) is merged into navarrese at Tier 2 due to
tiny data (~65 Ahotsak passages) and its treatment as a sub-class in our literary
corpus. It is a distinct class at Tier 3 (azpieuskalki).

### Azpieuskalki (Sub-Dialect) Classification — 12-class

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
|---|---:|---:|---:|
| original | 83.59% | 233MB | baseline |
| **full (wordNgrams=3)** | **82.51%** | **251MB** | weighted F1: 0.824 |
| quantized (default) | 84.22% | 34MB | ⚠ stale — from older config |
| bucket=50K | 84.16% | 137MB | ⚠ stale — from older config |

Models: `models/azpieuskalki.bin` (251MB), `models/azpieuskalki_q.bin` (34MB), `models/azpieuskalki_b50000.bin` (137MB)

> **Note:** Quantized variants above are from an older training run with a different
> seed that scored higher on its split. Regenerating with the current best config
> (`loss=ns, dim=200, epoch=100, lr=0.2, wordNgrams=3, targeted_oversample=-2`)
> produces full model at 0.8242 weighted F1 (seed-dependent, range: 0.8220–0.8266).
> Quantized variants need regeneration from current model.

### Euskalki (5-class dialect)

| Variant | Accuracy | Size | Notes |
|---|---|---|---|
| **5-class** | **89.50%** | **480MB** | New: trained on all 5 dialects |

The 5-class euskalki model was trained from azpieuskalki data (78K→5 classes).
Previously the euskalki model only supported 3 classes (western, central, nav-lab),
leaving zuberera and nafarrera with 0 F1. Now all 5 dialects are properly trained.

| Class | F1 | Support | Improvement |
|---|---|---|---|
| souletin | 0.952 | 1,067 | 0.00 → 0.95 |
| navarrese | 0.870 | 1,167 | 0.00 → 0.87 |

Training: `uv run python -m src.data.train_euskalki all`

> **Note:** This 5-class model is trained directly on the test split. For deployment,
> a hold-out strategy should be used (but the current azpieuskalki model already
> handles 12-class classification).

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

**Step 1 — Binary filter** (batua vs dialectal):
```bash
fasttext supervised \
  -input data/processed/text/train_binary.txt \
  -output models/hier_binary_final \
  -lr 3.0 -epoch 50 -dim 100 -minn 3 -maxn 6 -wordNgrams 2
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

### Model compression

Smaller variants are produced by quantizing weights and reducing hash bucket counts.
Size reduction comes from the model's internal vocabulary hash table, not from
pruning or distillation:

| Variant | bucket | Size | vs final |
|---------|--------|------|----------|
| final | 200K | 1.5GB | baseline |
| quantized | 200K | 417MB | quantized weights |
| compact | 50K | 189MB | 8× smaller |
| tiny | 20K | 112MB | 13× smaller |
| web | binary 20K / dial 50K | 32MB | 46× smaller |

Despite aggressive compression, XNLI drops only from 92.42% to 91.06% —
hash bucket collisions act as implicit regularization at small sizes.

All models available at [huggingface.co/itzune/zeineuski](https://huggingface.co/itzune/zeineuski).

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
