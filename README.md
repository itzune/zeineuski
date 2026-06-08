# Zeineuski — Basque Dialect Identification

Fine-grained dialect identification (DID) system for Basque (Euskara). Given a text or speech sample, classifies it into one of six dialect categories: Western (Bizkaiera), Central (Gipuzkera), Navarrese, Navarrese-Labourdin, Souletin (Zuberera), or Standard Basque (Batua).

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

## Results

### Euskalki (Dialect) Classification — 6-class

Hierarchical 2-step classifier (binary batua/dialectal → 5-class euskalkiak):

| Variant | Size | XNLI (3-class) | Test (4-class) | Batua F1 |
|---------|------|:---:|:---:|:---:|
| final | 1.5GB | 92.42% | 95.18% | 0.962 |
| quantized | 417MB | 92.38% | 95.16% | 0.961 |
| compact | 189MB | 91.78% | 94.71% | 0.957 |
| tiny | 112MB | 91.90% | 94.88% | 0.961 |
| **web** | **32MB** | **91.06%** | **94.33%** | **0.952** |

Per-class F1 (final): Western 0.953, Central 0.933, Nav-Lab 0.949, Batua 0.962.
(Navarrese/Souletin: no clean test data — all samples leaked into val from train.)

### Azpieuskalki (Sub-Dialect) Classification — 9 to 12-class

Fine-grained sub-dialect classifier trained on Ahotsak.eus oral history transcriptions, using Zuazo's
sub-dialect taxonomy mapped through official Ahotsak.eus municipality→azpieuskalki assignments.

| Variant | Classes | Accuracy | Model |
|---------|:---:|:---:|---|
| 12-class (all) | 12 | 82.08% | char+word n-grams, epoch=75 |
| 9-class (min_samples=600) | 9 | **83.55%** | drops 3 weakest (<600 train) |

**Optimal config:** fastText with `dim=200, lr=0.2, epoch=75, wordNgrams=2, minn=2, maxn=6, loss=ns`.
NO autotune — aggressive LR decay overfits to dominant classes.

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
| **quantized** | **83.28%** | **31MB** | -0.31pp, 7.5× smaller |
| bucket=50K | 83.22% | 119MB | -0.37pp, 2× smaller |
| bucket=50K quantized | 82.80% | 17MB | -0.79pp, 13.7× smaller |

Models: `models/azpieuskalki.bin` (233MB), `models/azpieuskalki_q.bin` (31MB)

## Training

The best classifier uses a **hierarchical 2-step** architecture discovered through
automated hyperparameter search (33 experiments via Pi Autoresearch):

**Step 1 — Binary filter** (batua vs dialectal):
```bash
fasttext supervised \
  -input data/processed/text/train_binary.txt \
  -output models/hier_binary_final \
  -lr 3.0 -epoch 50 -dim 100 -minn 3 -maxn 6 -wordNgrams 2
```

**Step 2 — 5-class dialect classifier** (trained without batua samples):
```bash
fasttext supervised \
  -input data/processed/text/train_dialectal_5class.txt \
  -output models/hier_dialect_final \
  -lr 0.2 -epoch 150 -dim 100 -minn 3 -maxn 6 -wordNgrams 2
```

Key insight: training the dialect model **without** batua samples prevents the model
from learning to distinguish batua from dialects, which the binary step already handles.

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

`val_6class.txt` was found to have 68.4% overlap with `train_6class.txt` and is **not used**
for evaluation. Navarrese and Souletin lack clean test splits — all their samples were
leaked into the validation set during dataset construction.

Models hosted at [huggingface.co/itzune/zeineuski](https://huggingface.co/itzune/zeineuski).

## Web Demo

Try it in your browser — no server, no install:

**[itzune.eus/euskalkid](https://itzune.eus/euskalkid)** ([source](https://github.com/itzune/euskalkid))

34MB of fastText models running via WebAssembly. Works offline after first load.

## Documentation

- [PROMPT.md](PROMPT.md) — Project prompt and requirements
- [PROJECT.md](PROJECT.md) — Project definition and architecture
- [PLAN.md](PLAN.md) — Detailed implementation plan

## License

MIT
