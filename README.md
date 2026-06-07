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
