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

| Variant | Size | XNLI (3-class) | Val (6-class) | Batua F1 |
|---------|------|:---:|:---:|:---:|
| final | 1.5GB | 92.42% | 98.83% | 0.996 |
| quantized | 417MB | 92.38% | 98.74% | 0.993 |
| compact | 189MB | 91.78% | 98.72% | 0.992 |
| tiny | 112MB | 91.90% | 98.61% | 0.989 |
| **web** | **32MB** | **91.06%** | **98.72%** | **0.993** |

Per-class F1 (final): Western 0.978, Central 0.977, Nav-Lab 0.990, Navarrese 1.000, Souletin 0.976.

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
