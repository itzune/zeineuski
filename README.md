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

| Variant | Size | XNLI (3-class) | Test (6-class) | Batua F1 |
|---------|------|:---:|:---:|:---:|
| final | 1,588MB | — | 97.83% | 0.962 |
| quantized | 438MB | 96.94% | — | — |
| compact | 198MB | 96.85% | — | — |
| tiny | 118MB | 96.68% | — | — |
| **web** | **34MB** | **96.84%** | — | — |

Per-class F1: Western 0.976, Central 0.958, Nav-Lab 0.968, Navarrese 0.972, Souletin 0.997.

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
