"""Zeineuski inference: hierarchical 6-class Basque dialect classification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import fasttext
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# ── Monkey-patch fastText for NumPy 2.0 compatibility ──
import fasttext.FastText as _ftm  # noqa: E402

_ftm_src = open(_ftm.__file__).read()
_ftm_src_fixed = _ftm_src.replace("np.array(probs, copy=False)", "np.asarray(probs)")
try:
    exec(_ftm_src_fixed, _ftm.__dict__)
except Exception:
    pass  # Patch may fail if already applied or on different fastText versions

HF_REPO = "itzune/zeineuski"
BINARY_MODEL_FILE = "models/hier_binary_final.bin"
DIALECT_MODEL_FILE = "models/hier_dialect_final.bin"
DEFAULT_MODEL_DIR = Path.home() / ".cache" / "zeineuski" / "models"

DIALECT_NAMES = {
    "batua": "Batua (Standard Basque)",
    "western": "Mendebaldekoa / Bizkaiera",
    "central": "Erdialdekoa / Gipuzkera",
    "navarrese": "Nafarrera",
    "nav-lab": "Nafar-Lapurtera",
    "souletin": "Zuberera",
}


def _download_model(filename: str, model_dir: Path) -> Path:
    """Download a model file from Hugging Face Hub if not cached."""
    model_dir.mkdir(parents=True, exist_ok=True)
    local_path = model_dir / filename
    if local_path.exists():
        logger.info(f"Using cached model: {local_path}")
        return local_path

    logger.info(f"Downloading {filename} from {HF_REPO}…")
    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=filename,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
    )
    logger.info(f"Downloaded to {path}")
    return Path(path)


def load_models(model_dir: Optional[Path] = None) -> tuple:
    """Load the hierarchical binary and dialect models.

    Downloads from Hugging Face Hub if not already cached.

    Returns:
        (binary_model, dialect_model) — fastText model objects.
    """
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    binary_path = _download_model(BINARY_MODEL_FILE, model_dir)
    dialect_path = _download_model(DIALECT_MODEL_FILE, model_dir)

    logger.info("Loading fastText models…")
    binary_model = fasttext.load_model(str(binary_path))
    dialect_model = fasttext.load_model(str(dialect_path))

    return binary_model, dialect_model


def predict(
    text: str,
    binary_model=None,
    dialect_model=None,
    threshold: float = 0.7,
    model_dir: Optional[Path] = None,
) -> dict:
    """Predict dialect for a single Basque text.

    Two-step hierarchical prediction:
    1. Binary model: batua vs dialectal
    2. Dialect model: 5-class euskalkiak classification

    Args:
        text: Basque text to classify.
        binary_model: Pre-loaded binary fastText model (auto-loaded if None).
        dialect_model: Pre-loaded dialect fastText model (auto-loaded if None).
        threshold: Confidence threshold for dialect predictions.
        model_dir: Directory to cache downloaded models.

    Returns:
        dict with keys: dialect, confidence, dialect_name, predictions (top-3).
    """
    if binary_model is None or dialect_model is None:
        binary_model, dialect_model = load_models(model_dir)

    text = text.replace("\n", " ").strip()

    # Step 1: Binary batua vs dialectal
    bin_labels, bin_probs = binary_model.predict(text, k=1)
    bin_conf = float(bin_probs[0])

    if bin_labels[0] == "__label__batua":
        return {
            "dialect": "batua",
            "confidence": bin_conf,
            "dialect_name": DIALECT_NAMES["batua"],
            "predictions": [("batua", bin_conf)],
        }

    # Step 2: Dialect 5-class
    dial_labels, dial_probs = dialect_model.predict(text, k=3)
    top_class = dial_labels[0].replace("__label__", "")
    top_conf = float(dial_probs[0])

    if top_conf < threshold:
        return {
            "dialect": "uncertain",
            "confidence": top_conf,
            "dialect_name": "Uncertain (below threshold)",
            "predictions": [
                (lbl.replace("__label__", ""), float(p))
                for lbl, p in zip(dial_labels, dial_probs)
            ],
        }

    return {
        "dialect": top_class,
        "confidence": top_conf,
        "dialect_name": DIALECT_NAMES.get(top_class, top_class),
        "predictions": [
            (lbl.replace("__label__", ""), float(p))
            for lbl, p in zip(dial_labels, dial_probs)
        ],
    }
