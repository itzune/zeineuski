#!/usr/bin/env python3
"""Quantize a trained fastText azpieuskalki model into compact variants.

Usage:
    python scripts/quantize_model.py                          # Default path
    python scripts/quantize_model.py models/azpieuskalki.bin  # Explicit path

Produces:
    models/azpieuskalki_q.bin       ~33MB  (default compression)
    models/azpieuskalki_b50000.bin  ~137MB (smaller bucket)
    models/azpieuskalki_b200000.bin ~???   (wider bucket)

Note: Models trained with loss='ns' (negative sampling) have compressed
internal matrices. Quantization with retrain=True requires the training
file and may fail with 'Matrix too small'. In that case, retrain the model
with loss='softmax' for quantization-friendly weights, or use retrain=False
(which works but is slower and may segfault on large models).

For ns-trained models, the recommended approach is to train directly
with bucket=50000 or bucket=200000 during the initial training step
to control model size without post-hoc quantization.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import fasttext

DEFAULT_MODEL = Path("models/azpieuskalki.bin")
DEFAULT_TRAIN = Path("data/processed/text/train_azpieuskalki.txt")


def quantize_retrain(
    model_path: Path,
    train_path: Path,
    qnorm: bool = True,
    qout: bool = True,
    cutoff: int = 100000,
    output_path: Path | None = None,
) -> Path | None:
    """Quantize with retrain — uses the original training data.

    This is fast and produces the best results, but requires train file
    and compatible model (not ns-compressed). Returns None on failure.
    """
    if output_path is None:
        suffix = f"_q{cutoff}" if cutoff != 100000 else "_q"
        output_path = model_path.parent / f"{model_path.stem}{suffix}.bin"

    print(f"Quantizing (retrain, cutoff={cutoff}): {model_path} → {output_path.name}")
    print(f"  Train: {train_path}")

    model = fasttext.load_model(str(model_path))
    t0 = time.time()
    try:
        model.quantize(
            input=str(train_path),
            retrain=True,
            qnorm=qnorm,
            cutoff=cutoff,
            qout=qout,
        )
        model.save_model(str(output_path))
        elapsed = time.time() - t0
        print(
            f"  Saved: {output_path} ({_size_mb(output_path):.1f} MB) in {elapsed:.1f}s"
        )
        return output_path
    except ValueError as e:
        print(f"  Failed: {e}")
        return None


def _size_mb(path: Path) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize fastText azpieuskalki model")
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to trained .bin model",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=None,
        help="Training data path (for retrain=True, default: auto-detect)",
    )
    args = parser.parse_args()

    model_path = args.model
    if not model_path.exists():
        print(f"Error: model not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    train_path = args.train_file or DEFAULT_TRAIN
    if not train_path.exists():
        print(f"Error: train file not found at {train_path}", file=sys.stderr)
        print(
            "  Specify with --train-file or generate training data first",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Model: {model_path} ({_size_mb(model_path):.1f} MB)")
    print(f"Train: {train_path}\n")

    results = []

    # Default quantized (~33MB)
    r = quantize_retrain(model_path, train_path, cutoff=100000)
    if r:
        results.append(r)

    # Smaller bucket (~137MB)
    r = quantize_retrain(
        model_path,
        train_path,
        cutoff=50000,
        output_path=model_path.parent / f"{model_path.stem}_b50000.bin",
    )
    if r:
        results.append(r)

    # Wider bucket for minority class coverage
    r = quantize_retrain(
        model_path,
        train_path,
        cutoff=200000,
        output_path=model_path.parent / f"{model_path.stem}_b200000.bin",
    )
    if r:
        results.append(r)

    if results:
        print(f"\nDone. {len(results)} variant(s) created.")
        print("Run 'python3 eval_3_models.py' to compare.")
    else:
        print("\nNo variants created — retrain failed for all cutoffs.")
        print("This is expected for ns-trained models with compressed matrices.")
        print("Workaround: re-train with loss='softmax' or bucket=50000 directly.")
        sys.exit(1)


if __name__ == "__main__":
    main()
