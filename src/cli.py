#!/usr/bin/env python3
"""Zeineuski CLI — Basque dialect identification from text.

Usage:
    uv run zeineuski predict "Gaur goizean goiz jaiki naiz"
    uv run zeineuski predict -f input.txt -o results.jsonl
    uv run zeineuski download  # Pre-download models
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from .inference import predict, load_models, HF_REPO, DEFAULT_MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("zeineuski")


@click.group()
@click.version_option(message="%(prog)s 0.1.0")
def cli():
    """Zeineuski — Basque (Euskara) dialect identification from text.

    Classifies Basque text into 6 categories: 5 regional dialects
    (euskalkiak) + standard Batua.
    """


@cli.command()
@click.argument("text", required=False)
@click.option(
    "-f",
    "--file",
    "input_file",
    type=click.Path(exists=True),
    help="Input file (one sentence per line).",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output JSONL file for batch results.",
)
@click.option(
    "-t",
    "--threshold",
    type=float,
    default=0.7,
    help="Confidence threshold for dialect predictions (default: 0.7).",
)
@click.option(
    "--model-dir",
    type=click.Path(),
    default=str(DEFAULT_MODEL_DIR),
    help="Directory to cache downloaded models.",
)
@click.option(
    "-v",
    "--variant",
    type=click.Choice(["final", "quantized", "compact", "tiny"]),
    default="compact",
    help="Model variant: final (1.6GB, best), quantized (438MB), compact (198MB, default), tiny (118MB).",
)
@click.option(
    "-n",
    "--top-n",
    type=int,
    default=3,
    help="Show top-N predictions (batch mode only).",
)
def predict_cmd(
    text: Optional[str],
    input_file: Optional[str],
    output: Optional[str],
    threshold: float,
    model_dir: str,
    variant: str,
    top_n: int,
):
    """Predict Basque dialect from TEXT or batch file.

    TEXT: Basque sentence to classify (if not using -f).
    """
    model_dir = Path(model_dir)

    if input_file:
        _batch_predict(input_file, output, threshold, model_dir, variant, top_n)
    elif text:
        _single_predict(text, threshold, model_dir, variant)
    elif not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if text:
            _single_predict(text, threshold, model_dir, variant)
        else:
            click.echo("Error: no input provided.", err=True)
            sys.exit(1)
    else:
        click.echo("Error: provide TEXT argument, -f FILE, or pipe input.", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--model-dir",
    type=click.Path(),
    default=str(DEFAULT_MODEL_DIR),
    help="Directory to cache downloaded models.",
)
@click.option(
    "-v",
    "--variant",
    type=click.Choice(["final", "quantized", "compact", "tiny"]),
    default="compact",
    help="Model variant to download.",
)
def download(model_dir: str, variant: str):
    """Pre-download models from Hugging Face Hub."""
    model_dir = Path(model_dir)
    logger.info(f"Downloading models (variant={variant}) from {HF_REPO}…")
    binary_model, dialect_model = load_models(model_dir, variant)
    logger.info("✓ Models downloaded and cached.")
    logger.info("  Binary: batua vs dialectal")
    logger.info("  Dialect: 5-class euskalkiak")
    logger.info(f"  Variant: {variant}")
    logger.info(f"  Cache: {model_dir}")


def _single_predict(text: str, threshold: float, model_dir: Path, variant: str):
    """Run single prediction and print result."""
    binary_model, dialect_model = load_models(model_dir, variant)
    result = predict(text, binary_model, dialect_model, threshold)

    click.echo(json.dumps(result, ensure_ascii=False, indent=2))


def _batch_predict(
    input_file: str,
    output: Optional[str],
    threshold: float,
    model_dir: Path,
    variant: str,
    top_n: int,
):
    """Run batch prediction on a file."""
    binary_model, dialect_model = load_models(model_dir, variant)

    with open(input_file) as f:
        lines = [line.strip() for line in f if line.strip()]

    out_fh = open(output, "w") if output else sys.stdout
    results = []

    for i, line in enumerate(lines):
        result = predict(line, binary_model, dialect_model, threshold)
        entry = {
            "id": i,
            "text": line[:200],
            "dialect": result["dialect"],
            "confidence": result["confidence"],
            "top3": result.get("predictions", [])[:top_n],
        }
        results.append(entry)

        if output:
            out_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        else:
            _ = result.get("dialect_name", result["dialect"])
            click.echo(f"[{result['dialect']}] {line[:80]}")

    if output:
        out_fh.close()
        logger.info(f"✓ {len(results)} predictions written to {output}")


if __name__ == "__main__":
    cli()
