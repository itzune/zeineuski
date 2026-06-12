"""
Cross-validate Ahotsak.eus municipality-derived dialect labels against
the hierarchical fastText classifier.

Task 3.5.2 — produces a validation report showing agreement rates,
confusion matrix, and identifies mislabeled or ambiguous samples.

Usage:
    uv run python -m src.data.ahotsak_label_validation validate
    uv run python -m src.data.ahotsak_label_validation report
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AHOTSAK_DIR = PROJECT_ROOT / "data" / "raw" / "speech" / "ahotsak"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "speech"

MODEL_PATH = MODELS_DIR / "hier_dialect_final.bin"

# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    passage_id: str
    town: str
    speaker: str
    transcription: str
    municipality_label: str  # from CSV mapping
    municipality_confidence: str  # high/medium/low
    text_model_prediction: str  # from fastText
    text_model_confidence: float  # probability
    text_model_top3: list[tuple[str, float]] = field(default_factory=list)
    outcome: str = ""  # agreement/flag_batua/flag_mismatch/ambiguous


# ── Model loading with numpy 2.x compatibility patch ──────────────────────────


def load_model():
    """Load the hierarchical fastText model, patching numpy 2.x compatibility."""
    import fasttext
    import fasttext.FastText as ft_mod

    # Read and patch the source to fix np.array(probs, copy=False) → np.asarray(probs)
    source = open(ft_mod.__file__).read()
    source = source.replace("np.array(probs, copy=False)", "np.asarray(probs)")
    exec(source, ft_mod.__dict__)

    model = fasttext.load_model(str(MODEL_PATH))
    logger.info(
        f"Loaded model: {len(model.labels)} labels → {[l.replace('__label__', '') for l in model.labels]}"
    )
    return model


# ── Load Ahotsak passages ─────────────────────────────────────────────────────


def load_passages(jsonl_path: Optional[Path] = None) -> list[dict]:
    """Load scraped Ahotsak passages from JSONL."""
    if jsonl_path is None:
        jsonl_files = sorted(AHOTSAK_DIR.glob("ahotsak_passages_*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No passage JSONL files found in {AHOTSAK_DIR}")
        jsonl_path = jsonl_files[-1]
        logger.info(f"Loading passages from {jsonl_path.name}")

    passages = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                passages.append(json.loads(line))

    logger.info(f"  {len(passages)} passages loaded")
    return passages


# ── Validation ────────────────────────────────────────────────────────────────


DIALECT_LABELS = ["western", "central", "navarrese", "nav-lab", "souletin"]


def strip_label(label: str) -> str:
    """Remove __label__ prefix from fastText label."""
    for prefix in ("__label__", "'__label__"):
        if prefix in label:
            label = label.replace(prefix, "")
    return label.strip("'")


def run_validation(
    model,
    passages: list[dict],
) -> list[ValidationResult]:
    """Run cross-validation: municipality label vs text model prediction."""
    results = []

    for i, p in enumerate(passages):
        text = p.get("transcription", "")
        muni_label = p.get("dialect_class", "").lower()
        muni_confidence = p.get("dialect_confidence", "")
        town = p.get("town_name", p.get("town_slug", ""))
        speaker = p.get("speaker_name", "")
        passage_id = p.get("passage_id", "")

        if not text or not muni_label:
            continue

        # Run fastText prediction (top-3) — strip newlines for fastText
        clean_text = text.strip().replace("\n", " ").replace("\r", " ")
        labels, probs = model.predict(clean_text, k=3)
        top3 = [(strip_label(l), float(prob)) for l, prob in zip(labels, probs)]

        prediction = top3[0][0] if top3 else ""
        confidence = top3[0][1] if top3 else 0.0

        # Classify outcome
        outcome = classify_outcome(muni_label, muni_confidence, prediction, confidence)

        results.append(
            ValidationResult(
                passage_id=passage_id,
                town=town,
                speaker=speaker,
                transcription=text,
                municipality_label=muni_label,
                municipality_confidence=muni_confidence,
                text_model_prediction=prediction,
                text_model_confidence=confidence,
                text_model_top3=top3,
                outcome=outcome,
            )
        )

        if (i + 1) % 50 == 0:
            logger.info(f"  Validated {i + 1}/{len(passages)} passages")

    logger.info(f"  Total: {len(results)} validated")

    return results


def classify_outcome(
    muni_label: str,
    muni_confidence: str,
    model_pred: str,
    model_conf: float,
) -> str:
    """Classify the outcome of municipality vs text model comparison.

    Returns:
        - agreement: both agree, high confidence → gold
        - agreement_medium: both agree, medium confidence
        - agreement_low: both agree, low confidence
        - flag_mismatch: disagree → needs review
        - ambiguous: low model confidence → uncertain
    """
    # Normalize
    muni_label = muni_label.lower()
    model_pred = model_pred.lower()

    if muni_label == model_pred:
        if muni_confidence == "high":
            return "agreement"
        elif muni_confidence == "medium":
            return "agreement_medium"
        else:
            return "agreement_low"

    # Disagreement
    if model_conf >= 0.5:
        return "flag_mismatch"
    else:
        return "ambiguous"


# ── Reporting ─────────────────────────────────────────────────────────────────


def generate_report(results: list[ValidationResult]) -> str:
    """Generate a validation report in Markdown format."""
    lines = []
    lines.append("# Ahotsak Label Cross-Validation Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Model: `{MODEL_PATH.name}`")
    lines.append(f"Total passages validated: {len(results)}")
    lines.append("")

    # ── Overall outcome distribution ──
    outcome_counts = Counter(r.outcome for r in results)
    lines.append("## Outcome Distribution")
    lines.append("")
    lines.append("| Outcome | Count | % |")
    lines.append("|---------|-------|---|")
    for outcome in [
        "agreement",
        "agreement_medium",
        "agreement_low",
        "flag_mismatch",
        "ambiguous",
    ]:
        count = outcome_counts.get(outcome, 0)
        pct = count / len(results) * 100 if results else 0
        lines.append(f"| {outcome} | {count} | {pct:.1f}% |")
    lines.append("")

    # ── Agreement rate summary ──
    agreement_total = sum(
        outcome_counts.get(o, 0)
        for o in ["agreement", "agreement_medium", "agreement_low"]
    )
    lines.append(
        f"**Total agreement (any confidence):** {agreement_total}/{len(results)} ({agreement_total / len(results) * 100:.1f}%)"
    )
    lines.append(
        f"**High-confidence agreement:** {outcome_counts.get('agreement', 0)}/{len(results)} ({outcome_counts.get('agreement', 0) / len(results) * 100:.1f}%)"
    )
    lines.append("")

    # ── Per-dialect agreement ──
    lines.append("## Per-Dialect Agreement")
    lines.append("")
    lines.append("| Dialect | Total | Agreement | Mismatch | Ambiguous | Agreement % |")
    lines.append("|---------|-------|-----------|----------|-----------|-------------|")

    by_dialect = defaultdict(list)
    for r in results:
        by_dialect[r.municipality_label].append(r)

    for dialect in sorted(by_dialect):
        dialect_results = by_dialect[dialect]
        total = len(dialect_results)
        agree = sum(1 for r in dialect_results if r.outcome.startswith("agreement"))
        mismatch = sum(1 for r in dialect_results if r.outcome == "flag_mismatch")
        ambiguous = sum(1 for r in dialect_results if r.outcome == "ambiguous")
        pct = agree / total * 100 if total else 0
        lines.append(
            f"| {dialect} | {total} | {agree} | {mismatch} | {ambiguous} | {pct:.1f}% |"
        )
    lines.append("")

    # ── Confusion matrix ──
    lines.append("## Confusion Matrix")
    lines.append("")
    lines.append("Municipality label → Text model prediction")
    lines.append("")
    header = "| Muni \\ Model | " + " | ".join(DIALECT_LABELS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(DIALECT_LABELS) + 1))

    matrix = defaultdict(Counter)
    for r in results:
        matrix[r.municipality_label][r.text_model_prediction] += 1

    for muni in DIALECT_LABELS:
        row = f"| {muni} |"
        for pred in DIALECT_LABELS:
            row += f" {matrix[muni].get(pred, 0)} |"
        lines.append(row)
    lines.append("")

    # ── Top disagreements ──
    mismatches = [r for r in results if r.outcome == "flag_mismatch"]
    if mismatches:
        lines.append(f"## Top Disagreements ({len(mismatches)} total)")
        lines.append("")
        lines.append(
            "| # | Town | Municipality | Model Pred | Model Conf | Transcription (first 120 chars) |"
        )
        lines.append(
            "|---|------|-------------|------------|------------|----------------------------------|"
        )

        # Sort by model confidence (most confident disagreements first)
        mismatches.sort(key=lambda r: r.text_model_confidence, reverse=True)
        for i, r in enumerate(mismatches[:20]):
            text_preview = r.transcription[:120].replace("\n", " ").replace("|", "\\|")
            lines.append(
                f"| {i + 1} | {r.town} | {r.municipality_label} ({r.municipality_confidence}) | "
                f"{r.text_model_prediction} | {r.text_model_confidence:.2f} | {text_preview} |"
            )
        lines.append("")

    # ── Recommendations ──
    lines.append("## Recommendations")
    lines.append("")
    agree_high = outcome_counts.get("agreement", 0)
    agree_med = outcome_counts.get("agreement_medium", 0)
    agree_low = outcome_counts.get("agreement_low", 0)

    lines.append(
        f"1. **Training-ready:** {agree_high} passages (agreement + high confidence) → use directly for speech training"
    )
    if agree_med:
        lines.append(
            f"2. **Include with caution:** {agree_med} passages (agreement + medium confidence) → include but note lower reliability"
        )
    if agree_low:
        lines.append(
            f"3. **Low confidence agreement:** {agree_low} passages → consider excluding from training, use for weakly-supervised evaluation"
        )

    flag_count = outcome_counts.get("flag_mismatch", 0)
    amb_count = outcome_counts.get("ambiguous", 0)
    if flag_count:
        lines.append(
            f"4. **Needs review:** {flag_count} passages where municipality label ≠ model prediction → manual review recommended before use"
        )
    if amb_count:
        lines.append(
            f"5. **Ambiguous:** {amb_count} passages where model has low confidence → may reflect transitional/contact dialects"
        )

    report = "\n".join(lines)
    return report


# ── Save results ──────────────────────────────────────────────────────────────


def save_results(results: list[ValidationResult], report: str):
    """Save validation results as CSV and report as Markdown."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV
    csv_path = OUTPUT_DIR / f"ahotsak_label_validation_{timestamp}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "passage_id",
                "town",
                "speaker",
                "municipality_label",
                "municipality_confidence",
                "text_model_prediction",
                "text_model_confidence",
                "top3_predictions",
                "outcome",
                "transcription",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "passage_id": r.passage_id,
                    "town": r.town,
                    "speaker": r.speaker,
                    "municipality_label": r.municipality_label,
                    "municipality_confidence": r.municipality_confidence,
                    "text_model_prediction": r.text_model_prediction,
                    "text_model_confidence": f"{r.text_model_confidence:.4f}",
                    "top3_predictions": json.dumps(r.text_model_top3),
                    "outcome": r.outcome,
                    "transcription": r.transcription.replace("\n", " "),
                }
            )
    logger.info(f"Results saved → {csv_path}")

    # Markdown report
    report_path = OUTPUT_DIR / f"ahotsak_label_validation_report_{timestamp}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved → {report_path}")

    # Also save a training-ready subset (agreement + high confidence only)
    train_ready = [r for r in results if r.outcome in ("agreement", "agreement_medium")]
    if train_ready:
        train_path = OUTPUT_DIR / f"ahotsak_train_ready_{timestamp}.jsonl"
        with open(train_path, "w", encoding="utf-8") as f:
            for r in train_ready:
                f.write(
                    json.dumps(
                        {
                            "passage_id": r.passage_id,
                            "town": r.town,
                            "speaker": r.speaker,
                            "dialect_class": r.municipality_label,
                            "dialect_confidence": r.municipality_confidence,
                            "text_model_agrees": True,
                            "text_model_prediction": r.text_model_prediction,
                            "text_model_confidence": r.text_model_confidence,
                            "transcription": r.transcription,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        logger.info(
            f"Training-ready subset ({len(train_ready)} passages) → {train_path}"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────


def cmd_validate():
    """Run validation: municipality labels vs text model."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load model
    model = load_model()

    # Load passages
    passages = load_passages()

    # Run validation
    results = run_validation(model, passages)

    # Generate report
    report = generate_report(results)
    print("\n" + report)

    # Save
    save_results(results, report)


def cmd_report():
    """Print the latest validation report."""
    reports = sorted(OUTPUT_DIR.glob("ahotsak_label_validation_report_*.md"))
    if not reports:
        logger.warning("No reports found. Run 'validate' first.")
        return

    latest = reports[-1]
    with open(latest) as f:
        print(f.read())


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or sys.argv[1] not in ("validate", "report"):
        print("Usage: python -m src.data.ahotsak_label_validation [validate|report]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "validate":
        cmd_validate()
    elif cmd == "report":
        cmd_report()
