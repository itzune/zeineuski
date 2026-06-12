"""
ECAPA-TDNN-based dialect classification for Basque speech.

Two modes:
1. embedding-only (default): Extract ECAPA-TDNN embeddings, train sklearn classifier on top.
   Fast, works on CPU, good baseline for small datasets.
2. fine-tune: Full fine-tuning with SpeechBrain (needs GPU, more data).

Usage:
    # Train (embedding-only mode)
    uv run python -m src.models.speech.ecapa_tdnn train \
      --train-manifest data/processed/speech/ahotsak/train.csv \
      --val-manifest data/processed/speech/ahotsak/val.csv \
      --test-manifest data/processed/speech/ahotsak/test.csv \
      --output models/speech/ecapa_dialect \
      --config configs/speech/ecapa.yaml \
      --embedding-only

    # Evaluate
    uv run python -m src.models.speech.ecapa_tdnn evaluate \
      --model models/speech/ecapa_dialect \
      --test-manifest data/processed/speech/ahotsak/test.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SPEECHBRAIN_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_DEVICE = "cpu"
TARGET_SR = 16000


def load_csv_manifest(csv_path: Path) -> list[dict]:
    """Load a CSV manifest into list of dicts."""
    samples = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    return samples


def extract_ecapa_embedding(
    audio: np.ndarray,
    sr: int,
    encoder=None,
    device: str = DEFAULT_DEVICE,
) -> np.ndarray:
    """Extract ECAPA-TDNN speaker embedding for an audio segment.

    Args:
        audio: Float32 audio array.
        sr: Sample rate (must be 16000).
        encoder: SpeechBrain EncoderClassifier or None (auto-load).
        device: 'cpu' or 'cuda'.

    Returns:
        192-dim embedding vector (float32).
    """
    import torch

    if encoder is None:
        from speechbrain.inference.speaker import EncoderClassifier
        encoder = EncoderClassifier.from_hparams(
            source=SPEECHBRAIN_MODEL,
            savedir=f"models/speech/speechbrain_cache",
            run_opts={"device": device},
        )

    # ECAPA-TDNN expects 16kHz
    if sr != TARGET_SR:
        import librosa
        audio = librosa.resample(
            audio.astype(np.float64), orig_sr=sr, target_sr=TARGET_SR
        ).astype(np.float32)

    # Convert to tensor
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(device)

    # Extract embedding
    with torch.no_grad():
        embedding = encoder.encode_batch(audio_tensor)

    return embedding.squeeze().cpu().numpy()


def extract_all_embeddings(
    manifest: list[dict],
    device: str = DEFAULT_DEVICE,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Extract ECAPA-TDNN embeddings for all audio files in manifest.

    Returns (embeddings_array, encoded_labels, label_encoder).
    """
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    logger.info(f"Loading ECAPA-TDNN encoder ({SPEECHBRAIN_MODEL})...")
    encoder = EncoderClassifier.from_hparams(
        source=SPEECHBRAIN_MODEL,
        savedir=f"models/speech/speechbrain_cache",
        run_opts={"device": device},
    )
    logger.info(f"  Device: {device}")

    le = LabelEncoder()
    labels = le.fit_transform([s["dialect"] for s in manifest])

    embeddings = []
    failed = 0

    for i, sample in enumerate(manifest):
        if (i + 1) % 50 == 0 or i == 0:
            logger.info(
                f"  Extracting [{i+1}/{len(manifest)}] "
                f"failed={failed}"
            )

        try:
            audio, sr = sf.read(sample["path"])
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            emb = extract_ecapa_embedding(audio, sr, encoder, device)
            embeddings.append(emb)

        except Exception as e:
            logger.debug(f"  Error extracting {sample['path']}: {e}")
            failed += 1
            # Use zero embedding as fallback
            embeddings.append(np.zeros(192, dtype=np.float32))

    logger.info(f"  Extracted {len(embeddings)} embeddings ({failed} failures)")

    return np.array(embeddings), np.array(labels), le


def train_embedding_classifier(
    train_manifest: Path,
    val_manifest: Path,
    test_manifest: Path,
    output_dir: Path,
    config: Optional[dict] = None,
    device: str = DEFAULT_DEVICE,
) -> dict:
    """Train a classifier on top of ECAPA-TDNN embeddings.

    Returns dict with metrics and model paths.
    """
    if config is None:
        config = {}

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifests
    train_samples = load_csv_manifest(train_manifest)
    val_samples = load_csv_manifest(val_manifest)
    test_samples = load_csv_manifest(test_manifest)

    logger.info(
        f"Loaded: train={len(train_samples)}, "
        f"val={len(val_samples)}, test={len(test_samples)}"
    )

    # Show dialect distribution
    from collections import Counter
    for name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        dialect_counts = Counter(s["dialect"] for s in samples)
        logger.info(f"  {name} dialects: {dict(dialect_counts)}")

    # Extract embeddings
    train_start = time.time()

    train_emb, train_labels, le = extract_all_embeddings(train_samples, device=device)
    val_emb, val_labels, _ = extract_all_embeddings(val_samples, device=device)
    test_emb, test_labels, _ = extract_all_embeddings(test_samples, device=device)

    extraction_time = time.time() - train_start
    logger.info(f"Embedding extraction: {extraction_time:.1f}s")

    # Normalize embeddings
    scaler = StandardScaler()
    train_emb = scaler.fit_transform(train_emb)
    val_emb = scaler.transform(val_emb)
    test_emb = scaler.transform(test_emb)

    # Train classifier
    classifier_type = config.get("classifier", "svm")
    logger.info(f"Training {classifier_type} classifier on {len(train_emb)} samples...")

    train_start = time.time()

    if classifier_type == "svm":
        clf = SVC(
            kernel=config.get("svm_kernel", "rbf"),
            C=config.get("svm_C", 1.0),
            gamma=config.get("svm_gamma", "scale"),
            probability=True,
            random_state=42,
        )
    else:
        # Default to SVM
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)

    clf.fit(train_emb, train_labels)

    train_time = time.time() - train_start
    logger.info(f"Training: {train_time:.1f}s")

    # Evaluate on test set
    test_preds = clf.predict(test_emb)
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_f1_macro = f1_score(test_labels, test_preds, average="macro")

    logger.info(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    logger.info(f"Test macro F1: {test_f1_macro:.4f}")

    # Per-class report
    class_names = le.classes_
    report = classification_report(
        test_labels, test_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    logger.info(f"\n{classification_report(test_labels, test_preds, target_names=class_names, zero_division=0)}")

    # Save model artifacts
    model_path = output_dir / "classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"classifier": clf, "scaler": scaler, "label_encoder": le}, f)

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "classifier_type": classifier_type,
            "num_classes": len(class_names),
            "classes": list(class_names),
            "num_train": len(train_samples),
            "num_val": len(val_samples),
            "num_test": len(test_samples),
            "test_accuracy": float(test_accuracy),
            "test_macro_f1": float(test_f1_macro),
            **config,
        }, f, indent=2)

    logger.info(f"Model saved → {model_path}")
    logger.info(f"Config saved → {config_path}")

    # Per-dialect F1 for METRIC output
    per_dialect_f1 = {}
    for cls_name in class_names:
        cls_idx = le.transform([cls_name])[0]
        tp = np.sum((test_labels == cls_idx) & (test_preds == cls_idx))
        fp = np.sum((test_labels != cls_idx) & (test_preds == cls_idx))
        fn = np.sum((test_labels == cls_idx) & (test_preds != cls_idx))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        per_dialect_f1[cls_name] = round(float(f1), 4)

    return {
        "accuracy": float(test_accuracy),
        "macro_f1": float(test_f1_macro),
        "num_classes": len(class_names),
        "classes": list(class_names),
        "per_dialect_f1": per_dialect_f1,
        "extraction_time_s": extraction_time,
        "train_time_s": train_time,
        "num_train": len(train_samples),
        "num_test": len(test_samples),
        "model_path": str(model_path),
    }


def evaluate_model(
    model_dir: Path,
    test_manifest: Path,
    device: str = DEFAULT_DEVICE,
) -> dict:
    """Evaluate a trained model on test data."""
    # Load model
    model_path = model_dir / "classifier.pkl"
    config_path = model_dir / "config.json"

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    with open(model_path, "rb") as f:
        artifacts = pickle.load(f)

    clf = artifacts["classifier"]
    scaler = artifacts["scaler"]
    le = artifacts["label_encoder"]

    # Load test data
    test_samples = load_csv_manifest(test_manifest)
    logger.info(f"Evaluating on {len(test_samples)} test samples")

    # Extract embeddings
    test_emb, test_labels, _ = extract_all_embeddings(test_samples, device=device)

    # Normalize
    test_emb = scaler.transform(test_emb)

    # Predict
    test_preds = clf.predict(test_emb)

    # Metrics
    accuracy = accuracy_score(test_labels, test_preds)
    macro_f1 = f1_score(test_labels, test_preds, average="macro")
    class_names = le.classes_

    # Per-dialect F1
    per_dialect_f1 = {}
    for cls_name in class_names:
        cls_idx = le.transform([cls_name])[0]
        tp = np.sum((test_labels == cls_idx) & (test_preds == cls_idx))
        fp = np.sum((test_labels != cls_idx) & (test_preds == cls_idx))
        fn = np.sum((test_labels == cls_idx) & (test_preds != cls_idx))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        per_dialect_f1[cls_name] = round(float(f1), 4)

    # Print report
    print(f"\nACCURACY: {accuracy:.6f}")
    print(f"MACRO_F1: {macro_f1:.6f}")
    print(f"NUM_CLASSES: {len(class_names)}")
    print(f"CLASSES: {','.join(class_names)}")
    print()
    print(classification_report(test_labels, test_preds, target_names=class_names, zero_division=0))

    # Per-dialect METRIC
    for cls_name, f1 in per_dialect_f1.items():
        print(f"METRIC f1_{cls_name}={f1}")

    # Audio hours
    total_duration = sum(float(s.get("duration_sec", 0)) for s in test_samples)
    print(f"AUDIO_HOURS: {total_duration / 3600:.2f}")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "num_classes": len(class_names),
        "per_dialect_f1": per_dialect_f1,
    }


def main():
    parser = argparse.ArgumentParser(description="ECAPA-TDNN dialect classifier")
    subparsers = parser.add_subparsers(dest="command")

    # Train
    train_parser = subparsers.add_parser("train", help="Train classifier")
    train_parser.add_argument("--train-manifest", required=True, type=Path)
    train_parser.add_argument("--val-manifest", required=True, type=Path)
    train_parser.add_argument("--test-manifest", required=True, type=Path)
    train_parser.add_argument("--output", required=True, type=Path)
    train_parser.add_argument("--config", type=Path)
    train_parser.add_argument("--embedding-only", action="store_true", default=True)
    train_parser.add_argument("--device", default=DEFAULT_DEVICE)
    train_parser.add_argument("--output-stats", action="store_true", help="Print METRIC lines")

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--model", required=True, type=Path)
    eval_parser.add_argument("--test-manifest", required=True, type=Path)
    eval_parser.add_argument("--device", default=DEFAULT_DEVICE)
    eval_parser.add_argument("--output-stats", action="store_true", help="Print METRIC lines")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "train":
        config = {}
        if args.config and args.config.exists():
            with open(args.config) as f:
                config = yaml.safe_load(f)

        results = train_embedding_classifier(
            train_manifest=args.train_manifest,
            val_manifest=args.val_manifest,
            test_manifest=args.test_manifest,
            output_dir=args.output,
            config=config,
            device=args.device,
        )

        if args.output_stats:
            print(f"\nACCURACY: {results['accuracy']:.6f}")
            print(f"MACRO_F1: {results['macro_f1']:.6f}")
            print(f"NUM_CLASSES: {results['num_classes']}")
            for cls_name, f1 in results["per_dialect_f1"].items():
                print(f"METRIC f1_{cls_name}={f1}")

    elif args.command == "evaluate":
        evaluate_model(
            model_dir=args.model,
            test_manifest=args.test_manifest,
            device=args.device,
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
