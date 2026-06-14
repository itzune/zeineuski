#!/usr/bin/env python3
"""
Audio + Text fusion experiment.
Extracts fastText logits from Ahotsak transcriptions and concatenates them
with Whisper audio embeddings, then trains an MLP on the combined features.

Usage:
    uv run python scripts/fusion_train.py \
        --audio-emb models/speech/whisper_merged_train_emb.pkl \
        --passages data/raw/speech/ahotsak/ahotsak_passages_20260608_213742.jsonl \
        --fasttext-model models/euskalki_5class.bin \
        --output models/speech/whisper_dialect_fusion
"""

import argparse
import json
import logging
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LABEL_ORDER = ["central", "nav-lab", "navarrese", "souletin", "western"]

# ── Path parsing ──────────────────────────────────────────────────────────────

def extract_passage_id(path: str) -> str | None:
    """Extract passage_id from an audio segment path."""
    stem = Path(path).stem.lower()
    if "mintzoak" in path:
        m = re.search(r"(\d+-\d+)", stem)
        if m:
            return "mintzoak:" + m.group(1)
    else:
        m = re.search(r"([a-z]{3}-\d{3}-\d{3})", stem)
        if m:
            return m.group(1)
    return None


def load_passage_texts(passages_jsonl: str) -> dict[str, str]:
    """Load {passage_id: cleaned_transcription} mapping."""
    texts = {}
    with open(passages_jsonl) as f:
        for line in f:
            p = json.loads(line)
            tid = p.get("passage_id", "").lower()
            txt = p.get("transcription", "")
            if txt:
                txt = re.sub(r"Egilea\(k\):.*$", "", txt)
                txt = re.sub(r"-\s*\w+\s*:", " ", txt)
                txt = re.sub(r"\s+", " ", txt).strip()
                if len(txt) > 20:
                    texts[tid] = txt
    logger.info(f"Loaded {len(texts)} passages with transcriptions")
    return texts


# ── fastText logit extraction ─────────────────────────────────────────────────

def get_fasttext_logits(model, text: str, label_order: list[str]) -> np.ndarray:
    """Get 5-class logit vector from fastText model."""
    labels, probs = model.predict(text.strip(), k=5)
    label_map = {l.replace("__label__", ""): float(p) for l, p in zip(labels, probs)}

    logits = np.zeros(len(label_order), dtype=np.float32)
    for i, label in enumerate(label_order):
        logits[i] = label_map.get(label, 0.01)  # small floor for missing
    return logits


# ── Fusion dataset builder ────────────────────────────────────────────────────

def build_fusion_dataset(
    audio_emb_path: str,
    val_emb_path: str,
    test_emb_path: str,
    passage_texts: dict[str, str],
    fasttext_model,
    label_order: list[str],
) -> tuple:
    """Build (X_audio + X_text) concatenated features for train/val/test."""

    def _process_split(emb_path):
        with open(emb_path, "rb") as f:
            samples = pickle.load(f)

        X_audio = []
        X_text = []
        y_raw = []
        matched = 0
        total = 0
        skipped_no_text = 0

        for s in samples:
            total += 1
            pid = extract_passage_id(s["path"])
            if pid is None or pid not in passage_texts:
                skipped_no_text += 1
                continue

            text = passage_texts[pid]
            text_logits = get_fasttext_logits(fasttext_model, text, label_order)

            X_audio.append(s["embedding"])
            X_text.append(text_logits)
            y_raw.append(s["label"])
            matched += 1

        logger.info(
            f"  {Path(emb_path).stem}: {matched}/{total} matched ({skipped_no_text} no text)"
        )
        return np.array(X_audio), np.array(X_text), y_raw, matched

    X_audio_train, X_text_train, y_train_raw, n_train = _process_split(audio_emb_path)
    X_audio_val, X_text_val, y_val_raw, n_val = _process_split(val_emb_path)
    X_audio_test, X_text_test, y_test_raw, n_test = _process_split(test_emb_path)

    return (
        X_audio_train,
        X_text_train,
        y_train_raw,
        X_audio_val,
        X_text_val,
        y_val_raw,
        X_audio_test,
        X_text_test,
        y_test_raw,
    )


# ── MLP Training ──────────────────────────────────────────────────────────────

class FusionMLP(nn.Module):
    def __init__(
        self, audio_dim: int, text_dim: int, hidden_dim: int, num_classes: int, dropout: float
    ):
        super().__init__()
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim // 4),
            nn.ReLU(),
        )
        combined_dim = hidden_dim + hidden_dim // 4
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, audio_emb, text_logits):
        a = self.audio_proj(audio_emb)
        t = self.text_proj(text_logits)
        combined = torch.cat([a, t], dim=1)
        return self.classifier(combined)


def train_fusion(
    X_audio_train: np.ndarray,
    X_text_train: np.ndarray,
    y_train_raw: list,
    X_audio_val: np.ndarray,
    X_text_val: np.ndarray,
    y_val_raw: list,
    X_audio_test: np.ndarray,
    X_text_test: np.ndarray,
    y_test_raw: list,
    config: dict,
    device: str,
    output_dir: str,
):
    """Train fusion MLP on combined audio+text features."""
    import gc

    label_encoder = LabelEncoder().fit(y_train_raw + y_val_raw + y_test_raw)
    y_train = label_encoder.transform(y_train_raw)
    y_val = label_encoder.transform(y_val_raw)
    y_test = label_encoder.transform(y_test_raw)
    num_classes = len(label_encoder.classes_)

    # Scale audio features
    scaler = StandardScaler().fit(X_audio_train)
    X_audio_train = scaler.transform(X_audio_train)
    X_audio_val = scaler.transform(X_audio_val)
    X_audio_test = scaler.transform(X_audio_test)

    logger.info(
        f"Train: {len(X_audio_train)}, Val: {len(X_audio_val)}, Test: {len(X_audio_test)}"
    )
    logger.info(f"Audio dim: {X_audio_train.shape[1]}, Text dim: {X_text_train.shape[1]}")

    # Convert to tensors
    X_audio_train_t = torch.tensor(X_audio_train, dtype=torch.float32)
    X_text_train_t = torch.tensor(X_text_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    X_audio_val_t = torch.tensor(X_audio_val, dtype=torch.float32).to(device)
    X_text_val_t = torch.tensor(X_text_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    X_audio_test_t = torch.tensor(X_audio_test, dtype=torch.float32).to(device)
    X_text_test_t = torch.tensor(X_text_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

    # Model
    model = FusionMLP(
        audio_dim=X_audio_train.shape[1],
        text_dim=X_text_train.shape[1],
        hidden_dim=config.get("hidden_dim", 512),
        num_classes=num_classes,
        dropout=config.get("dropout", 0.3),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 5e-4))
    batch_size = config.get("batch_size", 64)
    epochs = config.get("epochs", 100)

    n_train = len(X_audio_train)
    best_val_acc = 0.0
    best_state = None

    logger.info(
        f"Config: lr={config['lr']}, hidden_dim={config['hidden_dim']}, "
        f"dropout={config['dropout']}, epochs={epochs}, batch_size={batch_size}"
    )

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0

        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            a_batch = X_audio_train_t[idx].to(device)
            t_batch = X_text_train_t[idx].to(device)
            y_batch = y_train_t[idx].to(device)

            optimizer.zero_grad()
            logits = model(a_batch, t_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        total_loss /= n_train

        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_audio_val_t, X_text_val_t)
                val_preds = val_logits.argmax(dim=1)
                val_acc = (val_preds == y_val_t).float().mean().item()

            logger.info(
                f"  Epoch {epoch + 1}/{epochs}  loss={total_loss:.3f}  val_acc={val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test
    model.eval()
    with torch.no_grad():
        test_logits = model(X_audio_test_t, X_text_test_t)
        test_probs = torch.softmax(test_logits, dim=1)
        test_preds = test_logits.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, test_preds)
    macro_f1 = f1_score(y_test, test_preds, average="macro")

    # Per-class
    class_names = label_encoder.classes_
    report = classification_report(
        y_test, test_preds, target_names=class_names, digits=4
    )
    per_class = {}
    for cls_name in class_names:
        cls_idx = list(class_names).index(cls_name)
        mask = y_test == cls_idx
        if mask.any():
            per_class[cls_name] = f1_score(
                y_test[mask], test_preds[mask], average="micro"
            )
        else:
            per_class[cls_name] = 0.0

    print(f"\nTest accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Test macro F1: {macro_f1:.4f}")
    print(report)

    # Save model
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_data = {
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "label_encoder": label_encoder,
        "scaler": scaler,
        "config": config,
        "audio_dim": X_audio_train.shape[1],
        "text_dim": X_text_train.shape[1],
        "num_classes": num_classes,
    }
    with open(output_dir / "classifier.pkl", "wb") as f:
        pickle.dump(model_data, f)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Model saved → {output_dir}/classifier.pkl")

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "num_classes": num_classes,
        "per_class_f1": per_class,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-emb", required=True, help="Train embeddings pickle")
    parser.add_argument("--val-emb", required=True)
    parser.add_argument("--test-emb", required=True)
    parser.add_argument("--passages", required=True, help="Ahotsak passages JSONL")
    parser.add_argument("--fasttext-model", required=True, help="fastText .bin model")
    parser.add_argument("--output", default="models/speech/whisper_dialect_fusion")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load passage texts
    passage_texts = load_passage_texts(args.passages)

    # Load fastText model (patched for numpy 2.x)
    import fasttext
    import fasttext.FastText as ft_mod

    source = open(ft_mod.__file__).read()
    if "np.array(probs, copy=False)" in source:
        source = source.replace("np.array(probs, copy=False)", "np.asarray(probs)")
        exec(source, ft_mod.__dict__)

    ft_model = fasttext.load_model(args.fasttext_model)
    logger.info(f"Loaded fastText model with {len(ft_model.labels)} labels")

    # Build fusion dataset
    (
        X_audio_train,
        X_text_train,
        y_train_raw,
        X_audio_val,
        X_text_val,
        y_val_raw,
        X_audio_test,
        X_text_test,
        y_test_raw,
    ) = build_fusion_dataset(
        args.audio_emb,
        args.val_emb,
        args.test_emb,
        passage_texts,
        ft_model,
        LABEL_ORDER,
    )

    # Train
    config = {
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "mode": "audio_text_fusion",
    }

    results = train_fusion(
        X_audio_train,
        X_text_train,
        y_train_raw,
        X_audio_val,
        X_text_val,
        y_val_raw,
        X_audio_test,
        X_text_test,
        y_test_raw,
        config,
        args.device,
        args.output,
    )

    print(f"ACCURACY: {results['accuracy']:.6f}")
    print(f"MACRO_F1: {results['macro_f1']:.6f}")
    print(f"NUM_CLASSES: {results['num_classes']}")
    for cls_name, f1 in results["per_class_f1"].items():
        print(f"METRIC f1_{cls_name}={f1}")


if __name__ == "__main__":
    main()
