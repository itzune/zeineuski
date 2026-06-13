"""
Whisper encoder → dialect classifier.
Uses the Whisper encoder (no decoder) as a frozen feature extractor,
then mean-pools the time dimension and trains a linear classifier on top.

Approach from ADI-20 paper (arxiv 2511.10070) adapted for Basque dialects.
"""

import argparse
import csv
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset
from transformers import WhisperModel, WhisperProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class WhisperEncoder:
    """Frozen Whisper encoder for feature extraction."""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = None

    def load(self):
        self.model = WhisperModel.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.float16,
        ).eval()

    def extract(self, audio_path: str) -> np.ndarray:
        """Extract 1280-dim pooled embedding from a single wav file."""
        wav, sr = sf.read(audio_path)
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)  # mono
        if sr != 16000:
            import torchaudio

            wav = (
                torchaudio.functional.resample(
                    torch.tensor(wav).unsqueeze(0), sr, 16000
                )
                .squeeze(0)
                .numpy()
            )

        mel = self.processor(
            wav, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device, torch.float16)

        with torch.no_grad():
            out = self.model.encoder(mel)
        # Mean pool across time dimension
        return out.last_hidden_state.mean(dim=1).cpu().float().numpy().squeeze(0)


class WhisperDialectDataset(Dataset):
    """Loads pre-extracted embeddings from a manifest CSV."""

    def __init__(self, manifest_path: str, label_encoder: LabelEncoder):
        self.samples = []
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                self.samples.append(
                    {
                        "path": row["path"],
                        "label": row["dialect"],
                    }
                )
        self.label_encoder = label_encoder
        self.labels = self.label_encoder.transform([s["label"] for s in self.samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(
            self.samples[idx]["embedding"], dtype=torch.float32
        ), torch.tensor(self.labels[idx], dtype=torch.long)


class MLPClassifier(nn.Module):
    """Simple MLP on top of pooled Whisper embeddings."""

    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 5,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def extract_all_embeddings(
    encoder: WhisperEncoder,
    manifest_path: str,
    output_path: str,
    batch_report_every: int = 200,
):
    """Extract and cache Whisper embeddings for all audio files."""
    samples = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            samples.append({"path": row["path"], "label": row["dialect"]})

    t0 = time.time()
    for i, sample in enumerate(samples):
        sample["embedding"] = encoder.extract(sample["path"])
        if (i + 1) % batch_report_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(f"  Extracted {i + 1}/{len(samples)} ({rate:.1f}/s)")

    with open(output_path, "wb") as f:
        pickle.dump(samples, f)

    elapsed = time.time() - t0
    logger.info(f"Saved {len(samples)} embeddings → {output_path} ({elapsed:.0f}s)")
    return samples


def train_mlp(
    train_emb: str,
    val_emb: str,
    test_emb: str,
    output_dir: str,
    config: dict,
    device: str = "cuda",
):
    """Train MLP classifier on pre-extracted train/val/test embeddings."""
    # Load train
    with open(train_emb, "rb") as f:
        train_samples = pickle.load(f)
    X_train = np.stack([s["embedding"] for s in train_samples], axis=0)
    y_train_raw = [s["label"] for s in train_samples]

    # Load val
    with open(val_emb, "rb") as f:
        val_samples = pickle.load(f)
    X_val = np.stack([s["embedding"] for s in val_samples], axis=0)
    y_val_raw = [s["label"] for s in val_samples]

    # Load test
    with open(test_emb, "rb") as f:
        test_samples = pickle.load(f)
    X_test = np.stack([s["embedding"] for s in test_samples], axis=0)
    y_test_raw = [s["label"] for s in test_samples]

    # Unified label encoder from all splits
    label_encoder = LabelEncoder().fit(y_train_raw + y_val_raw + y_test_raw)
    y_train = label_encoder.transform(y_train_raw)
    y_val = label_encoder.transform(y_val_raw)
    y_test = label_encoder.transform(y_test_raw)

    logger.info(
        f"Train: {len(train_samples)} samples, Val: {len(val_samples)}, Test: {len(test_samples)}"
    )
    logger.info(f"Embedding dim: {X_train.shape[1]}")

    # Scale
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    num_classes = len(label_encoder.classes_)
    batch_size = int(config.get("batch_size", 32))
    lr = float(config.get("learning_rate", 1e-3))
    epochs = int(config.get("epochs", 30))
    hidden_dim = int(config.get("hidden_dim", 512))
    dropout = float(config.get("dropout", 0.3))

    model = MLPClassifier(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # Loss function
    loss_type = config.get("loss", "crossentropy")
    if loss_type == "focal":
        gamma = float(config.get("focal_gamma", 2.0))
        alpha = float(config.get("focal_alpha", 0.25))

        # Focal Loss implementation: FL(p) = -alpha * (1-p)^gamma * log(p)
        class FocalLoss(nn.Module):
            def __init__(self, alpha=0.25, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, inputs, targets):
                ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
                pt = torch.exp(-ce_loss)
                return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

        criterion = FocalLoss(alpha=alpha, gamma=gamma)
        logger.info(f"Using Focal Loss (alpha={alpha}, gamma={gamma})")
    else:
        criterion = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    best_val_acc = 0.0
    train_start = time.time()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train_t))
        total_loss = 0.0
        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Evaluate on val
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t.to(device)).argmax(dim=1).cpu()
            val_acc = accuracy_score(y_val_t, val_preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"  Epoch {epoch + 1}/{epochs}  loss={total_loss:.3f}  val_acc={val_acc:.4f}"
            )

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_preds = model(X_test_t.to(device)).argmax(dim=1).cpu()
        acc = accuracy_score(y_test_t, test_preds)
        macro_f1 = f1_score(y_test_t, test_preds, average="macro")

    train_time = time.time() - train_start

    logger.info(f"\nTest accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    logger.info(f"Test macro F1: {macro_f1:.4f}")

    class_names = label_encoder.classes_
    report = classification_report(
        y_test_t, test_preds, target_names=class_names, zero_division=0
    )
    logger.info(f"\n{report}")

    # Save model
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model_state": {
                    k: v.numpy() for k, v in model.cpu().state_dict().items()
                },
                "config": config,
                "classes": list(class_names),
                "label_encoder": label_encoder,
                "scaler": scaler,
                "input_dim": X_train.shape[1],
                "hidden_dim": hidden_dim,
                "num_classes": num_classes,
            },
            f,
        )

    cfg_path = output_dir / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(
            {
                "model": "whisper_encoder_mlp",
                "whisper_model": config["whisper_model"],
                "num_classes": num_classes,
                "classes": list(class_names),
                "num_train": len(y_train),
                "num_val": len(y_val),
                "num_test": len(y_test),
                "test_accuracy": float(acc),
                "test_macro_f1": float(macro_f1),
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "batch_size": batch_size,
                "learning_rate": lr,
                "epochs": epochs,
                "train_time_s": round(train_time, 1),
            },
            f,
            indent=2,
        )

    logger.info(f"Model saved → {model_path}")
    logger.info(f"Config saved → {cfg_path}")

    # Per-class METRIC output
    per_class = {}
    for cls_name in class_names:
        cls_idx = label_encoder.transform([cls_name])[0]
        tp = ((y_test_t == cls_idx) & (test_preds == cls_idx)).sum().item()
        fp = ((y_test_t != cls_idx) & (test_preds == cls_idx)).sum().item()
        fn = ((y_test_t == cls_idx) & (test_preds != cls_idx)).sum().item()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        per_class[cls_name] = round(float(f1), 4)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "num_classes": num_classes,
        "classes": list(class_names),
        "per_class_f1": per_class,
        "train_time_s": round(train_time, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    # extract
    p_extract = sub.add_parser("extract")
    p_extract.add_argument("--manifest", required=True)
    p_extract.add_argument("--output", default="models/speech/whisper_embeddings.pkl")
    p_extract.add_argument("--whisper-model", default="xezpeleta/whisper-large-v3-eu")
    p_extract.add_argument("--device", default="cuda")

    # train
    p_train = sub.add_parser("train")
    p_train.add_argument("--train-emb", default="models/speech/whisper_train_emb.pkl")
    p_train.add_argument("--val-emb", default="models/speech/whisper_val_emb.pkl")
    p_train.add_argument("--test-emb", default="models/speech/whisper_test_emb.pkl")
    p_train.add_argument("--output", default="models/speech/whisper_dialect")
    p_train.add_argument("--config", default="configs/speech/whisper.yaml")
    p_train.add_argument("--device", default="cuda")

    args = parser.parse_args()

    if args.cmd == "extract":
        encoder = WhisperEncoder(args.whisper_model, args.device)
        encoder.load()
        extract_all_embeddings(encoder, args.manifest, args.output)
        torch.cuda.empty_cache()

    elif args.cmd == "train":
        import yaml

        with open(args.config) as f:
            config = yaml.safe_load(f)
        config["whisper_model"] = config.get(
            "whisper_model", "xezpeleta/whisper-large-v3-eu"
        )
        results = train_mlp(
            args.train_emb,
            args.val_emb,
            args.test_emb,
            args.output,
            config,
            args.device,
        )

        # Print METRIC lines for autoresearch
        print(f"ACCURACY: {results['accuracy']:.6f}")
        print(f"MACRO_F1: {results['macro_f1']:.6f}")
        print(f"NUM_CLASSES: {results['num_classes']}")
        for cls_name, f1 in results["per_class_f1"].items():
            print(f"METRIC f1_{cls_name}={f1}")


if __name__ == "__main__":
    main()
