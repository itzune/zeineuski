"""
Whisper encoder fine-tuning for Basque dialect classification.

Unfreezes the last N encoder layers and trains a classification head
end-to-end on the Ahotsak audio dataset. Uses gradient accumulation
to simulate larger batches without OOM.

Unlike whisper_did.py (frozen encoder + cached embeddings approach),
this jointly optimizes encoder layers + classification head.
"""

import argparse
import csv
import json
import logging
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperModel, WhisperProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class AudioDataset(Dataset):
    """Loads raw audio on-the-fly (no pre-extraction needed)."""

    def __init__(
        self,
        manifest_path: str,
        label_encoder: LabelEncoder,
        segments_dir: str,
        max_samples_per_class: int | None = None,
        seed: int = 42,
    ):
        self.segments_dir = segments_dir
        self.label_encoder = label_encoder

        samples = []
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                samples.append({"fname": row["filename"], "dialect": row["dialect"]})

        if max_samples_per_class is not None:
            random.seed(seed)
            capped = []
            counter = Counter()
            random.shuffle(samples)
            for s in samples:
                if counter[s["dialect"]] < max_samples_per_class:
                    capped.append(s)
                    counter[s["dialect"]] += 1
            samples = capped
            logger.info(f"Capped to {len(samples)} samples ({dict(counter)})")

        self.samples = samples
        self.labels = label_encoder.transform([s["dialect"] for s in samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            wav, sr = sf.read(f"{self.segments_dir}/{s['fname']}")
        except Exception:
            return torch.zeros(16000), self.labels[idx]

        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)
        return torch.tensor(wav, dtype=torch.float32), self.labels[idx]


class WhisperDialectFineTuner(nn.Module):
    """Whisper encoder + classification head with configurable unfreezing."""

    def __init__(
        self,
        model_name: str,
        num_classes: int = 5,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        unfreeze_layers: int = 3,
    ):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(model_name)
        self.processor = WhisperProcessor.from_pretrained(model_name)

        # Freeze all encoder layers first
        for param in self.whisper.encoder.parameters():
            param.requires_grad = False

        # Unfreeze last N layers
        num_layers = self.whisper.config.encoder_layers
        unfreeze_from = num_layers - unfreeze_layers
        for i, layer in enumerate(self.whisper.encoder.layers):
            if i >= unfreeze_from:
                for param in layer.parameters():
                    param.requires_grad = True

        # Also unfreeze final layer norm
        for param in self.whisper.encoder.layer_norm.parameters():
            param.requires_grad = True

        trainable = sum(
            p.numel() for p in self.whisper.encoder.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in self.whisper.encoder.parameters())
        logger.info(
            f"Encoder trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)"
        )

        self.input_dim = self.whisper.config.d_model  # 1280 for large-v3

        # Keep classifier in float32 precision for stability
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, mel_features):
        """mel_features: (B, 80, 3000) — pre-computed log-mel spectrograms."""
        encoder_out = self.whisper.encoder(mel_features)
        hidden = encoder_out.last_hidden_state  # (B, T, 1280)
        pooled = hidden.mean(dim=1)  # (B, 1280)
        return self.classifier(pooled)


def pad_wav(wav, target_len):
    """Pad or truncate wav to target length."""
    if len(wav) < target_len:
        return torch.cat([wav, torch.zeros(target_len - len(wav))])
    return wav[:target_len]


def wavs_to_mels(wavs, labels_list, processor, device):
    """Convert padded wavs + labels to mel spectrogram batch, padded to 3000 frames."""
    wavs_np = wavs.numpy()
    mels = processor(
        wavs_np,
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",
        max_length=480000,
        truncation=True,
    )  # 30s * 16000 = 480000 samples
    # Pad mel to exactly 3000 frames (Whisper requirement)
    mel = mels.input_features  # (B, 80, T)
    if mel.shape[-1] < 3000:
        pad = torch.zeros(mel.shape[0], mel.shape[1], 3000 - mel.shape[-1])
        mel = torch.cat([mel, pad], dim=-1)
    elif mel.shape[-1] > 3000:
        mel = mel[..., :3000]
    return mel.to(device), torch.tensor(labels_list, dtype=torch.long).to(device)


def evaluate(model, dataloader, processor, device):
    """Evaluate on a full dataloader."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for wavs, labels in dataloader:
            mels, lbls = wavs_to_mels(wavs, labels.tolist(), processor, device)
            logits = model(mels)
            preds = logits.argmax(dim=-1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(lbls.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-manifest", default="data/processed/speech/ahotsak_full/train.csv"
    )
    parser.add_argument(
        "--val-manifest", default="data/processed/speech/ahotsak_full/val.csv"
    )
    parser.add_argument(
        "--test-manifest", default="data/processed/speech/ahotsak_full/test.csv"
    )
    parser.add_argument(
        "--segments-dir", default="data/processed/speech/ahotsak_full/segments"
    )
    parser.add_argument("--whisper-model", default="xezpeleta/whisper-large-v3-eu")
    parser.add_argument("--output-dir", default="models/speech/whisper_finetuned")
    parser.add_argument("--unfreeze-layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Label encoder from all splits
    all_labels = []
    for manifest in [args.train_manifest, args.val_manifest, args.test_manifest]:
        with open(manifest) as f:
            for row in csv.DictReader(f):
                all_labels.append(row["dialect"])
    label_encoder = LabelEncoder().fit(all_labels)
    num_classes = len(label_encoder.classes_)
    logger.info(f"Classes: {list(label_encoder.classes_)} ({num_classes})")

    # Build datasets
    train_ds = AudioDataset(
        args.train_manifest,
        label_encoder,
        args.segments_dir,
        max_samples_per_class=args.max_samples_per_class,
        seed=args.seed,
    )
    val_ds = AudioDataset(
        args.val_manifest,
        label_encoder,
        args.segments_dir,
        max_samples_per_class=args.max_samples_per_class,
        seed=args.seed,
    )
    test_ds = AudioDataset(
        args.test_manifest,
        label_encoder,
        args.segments_dir,
        max_samples_per_class=args.max_samples_per_class,
        seed=args.seed,
    )

    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Setup model
    logger.info(f"Loading {args.whisper_model}...")
    model = WhisperDialectFineTuner(
        model_name=args.whisper_model,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        unfreeze_layers=args.unfreeze_layers,
    )
    model.to(args.device)
    processor = model.processor

    def data_collate(batch):
        """Custom collate: pad wavs to max length, return wavs + labels."""
        wavs, labels = zip(*batch)
        max_len = max(w.shape[0] for w in wavs)
        wavs_padded = torch.stack([pad_wav(w, max_len) for w in wavs])
        return wavs_padded, torch.tensor(labels, dtype=torch.long)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=data_collate
    )

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    effective_batch = args.batch_size * args.grad_accum
    logger.info(
        f"Effective batch size: {effective_batch} ({args.batch_size} × {args.grad_accum})"
    )
    logger.info(f"Training for {args.epochs} epochs...")

    best_val_acc = 0.0
    best_state = None
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()

        for step, (wavs, labels) in enumerate(train_loader):
            mels, lbls = wavs_to_mels(wavs, labels.tolist(), processor, args.device)
            logits = model(mels)
            loss = criterion(logits, lbls) / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.grad_accum
            preds = logits.argmax(dim=-1)
            train_correct += (preds == lbls).sum().item()
            train_total += len(lbls)

        scheduler.step()

        # Validation
        val_preds, val_labels = evaluate(model, val_loader, processor, args.device)
        val_acc = accuracy_score(val_labels, val_preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        logger.info(
            f"Epoch {epoch + 1:2d}/{args.epochs} | "
            f"loss={total_loss / len(train_loader):.3f} | "
            f"train_acc={train_correct / max(train_total, 1):.3f} | "
            f"val_acc={val_acc:.4f}"
        )

    # Restore best and evaluate on test
    model.load_state_dict(best_state)
    test_preds, test_labels = evaluate(
        model,
        DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=data_collate
        ),
        processor,
        args.device,
    )

    acc = accuracy_score(test_labels, test_preds)
    macro_f1 = f1_score(test_labels, test_preds, average="macro")
    train_time = time.time() - t_start

    class_names = list(label_encoder.classes_)
    report = classification_report(
        test_labels, test_preds, target_names=class_names, zero_division=0
    )

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Test accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    logger.info(f"Test macro F1: {macro_f1:.4f}")
    logger.info(f"Best val accuracy: {best_val_acc:.4f}")
    logger.info(f"Train time: {train_time:.0f}s")
    logger.info(f"\n{report}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(best_state, output_dir / "model.pt")

    with open(output_dir / "config.json", "w") as f:
        json.dump(
            {
                "whisper_model": args.whisper_model,
                "unfreeze_layers": args.unfreeze_layers,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "effective_batch": effective_batch,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "num_classes": num_classes,
                "classes": class_names,
                "num_train": len(train_ds),
                "num_val": len(val_ds),
                "num_test": len(test_ds),
                "test_accuracy": float(acc),
                "test_macro_f1": float(macro_f1),
                "best_val_accuracy": float(best_val_acc),
                "train_time_s": round(train_time, 1),
            },
            f,
            indent=2,
        )

    logger.info(f"Saved → {output_dir}")

    # Per-class F1 output
    per_class = {}
    for cls_name in class_names:
        cls_idx = label_encoder.transform([cls_name])[0]
        tp = ((test_labels == cls_idx) & (test_preds == cls_idx)).sum()
        fp = ((test_labels != cls_idx) & (test_preds == cls_idx)).sum()
        fn = ((test_labels == cls_idx) & (test_preds != cls_idx)).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        per_class[cls_name] = round(float(f1), 4)

    print(f"ACCURACY: {acc:.6f}")
    print(f"MACRO_F1: {macro_f1:.6f}")
    for cls_name, f1 in per_class.items():
        print(f"METRIC f1_{cls_name}={f1}")


if __name__ == "__main__":
    main()
