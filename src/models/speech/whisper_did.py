"""
Whisper encoder → dialect classifier.
Uses the Whisper encoder (no decoder) as a frozen feature extractor,
then mean-pools the time dimension and trains a linear classifier on top.

Approach from ADI-20 paper (arxiv 2511.10070) adapted for Basque dialects.
"""

import argparse
import csv
import gc
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

    def extract(self, audio_path: str, return_frames: bool = False) -> np.ndarray:
        """Extract embeddings from a single wav file.

        With return_frames=False (default): returns mean-pooled (1280,).
        With return_frames=True: returns full frame-level (seq_len, 1280).
        """
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

        frames = out.last_hidden_state.cpu().float().squeeze(0)  # (seq_len, 1280)
        if return_frames:
            return frames.numpy()
        return frames.mean(dim=0).numpy()


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
        num_layers: int = 2,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim // (2**i) if i > 0 else hidden_dim
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_dim),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AttentionPooling(nn.Module):
    """Learned attention pooling over per-segment mean embeddings.

    Input: (batch, num_segments, embed_dim) — per-segment mean-pooled vectors.
    Output: (batch, embed_dim) — attention-weighted sum of segments.

    Uses a 2-layer bottleneck attention mechanism:
      1. Project each segment to a lower-dim space
      2. Score each segment with a learned query vector
      3. Softmax over segments → weighted sum
    """

    def __init__(self, embed_dim: int = 1280, attention_dim: int = 256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, segments: torch.Tensor) -> torch.Tensor:
        """
        Args:
            segments: (batch, num_segments, embed_dim)
        Returns:
            pooled: (batch, embed_dim) weighted sum
        """
        # Compute attention scores: (batch, num_segments, 1)
        scores = self.attention(segments)
        # Softmax over segment dimension
        weights = torch.softmax(scores, dim=1)  # (batch, num_segments, 1)
        # Weighted sum: (batch, embed_dim)
        return (segments * weights).sum(dim=1)


def extract_all_embeddings(
    encoder: WhisperEncoder,
    manifest_path: str,
    output_path: str,
    pooling: str = "mean_std_max",
    num_segments: int = 8,
    batch_report_every: int = 200,
):
    """Extract and cache Whisper embeddings for all audio files.

    Args:
        pooling: "mean_std_max" (3840-dim) or "attention" (segmented frame-level).
        num_segments: Number of equal temporal segments for attention pooling.
    """
    samples = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            samples.append({"path": row["path"], "label": row["dialect"]})

    t0 = time.time()
    short_segments = 0
    for i, sample in enumerate(samples):
        if pooling == "attention":
            # Extract full frame-level embeddings
            frames = encoder.extract(
                sample["path"], return_frames=True
            )  # (seq_len, 1280)
            seq_len = frames.shape[0]
            if seq_len < num_segments:
                short_segments += 1
                # Pad with zeros to reach num_segments
                pad = np.zeros(
                    (num_segments - seq_len, frames.shape[1]), dtype=frames.dtype
                )
                frames = np.concatenate([frames, pad], axis=0)
                seq_len = num_segments

            # Split into num_segments equal parts and mean-pool each
            seg_size = seq_len // num_segments
            segments = np.zeros((num_segments, 1280), dtype=np.float32)
            for s in range(num_segments):
                start = s * seg_size
                end = (s + 1) * seg_size if s < num_segments - 1 else seq_len
                segments[s] = frames[start:end].mean(axis=0)

            sample["embedding"] = segments  # (num_segments, 1280)
            sample["_seq_len"] = min(
                seq_len, seq_len
            )  # original seq_len before padding
        elif pooling == "mean":
            # Plain mean pooling (1280-dim)
            frames = encoder.extract(sample["path"], return_frames=True)
            sample["embedding"] = frames.mean(axis=0)
        else:
            # Original mean_std_max pooling
            frames = encoder.extract(
                sample["path"], return_frames=True
            )  # (seq_len, 1280)
            mean_vec = frames.mean(axis=0)
            std_vec = frames.std(axis=0)
            max_vec = frames.max(axis=0)
            sample["embedding"] = np.concatenate(
                [mean_vec, std_vec, max_vec]
            )  # (3840,)

        if (i + 1) % batch_report_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(f"  Extracted {i + 1}/{len(samples)} ({rate:.1f}/s)")

    if pooling == "attention" and short_segments > 0:
        logger.info(
            f"  {short_segments}/{len(samples)} segments < {num_segments} frames (zero-padded)"
        )

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

    pooling = config.get("pooling", "mean_std_max")

    # Load labels first (cheap) to build label encoder
    with open(train_emb, "rb") as f:
        y_train_raw = [s["label"] for s in pickle.load(f)]
    with open(val_emb, "rb") as f:
        y_val_raw = [s["label"] for s in pickle.load(f)]
    with open(test_emb, "rb") as f:
        y_test_raw = [s["label"] for s in pickle.load(f)]

    label_encoder = LabelEncoder().fit(y_train_raw + y_val_raw + y_test_raw)
    y_train = label_encoder.transform(y_train_raw)
    y_val = label_encoder.transform(y_val_raw)
    y_test = label_encoder.transform(y_test_raw)
    num_classes = len(label_encoder.classes_)

    # ── Determine subsample indices (from labels only, before loading embeddings) ──
    subsample_per_class = config.get("balanced_subsample", 0)
    train_indices = None
    if subsample_per_class > 0:
        train_indices = []
        for cls_idx in range(num_classes):
            cls_mask = np.array(y_train) == cls_idx
            cls_indices = np.where(cls_mask)[0]
            n_available = len(cls_indices)
            n_sample = min(subsample_per_class, n_available)
            if n_sample < n_available:
                sampled = np.random.choice(cls_indices, n_sample, replace=False)
            else:
                sampled = cls_indices
            train_indices.append(sampled)
            logger.info(
                f"  Class {label_encoder.classes_[cls_idx]}: {n_sample}/{n_available} samples"
            )
        train_indices = np.sort(np.concatenate(train_indices))
        logger.info(f"  Total after subsampling: {len(train_indices)} samples")

    # ── Load embeddings (only needed subset for train) ──
    def _load_split(emb_path, indices=None):
        with open(emb_path, "rb") as f:
            all_samples = pickle.load(f)
        if indices is not None:
            embeddings = [all_samples[i]["embedding"] for i in indices]
        else:
            embeddings = [s["embedding"] for s in all_samples]
        del all_samples
        gc.collect()
        return np.stack(embeddings, axis=0)

    logger.info("Loading train embeddings...")
    X_train = _load_split(train_emb, train_indices)
    if train_indices is not None:
        y_train = y_train[train_indices]

    logger.info("Loading val/test embeddings...")
    X_val = _load_split(val_emb)
    X_test = _load_split(test_emb)

    logger.info(
        f"Train: {len(X_train)} samples, Val: {len(X_val)}, Test: {len(X_test)}"
    )
    logger.info(f"Embedding shape: {X_train.shape}")

    # Scale (only for flat 2D embeddings; attention uses 3D segment tensors)
    pooling = config.get("pooling", "mean_std_max")
    if pooling != "attention":
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    else:
        # For attention pooling: keep raw segment tensors, no StandardScaler
        scaler = None
        logger.info("Attention pooling: skipping StandardScaler (3D segment input)")

    num_classes = len(label_encoder.classes_)

    # ── Class weights ──
    use_class_weights = config.get("class_weights", False)
    class_weights_tensor = None
    if use_class_weights:
        from sklearn.utils.class_weight import compute_class_weight

        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(
            device
        )
        logger.info(
            f"  Class weights: {dict(zip(label_encoder.classes_, class_weights))}"
        )

    batch_size = int(config.get("batch_size", 32))
    lr = float(config.get("learning_rate", 1e-3))
    epochs = int(config.get("epochs", 30))
    hidden_dim = int(config.get("hidden_dim", 512))
    dropout = float(config.get("dropout", 0.3))

    logger.info(
        f"Config: lr={lr}, hidden_dim={hidden_dim}, dropout={dropout}, epochs={epochs}, batch_size={batch_size}"
    )

    pooling = config.get("pooling", "mean_std_max")
    if pooling == "attention":
        num_segments = config.get("num_segments", 8)
        attention_dim = int(config.get("attention_dim", 256))
        logger.info(
            f"Attention pooling: num_segments={num_segments}, attention_dim={attention_dim}"
        )
        # Build attention pooling + MLP
        attn_pool = AttentionPooling(embed_dim=1280, attention_dim=attention_dim).to(
            device
        )
        mlp = MLPClassifier(
            input_dim=1280,  # attention output is 1280-dim
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=int(config.get("num_layers", 2)),
        ).to(device)
        model = nn.ModuleList([attn_pool, mlp])
    else:
        model = MLPClassifier(
            input_dim=X_train.shape[1],
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=int(config.get("num_layers", 2)),
        ).to(device)

    # Collect all parameters for optimizer
    if isinstance(model, nn.ModuleList):
        all_params = list(model[0].parameters()) + list(model[1].parameters())
    else:
        all_params = model.parameters()

    optimizer = optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
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
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_tensor if use_class_weights else None
        )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    best_val_acc = 0.0
    train_start = time.time()

    def model_forward(x):
        """Handle both attention pooling and direct MLP forwarding."""
        if isinstance(model, nn.ModuleList):
            # x: (batch, num_segments, 1280)
            pooled = model[0](x)  # AttentionPooling → (batch, 1280)
            return model[1](pooled)  # MLP → (batch, num_classes)
        else:
            return model(x)

    for epoch in range(epochs):
        model.train()
        if isinstance(model, nn.ModuleList):
            model[0].train()
            model[1].train()
        perm = torch.randperm(len(X_train_t))
        total_loss = 0.0
        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)
            optimizer.zero_grad()
            loss = criterion(model_forward(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Evaluate on val
        model.eval()
        if isinstance(model, nn.ModuleList):
            model[0].eval()
            model[1].eval()
        with torch.no_grad():
            val_preds = model_forward(X_val_t.to(device)).argmax(dim=1).cpu()
            val_acc = accuracy_score(y_val_t, val_preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"  Epoch {epoch + 1}/{epochs}  loss={total_loss:.3f}  val_acc={val_acc:.4f}"
            )

    model = (
        model if not isinstance(model, nn.ModuleList) else model
    )  # keep model reference
    model_state = best_state
    if isinstance(model, nn.ModuleList):
        model[0].load_state_dict(
            {k: v for k, v in model_state.items() if k.startswith("0.")}
        )
        model[1].load_state_dict(
            {
                k.replace("1.", ""): v
                for k, v in model_state.items()
                if k.startswith("1.")
            }
        )
        model[0].eval()
        model[1].eval()
    else:
        model.load_state_dict(best_state)
        model.eval()

    with torch.no_grad():
        test_preds = model_forward(X_test_t.to(device)).argmax(dim=1).cpu()
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

    # Build serializable state dict
    if isinstance(model, nn.ModuleList):
        serializable_state = {
            "attention_pool": {
                k: v.cpu().numpy() for k, v in model[0].state_dict().items()
            },
            "mlp": {k: v.cpu().numpy() for k, v in model[1].state_dict().items()},
        }
    else:
        serializable_state = {k: v.cpu().numpy() for k, v in model.state_dict().items()}

    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model_state": serializable_state,
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
                "pooling": pooling,
                "num_segments": int(config.get("num_segments", 8))
                if pooling == "attention"
                else None,
                "attention_dim": int(config.get("attention_dim", 256))
                if pooling == "attention"
                else None,
                "balanced_subsample": subsample_per_class,
                "class_weights": use_class_weights,
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


# ── Inference ──

DIALECT_NAMES = {
    "western": "Mendebaldekoa / Bizkaiera",
    "central": "Erdialdekoa / Gipuzkera",
    "navarrese": "Nafarrera",
    "nav-lab": "Napar-Lapurtera",
    "souletin": "Zuberera",
}


def load_speech_model(
    model_dir: str = "models/speech/whisper_dialect_merged",
    device: str = "cuda",
) -> tuple:
    """Load a trained speech dialect classifier.

    Returns (encoder, mlp_model, label_encoder, scaler, config).
    """
    import pickle as _pickle

    model_dir = Path(model_dir)
    model_path = model_dir / "classifier.pkl"

    with open(model_path, "rb") as f:
        saved = _pickle.load(f)

    # Rebuild encoder
    whisper_model = saved["config"]["whisper_model"]
    encoder = WhisperEncoder(whisper_model, device)
    encoder.load()

    # Rebuild MLP / attention pooling
    model_state = saved["model_state"]
    pooling = saved["config"].get("pooling", "mean_std_max")

    if pooling == "attention":
        attention_dim = int(saved["config"].get("attention_dim", 256))
        attn_pool = AttentionPooling(embed_dim=1280, attention_dim=attention_dim)
        mlp = MLPClassifier(
            input_dim=1280,
            num_classes=saved["num_classes"],
            hidden_dim=saved["hidden_dim"],
            dropout=saved["config"].get("dropout", 0.3),
            num_layers=int(saved["config"].get("num_layers", 2)),
        )
        attn_state = {
            k: torch.from_numpy(v) for k, v in model_state["attention_pool"].items()
        }
        mlp_state = {k: torch.from_numpy(v) for k, v in model_state["mlp"].items()}
        attn_pool.load_state_dict(attn_state)
        mlp.load_state_dict(mlp_state)
        attn_pool.to(device).eval()
        mlp.to(device).eval()
        mlp = (attn_pool, mlp)
    else:
        mlp = MLPClassifier(
            input_dim=saved["input_dim"],
            num_classes=saved["num_classes"],
            hidden_dim=saved["hidden_dim"],
            dropout=saved["config"].get("dropout", 0.3),
            num_layers=int(saved["config"].get("num_layers", 2)),
        )
        state = {k: torch.from_numpy(v) for k, v in model_state.items()}
        mlp.load_state_dict(state)
        mlp.to(device)
        mlp.eval()

    return encoder, mlp, saved["label_encoder"], saved["scaler"], saved["config"]


def predict_speech(
    audio_path: str,
    encoder: WhisperEncoder = None,
    mlp_model=None,
    label_encoder=None,
    scaler=None,
    config=None,
    model_dir: str = "models/speech/whisper_dialect_merged",
    device: str = "cuda",
) -> dict:
    """Predict dialect from an audio file.

    Args:
        audio_path: Path to a WAV file (16kHz mono recommended).
        encoder, mlp_model, label_encoder, scaler: Pre-loaded model parts
            (auto-loaded from model_dir if None).
        model_dir: Directory containing classifier.pkl.
        device: 'cuda' or 'cpu'.

    Returns:
        dict with keys: dialect, dialect_name, confidence, predictions (top-3).
    """
    import numpy as _np

    # Auto-load if needed
    if encoder is None or mlp_model is None:
        encoder, mlp_model, label_encoder, scaler, config = load_speech_model(
            model_dir, device
        )

    # Extract embedding
    pooling = (
        config.get("pooling", "mean_std_max")
        if isinstance(mlp_model, tuple)
        else "mean_std_max"
    )
    if pooling == "attention":
        frames = encoder.extract(audio_path, return_frames=True)  # (seq_len, 1280)
        num_segments = int(config.get("num_segments", 8))
        seq_len = frames.shape[0]
        if seq_len < num_segments:
            pad = _np.zeros(
                (num_segments - seq_len, frames.shape[1]), dtype=frames.dtype
            )
            frames = _np.concatenate([frames, pad], axis=0)
            seq_len = num_segments
        seg_size = seq_len // num_segments
        segments = _np.zeros((num_segments, 1280), dtype=_np.float32)
        for s in range(num_segments):
            start = s * seg_size
            end = (s + 1) * seg_size if s < num_segments - 1 else seq_len
            segments[s] = frames[start:end].mean(axis=0)
        embedding = segments  # (num_segments, 1280)
    else:
        embedding = encoder.extract(audio_path)  # (1280,) or (3840,)

    # Scale (only for flat embeddings; attention uses unscaled segments)
    if pooling != "attention":
        embedding = scaler.transform(embedding.reshape(1, -1)).squeeze(0)

    # Predict
    X = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        if isinstance(mlp_model, tuple):
            attn_pool, mlp = mlp_model
            pooled = attn_pool(X)  # (1, 1280)
            logits = mlp(pooled)
        else:
            logits = mlp_model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze(0)

    # Top-3
    top3_idx = _np.argsort(probs)[::-1][:3]
    predictions = []
    for idx in top3_idx:
        cls_name = label_encoder.classes_[idx]
        predictions.append(
            {
                "dialect": cls_name,
                "confidence": round(float(probs[idx]), 4),
                "dialect_name": DIALECT_NAMES.get(cls_name, cls_name),
            }
        )

    top = predictions[0]
    return {
        "dialect": top["dialect"],
        "confidence": top["confidence"],
        "dialect_name": top["dialect_name"],
        "predictions": predictions,
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
    p_extract.add_argument(
        "--pooling",
        default="mean_std_max",
        choices=["mean", "mean_std_max", "attention"],
    )
    p_extract.add_argument("--num-segments", type=int, default=8)
    p_extract.add_argument(
        "--config", default=None, help="YAML config with pooling settings"
    )

    # train
    p_train = sub.add_parser("train")
    p_train.add_argument("--train-emb", default="models/speech/whisper_train_emb.pkl")
    p_train.add_argument("--val-emb", default="models/speech/whisper_val_emb.pkl")
    p_train.add_argument("--test-emb", default="models/speech/whisper_test_emb.pkl")
    p_train.add_argument("--output", default="models/speech/whisper_dialect")
    p_train.add_argument("--config", default="configs/speech/whisper.yaml")
    p_train.add_argument("--device", default="cuda")

    p_train.add_argument("--seed", type=int, default=42)

    # predict
    p_predict = sub.add_parser("predict")
    p_predict.add_argument("audio", help="Path to audio file (WAV)")
    p_predict.add_argument(
        "--model-dir",
        default="models/speech/whisper_dialect_merged",
        help="Directory with classifier.pkl",
    )
    p_predict.add_argument("--device", default="cuda")

    args = parser.parse_args()

    if args.cmd == "extract":
        encoder = WhisperEncoder(args.whisper_model, args.device)
        encoder.load()

        # Determine pooling settings from args or config
        pooling = args.pooling
        num_segments = args.num_segments
        if args.config:
            import yaml

            with open(args.config) as f:
                cfg = yaml.safe_load(f)
            pooling = cfg.get("pooling", pooling)
            num_segments = int(cfg.get("num_segments", num_segments))

        extract_all_embeddings(
            encoder,
            args.manifest,
            args.output,
            pooling=pooling,
            num_segments=num_segments,
        )
        torch.cuda.empty_cache()

    elif args.cmd == "train":
        import yaml

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

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

    elif args.cmd == "predict":
        result = predict_speech(
            args.audio, model_dir=args.model_dir, device=args.device
        )
        print(f"Dialect: {result['dialect']} ({result['dialect_name']})")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nTop predictions:")
        for p in result["predictions"]:
            print(f"  {p['dialect']:12s} — {p['confidence']:.4f} — {p['dialect_name']}")


if __name__ == "__main__":
    main()
