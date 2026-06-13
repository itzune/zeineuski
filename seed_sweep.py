"""Quick diagnostic: train MLP with different seeds to check stochastic variance."""

import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

with open("models/speech/whisper_train_emb3d.pkl", "rb") as f:
    train = pickle.load(f)
with open("models/speech/whisper_val_emb3d.pkl", "rb") as f:
    val_data = pickle.load(f)
with open("models/speech/whisper_test_emb3d.pkl", "rb") as f:
    test_data = pickle.load(f)

X_tr = np.stack([s["embedding"] for s in train])
X_va = np.stack([s["embedding"] for s in val_data])
X_te = np.stack([s["embedding"] for s in test_data])
y_tr_raw = [s["label"] for s in train]
y_va_raw = [s["label"] for s in val_data]
y_te_raw = [s["label"] for s in test_data]

le = LabelEncoder().fit(y_tr_raw + y_va_raw + y_te_raw)
y_tr = le.transform(y_tr_raw)
y_va = le.transform(y_va_raw)
y_te = le.transform(y_te_raw)

scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr)
X_va = scaler.transform(X_va)
X_te = scaler.transform(X_te)

X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.long)
X_va_t = torch.tensor(X_va, dtype=torch.float32)
y_va_t = torch.tensor(y_va, dtype=torch.long)
X_te_t = torch.tensor(X_te, dtype=torch.float32)
y_te_t = torch.tensor(y_te, dtype=torch.long)

device = "cuda"
nc = len(le.classes_)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3840, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, nc),
        )

    def forward(self, x):
        return self.net(x)


class FL(nn.Module):
    def forward(self, x, y):
        ce = nn.functional.cross_entropy(x, y, reduction="none")
        return (0.25 * (1 - torch.exp(-ce)) ** 2.0 * ce).mean()


seed = int(sys.argv[1])
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

model = MLP().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
crit = FL()

best = 0.0
best_state = None

for ep in range(100):
    model.train()
    perm = torch.randperm(len(X_tr_t))
    for i in range(0, len(X_tr_t), 64):
        idx = perm[i : i + 64]
        opt.zero_grad()
        crit(model(X_tr_t[idx].to(device)), y_tr_t[idx].to(device)).backward()
        opt.step()
    sch.step()
    model.eval()
    with torch.no_grad():
        va = accuracy_score(y_va_t, model(X_va_t.to(device)).argmax(dim=1).cpu())
        if va > best:
            best = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    preds = model(X_te_t.to(device)).argmax(dim=1).cpu()
    acc = accuracy_score(y_te_t, preds)
    f1 = f1_score(y_te_t, preds, average="macro")

print(f"seed={seed}: acc={acc:.4f} ({acc * 100:.2f}%) macro_f1={f1:.4f}")
