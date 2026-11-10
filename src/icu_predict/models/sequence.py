"""GRU sequence model — a benchmark against the tree models.

Tree models see each hour as one row of hand-made summaries (6-hour mean, change over 6 h, ...).
The GRU instead reads the last `window` hours in order, step by step, and learns its own summaries.
It is trained and evaluated on exactly the same rows, splits and labels as the tree models so the
comparison is fair. It is reported as a benchmark only: the API and UIs keep using the selected tree
model, which supports per-prediction explanations.

Requires PyTorch (optional dependency). If it is missing, the benchmark is skipped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEQ_VARS = ["heart_rate", "sbp", "map", "resp_rate", "spo2", "temp_c", "gcs_total",
            "lactate", "creatinine", "wbc", "bicarbonate", "platelets"]
STATIC = ["age", "male", "emergency_admission"]


def available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _step_features(ds: pd.DataFrame) -> list[str]:
    cols = [f"{v}_last" for v in SEQ_VARS] + [f"{v}_hrs_since" for v in SEQ_VARS]
    cols += ["vasopressor_now", "hours_in_icu"] + STATIC
    return [c for c in cols if c in ds.columns]


class SequenceBuilder:
    """Turns (stay_id, hour) rows into [window x features] arrays from the full hourly dataset."""

    def __init__(self, ds: pd.DataFrame, window: int = 12):
        self.window = window
        self.cols = _step_features(ds)
        d = ds.sort_values(["stay_id", "hour"])
        self.arrays, self.offsets = {}, {}
        vals = d[self.cols].to_numpy(dtype=np.float32)
        for sid, idx in d.groupby("stay_id", sort=False).indices.items():
            self.arrays[sid] = vals[idx]
            self.offsets[sid] = int(d["hour"].iloc[idx[0]])
        self.mean = self.std = None

    def fit_scaler(self, stay_ids) -> "SequenceBuilder":
        stack = np.concatenate([self.arrays[s] for s in stay_ids if s in self.arrays])
        self.mean = np.nanmean(stack, axis=0)
        self.std = np.nanstd(stack, axis=0) + 1e-6
        return self

    def build(self, rows: pd.DataFrame) -> np.ndarray:
        W, F = self.window, len(self.cols)
        out = np.zeros((len(rows), W, F + 1), dtype=np.float32)   # last channel = "real step" flag
        for i, (sid, hour) in enumerate(zip(rows["stay_id"].to_numpy(), rows["hour"].to_numpy())):
            a = self.arrays[sid]
            end = int(hour) - self.offsets[sid] + 1
            seg = a[max(0, end - W):end]
            seg = (seg - self.mean) / self.std
            seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
            out[i, W - len(seg):, :F] = seg
            out[i, W - len(seg):, F] = 1.0
        return out


def _net(n_in: int, hidden: int):
    import torch.nn as nn

    class GRUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(n_in, hidden, batch_first=True)
            self.drop = nn.Dropout(0.2)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            _, h = self.gru(x)
            return self.head(self.drop(h[-1])).squeeze(-1)

    return GRUNet()


class GRUClassifier:
    """Minimal sklearn-style wrapper: fit(rows, y) / predict_proba(rows) on (stay_id, hour) rows."""

    def __init__(self, builder: SequenceBuilder, hidden: int = 64, epochs: int = 30, lr: float = 1e-3,
                 batch_size: int = 256, patience: int = 5, seed: int = 0):
        self.b, self.hidden, self.epochs, self.lr = builder, hidden, epochs, lr
        self.batch_size, self.patience, self.seed = batch_size, patience, seed
        self.net = None

    def fit(self, rows, y, val_rows=None, val_y=None, max_rows: int | None = 300_000):
        import torch
        from sklearn.metrics import average_precision_score
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        y = np.asarray(y, dtype=np.float32)
        if max_rows and len(rows) > max_rows:   # keep memory bounded on full MIMIC-IV
            keep = rng.choice(len(rows), max_rows, replace=False)
            rows, y = rows.iloc[keep], y[keep]
        X = torch.from_numpy(self.b.build(rows))
        Y = torch.from_numpy(y)
        Xv = torch.from_numpy(self.b.build(val_rows)) if val_rows is not None else None
        self.net = _net(X.shape[-1], self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        pos_w = torch.tensor(float((1 - y).sum() / max(y.sum(), 1)))
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
        best, best_state, bad = -1.0, None, 0
        for _ in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(X))
            for i in range(0, len(X), self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(X[idx]), Y[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            if Xv is not None:
                score = average_precision_score(val_y, self._proba_tensor(Xv))
                if score > best + 1e-4:
                    best, bad = score, 0
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def _proba_tensor(self, X) -> np.ndarray:
        import torch
        self.net.eval()
        with torch.no_grad():
            return torch.sigmoid(torch.cat([self.net(X[i:i + 4096]) for i in range(0, len(X), 4096)])).numpy()

    def predict_proba(self, rows) -> np.ndarray:
        import torch
        p = self._proba_tensor(torch.from_numpy(self.b.build(rows)))
        return np.column_stack([1 - p, p])
