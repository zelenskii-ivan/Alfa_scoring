"""Alfa credit scoring v3: collapse each client's credit history into one row.

Every feature is a small categorical code, so per client we count how many of
their credits have each value (one-hot sum), plus the same counts over the
client's last 3 credits, and the number of credits. One row per id.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(exist_ok=True)

files = sorted((DATA / "train_data").glob("*.pq")) + sorted((DATA / "test_data").glob("*.pq"))
cols = None
card = None
for f in files:  # global cardinality, so every file gets the same columns
    m = pd.read_parquet(f).drop(columns=["id", "rn"]).max()
    cols = list(m.index)
    card = m if card is None else np.maximum(card, m)
card = card.astype(int) + 1
offsets = np.concatenate([[0], np.cumsum(card.values)[:-1]])
width = int(card.sum())
names = [f"{c}_{v}" for c in cols for v in range(card[c])]
print("features per client:", width * 2 + 2, flush=True)


def onehot_counts(codes: np.ndarray, row: np.ndarray, n: int) -> np.ndarray:
    idx = (row.astype(np.int64)[:, None] * width + (codes + offsets)).ravel()
    return np.bincount(idx, minlength=n * width).astype(np.uint8).reshape(n, width)


def client_rows(df: pd.DataFrame) -> pd.DataFrame:
    ids, row = np.unique(df["id"].values, return_inverse=True)
    n = len(ids)
    codes = df[cols].values.astype(np.int32)
    allc = onehot_counts(codes, row, n)
    # last 3 credits of each client (largest rn)
    last = df.groupby("id")["rn"].transform("max") - df["rn"] < 3
    r_last = row[last.values]
    lastc = onehot_counts(codes[last.values], r_last, n)
    feats = pd.DataFrame(np.hstack([allc, lastc]),
                         columns=names + [f"last3_{x}" for x in names])
    feats.insert(0, "n_credits", np.bincount(row, minlength=n).astype(np.uint8))
    feats.insert(0, "id", ids)
    return feats


for f in files:
    if (OUT / f.name).exists():
        continue
    df = pd.read_parquet(f)
    bounds = np.quantile(df["id"].unique(), np.linspace(0, 1, 6))  # ~50k clients per piece
    parts = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        sel = (df["id"] >= lo) & ((df["id"] < hi) | (hi == bounds[-1]))
        parts.append(client_rows(df[sel]))
    del df
    feats = pd.concat(parts, ignore_index=True)
    feats.to_parquet(OUT / f.name, index=False)
    print(f.name, feats.shape, flush=True)
    del parts, feats
