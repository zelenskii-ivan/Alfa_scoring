"""Train LightGBM on per-client features; hold out the last train file for validation."""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DATA, FEATS = Path(sys.argv[1]), Path(sys.argv[2])
target = pd.read_csv(DATA / "train_target.csv").set_index("id")["target"]


class Block(lgb.Sequence):
    """One file of uint8 features, handed to LightGBM in float64 batches."""

    batch_size = 50_000

    def __init__(self, x: np.ndarray):
        self.x = x

    def __getitem__(self, idx):
        return self.x[idx].astype(np.float64)

    def __len__(self):
        return len(self.x)


def load(name):
    df = pd.read_parquet(FEATS / name)
    return df["id"].values, df[keep] if "keep" in globals() else df.drop(columns="id")


train_files = [f"train_data_{i}.pq" for i in range(7)]
probe = pd.read_parquet(FEATS / "train_data_0.pq").drop(columns="id")
keep = list(probe.columns[(probe > 0).mean() > 0.002])  # drop almost-empty columns
del probe
blocks, ys = [], []
for name in train_files:
    ids, x = load(name)
    cols = list(x.columns)
    blocks.append(Block(x.values))
    ys.append(target.loc[ids].values)
y = np.concatenate(ys)
val_ids, xv = load("train_data_11.pq")
yv = target.loc[val_ids].values
print("train", len(y), "val", len(yv), "features", len(cols), "default rate", round(y.mean(), 4), flush=True)

params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=5, max_bin=32, num_threads=2, verbose=-1, seed=42)
dtrain = lgb.Dataset(blocks, label=y, feature_name=cols, params={"max_bin": 32, "verbose": -1}, free_raw_data=True)
dtrain.construct()
del blocks
import gc; gc.collect()
dval = lgb.Dataset(xv.values.astype(np.float32), label=yv, reference=dtrain)
t = time.time()
model = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)])
print(f"trained in {(time.time() - t) / 60:.1f} min, best iter {model.best_iteration}", flush=True)
auc = roc_auc_score(yv, model.predict(xv.values.astype(np.float32), num_iteration=model.best_iteration))
print(f"VALIDATION ROC-AUC: {auc:.4f}", flush=True)
model.save_model(str(FEATS.parent / "model_v3.txt"), num_iteration=model.best_iteration)

imp = pd.Series(model.feature_importance("gain"), index=cols).sort_values(ascending=False)
print("top features:\n", imp.head(15).round(0).to_string(), flush=True)

preds = []
for name in ["test_data_0.pq", "test_data_1.pq"]:
    ids, x = load(name)
    preds.append(pd.DataFrame({"id": ids, "target": model.predict(x.values.astype(np.float32),
                                                                 num_iteration=model.best_iteration)}))
pd.concat(preds).sort_values("id").to_csv(FEATS.parent / "submission_v3.csv", index=False)
print("submission saved", flush=True)
