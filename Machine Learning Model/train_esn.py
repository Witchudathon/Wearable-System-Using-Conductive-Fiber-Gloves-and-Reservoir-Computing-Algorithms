#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_esn.py — Echo State Network (ESN) Training
═══════════════════════════════════════════════════
Reservoir computing approach: random sparse reservoir (800 neurons)
with Ridge classifier readout. Fixed random weights — only readout trained.
"""

import json, pickle, time
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from utils import (
    read_all_csvs, add_timestep_features, get_enhanced_sensor_cols,
    pick_test_files, get_rounds, save_model_outputs,
    BASE_OUT, SEED, TEST_FILES_PER_CLASS_PER_ROUND,
)
from sklearn.metrics import confusion_matrix, accuracy_score

# ═══════════════════════════════════════════════════════════════
#  ESN CONFIG
# ═══════════════════════════════════════════════════════════════

ESN_RESERVOIR  = 800
ESN_SR         = 0.95   # spectral radius
ESN_LEAK       = 0.2
ESN_SPARSITY   = 0.1
ESN_INPUT_SCALE = 0.3
ESN_RIDGE_ALPHA = 1.0


# ═══════════════════════════════════════════════════════════════
#  ESN CLASS
# ═══════════════════════════════════════════════════════════════

class EchoStateNetwork:
    def __init__(self, n_inputs, seed=42):
        rng = np.random.RandomState(seed)
        N = ESN_RESERVOIR
        self.N = N
        self.leak = ESN_LEAK

        # Input weights
        self.W_in = rng.uniform(-ESN_INPUT_SCALE, ESN_INPUT_SCALE, (N, n_inputs))

        # Reservoir weights (sparse)
        W = rng.randn(N, N)
        mask = rng.rand(N, N) < ESN_SPARSITY
        W *= mask
        rho = np.max(np.abs(np.linalg.eigvals(W)))
        if rho > 0:
            W = W * (ESN_SR / rho)
        self.W = W

        self.bias = rng.uniform(-0.1, 0.1, N)
        self.readout = RidgeClassifier(alpha=ESN_RIDGE_ALPHA)

    def _run_reservoir(self, X_seq):
        T, D = X_seq.shape
        states = np.zeros((T, self.N))
        h = np.zeros(self.N)
        for t in range(T):
            pre = np.tanh(self.W_in @ X_seq[t] + self.W @ h + self.bias)
            h = (1 - self.leak) * h + self.leak * pre
            states[t] = h
        return states

    def _extract_features(self, states):
        return np.concatenate([states.mean(0), states.std(0),
                               states.max(0), states[-1]])

    def fit_sequences(self, seqs, labels):
        X = np.array([self._extract_features(self._run_reservoir(s)) for s in seqs])
        self.readout.fit(X, labels)

    def predict_sequences(self, seqs):
        X = np.array([self._extract_features(self._run_reservoir(s)) for s in seqs])
        return self.readout.predict(X)


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def make_file_sequences(df, cols, fit_scaler=False, scaler=None):
    seqs, labels, fids = [], [], []
    for fid, g in df.groupby("file_id"):
        vals = g[cols].values.astype(np.float32)
        seqs.append(vals)
        labels.append(g["label"].iloc[0])
        fids.append(fid)
    if not seqs:
        return [], [], [], scaler

    if fit_scaler:
        scaler = StandardScaler()
        all_vals = np.vstack(seqs)
        scaler.fit(all_vals)

    seqs = [np.nan_to_num(scaler.transform(s).astype(np.float32)) for s in seqs]
    return seqs, labels, fids, scaler


# ═══════════════════════════════════════════════════════════════
#  TRAIN
# ═══════════════════════════════════════════════════════════════

def train_esn(df_ts, enhanced_cols, file_groups, classes, rounds):
    out = BASE_OUT / "esn"
    out.mkdir(parents=True, exist_ok=True)
    n_cls = len(classes)
    cm_agg = np.zeros((n_cls, n_cls), dtype=int)
    y_true_all, y_pred_all, round_accs = [], [], []

    for r in range(rounds):
        test_files, train_files = pick_test_files(file_groups, TEST_FILES_PER_CLASS_PER_ROUND, SEED, r)
        df_train = df_ts[df_ts["file_id"].isin(train_files)]
        df_test  = df_ts[df_ts["file_id"].isin(test_files)]

        seqs_train, labels_train, _, scaler = make_file_sequences(df_train, enhanced_cols, fit_scaler=True)
        seqs_test, labels_test, _, _ = make_file_sequences(df_test, enhanced_cols, scaler=scaler)

        if not seqs_train or not seqs_test:
            continue

        esn = EchoStateNetwork(n_inputs=len(enhanced_cols), seed=SEED + r)
        esn.fit_sequences(seqs_train, labels_train)
        y_pred = esn.predict_sequences(seqs_test)

        round_acc = accuracy_score(labels_test, y_pred)
        round_accs.append(round_acc)
        y_true_all.extend(labels_test); y_pred_all.extend(y_pred)
        cm_agg += confusion_matrix(labels_test, y_pred, labels=classes)

        if (r + 1) % 5 == 0 or r == rounds - 1:
            print(f"    ESN Round {r+1}/{rounds}  (round acc={round_acc:.2%})", flush=True)

    def save_esn(esn, d):
        with open(d / "esn_model.pkl", "wb") as f:
            pickle.dump({"esn": esn, "classes": classes, "sensor_cols": enhanced_cols}, f)

    acc, _ = save_model_outputs("ESN", out, cm_agg,
        y_true_all, y_pred_all, classes,
        {"n_reservoir": ESN_RESERVOIR, "spectral_radius": ESN_SR,
         "leak_rate": ESN_LEAK, "input_channels": len(enhanced_cols),
         "features_per_sequence": f"mean+std+max+last = {ESN_RESERVOIR*4}"},
        esn, save_esn)
    return acc, round_accs


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 70)
    print("  ESN Training — Echo State Network (800 neurons)")
    print("=" * 70)

    print("\n  [1/3] Loading data ...", flush=True)
    df_raw = read_all_csvs()
    classes = sorted(df_raw["label"].unique())
    file_groups = {cls: list(df_raw.loc[df_raw["label"] == cls, "file_id"].unique()) for cls in classes}
    rounds = get_rounds()
    print(f"         Shape: {df_raw.shape}, Classes: {len(classes)}, Rounds: {rounds}")

    print("\n  [2/3] Building features (30 channels) ...", flush=True)
    df_ts = add_timestep_features(df_raw)
    enhanced_cols = get_enhanced_sensor_cols()
    print(f"         Channels: {len(enhanced_cols)}")

    print(f"\n  [3/3] Training ESN ...", flush=True)
    acc, round_accs = train_esn(df_ts, enhanced_cols, file_groups, classes, rounds)

    # Save round metrics
    metrics = {"ESN": {"overall_accuracy": acc, "round_accuracies": round_accs}}
    mf = BASE_OUT / "esn" / "round_metrics.json"
    mf.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  ESN Accuracy: {acc:.2%}  (elapsed: {elapsed/60:.1f} min)")
    print(f"  Per-round: {np.mean(round_accs):.2%} ± {np.std(round_accs):.2%}")
    print(f"{'='*70}")
