#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py — Shared utilities for all training scripts
═══════════════════════════════════════════════════════
Data loading, feature engineering, evaluation protocol,
plotting, and output functions.
"""

import json, math, random, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASE_OUT = Path(__file__).resolve().parent.parent / "optimized"

EMA_ALPHA  = 0.2
ROLL_SHORT = 5
ROLL_LONG  = 15
SEED       = 42

TARGET_TEST_FILES_PER_CLASS    = 50
TEST_FILES_PER_CLASS_PER_ROUND = 2
FINGERS = ["Thumb", "Index", "Middle", "Ring", "Little"]

random.seed(SEED)
np.random.seed(SEED)

rcParams.update({
    "font.family": "serif", "font.size": 12,
    "axes.titlesize": 16, "axes.titleweight": "bold",
    "axes.labelsize": 13, "axes.labelweight": "bold",
    "axes.linewidth": 1.2, "axes.edgecolor": "#333333",
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "legend.framealpha": 0.9, "legend.edgecolor": "#CCCCCC",
    "figure.titlesize": 20, "figure.titleweight": "bold",
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.bbox": "tight", "savefig.dpi": 200,
    "grid.alpha": 0.3, "grid.linewidth": 0.6,
})


# ═══════════════════════════════════════════════════════════════
#  DATA I/O
# ═══════════════════════════════════════════════════════════════

def read_all_csvs():
    rows = []
    for class_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        label = class_dir.name
        for csv in class_dir.rglob("*.csv"):
            d = pd.read_csv(csv)
            d["label"] = label
            d["file_id"] = str(csv)
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


def ensure_ema(g):
    for f in FINGERS:
        ohm, ema_c = f"ohm_{f}", f"ema_{f}"
        if ema_c not in g.columns and ohm in g.columns:
            g[ema_c] = g[ohm].ewm(alpha=EMA_ALPHA, adjust=False).mean()
    return g


def pick_test_files(file_groups, k_per_class, seed, round_idx):
    test_files, train_files = [], []
    for cls, files in file_groups.items():
        fl = files[:]
        rnd = random.Random(seed + round_idx * 997 + hash(cls) % 100000)
        rnd.shuffle(fl)
        test_files.extend(fl[:k_per_class])
        train_files.extend(fl[k_per_class:])
    return test_files, train_files


def get_rounds():
    return math.ceil(TARGET_TEST_FILES_PER_CLASS / TEST_FILES_PER_CLASS_PER_ROUND)


# ═══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING — ROW-LEVEL (SVM, RF)
# ═══════════════════════════════════════════════════════════════

def add_row_features(df):
    """72 row-level features for SVM/RF (row-level + mode voting)."""
    df = df.copy()

    def per_file_features(g):
        g = ensure_ema(g)
        g["ohm_sum"] = g[[f"ohm_{f}" for f in FINGERS]].sum(axis=1).replace(0, 1e-6)
        g["ema_sum"] = g[[f"ema_{f}" for f in FINGERS]].sum(axis=1).replace(0, 1e-6)

        for f in FINGERS:
            o, e = f"ohm_{f}", f"ema_{f}"
            if o not in g.columns: g[o] = np.nan
            if e not in g.columns: g[e] = np.nan
            g[f"{o}_rel"] = g[o] / g["ohm_sum"]
            g[f"{e}_rel"] = g[e] / g["ema_sum"]
            g[f"{o}_diff1"] = g[o].diff().fillna(0)
            g[f"{o}_diff3"] = g[o].diff(ROLL_SHORT).fillna(0)
            g[f"{o}_pct"] = g[o].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
            g[f"{o}_stdS"] = g[o].rolling(ROLL_SHORT).std().fillna(0)
            g[f"{o}_stdL"] = g[o].rolling(ROLL_LONG).std().fillna(0)
            g[f"{o}_madS"] = (g[o] - g[o].rolling(ROLL_SHORT).median()).abs().rolling(ROLL_SHORT).median().fillna(0)
            g[f"{f}_resid"] = (g[o] - g[e]).fillna(0)
            g[f"{f}_resid_diff"] = g[f"{f}_resid"].diff().fillna(0)
            g[f"{o}_slopeS"] = (g[o] - g[o].shift(ROLL_SHORT)).fillna(0) / max(1, ROLL_SHORT)

        for a, b in [("Thumb","Index"),("Index","Middle"),("Middle","Ring"),("Ring","Little"),("Thumb","Little")]:
            g[f"ratio_{a}_{b}"] = (g[f"ohm_{a}"] + 1e-6) / (g[f"ohm_{b}"] + 1e-6)

        diffs = [f"ohm_{f}_diff1" for f in FINGERS]
        stds  = [f"ohm_{f}_stdS" for f in FINGERS]
        g["move_energyS"] = g[diffs].abs().sum(axis=1)
        g["steadinessS"]  = 1.0 / (g[stds].sum(axis=1) + 1e-6)
        return g

    _fid = df["file_id"].copy()
    df = df.groupby("file_id", group_keys=False).apply(per_file_features)
    if "file_id" not in df.columns:
        df["file_id"] = _fid

    feature_cols = []
    for f in FINGERS:
        feature_cols += [f"ohm_{f}", f"ema_{f}", f"ohm_{f}_rel", f"ema_{f}_rel",
                         f"ohm_{f}_diff1", f"ohm_{f}_diff3", f"ohm_{f}_pct",
                         f"ohm_{f}_stdS", f"ohm_{f}_stdL", f"ohm_{f}_madS",
                         f"{f}_resid", f"{f}_resid_diff", f"ohm_{f}_slopeS"]
    feature_cols += [f"ratio_{a}_{b}" for a, b in
                     [("Thumb","Index"),("Index","Middle"),("Middle","Ring"),("Ring","Little"),("Thumb","Little")]]
    feature_cols += ["move_energyS", "steadinessS"]
    return df, feature_cols


# ═══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING — SEQUENCE-LEVEL (ESN, LSTM, GRU)
# ═══════════════════════════════════════════════════════════════

def get_enhanced_sensor_cols():
    cols = []
    for f in FINGERS:
        cols += [f"ohm_{f}", f"ema_{f}", f"ohm_{f}_diff1",
                 f"ohm_{f}_pct", f"ohm_{f}_rstd3", f"ohm_{f}_rel"]
    return cols


def add_timestep_features(df):
    df = df.copy()
    _fid = df["file_id"].copy()
    df = df.groupby("file_id", group_keys=False).apply(ensure_ema)
    if "file_id" not in df.columns:
        df["file_id"] = _fid
    ohm_sum = df[[f"ohm_{f}" for f in FINGERS]].sum(axis=1).replace(0, 1e-6)

    for f in FINGERS:
        o = f"ohm_{f}"
        df[f"{o}_diff1"] = df.groupby("file_id")[o].diff().fillna(0)
        df[f"{o}_pct"]   = df.groupby("file_id")[o].pct_change().replace([np.inf,-np.inf],0).fillna(0)
        df[f"{o}_rstd3"] = df.groupby("file_id")[o].transform(
            lambda s: s.rolling(3, min_periods=1).std().fillna(0))
        df[f"{o}_rel"]   = df[o] / ohm_sum
    return df


# ═══════════════════════════════════════════════════════════════
#  PLOTTING & OUTPUT
# ═══════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, classes, title, path, normalized=False):
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(cmap="YlGnBu" if normalized else "Blues", ax=ax,
              xticks_rotation=45, colorbar=True,
              values_format=".0%" if normalized else "d")
    ax.set_title(title, pad=15)
    ax.set_xlabel("Predicted Label", labelpad=10)
    ax.set_ylabel("True Label", labelpad=10)
    fs = 7 if normalized else 10
    for t in disp.text_.ravel(): t.set_fontsize(fs)
    plt.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_per_class_metrics(report_dict, classes, title, path):
    metrics = ["precision", "recall", "f1-score"]
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(12, n * 1.2), 6))
    x = np.arange(n); w = 0.22; offs = np.arange(3) - 1
    colors = ["#1B3A5C", "#2E6E9E", "#3D8B7A"]
    for i, (m, c) in enumerate(zip(metrics, colors)):
        vals = [report_dict.get(cls, {}).get(m, 0) for cls in classes]
        bars = ax.bar(x + offs[i]*w, vals, w, label=m.title(), color=c, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score"); ax.set_title(title, pad=12)
    ax.legend(loc="upper right", ncol=3, frameon=True)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    plt.tight_layout(); fig.savefig(path); plt.close(fig)


def save_model_outputs(model_name, out_dir, cm_agg, y_true_all, y_pred_all,
                       classes, meta_extra, model_obj=None, save_fn=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    acc = accuracy_score(y_true_all, y_pred_all)
    report = classification_report(y_true_all, y_pred_all, labels=classes,
                                   output_dict=True, zero_division=0)
    cm_norm = (cm_agg.T / np.maximum(cm_agg.sum(axis=1), 1)).T

    # Save CM data for future regeneration
    np.save(out_dir / "cm_agg.npy", cm_agg)
    np.save(out_dir / "cm_norm.npy", cm_norm)

    plot_confusion_matrix(cm_agg, classes, f"{model_name} — Confusion Matrix (Counts)",
                          out_dir / "confusion_matrix_counts.png")
    plot_confusion_matrix(cm_norm, classes, f"{model_name} — Confusion Matrix (Normalized %)",
                          out_dir / "confusion_matrix_normalized.png", normalized=True)
    plot_per_class_metrics(report, classes,
                           f"{model_name} — Per-Class Metrics", out_dir / "per_class_metrics.png")

    report_str = classification_report(y_true_all, y_pred_all, labels=classes,
                                       zero_division=0, digits=4)
    txt = f"{'='*72}\n  {model_name} — CLASSIFICATION REPORT\n{'='*72}\n\n"
    txt += f"  Overall Accuracy : {acc:.4f}  ({acc:.2%})\n\n{'-'*72}\n{report_str}\n{'='*72}\n"
    (out_dir / "classification_report.txt").write_text(txt, encoding="utf-8")

    if save_fn and model_obj:
        save_fn(model_obj, out_dir)

    meta = {
        "model": model_name, "accuracy": float(acc), "classes": classes,
        "per_class": {
            cls: {"precision": round(report[cls]["precision"], 4),
                  "recall": round(report[cls]["recall"], 4),
                  "f1_score": round(report[cls]["f1-score"], 4),
                  "support": int(report[cls]["support"])}
            for cls in classes
        },
        "evaluation": f"25-round file-level holdout, {TARGET_TEST_FILES_PER_CLASS} votes/class",
    }
    meta.update(meta_extra)
    safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    (out_dir / f"{safe_name}_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"    Saved: {out_dir}/  (Accuracy: {acc:.2%})")
    return acc, report
