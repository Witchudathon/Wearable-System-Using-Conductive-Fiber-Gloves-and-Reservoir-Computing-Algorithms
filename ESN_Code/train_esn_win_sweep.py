#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESN Win Sweep (Paper-style Table + PNG) — Times New Roman

- Sweep Weight input (W^in) by setting input_scale in ESNConfig
- Report: Accuracy, Precision Macro, Recall Macro, F1 Macro (percent)
- Output:
  1) Prints a table to terminal
  2) Saves CSV: esn_win_sweep_results.csv
  3) Saves PNG table (Times New Roman): esn_win_table.png
  4) (Optional) Saves best model .pkl

CSV input:
- Columns: feature_1, ..., feature_n, label
  OR label column name = 'label'
  OR label is the last column

Usage:
  python train_esn_win_sweep.py --csv dataset.csv
  python train_esn_win_sweep.py --csv dataset.csv --label_col label
  python train_esn_win_sweep.py --csv dataset.csv --win_list 0.1,0.3,0.5,0.7,0.9 --seed 42
  python train_esn_win_sweep.py --csv dataset.csv --save_best_model
"""

import argparse
import os
import pickle
import numpy as np

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -----------------------------
# ESN Core
# -----------------------------
@dataclass
class ESNConfig:
    n_reservoir: int = 400
    spectral_radius: float = 0.9
    sparsity: float = 0.9
    leak_rate: float = 0.25
    input_scale: float = 0.5  # <-- this is W^in in your table
    bias_scale: float = 0.2
    ridge_alpha: float = 1e-3
    seed: int = 42
    washout: int = 0


class ESNClassifier:
    def __init__(self, n_inputs: int, n_classes: int, cfg: ESNConfig):
        self.n_inputs = int(n_inputs)
        self.n_classes = int(n_classes)
        self.cfg = cfg

        rng = np.random.default_rng(cfg.seed)

        # Input weights Win (scaled by input_scale = Win in table)
        self.Win = (
            rng.uniform(-1, 1, size=(cfg.n_reservoir, self.n_inputs)) * cfg.input_scale
        ).astype(np.float32)

        self.bias = (
            rng.uniform(-1, 1, size=(cfg.n_reservoir,)) * cfg.bias_scale
        ).astype(np.float32)

        # Reservoir weights W: sparse random
        W = rng.uniform(-1, 1, size=(cfg.n_reservoir, cfg.n_reservoir)).astype(
            np.float32
        )
        if cfg.sparsity > 0:
            mask = rng.random((cfg.n_reservoir, cfg.n_reservoir)) < cfg.sparsity
            W[mask] = 0.0

        # Scale to spectral radius
        self.W = self._scale_to_spectral_radius(W, cfg.spectral_radius, rng).astype(
            np.float32
        )

        self.Wout = None

    @staticmethod
    def _scale_to_spectral_radius(
        W: np.ndarray, target_radius: float, rng: np.random.Generator
    ) -> np.ndarray:
        # Power iteration approximate spectral radius
        v = rng.normal(size=(W.shape[0],)).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-12

        for _ in range(50):
            v = W @ v
            v /= np.linalg.norm(v) + 1e-12

        wv = W @ v
        radius = float(np.linalg.norm(wv) / (np.linalg.norm(v) + 1e-12))
        if radius < 1e-8:
            return W
        return W * (target_radius / radius)

    def _step(self, state: np.ndarray, x: np.ndarray) -> np.ndarray:
        pre = self.W @ state + self.Win @ x + self.bias
        new_state = np.tanh(pre).astype(np.float32)
        return (
            (1.0 - self.cfg.leak_rate) * state + self.cfg.leak_rate * new_state
        ).astype(np.float32)

    def _collect_states(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        state = np.zeros((self.cfg.n_reservoir,), dtype=np.float32)
        S = np.zeros((N, self.cfg.n_reservoir), dtype=np.float32)

        for i in range(N):
            state = self._step(state, X[i].astype(np.float32))
            S[i] = state

        if self.cfg.washout > 0 and self.cfg.washout < N:
            S[: self.cfg.washout] = 0.0
        return S

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ESNClassifier":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        S = self._collect_states(X)

        ones = np.ones((X.shape[0], 1), dtype=np.float32)
        Z = np.concatenate([ones, X, S], axis=1).astype(np.float32)

        Y = np.zeros((X.shape[0], self.n_classes), dtype=np.float32)
        Y[np.arange(X.shape[0]), y] = 1.0

        A = Z.T @ Z
        A += self.cfg.ridge_alpha * np.eye(A.shape[0], dtype=np.float32)
        B = Z.T @ Y

        self.Wout = np.linalg.solve(A, B).T.astype(np.float32)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.Wout is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        X = np.asarray(X, dtype=np.float32)

        S = self._collect_states(X)
        ones = np.ones((X.shape[0], 1), dtype=np.float32)
        Z = np.concatenate([ones, X, S], axis=1).astype(np.float32)

        logits = (Z @ self.Wout.T).astype(np.float32)
        return np.argmax(logits, axis=1).astype(np.int64)


# -----------------------------
# Data loader (CSV)
# -----------------------------
def load_csv_dataset(
    csv_path: str, label_col: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    import pandas as pd

    df = pd.read_csv(csv_path)

    if label_col is not None:
        if label_col not in df.columns:
            raise ValueError(
                f"label_col '{label_col}' not found. Available: {list(df.columns)}"
            )
        y_raw = df[label_col].values
        X_df = df.drop(columns=[label_col])
    else:
        if "label" in df.columns:
            y_raw = df["label"].values
            X_df = df.drop(columns=["label"])
        else:
            y_raw = df.iloc[:, -1].values
            X_df = df.iloc[:, :-1]

    X = X_df.values.astype(np.float32)

    meta = {
        "columns": list(X_df.columns),
        "label_col": (
            label_col or ("label" if "label" in df.columns else df.columns[-1])
        ),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }
    return X, y_raw, meta


# -----------------------------
# Helpers
# -----------------------------
def parse_win_list(s: str) -> List[float]:
    s = s.replace(" ", ",")
    items = [x.strip() for x in s.split(",") if x.strip() != ""]
    return [float(x) for x in items]


def print_table(rows: List[Dict[str, Any]]):
    headers = [
        "Model",
        "Weight input (Win)",
        "Accuracy (%)",
        "Precision Macro (%)",
        "Recall Macro (%)",
        "F1 Macro (%)",
    ]
    widths = [6, 18, 14, 20, 17, 12]
    sep = "+".join(["-" * w for w in widths])

    def fmt(vals):
        return (
            f"{str(vals[0]):>6}"
            f"{str(vals[1]):>18}"
            f"{str(vals[2]):>14}"
            f"{str(vals[3]):>20}"
            f"{str(vals[4]):>17}"
            f"{str(vals[5]):>12}"
        )

    print(sep)
    print(fmt(headers))
    print(sep)
    for r in rows:
        print(
            fmt(
                [
                    r["model_id"],
                    f'{r["win"]:.1f}',
                    f'{r["acc"]:.2f}',
                    f'{r["prec_macro"]:.2f}',
                    f'{r["rec_macro"]:.2f}',
                    f'{r["f1_macro"]:.2f}',
                ]
            )
        )
    print(sep)


def save_csv(rows: List[Dict[str, Any]], path: str):
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(
            ["Model", "Win", "Accuracy", "Precision_Macro", "Recall_Macro", "F1_Macro"]
        )
        for r in rows:
            wr.writerow(
                [
                    r["model_id"],
                    r["win"],
                    r["acc"],
                    r["prec_macro"],
                    r["rec_macro"],
                    r["f1_macro"],
                ]
            )


def save_table_image(
    rows: List[Dict[str, Any]],
    path: str = "esn_win_table.png",
    font_name: str = "Times New Roman",
    dpi: int = 300,
):
    """
    Save paper-style table as PNG with Times New Roman.

    Note:
    - If Times New Roman isn't installed on your OS, matplotlib will fallback.
    - On macOS, it should exist by default. On Linux, install ttf-mscorefonts or use Times/DejaVu Serif.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    # Try to use Times New Roman; fallback to serif if not found.
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    if font_name in available_fonts:
        matplotlib.rcParams["font.family"] = font_name
    else:
        matplotlib.rcParams["font.family"] = "serif"

    headers = [
        "Model",
        r"Weight input ($W^{in}$)",
        "Accuracy (%)",
        "Precision Macro (%)",
        "Recall Macro (%)",
        "F1 Macro (%)",
    ]

    table_data = []
    for r in rows:
        table_data.append(
            [
                str(r["model_id"]),
                f'{r["win"]:.1f}',
                f'{r["acc"]:.2f}',
                f'{r["prec_macro"]:.2f}',
                f'{r["rec_macro"]:.2f}',
                f'{r["f1_macro"]:.2f}',
            ]
        )

    # Figure sizing: scale with number of rows
    fig_h = max(2.8, 0.55 + 0.55 * (len(rows) + 1))
    fig_w = 16
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )

    # Style
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 1.8)

    # Make header bold + thicker borders
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(1.2)
        if r == 0:
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved table image -> {path}")


# -----------------------------
# Sweep / Train / Evaluate
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to dataset CSV")
    ap.add_argument("--label_col", default=None, help="Label column name (optional)")

    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--win_list", default="0.1,0.3,0.5,0.7,0.9", help="Comma-separated Win values"
    )

    # ESN base hyperparams (kept constant while sweeping Win)
    ap.add_argument("--n_reservoir", type=int, default=400)
    ap.add_argument("--spectral_radius", type=float, default=0.9)
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--leak_rate", type=float, default=0.25)
    ap.add_argument("--bias_scale", type=float, default=0.2)
    ap.add_argument("--ridge_alpha", type=float, default=1e-3)
    ap.add_argument("--washout", type=int, default=0)

    ap.add_argument("--out_csv", default="esn_win_sweep_results.csv")
    ap.add_argument("--out_png", default="esn_win_table.png")
    ap.add_argument("--font", default="Times New Roman")

    ap.add_argument("--save_best_model", action="store_true")
    ap.add_argument("--best_model_path", default="esn_best_win.pkl")

    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(args.csv)

    # Load
    X, y_raw, meta = load_csv_dataset(args.csv, args.label_col)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Fixed split (important for fair sweep)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    # Fixed scaling (important for ESN stability)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    win_values = parse_win_list(args.win_list)

    rows: List[Dict[str, Any]] = []
    best = None  # (acc, bundle, win)

    for idx, win in enumerate(win_values, start=1):
        cfg = ESNConfig(
            n_reservoir=args.n_reservoir,
            spectral_radius=args.spectral_radius,
            sparsity=args.sparsity,
            leak_rate=args.leak_rate,
            input_scale=float(win),
            bias_scale=args.bias_scale,
            ridge_alpha=args.ridge_alpha,
            seed=args.seed,
            washout=args.washout,
        )

        model = ESNClassifier(
            n_inputs=X_train_s.shape[1], n_classes=len(le.classes_), cfg=cfg
        )
        model.fit(X_train_s, y_train)

        pred = model.predict(X_test_s)

        acc = accuracy_score(y_test, pred) * 100.0
        prec_macro = (
            precision_score(y_test, pred, average="macro", zero_division=0) * 100.0
        )
        rec_macro = recall_score(y_test, pred, average="macro", zero_division=0) * 100.0
        f1_macro = f1_score(y_test, pred, average="macro", zero_division=0) * 100.0

        rows.append(
            {
                "model_id": idx,
                "win": float(win),
                "acc": float(acc),
                "prec_macro": float(prec_macro),
                "rec_macro": float(rec_macro),
                "f1_macro": float(f1_macro),
            }
        )

        if best is None or acc > best[0]:
            bundle = {
                "meta": meta,
                "label_encoder": le,
                "scaler": scaler,
                "esn_config": cfg,
                "Win": model.Win,
                "bias": model.bias,
                "W": model.W,
                "Wout": model.Wout,
                "n_inputs": model.n_inputs,
                "n_classes": model.n_classes,
                "classes": le.classes_,
            }
            best = (acc, bundle, float(win))

    # Print terminal table
    print("========== ESN Win Sweep Results ==========")
    print(
        f"Samples: {meta['n_samples']} | Features: {meta['n_features']} | Classes: {len(le.classes_)}"
    )
    print(f"Test size: {args.test_size} | Seed: {args.seed}")
    print_table(rows)

    # Save CSV + PNG table
    save_csv(rows, args.out_csv)
    print(f"Saved table CSV -> {args.out_csv}")

    save_table_image(rows, path=args.out_png, font_name=args.font, dpi=300)

    # Save best model bundle
    if args.save_best_model and best is not None:
        with open(args.best_model_path, "wb") as f:
            pickle.dump(best[1], f)
        print(
            f"Saved best model -> {args.best_model_path} (best Win={best[2]:.1f}, acc={best[0]:.2f}%)"
        )


if __name__ == "__main__":
    main()
