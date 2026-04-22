#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_svm.py — Support Vector Machine (SVM) Training
═════════════════════════════════════════════════════════
SVM with RBF kernel using row-level 72 features + mode voting.
Each row is classified independently, then file-level prediction
is determined by majority vote across all rows in the file.
"""

import json, time
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score

from utils import (
    read_all_csvs, add_row_features, pick_test_files, get_rounds,
    save_model_outputs, BASE_OUT, SEED, TEST_FILES_PER_CLASS_PER_ROUND,
)

# ═══════════════════════════════════════════════════════════════
#  SVM CONFIG
# ═══════════════════════════════════════════════════════════════

SVM_C      = 10.0
SVM_GAMMA  = "scale"
SVM_KERNEL = "rbf"


# ═══════════════════════════════════════════════════════════════
#  TRAIN
# ═══════════════════════════════════════════════════════════════

def train_svm(df_feat, feature_cols, file_groups, classes, rounds):
    out = BASE_OUT / "svm"
    X, y = df_feat[feature_cols], df_feat["label"]
    n_cls = len(classes)
    cm_agg = np.zeros((n_cls, n_cls), dtype=int)
    y_true_all, y_pred_all, round_accs = [], [], []

    for r in range(rounds):
        test_files, train_files = pick_test_files(file_groups, TEST_FILES_PER_CLASS_PER_ROUND, SEED, r)
        train_mask = df_feat["file_id"].isin(train_files)
        test_mask  = df_feat["file_id"].isin(test_files)
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        file_test = df_feat.loc[test_mask, "file_id"].values

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(C=SVM_C, kernel=SVM_KERNEL, gamma=SVM_GAMMA,
                        class_weight="balanced", random_state=SEED + r)),
        ])
        pipe.fit(X_train, y_train)

        # Row-level prediction → mode voting per file
        y_pred_row = pipe.predict(X_test)
        pred_df = pd.DataFrame({"file_id": file_test, "y_true": y_test.values, "y_pred": y_pred_row})
        true_pf = pred_df.groupby("file_id")["y_true"].agg(lambda s: s.mode().iat[0])
        pred_pf = pred_df.groupby("file_id")["y_pred"].agg(lambda s: s.mode().iat[0])

        round_acc = accuracy_score(true_pf.values, pred_pf.values)
        round_accs.append(round_acc)
        y_true_all.extend(true_pf.values); y_pred_all.extend(pred_pf.values)
        cm_agg += confusion_matrix(true_pf.values, pred_pf.values, labels=classes)

        if (r + 1) % 5 == 0 or r == rounds - 1:
            print(f"    SVM Round {r+1}/{rounds}  (round acc={round_acc:.2%})", flush=True)

    def save_svm(pipe, d):
        joblib.dump({"pipeline": pipe, "classes": classes, "feature_cols": feature_cols},
                    d / "svm_model.joblib")

    acc, _ = save_model_outputs("SVM", out, cm_agg, y_true_all, y_pred_all,
        classes,
        {"kernel": SVM_KERNEL, "C": SVM_C, "gamma": str(SVM_GAMMA),
         "n_features": len(feature_cols), "approach": "row-level + mode voting"},
        pipe, save_svm)
    return acc, round_accs


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 70)
    print("  SVM Training — RBF Kernel, C=10, Row-Level + Mode Voting")
    print("=" * 70)

    print("\n  [1/3] Loading data ...", flush=True)
    df_raw = read_all_csvs()
    classes = sorted(df_raw["label"].unique())
    file_groups = {cls: list(df_raw.loc[df_raw["label"] == cls, "file_id"].unique()) for cls in classes}
    rounds = get_rounds()
    print(f"         Shape: {df_raw.shape}, Classes: {len(classes)}, Rounds: {rounds}")

    print("\n  [2/3] Feature engineering (72 row-level features) ...", flush=True)
    df_feat, feature_cols = add_row_features(df_raw)
    print(f"         Features: {len(feature_cols)}")

    print(f"\n  [3/3] Training SVM ...", flush=True)
    acc, round_accs = train_svm(df_feat, feature_cols, file_groups, classes, rounds)

    # Save round metrics
    metrics = {"SVM": {"overall_accuracy": acc, "round_accuracies": round_accs}}
    mf = BASE_OUT / "svm" / "round_metrics.json"
    mf.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  SVM Accuracy: {acc:.2%}  (elapsed: {elapsed/60:.1f} min)")
    print(f"  Per-round: {np.mean(round_accs):.2%} ± {np.std(round_accs):.2%}")
    print(f"{'='*70}")
