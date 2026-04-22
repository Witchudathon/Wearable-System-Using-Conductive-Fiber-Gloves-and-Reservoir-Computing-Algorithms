#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_lstm.py — LSTM Training (NOT Bidirectional)
══════════════════════════════════════════════════
Standard uni-directional LSTM with 2 layers (128→64),
BatchNormalization, Dropout, and probability voting.
Sliding window (30 steps, stride 15) over 30 sensor channels.
"""

import json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from utils import (
    read_all_csvs, add_timestep_features, get_enhanced_sensor_cols,
    pick_test_files, get_rounds, save_model_outputs,
    BASE_OUT, SEED, TEST_FILES_PER_CLASS_PER_ROUND,
)

# ═══════════════════════════════════════════════════════════════
#  LSTM CONFIG
# ═══════════════════════════════════════════════════════════════

DL_WINDOW  = 30
DL_STRIDE  = 15
DL_UNITS1  = 128
DL_UNITS2  = 64
DL_DROPOUT = 0.3
DL_EPOCHS  = 50
DL_BATCH   = 128
DL_LR      = 1e-3


# ═══════════════════════════════════════════════════════════════
#  WINDOWING
# ═══════════════════════════════════════════════════════════════

def make_windows(df, sensor_cols, window, stride, le, scaler=None, fit_scaler=False):
    X_all, y_all, fids = [], [], []
    for fid, g in df.groupby("file_id"):
        vals = g[sensor_cols].values.astype(np.float32)
        label_str = g["label"].iloc[0]
        for start in range(0, len(vals) - window + 1, stride):
            X_all.append(vals[start:start + window])
            y_all.append(label_str)
            fids.append(fid)
    if not X_all:
        return np.array([]), np.array([]), np.array([]), scaler

    X_3d = np.array(X_all, dtype=np.float32)
    y_str = np.array(y_all)
    N, W, F = X_3d.shape
    X_flat = X_3d.reshape(N * W, F)

    if fit_scaler:
        scaler = StandardScaler()
        X_flat = scaler.fit_transform(X_flat)
    else:
        X_flat = scaler.transform(X_flat)

    X_3d = np.nan_to_num(X_flat.reshape(N, W, F).astype(np.float32))
    y_int = le.transform(y_str)
    return X_3d, y_int, np.array(fids), scaler


# ═══════════════════════════════════════════════════════════════
#  MODEL BUILDER — Standard LSTM (NO Bidirectional)
# ═══════════════════════════════════════════════════════════════

def build_lstm(n_timesteps, n_features, n_classes):
    import tensorflow as tf
    tf.random.set_seed(SEED)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_timesteps, n_features)),
        tf.keras.layers.LSTM(DL_UNITS1, return_sequences=True,
                             kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DL_DROPOUT),
        tf.keras.layers.LSTM(DL_UNITS2, return_sequences=False,
                             kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DL_DROPOUT),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=DL_LR),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ═══════════════════════════════════════════════════════════════
#  TRAIN
# ═══════════════════════════════════════════════════════════════

def train_lstm_model(df_ts, enhanced_cols, file_groups, classes, rounds):
    import tensorflow as tf
    tf.random.set_seed(SEED)

    out = BASE_OUT / "lstm"
    out.mkdir(parents=True, exist_ok=True)

    le = LabelEncoder()
    le.fit(sorted(classes))
    n_classes = len(classes)

    cm_agg = np.zeros((n_classes, n_classes), dtype=int)
    y_true_all, y_pred_all, round_accs = [], [], []
    last_history = None

    for r in range(rounds):
        test_files, train_files = pick_test_files(file_groups, TEST_FILES_PER_CLASS_PER_ROUND, SEED, r)
        df_train = df_ts[df_ts["file_id"].isin(train_files)]
        df_test  = df_ts[df_ts["file_id"].isin(test_files)]

        X_train, y_train, fid_train, scaler = make_windows(
            df_train, enhanced_cols, DL_WINDOW, DL_STRIDE, le, fit_scaler=True)
        X_test, y_test, fid_test, _ = make_windows(
            df_test, enhanced_cols, DL_WINDOW, DL_STRIDE, le, scaler=scaler)

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        try:
            cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
            cw_dict = dict(enumerate(cw))
        except Exception:
            cw_dict = None

        model = build_lstm(DL_WINDOW, len(enhanced_cols), n_classes)

        hist = model.fit(X_train, y_train, validation_split=0.15,
                         epochs=DL_EPOCHS, batch_size=DL_BATCH, verbose=0,
                         class_weight=cw_dict,
                         callbacks=[
                             tf.keras.callbacks.EarlyStopping(
                                 patience=8, restore_best_weights=True, monitor="val_loss"),
                             tf.keras.callbacks.ReduceLROnPlateau(
                                 factor=0.5, patience=4, min_lr=1e-6, monitor="val_loss"),
                         ])
        last_history = hist.history

        # Probability voting (average softmax per file, then argmax)
        y_pred_prob = model.predict(X_test, verbose=0)
        prob_df = pd.DataFrame(y_pred_prob, columns=le.classes_)
        prob_df["file_id"] = fid_test
        prob_df["y_true"] = le.inverse_transform(y_test)

        avg_probs = prob_df.groupby("file_id")[list(le.classes_)].mean()
        pred_per_file = avg_probs.idxmax(axis=1)
        true_per_file = prob_df.groupby("file_id")["y_true"].first()

        round_acc = accuracy_score(true_per_file.values, pred_per_file.values)
        round_accs.append(round_acc)
        y_true_all.extend(true_per_file.values)
        y_pred_all.extend(pred_per_file.values)
        cm_agg += confusion_matrix(true_per_file.values, pred_per_file.values, labels=classes)

        print(f"    LSTM Round {r+1}/{rounds}  (round acc={round_acc:.2%})", flush=True)
        tf.keras.backend.clear_session()

    # Training history plot (last round)
    if last_history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ep = range(1, len(last_history["loss"]) + 1)
        ax1.plot(ep, last_history["loss"], color="#1B3A5C", label="Train Loss", linewidth=2)
        ax1.plot(ep, last_history["val_loss"], color="#A0522D", label="Val Loss", linewidth=2, linestyle="--")
        ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.legend(); ax1.grid(True, alpha=0.25, linestyle="--")
        ax2.plot(ep, last_history["accuracy"], color="#1B3A5C", label="Train Acc", linewidth=2)
        ax2.plot(ep, last_history["val_accuracy"], color="#2E6E9E", label="Val Acc", linewidth=2, linestyle="--")
        ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
        ax2.legend(); ax2.grid(True, alpha=0.25, linestyle="--")
        fig.suptitle("LSTM — Training History (Last Round)", fontsize=16, fontweight="bold")
        plt.tight_layout(); fig.savefig(out / "training_history.png"); plt.close(fig)

    acc, _ = save_model_outputs("LSTM", out, cm_agg,
        y_true_all, y_pred_all, classes,
        {"architecture": f"LSTM({DL_UNITS1})→LSTM({DL_UNITS2})",
         "window": DL_WINDOW, "stride": DL_STRIDE,
         "epochs": DL_EPOCHS, "batch": DL_BATCH, "lr": DL_LR,
         "input_channels": len(enhanced_cols),
         "voting": "probability (softmax average)"},
        None, None)
    return acc, round_accs


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 70)
    print(f"  LSTM Training — LSTM({DL_UNITS1})→LSTM({DL_UNITS2}), 30ch, Prob Voting")
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

    print(f"\n  [3/3] Training LSTM ...", flush=True)
    acc, round_accs = train_lstm_model(df_ts, enhanced_cols, file_groups, classes, rounds)

    # Save round metrics
    metrics = {"LSTM": {"overall_accuracy": acc, "round_accuracies": round_accs}}
    mf = BASE_OUT / "lstm" / "round_metrics.json"
    mf.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  LSTM Accuracy: {acc:.2%}  (elapsed: {elapsed/60:.1f} min)")
    print(f"  Per-round: {np.mean(round_accs):.2%} ± {np.std(round_accs):.2%}")
    print(f"{'='*70}")
