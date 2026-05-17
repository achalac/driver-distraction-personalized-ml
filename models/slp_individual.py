"""
Driver Distraction Detection using Personalized 
Machine Learning

Authors: Achala Aponso, Craig Speelman, 
         Michael N. Johnstone
Institution: Edith Cowan University, 
             Joondalup, WA, Australia
Contact: aaponso@our.ecu.edu.au

Associated Publication:
"Driver Distraction Detection using Personalized 
Machine Learning"
Submitted to IEEE Access, 2026

Dataset: https://doi.org/10.5281/zenodo.20233645
Code: https://github.com/achalac/
      driver-distraction-personalized-ml

License: CC BY-NC 4.0

SLP INDIVIDUAL (Personalised) MODEL - IEEE Access Q1 reviewer-safe (FIXED split)

✅ Window-level dataset (5s non-overlapping; windows never cross file boundaries)
✅ IMPORTANT FIX: Temporal split is now done at WINDOW level *within each file/condition*:
   - Normal windows: first X% train, last (1-X)% test
   - Load windows:   first X% train, last (1-X)% test
   => ensures BOTH classes appear in train AND test (no {1:N} test pathology)
✅ Blockwise window validation split (last 10% windows per file, inside TRAIN only)
✅ Balanced training windows to avoid collapse
✅ Threshold tuned ONLY on validation (per participant) to maximize F1
✅ Metrics (REAL + SHUFFLE): Accuracy, Balanced Acc, Precision, Recall, F1, Macro-F1, MCC
✅ Confusion matrices (counts + normalized)
✅ Shuffle negative control (shuffle training labels) evaluated on same test windows
✅ Outputs:
   <project_root>/OUTPUT_SLP_INDIVIDUAL_FIXED/
     - SLP_INDIV_pid_XXX.xlsx
     - LOSS_INDIV_pid_XXX.png
     - SUMMARY_SLP_INDIVIDUAL_FIXED.xlsx
"""

import os
import re
import gc
import time
import logging
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, balanced_accuracy_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PARTICIPANT_RE = re.compile(r"GAC(\d{3})_")

def parse_participant_id(filename: str) -> int:
    m = PARTICIPANT_RE.search(filename)
    if not m:
        raise ValueError(f"Cannot parse participant id from filename: {filename}")
    return int(m.group(1))

def parse_label(filename: str) -> int:
    if "Normal" in filename:
        return 0
    if "Load" in filename:
        return 1
    raise ValueError(f"Cannot determine label from filename: {filename}")

def participant_files(pid: int):
    return [
        f"GAC{pid:03d}_Normal_F.csv",
        f"GAC{pid:03d}_Load_F.csv",
    ]

# ----------------------------
# Utilities
# ----------------------------
def cm_normalized(cm):
    cm = cm.astype(float)
    denom = cm.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return cm / denom

def dist_str(y):
    u, c = np.unique(y, return_counts=True)
    return str({int(k): int(v) for k, v in zip(u, c)})

def balance_windows(X, y, seed=SEED):
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return X, y
    n = min(len(idx0), len(idx1))
    sel = np.concatenate([rng.choice(idx0, n, replace=False), rng.choice(idx1, n, replace=False)])
    rng.shuffle(sel)
    return X[sel], y[sel]

def choose_threshold_on_val(y_true, y_prob):
    best_t, best_f1v = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        pred = (y_prob >= t).astype(int)
        f1v = f1_score(y_true, pred, zero_division=0)
        if f1v > best_f1v:
            best_f1v = f1v
            best_t = float(t)
    return best_t, float(best_f1v)

# ----------------------------
# Load participant sample-level data PER FILE (keeps strict time order)
# ----------------------------
def load_one_file(data_folder, pid, file_id, n_features=16):
    fp = os.path.join(data_folder, file_id)
    if not os.path.exists(fp):
        logging.warning(f"Missing: {fp}")
        return None

    df = pd.read_csv(fp, usecols=range(n_features))

    # Your cleaning
    if "GSR" in df.columns:
        df = df[df["GSR"] >= 0]

    df = df.reset_index(drop=True)
    df = df.fillna(df.median(numeric_only=True))

    df["label"] = parse_label(file_id)
    df["participant_id"] = pid
    df["file_id"] = file_id
    return df

# ----------------------------
# Windowing: make WINDOW-level dataset (mean features) PER FILE
# ----------------------------
def make_window_dataset_per_file(df, fs=128, window_sec=5, feature_cols=None):
    """
    Returns:
      Xw: [n_win, n_features]
      yw: [n_win]
      meta: DataFrame with file_id, participant_id, window_index, window_start_sample, label
    """
    if df is None or len(df) == 0:
        return np.empty((0, 16), dtype=np.float32), np.empty((0,), dtype=int), pd.DataFrame()

    if feature_cols is None:
        feature_cols = list(df.columns[:16])

    win_len = int(fs * window_sec)

    X_list, y_list, meta_rows = [], [], []
    file_id = str(df["file_id"].iloc[0])
    pid = int(df["participant_id"].iloc[0])

    n = len(df)
    w_idx = 0
    for start in range(0, n - win_len + 1, win_len):
        w = df.iloc[start:start + win_len]
        xw = w[feature_cols].mean(axis=0).values.astype(np.float32)
        yw = int(np.round(w["label"].mean()))

        X_list.append(xw)
        y_list.append(yw)
        meta_rows.append({
            "file_id": file_id,
            "participant_id": pid,
            "window_index": int(w_idx),
            "window_start_sample": int(start),
            "label": int(yw),
        })
        w_idx += 1

    Xw = np.asarray(X_list, dtype=np.float32)
    yw = np.asarray(y_list, dtype=int)
    meta = pd.DataFrame(meta_rows)
    return Xw, yw, meta

# ----------------------------
# FIXED temporal split: within each FILE at WINDOW level
# ----------------------------
def split_windows_within_each_file(Xw, yw, meta, train_fraction=0.70):
    """
    For each file_id, take early windows for train, later windows for test.
    Ensures both classes exist in test as long as both files have windows.
    """
    if len(yw) == 0:
        return (np.empty((0, Xw.shape[1]), dtype=np.float32), np.empty((0,), dtype=int), pd.DataFrame(),
                np.empty((0, Xw.shape[1]), dtype=np.float32), np.empty((0,), dtype=int), pd.DataFrame())

    train_idx = []
    test_idx = []

    for file_id, g in meta.groupby("file_id", sort=False):
        idx = g.index.to_numpy()
        # idx is already in time order by construction (window_index increasing)
        cut = int(np.floor(train_fraction * len(idx)))
        cut = max(1, min(cut, len(idx) - 1)) if len(idx) >= 2 else len(idx)
        train_idx.extend(idx[:cut])
        test_idx.extend(idx[cut:])

    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    X_tr, y_tr, m_tr = Xw[train_idx], yw[train_idx], meta.loc[train_idx].reset_index(drop=True)
    X_te, y_te, m_te = Xw[test_idx], yw[test_idx], meta.loc[test_idx].reset_index(drop=True)
    return X_tr, y_tr, m_tr, X_te, y_te, m_te

# ----------------------------
# Validation split inside TRAIN only (blockwise per file)
# ----------------------------
def blockwise_val_mask(meta, val_fraction=0.1):
    mask = np.zeros(len(meta), dtype=bool)
    for file_id, g in meta.groupby("file_id", sort=False):
        idx = g.index.to_numpy()
        cut = int((1.0 - val_fraction) * len(idx))
        mask[idx[cut:]] = True
    return mask

# ----------------------------
# Model (SLP)
# ----------------------------
def build_slp(input_dim, lr=0.001, l2=0.001, units=128):
    model = Sequential([
        Dense(units, activation="relu",
              kernel_regularizer=tf.keras.regularizers.l2(l2),
              input_shape=(input_dim,)),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy")
    return model

# ----------------------------
# Plots / Excel
# ----------------------------
def save_loss_plot(out_folder, pid, history):
    png_path = os.path.join(out_folder, f"LOSS_INDIV_pid_{pid:03d}.png")
    plt.figure()
    plt.plot(history.history.get("loss", []), label="train_loss")
    if "val_loss" in history.history:
        plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss (Individual {pid:03d})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()
    return png_path

def save_participant_excel(out_folder, pid, settings, metrics_main, cm_counts, cm_norm,
                           metrics_shuffle, distributions, history_df=None):
    os.makedirs(out_folder, exist_ok=True)
    xlsx_path = os.path.join(out_folder, f"SLP_INDIV_pid_{pid:03d}.xlsx")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame([settings]).to_excel(writer, sheet_name="Settings", index=False)
        pd.DataFrame([metrics_main]).to_excel(writer, sheet_name="WindowMetrics", index=False)
        pd.DataFrame([distributions]).to_excel(writer, sheet_name="Distributions", index=False)
        pd.DataFrame(cm_counts, index=["True0", "True1"], columns=["Pred0", "Pred1"]).to_excel(writer, sheet_name="CM_Counts")
        pd.DataFrame(cm_norm, index=["True0", "True1"], columns=["Pred0", "Pred1"]).to_excel(writer, sheet_name="CM_Normalized")
        pd.DataFrame([metrics_shuffle]).to_excel(writer, sheet_name="ShuffleControl", index=False)
        if history_df is not None:
            history_df.to_excel(writer, sheet_name="TrainHistory", index=False)

    logging.info(f"Saved participant Excel: {xlsx_path}")

# ----------------------------
# MAIN
# ----------------------------
def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_folder = os.path.join(project_root, "data", "ALLDATAFINALPAS")
    out_folder = os.path.join(project_root, "OUTPUT_SLP_INDIVIDUAL_FIXED")
    os.makedirs(out_folder, exist_ok=True)

    # Settings
    FS = 128
    WINDOW_SEC = 5
    N_FEATURES = 16

    # ✅ FIXED: train fraction applied within each file at WINDOW level
    TRAIN_FRACTION_WINDOWS = 0.70

    LR = 0.001
    L2 = 0.001
    UNITS = 128

    EPOCHS = 50
    BATCH = 2048
    VAL_FRAC = 0.10
    PATIENCE = 6
    SHUFFLE_EPOCHS = 10

    summary_rows = []

    for pid in range(1, 51):
        t0 = time.time()
        logging.info(f"\n=== INDIV participant {pid:03d} ===")

        # Load per file (keeps time order inside each condition)
        dfs = []
        for f in participant_files(pid):
            dfi = load_one_file(data_folder, pid, f, n_features=N_FEATURES)
            if dfi is not None and len(dfi) > 0:
                dfs.append(dfi)

        if not dfs:
            logging.warning(f"No data loaded for participant {pid:03d}. Skipping.")
            continue

        feature_cols = list(dfs[0].columns[:N_FEATURES])

        # Window each file separately, then concat windows
        X_all, y_all, meta_all = [], [], []
        for dfi in dfs:
            Xw, yw, meta = make_window_dataset_per_file(
                dfi, fs=FS, window_sec=WINDOW_SEC, feature_cols=feature_cols
            )
            if len(yw) == 0:
                continue
            X_all.append(Xw); y_all.append(yw); meta_all.append(meta)

        if not X_all:
            logging.warning(f"Participant {pid:03d}: not enough windows. Skipping.")
            continue

        Xw = np.concatenate(X_all, axis=0).astype(np.float32)
        yw = np.concatenate(y_all, axis=0).astype(int)
        meta = pd.concat(meta_all, axis=0, ignore_index=True)

        # ✅ FIXED split: within each file at WINDOW level
        Xw_train, yw_train, meta_train, Xw_test, yw_test, meta_test = split_windows_within_each_file(
            Xw, yw, meta, train_fraction=TRAIN_FRACTION_WINDOWS
        )

        logging.info(
            f"Windows: train={len(yw_train):,} ({dist_str(yw_train)}), "
            f"test={len(yw_test):,} ({dist_str(yw_test)})"
        )

        # Hard guard: both classes must be present in test for valid classification metrics
        if len(np.unique(yw_test)) < 2 or len(np.unique(yw_train)) < 2:
            logging.warning(
                f"Participant {pid:03d}: train/test missing a class "
                f"(train={dist_str(yw_train)}, test={dist_str(yw_test)}). Skipping."
            )
            continue

        # Scale on TRAIN only
        scaler = StandardScaler()
        Xw_train_s = scaler.fit_transform(Xw_train)
        Xw_test_s  = scaler.transform(Xw_test)

        # Blockwise validation INSIDE TRAIN only
        val_mask = blockwise_val_mask(meta_train, val_fraction=VAL_FRAC)
        X_tr, y_tr = Xw_train_s[~val_mask], yw_train[~val_mask]
        X_val, y_val = Xw_train_s[val_mask], yw_train[val_mask]

        # Balance TRAIN only
        X_tr_bal, y_tr_bal = balance_windows(X_tr, y_tr, seed=SEED)

        logging.info(
            f"Train split: X_tr={X_tr.shape}, X_val={X_val.shape}, "
            f"balanced_train={X_tr_bal.shape}, val_dist={dist_str(y_val)}"
        )

        # Train REAL model
        model = build_slp(X_tr_bal.shape[1], lr=LR, l2=L2, units=UNITS)
        early = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)

        logging.info("Training REAL model...")
        hist = model.fit(
            X_tr_bal, y_tr_bal,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH,
            callbacks=[early],
            verbose=1
        )
        hist_df = pd.DataFrame(hist.history)

        loss_png = save_loss_plot(out_folder, pid, hist)
        logging.info(f"Saved loss plot: {loss_png}")

        # Threshold tuning on validation
        val_prob = model.predict(X_val, verbose=0).reshape(-1)
        best_thr, best_val_f1 = choose_threshold_on_val(y_val, val_prob)
        logging.info(f"Best threshold={best_thr:.2f} (val F1={best_val_f1:.3f})")

        # REAL test metrics
        test_prob = model.predict(Xw_test_s, verbose=0).reshape(-1)
        y_pred = (test_prob >= best_thr).astype(int)

        metrics_main = {
            "accuracy": float(accuracy_score(yw_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(yw_test, y_pred)),
            "precision": float(precision_score(yw_test, y_pred, zero_division=0)),
            "recall": float(recall_score(yw_test, y_pred, zero_division=0)),
            "f1": float(f1_score(yw_test, y_pred, zero_division=0)),
            "macro_f1": float(f1_score(yw_test, y_pred, average="macro", zero_division=0)),
            "mcc": float(matthews_corrcoef(yw_test, y_pred)),
            "n_test_windows": int(len(yw_test)),
            "threshold_used": float(best_thr),
            "val_f1_at_threshold": float(best_val_f1),
        }

        cm = confusion_matrix(yw_test, y_pred, labels=[0, 1])
        cmn = cm_normalized(cm)

        distributions = {
            "train_windows_total": int(len(yw_train)),
            "train_windows_val": int(val_mask.sum()),
            "train_label_dist_all": dist_str(yw_train),
            "train_label_dist_train": dist_str(y_tr),
            "train_label_dist_train_balanced": dist_str(y_tr_bal),
            "val_label_dist": dist_str(y_val),
            "test_label_dist": dist_str(yw_test),
            "test_pred_dist": dist_str(y_pred),
        }

        # SHUFFLE negative control (shuffle TRAIN labels only)
        logging.info("Training SHUFFLE control model...")
        y_tr_shuf = y_tr_bal.copy()
        np.random.shuffle(y_tr_shuf)

        model_s = build_slp(X_tr_bal.shape[1], lr=LR, l2=L2, units=UNITS)
        model_s.fit(
            X_tr_bal, y_tr_shuf,
            validation_data=(X_val, y_val),
            epochs=SHUFFLE_EPOCHS,
            batch_size=BATCH,
            callbacks=[early],
            verbose=0
        )
        test_prob_s = model_s.predict(Xw_test_s, verbose=0).reshape(-1)
        y_pred_s = (test_prob_s >= best_thr).astype(int)

        metrics_shuffle = {
            "accuracy": float(accuracy_score(yw_test, y_pred_s)),
            "balanced_accuracy": float(balanced_accuracy_score(yw_test, y_pred_s)),
            "precision": float(precision_score(yw_test, y_pred_s, zero_division=0)),
            "recall": float(recall_score(yw_test, y_pred_s, zero_division=0)),
            "f1": float(f1_score(yw_test, y_pred_s, zero_division=0)),
            "macro_f1": float(f1_score(yw_test, y_pred_s, average="macro", zero_division=0)),
            "mcc": float(matthews_corrcoef(yw_test, y_pred_s)),
        }

        settings = {
            "participant": pid,
            "fs": FS,
            "window_sec": WINDOW_SEC,
            "n_features": N_FEATURES,
            "train_fraction_windows_within_each_file": TRAIN_FRACTION_WINDOWS,
            "lr": LR,
            "l2": L2,
            "units": UNITS,
            "epochs": EPOCHS,
            "batch": BATCH,
            "val_frac_windows": VAL_FRAC,
            "early_stopping_patience": PATIENCE,
            "shuffle_epochs": SHUFFLE_EPOCHS,
        }

        save_participant_excel(
            out_folder, pid,
            settings, metrics_main,
            cm, cmn,
            metrics_shuffle, distributions,
            history_df=hist_df
        )

        summary_rows.append({
            "participant": pid,
            "accuracy": metrics_main["accuracy"],
            "balanced_accuracy": metrics_main["balanced_accuracy"],
            "precision": metrics_main["precision"],
            "recall": metrics_main["recall"],
            "f1": metrics_main["f1"],
            "macro_f1": metrics_main["macro_f1"],
            "mcc": metrics_main["mcc"],
            "threshold": metrics_main["threshold_used"],
            "n_test_windows": metrics_main["n_test_windows"],
            "shuffle_accuracy": metrics_shuffle["accuracy"],
            "shuffle_balanced_accuracy": metrics_shuffle["balanced_accuracy"],
            "shuffle_precision": metrics_shuffle["precision"],
            "shuffle_recall": metrics_shuffle["recall"],
            "shuffle_f1": metrics_shuffle["f1"],
            "shuffle_macro_f1": metrics_shuffle["macro_f1"],
            "shuffle_mcc": metrics_shuffle["mcc"],
            "test_label_dist": distributions["test_label_dist"],
            "test_pred_dist": distributions["test_pred_dist"],
        })

        elapsed = time.time() - t0
        logging.info(
            f"✅ Participant {pid:03d} done in {elapsed/60:.1f} min | "
            f"REAL F1={metrics_main['f1']:.3f} MCC={metrics_main['mcc']:.3f} | "
            f"SHUFFLE F1={metrics_shuffle['f1']:.3f}"
        )

        tf.keras.backend.clear_session()
        gc.collect()

    # ----------------------------
    # SUMMARY EXCEL
    # ----------------------------
    if not summary_rows:
        logging.error("No participants completed.")
        return

    summary_df = pd.DataFrame(summary_rows).sort_values("participant")
    summary_path = os.path.join(out_folder, "SUMMARY_SLP_INDIVIDUAL_FIXED.xlsx")

    def mean_std(col):
        return float(summary_df[col].mean()), float(summary_df[col].std(ddof=1))

    stats = {"n_participants": int(len(summary_df))}

    for col in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "macro_f1", "mcc"]:
        m, s = mean_std(col)
        stats[f"{col}_mean"] = m
        stats[f"{col}_std"] = s

    for col in ["shuffle_accuracy", "shuffle_balanced_accuracy", "shuffle_precision", "shuffle_recall",
                "shuffle_f1", "shuffle_macro_f1", "shuffle_mcc"]:
        m, s = mean_std(col)
        stats[f"{col}_mean"] = m
        stats[f"{col}_std"] = s

    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame([stats]).to_excel(writer, sheet_name="SummaryStats", index=False)

    logging.info(f"✅ Saved summary: {summary_path}")

if __name__ == "__main__":
    main()
