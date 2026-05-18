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

LSTM INDIVIDUAL (Personalised) MODEL 


"""

import os
import gc
import time
import logging
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, balanced_accuracy_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ----------------------------
# Filename helpers
# ----------------------------
def participant_files(pid: int):
    return [
        f"GAC{pid:03d}_Normal_F.csv",
        f"GAC{pid:03d}_Load_F.csv",
    ]

def parse_label(filename: str) -> int:
    if "Normal" in filename:
        return 0
    if "Load" in filename:
        return 1
    raise ValueError(f"Cannot infer label from {filename}")


# ----------------------------
# Utils
# ----------------------------
def dist_str(y):
    u, c = np.unique(y, return_counts=True)
    return str({int(k): int(v) for k, v in zip(u, c)})

def cm_normalized(cm):
    cm = cm.astype(float)
    denom = cm.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return cm / denom

def blockwise_val_mask(meta_df: pd.DataFrame, val_fraction: float = 0.1) -> np.ndarray:
    """
    meta_df must include column: file_id
    mark last val_fraction windows of each file as validation.
    """
    mask = np.zeros(len(meta_df), dtype=bool)
    for fid, g in meta_df.groupby("file_id", sort=False):
        idx = g.index.to_numpy()
        cut = int((1.0 - val_fraction) * len(idx))
        mask[idx[cut:]] = True
    return mask

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
# Load single file (keeps strict time order)
# ----------------------------
def load_one_file(data_folder, pid, file_id, n_features=16):
    fp = os.path.join(data_folder, file_id)
    if not os.path.exists(fp):
        logging.warning(f"Missing: {fp}")
        return None

    df = pd.read_csv(fp, usecols=range(n_features))

    # your cleaning
    if "GSR" in df.columns:
        df = df[df["GSR"] >= 0]

    df = df.reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))

    df["label"] = parse_label(file_id)
    df["participant_id"] = pid
    df["file_id"] = file_id
    return df


# ----------------------------
# Window into SEQUENCES per file (5s non-overlapping)
# ----------------------------
def make_sequence_window_dataset(df, fs=128, window_sec=5, n_features=16, downsample=4):
    """
    Returns:
      X_seq: [n_windows, timesteps, n_features]
      y_w:   [n_windows]
      meta:  DataFrame with file_id, participant_id, label, window_start_sample
    """
    if df is None or len(df) == 0:
        return np.empty((0, 1, n_features), dtype=np.float32), np.empty((0,), dtype=int), pd.DataFrame()

    win_len = int(fs * window_sec)
    step = win_len  # non-overlapping
    ds = int(max(1, downsample))
    timesteps = win_len // ds

    feat_cols = list(df.columns[:n_features])

    X_list, y_list, meta_rows = [], [], []

    pid = int(df["participant_id"].iloc[0])
    fid = str(df["file_id"].iloc[0])
    label = int(df["label"].iloc[0])

    n = len(df)
    widx = 0
    for start in range(0, n - win_len + 1, step):
        window = df.iloc[start:start + win_len][feat_cols].to_numpy(dtype=np.float32, copy=False)
        window = window[::ds]
        window = window[:timesteps]
        if window.shape[0] != timesteps:
            continue

        X_list.append(window)
        y_list.append(label)
        meta_rows.append({
            "file_id": fid,
            "participant_id": pid,
            "label": label,
            "window_index": int(widx),
            "window_start_sample": int(start),
        })
        widx += 1

    X_seq = np.asarray(X_list, dtype=np.float32)
    y_w = np.asarray(y_list, dtype=int)
    meta = pd.DataFrame(meta_rows)
    return X_seq, y_w, meta


# ----------------------------
# ✅ FIX: Split windows WITHIN each file
# ----------------------------
def split_windows_within_each_file(X, y, meta, train_fraction=0.70):
    """
    For each file_id: early windows -> train, later windows -> test.
    """
    if len(y) == 0:
        return (np.empty((0,) + X.shape[1:], dtype=np.float32), np.empty((0,), dtype=int), pd.DataFrame(),
                np.empty((0,) + X.shape[1:], dtype=np.float32), np.empty((0,), dtype=int), pd.DataFrame())

    train_idx, test_idx = [], []

    for fid, g in meta.groupby("file_id", sort=False):
        idx = g.index.to_numpy()  # already chronological due to construction
        if len(idx) < 2:
            train_idx.extend(idx.tolist())
            continue

        cut = int(np.floor(train_fraction * len(idx)))
        cut = max(1, min(cut, len(idx) - 1))  # ensure at least 1 window in train and test
        train_idx.extend(idx[:cut].tolist())
        test_idx.extend(idx[cut:].tolist())

    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    X_tr, y_tr, m_tr = X[train_idx], y[train_idx], meta.loc[train_idx].reset_index(drop=True)
    X_te, y_te, m_te = X[test_idx], y[test_idx], meta.loc[test_idx].reset_index(drop=True)
    return X_tr, y_tr, m_tr, X_te, y_te, m_te


# ----------------------------
# Sequence scaling (fit on TRAIN only)
# ----------------------------
def fit_transform_sequence_scaler(X_train_seq):
    """
    Standardize per-feature using TRAIN windows only:
    flatten across (windows * timesteps), compute mean/std for each feature,
    then apply.
    """
    X = X_train_seq.reshape(-1, X_train_seq.shape[-1])
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0

    def transform(Xseq):
        Xf = Xseq.reshape(-1, Xseq.shape[-1])
        Xf = (Xf - mu) / sd
        return Xf.reshape(Xseq.shape)

    return (mu, sd), transform(X_train_seq)

def transform_sequence_scaler(scaler, X_seq):
    mu, sd = scaler
    Xf = X_seq.reshape(-1, X_seq.shape[-1])
    Xf = (Xf - mu) / sd
    return Xf.reshape(X_seq.shape)


# ----------------------------
# Model
# ----------------------------
def build_lstm_model(timesteps, n_features, lr=0.001, lstm_units=64, dropout=0.2):
    model = Sequential([
        LSTM(lstm_units, input_shape=(timesteps, n_features)),
        Dropout(dropout),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy")
    return model


# ----------------------------
# Plot + Excel helpers
# ----------------------------
def save_loss_plot(out_folder, pid, history):
    png_path = os.path.join(out_folder, f"LOSS_LSTM_INDIV_pid_{pid:03d}.png")
    plt.figure()
    plt.plot(history.history.get("loss", []), label="train_loss")
    if "val_loss" in history.history:
        plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"LSTM Individual {pid:03d} Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()
    return png_path

def save_participant_excel(out_folder, pid, settings, metrics_main, cm_counts, cm_norm,
                           metrics_shuffle, distributions, history_df=None):
    xlsx_path = os.path.join(out_folder, f"LSTM_INDIV_pid_{pid:03d}.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame([settings]).to_excel(writer, sheet_name="Settings", index=False)
        pd.DataFrame([metrics_main]).to_excel(writer, sheet_name="WindowMetrics", index=False)
        pd.DataFrame([distributions]).to_excel(writer, sheet_name="Distributions", index=False)
        pd.DataFrame(cm_counts, index=["True0", "True1"], columns=["Pred0", "Pred1"]).to_excel(writer, sheet_name="CM_Counts")
        pd.DataFrame(cm_norm, index=["True0", "True1"], columns=["Pred0", "Pred1"]).to_excel(writer, sheet_name="CM_Normalized")
        pd.DataFrame([metrics_shuffle]).to_excel(writer, sheet_name="ShuffleControl", index=False)
        if history_df is not None:
            history_df.to_excel(writer, sheet_name="TrainHistory", index=False)
    logging.info(f"Saved: {xlsx_path}")


# ----------------------------
# MAIN
# ----------------------------
def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_folder = os.path.join(project_root, "data", "ALLDATAFINALPAS")
    out_folder = os.path.join(project_root, "OUTPUT_LSTM_INDIVIDUAL_FIXED")
    os.makedirs(out_folder, exist_ok=True)

    # Settings (keep your choices)
    FS = 128
    WINDOW_SEC = 5
    DOWNSAMPLE = 4

    TRAIN_FRACTION_WINDOWS = 0.70   # ✅ applied within each file at WINDOW level

    LR = 0.001
    LSTM_UNITS = 64
    DROPOUT = 0.2

    EPOCHS = 40
    BATCH = 128
    VAL_FRAC = 0.10
    PATIENCE = 6

    SHUFFLE_EPOCHS = 12

    summary_rows = []

    for pid in range(1, 51):
        t0 = time.time()
        logging.info(f"\n=== LSTM INDIV participant {pid:03d} ===")

        # Load each file separately
        dfs = []
        for f in participant_files(pid):
            dfi = load_one_file(data_folder, pid, f, n_features=16)
            if dfi is not None and len(dfi) > 0:
                dfs.append(dfi)

        if not dfs:
            logging.warning(f"pid {pid:03d}: no data. Skipping.")
            continue

        # Window each file into sequences, then concat windows
        X_all, y_all, meta_all = [], [], []
        for dfi in dfs:
            Xw, yw, meta = make_sequence_window_dataset(
                dfi, fs=FS, window_sec=WINDOW_SEC, n_features=16, downsample=DOWNSAMPLE
            )
            if len(yw) == 0:
                continue
            X_all.append(Xw); y_all.append(yw); meta_all.append(meta)

        if not X_all:
            logging.warning(f"pid {pid:03d}: not enough windows. Skipping.")
            continue

        X = np.concatenate(X_all, axis=0).astype(np.float32)
        y = np.concatenate(y_all, axis=0).astype(int)
        meta = pd.concat(meta_all, axis=0, ignore_index=True)

        # ✅ FIXED split: within each file at WINDOW level
        X_train, y_train, meta_train, X_test, y_test, meta_test = split_windows_within_each_file(
            X, y, meta, train_fraction=TRAIN_FRACTION_WINDOWS
        )

        if len(y_train) == 0 or len(y_test) == 0:
            logging.warning(f"pid {pid:03d}: empty train/test after split. Skipping.")
            continue

        # Guard: both classes must exist in train and test
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            logging.warning(
                f"pid {pid:03d}: missing class (train={dist_str(y_train)}, test={dist_str(y_test)}). Skipping."
            )
            continue

        timesteps = X_train.shape[1]
        n_features = X_train.shape[2]

        logging.info(
            f"Windows: train={len(y_train)} {dist_str(y_train)} | "
            f"test={len(y_test)} {dist_str(y_test)} | T={timesteps}, F={n_features}"
        )

        # Scale using TRAIN only
        scaler, X_train = fit_transform_sequence_scaler(X_train)
        X_test = transform_sequence_scaler(scaler, X_test)

        # Blockwise validation inside training
        val_mask = blockwise_val_mask(meta_train, val_fraction=VAL_FRAC)
        X_tr, y_tr = X_train[~val_mask], y_train[~val_mask]
        X_val, y_val = X_train[val_mask], y_train[val_mask]

        # Balance TRAIN only
        X_tr_bal, y_tr_bal = balance_windows(X_tr, y_tr, seed=SEED)

        logging.info(
            f"Train split: X_tr={X_tr.shape} {dist_str(y_tr)} | "
            f"balanced={X_tr_bal.shape} {dist_str(y_tr_bal)} | "
            f"val={X_val.shape} {dist_str(y_val)}"
        )

        # Train REAL model
        model = build_lstm_model(timesteps, n_features, lr=LR, lstm_units=LSTM_UNITS, dropout=DROPOUT)
        early = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)

        hist = model.fit(
            X_tr_bal, y_tr_bal,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH,
            callbacks=[early],
            verbose=1
        )
        hist_df = pd.DataFrame(hist.history)
        save_loss_plot(out_folder, pid, hist)

        # Threshold tuning on validation only
        val_prob = model.predict(X_val, verbose=0).reshape(-1)
        thr, val_f1 = choose_threshold_on_val(y_val, val_prob)

        # REAL test metrics
        test_prob = model.predict(X_test, verbose=0).reshape(-1)
        y_pred = (test_prob >= thr).astype(int)

        metrics_main = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "mcc": float(matthews_corrcoef(y_test, y_pred)),
            "threshold_used": float(thr),
            "val_f1_at_threshold": float(val_f1),
            "n_test_windows": int(len(y_test)),
        }

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cmn = cm_normalized(cm)

        distributions = {
            "train_label_dist_all": dist_str(y_train),
            "train_label_dist_train": dist_str(y_tr),
            "train_label_dist_train_balanced": dist_str(y_tr_bal),
            "val_label_dist": dist_str(y_val),
            "test_label_dist": dist_str(y_test),
            "test_pred_dist": dist_str(y_pred),
        }

        # SHUFFLE control: shuffle TRAIN labels only
        y_tr_shuf = y_tr_bal.copy()
        np.random.shuffle(y_tr_shuf)

        model_s = build_lstm_model(timesteps, n_features, lr=LR, lstm_units=LSTM_UNITS, dropout=DROPOUT)
        model_s.fit(
            X_tr_bal, y_tr_shuf,
            validation_data=(X_val, y_val),
            epochs=SHUFFLE_EPOCHS,
            batch_size=BATCH,
            callbacks=[early],
            verbose=0
        )

        test_prob_s = model_s.predict(X_test, verbose=0).reshape(-1)
        y_pred_s = (test_prob_s >= thr).astype(int)

        metrics_shuffle = {
            "accuracy": float(accuracy_score(y_test, y_pred_s)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred_s)),
            "precision": float(precision_score(y_test, y_pred_s, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_s, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred_s, zero_division=0)),
            "macro_f1": float(f1_score(y_test, y_pred_s, average="macro", zero_division=0)),
            "mcc": float(matthews_corrcoef(y_test, y_pred_s)),
        }

        settings = {
            "participant": pid,
            "fs": FS,
            "window_sec": WINDOW_SEC,
            "downsample": DOWNSAMPLE,
            "train_fraction_windows_within_each_file": TRAIN_FRACTION_WINDOWS,
            "lr": LR,
            "lstm_units": LSTM_UNITS,
            "dropout": DROPOUT,
            "epochs": EPOCHS,
            "batch": BATCH,
            "val_frac_windows": VAL_FRAC,
            "early_stopping_patience": PATIENCE,
            "shuffle_epochs": SHUFFLE_EPOCHS,
        }

        save_participant_excel(
            out_folder, pid,
            settings, metrics_main, cm, cmn,
            metrics_shuffle, distributions,
            history_df=hist_df
        )

        summary_rows.append({
            "participant": pid,
            **metrics_main,
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
            f"✅ pid {pid:03d} done in {elapsed/60:.1f} min | "
            f"REAL F1={metrics_main['f1']:.3f} MCC={metrics_main['mcc']:.3f} | "
            f"SHUFFLE F1={metrics_shuffle['f1']:.3f}"
        )

        tf.keras.backend.clear_session()
        gc.collect()

    # Summary Excel
    if not summary_rows:
        logging.error("No participants completed.")
        return

    summary_df = pd.DataFrame(summary_rows).sort_values("participant")
    summary_path = os.path.join(out_folder, "SUMMARY_LSTM_INDIVIDUAL_FIXED.xlsx")

    def mean_std(col):
        return float(summary_df[col].mean()), float(summary_df[col].std(ddof=1))

    stats = {"n_participants": int(len(summary_df))}
    for col in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "macro_f1", "mcc",
                "shuffle_accuracy", "shuffle_balanced_accuracy", "shuffle_precision", "shuffle_recall",
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
