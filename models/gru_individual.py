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

GRU INDIVIDUAL (Personalised) MODEL 
Outputs:
  <project_root>/OUTPUT_GRU_INDIVIDUAL/
  <project_root>/CACHE_WINDOWS_GRU_INDIVIDUAL/

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

from tensorflow.keras import layers, Model
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
# Cache utilities (per file)
# ----------------------------
def window_cache_path(cache_dir, file_id, fs, window_sec, n_features):
    safe = file_id.replace(".csv", "")
    return os.path.join(cache_dir, f"{safe}_fs{fs}_w{window_sec}_f{n_features}.npz")

def build_window_tensor_from_csv(csv_path, file_id, participant_id, label,
                                 fs=128, window_sec=5, n_features=16):
    """
    Builds:
      Xw: [n_windows, win_len, n_features]  (sequence per window)
      yw: [n_windows]
      meta: dict with window_start_sample
    """
    df = pd.read_csv(csv_path, usecols=range(n_features))

    # keep your cleaning
    if "GSR" in df.columns:
        df = df[df["GSR"] >= 0]

    df = df.reset_index(drop=True)
    df = df.fillna(df.median(numeric_only=True))

    win_len = int(fs * window_sec)
    n = len(df)
    feat_cols = list(df.columns[:n_features])

    X_list, y_list, start_list = [], [], []

    for start in range(0, n - win_len + 1, win_len):
        window = df.iloc[start:start + win_len][feat_cols].values.astype(np.float32)
        X_list.append(window)
        y_list.append(int(label))
        start_list.append(int(start))

    Xw = np.asarray(X_list, dtype=np.float32)
    yw = np.asarray(y_list, dtype=np.int32)

    meta = {
        "file_id": file_id,
        "participant_id": int(participant_id),
        "label": int(label),
        "window_start_sample": np.asarray(start_list, dtype=np.int32)
    }
    return Xw, yw, meta

def load_or_build_cached_windows(data_folder, file_id, cache_dir,
                                 fs=128, window_sec=5, n_features=16):
    os.makedirs(cache_dir, exist_ok=True)

    participant_id = parse_participant_id(file_id)
    label = parse_label(file_id)
    npz_path = window_cache_path(cache_dir, file_id, fs, window_sec, n_features)

    if os.path.exists(npz_path):
        z = np.load(npz_path, allow_pickle=True)
        return z["Xw"], z["yw"], z["meta"].item()

    csv_path = os.path.join(data_folder, file_id)
    if not os.path.exists(csv_path):
        logging.warning(f"Missing file: {csv_path}")
        return None

    Xw, yw, meta = build_window_tensor_from_csv(
        csv_path, file_id, participant_id, label,
        fs=fs, window_sec=window_sec, n_features=n_features
    )
    np.savez_compressed(npz_path, Xw=Xw, yw=yw, meta=meta)
    return Xw, yw, meta

def concat_windows(items):
    """
    Concatenate list of (Xw,yw,meta) into:
      X: [N,T,F], y: [N], meta_df
    """
    Xs, ys, metas = [], [], []
    for it in items:
        if it is None:
            continue
        Xw, yw, meta = it
        if len(yw) == 0:
            continue
        Xs.append(Xw)
        ys.append(yw)

        mdf = pd.DataFrame({
            "file_id": [meta["file_id"]] * len(yw),
            "participant_id": [meta["participant_id"]] * len(yw),
            "label": [meta["label"]] * len(yw),
            "window_start_sample": meta["window_start_sample"]
        })
        metas.append(mdf)

    if not Xs:
        return np.empty((0, 0, 0), dtype=np.float32), np.empty((0,), dtype=np.int32), pd.DataFrame()

    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.int32)
    meta_df = pd.concat(metas, axis=0, ignore_index=True)
    return X, y, meta_df

# ----------------------------
# Temporal split: earliest TRAIN_FRACTION windows per file -> train; rest -> test
# (prevents mixing early/late within same file)
# ----------------------------
def split_windows_timewise_per_file(meta_df, train_fraction=0.70):
    train_mask = np.zeros(len(meta_df), dtype=bool)
    for file_id, g in meta_df.groupby("file_id", sort=False):
        idx = g.index.to_numpy()
        order = np.argsort(g["window_start_sample"].values)
        idx_sorted = idx[order]
        cut = int(train_fraction * len(idx_sorted))
        train_mask[idx_sorted[:cut]] = True
    test_mask = ~train_mask
    return train_mask, test_mask

# ----------------------------
# Validation split (block-wise per file, within TRAIN)
# ----------------------------
def blockwise_val_mask(meta_df, val_fraction=0.10):
    mask = np.zeros(len(meta_df), dtype=bool)
    for file_id, g in meta_df.groupby("file_id", sort=False):
        idx = g.index.to_numpy()
        idx_sorted = idx[np.argsort(g["window_start_sample"].values)]
        cut = int((1.0 - val_fraction) * len(idx_sorted))
        mask[idx_sorted[cut:]] = True
    return mask

# ----------------------------
# Balancing / threshold
# ----------------------------
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
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        pred = (y_prob >= t).astype(int)
        f1v = f1_score(y_true, pred, zero_division=0)
        if f1v > best_f1:
            best_f1 = f1v
            best_t = float(t)
    return best_t, float(best_f1)

# ----------------------------
# Scaling for sequences
# ----------------------------
def fit_sequence_scaler(X_train_seq):
    N, T, F = X_train_seq.shape
    flat = X_train_seq.reshape(N * T, F)
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler

def transform_sequence_scaler(X_seq, scaler):
    N, T, F = X_seq.shape
    flat = X_seq.reshape(N * T, F)
    flat = scaler.transform(flat)
    return flat.reshape(N, T, F).astype(np.float32)

# ----------------------------
# GRU model
# ----------------------------
def build_gru_classifier(seq_len, n_features,
                         gru_units=64, dropout=0.3,
                         lr=1e-3, l2=1e-4):
    inp = layers.Input(shape=(seq_len, n_features), name="window_seq")
    x = layers.GRU(
        gru_units,
        dropout=dropout,
        recurrent_dropout=0.0,
        kernel_regularizer=tf.keras.regularizers.l2(l2),
        recurrent_regularizer=tf.keras.regularizers.l2(l2),
        return_sequences=False,
    )(inp)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="binary_crossentropy")
    return model

# ----------------------------
# Metrics / saving
# ----------------------------
def cm_normalized(cm):
    cm = cm.astype(float)
    denom = cm.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return cm / denom

def dist_str(y):
    u, c = np.unique(y, return_counts=True)
    return str({int(k): int(v) for k, v in zip(u, c)})

def save_loss_plot(out_folder, pid, history):
    png_path = os.path.join(out_folder, f"LOSS_GRU_INDIV_{pid:03d}.png")
    plt.figure()
    plt.plot(history.history.get("loss", []), label="train_loss")
    if "val_loss" in history.history:
        plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"GRU Loss (Individual {pid:03d})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()
    return png_path

def save_participant_excel(out_folder, pid, settings, metrics_main, cm_counts, cm_norm,
                           metrics_shuffle, distributions, history_df=None):
    os.makedirs(out_folder, exist_ok=True)
    xlsx_path = os.path.join(out_folder, f"GRU_INDIV_{pid:03d}.xlsx")
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
# MAIN (Individual)
# ----------------------------
def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_folder = os.path.join(project_root, "data", "ALLDATAFINALPAS")

    out_folder = os.path.join(project_root, "OUTPUT_GRU_INDIVIDUAL")
    cache_dir  = os.path.join(project_root, "CACHE_WINDOWS_GRU_INDIVIDUAL")
    os.makedirs(out_folder, exist_ok=True)

    # --- Window settings ---
    FS = 128
    WINDOW_SEC = 5
    N_FEATURES = 16
    SEQ_LEN = FS * WINDOW_SEC

    # --- Temporal split ---
    TRAIN_FRACTION = 0.70  # early windows per file -> train, later -> test

    # --- Model hyperparams ---
    LR = 1e-3
    L2 = 1e-4
    GRU_UNITS = 64
    DROPOUT = 0.3

    # --- Training ---
    EPOCHS = 30
    BATCH = 64
    VAL_FRAC = 0.10
    PATIENCE = 6
    SHUFFLE_EPOCHS = 10

    summary_rows = []

    for pid in range(1, 51):
        t0 = time.time()
        logging.info(f"\n=== GRU INDIV participant {pid:03d} ===")

        # Load cached window tensors for this participant's 2 files
        items = [load_or_build_cached_windows(data_folder, f, cache_dir, FS, WINDOW_SEC, N_FEATURES)
                 for f in participant_files(pid)]
        X_all, y_all, meta_all = concat_windows(items)

        if len(y_all) == 0:
            logging.warning(f"Participant {pid:03d}: no windows. Skipping.")
            continue

        # Split timewise per file -> train vs test
        train_mask, test_mask = split_windows_timewise_per_file(meta_all, train_fraction=TRAIN_FRACTION)
        X_train, y_train, meta_train = X_all[train_mask], y_all[train_mask], meta_all.loc[train_mask].reset_index(drop=True)
        X_test,  y_test,  meta_test  = X_all[test_mask],  y_all[test_mask],  meta_all.loc[test_mask].reset_index(drop=True)

        logging.info(
            f"Windows total={len(y_all):,} ({dist_str(y_all)}), "
            f"train={len(y_train):,} ({dist_str(y_train)}), "
            f"test={len(y_test):,} ({dist_str(y_test)})"
        )

        if len(y_train) == 0 or len(y_test) == 0:
            logging.warning(f"Participant {pid:03d}: not enough train/test windows. Skipping.")
            continue

        # Validation split within TRAIN (block-wise per file)
        val_mask = blockwise_val_mask(meta_train, val_fraction=VAL_FRAC)
        X_tr, y_tr = X_train[~val_mask], y_train[~val_mask]
        X_val, y_val = X_train[val_mask], y_train[val_mask]

        # Balance training windows
        X_tr_bal, y_tr_bal = balance_windows(X_tr, y_tr, seed=SEED)
        logging.info(f"Split: X_tr={X_tr.shape}, X_val={X_val.shape}, balanced_train={X_tr_bal.shape}, val_dist={dist_str(y_val)}")

        # Scale (fit on TRAIN only)
        scaler = fit_sequence_scaler(X_tr_bal)
        X_tr_bal = transform_sequence_scaler(X_tr_bal, scaler)
        X_val    = transform_sequence_scaler(X_val, scaler)
        X_test_s = transform_sequence_scaler(X_test, scaler)

        # Train REAL model
        model = build_gru_classifier(
            seq_len=SEQ_LEN, n_features=N_FEATURES,
            gru_units=GRU_UNITS, dropout=DROPOUT,
            lr=LR, l2=L2
        )
        early = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)

        logging.info("Training REAL GRU model...")
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

        # Threshold on validation only
        val_prob = model.predict(X_val, verbose=0).reshape(-1)
        best_thr, best_val_f1 = choose_threshold_on_val(y_val, val_prob)
        logging.info(f"Best threshold={best_thr:.2f} (val F1={best_val_f1:.3f})")

        # Test (publishable)
        test_prob = model.predict(X_test_s, verbose=0).reshape(-1)
        y_pred = (test_prob >= best_thr).astype(int)

        metrics_main = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "mcc": float(matthews_corrcoef(y_test, y_pred)),
            "n_test_windows": int(len(y_test)),
            "threshold_used": float(best_thr),
            "val_f1_at_threshold": float(best_val_f1),
        }

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cmn = cm_normalized(cm)

        distributions = {
            "participant": pid,
            "train_fraction_windows_per_file": TRAIN_FRACTION,
            "train_windows_total": int(len(y_train)),
            "train_windows_val": int(val_mask.sum()),
            "train_label_dist_all": dist_str(y_train),
            "train_label_dist_train": dist_str(y_tr),
            "train_label_dist_train_balanced": dist_str(y_tr_bal),
            "val_label_dist": dist_str(y_val),
            "test_windows_total": int(len(y_test)),
            "test_label_dist": dist_str(y_test),
            "test_pred_dist": dist_str(y_pred),
        }

        # SHUFFLE control
        logging.info("Training SHUFFLE control GRU (cheaper epochs)...")
        y_tr_shuf = y_tr_bal.copy()
        np.random.shuffle(y_tr_shuf)

        model_s = build_gru_classifier(
            seq_len=SEQ_LEN, n_features=N_FEATURES,
            gru_units=GRU_UNITS, dropout=DROPOUT,
            lr=LR, l2=L2
        )
        model_s.fit(
            X_tr_bal, y_tr_shuf,
            validation_data=(X_val, y_val),
            epochs=SHUFFLE_EPOCHS,
            batch_size=BATCH,
            callbacks=[early],
            verbose=0
        )
        test_prob_s = model_s.predict(X_test_s, verbose=0).reshape(-1)
        y_pred_s = (test_prob_s >= best_thr).astype(int)

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
            "seq_len": SEQ_LEN,
            "n_features": N_FEATURES,
            "train_fraction_windows_per_file": TRAIN_FRACTION,

            "lr": LR,
            "l2": L2,
            "gru_units": GRU_UNITS,
            "dropout": DROPOUT,

            "epochs": EPOCHS,
            "batch": BATCH,
            "val_frac_windows": VAL_FRAC,
            "early_stopping_patience": PATIENCE,
            "shuffle_epochs": SHUFFLE_EPOCHS,

            "cache_dir": cache_dir
        }

        save_participant_excel(
            out_folder, pid,
            settings, metrics_main, cm, cmn,
            metrics_shuffle, distributions,
            history_df=hist_df
        )

        summary_rows.append({
            "participant": pid,

            "indiv_accuracy": metrics_main["accuracy"],
            "indiv_precision": metrics_main["precision"],
            "indiv_recall": metrics_main["recall"],
            "indiv_f1": metrics_main["f1"],
            "indiv_macro_f1": metrics_main["macro_f1"],
            "indiv_balanced_accuracy": metrics_main["balanced_accuracy"],
            "indiv_mcc": metrics_main["mcc"],

            "threshold": metrics_main["threshold_used"],

            "shuffle_accuracy": metrics_shuffle["accuracy"],
            "shuffle_f1": metrics_shuffle["f1"],
            "shuffle_mcc": metrics_shuffle["mcc"],

            "n_test_windows": metrics_main["n_test_windows"],
            "test_label_dist": distributions["test_label_dist"],
            "test_pred_dist": distributions["test_pred_dist"],
        })

        elapsed = time.time() - t0
        logging.info(
            f"✅ Participant {pid:03d} done in {elapsed/60:.1f} min | "
            f"REAL F1={metrics_main['f1']:.3f} | SHUFFLE F1={metrics_shuffle['f1']:.3f}"
        )

        tf.keras.backend.clear_session()
        gc.collect()

    # SUMMARY Excel
    if not summary_rows:
        logging.error("No participants completed. Check your data.")
        return

    summary_df = pd.DataFrame(summary_rows).sort_values("participant")
    summary_path = os.path.join(out_folder, "SUMMARY_GRU_INDIVIDUAL.xlsx")

    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        stats = {
            "n_participants": int(len(summary_df)),
            "indiv_f1_mean": float(summary_df["indiv_f1"].mean()),
            "indiv_f1_std": float(summary_df["indiv_f1"].std(ddof=1)),
            "indiv_acc_mean": float(summary_df["indiv_accuracy"].mean()),
            "indiv_prec_mean": float(summary_df["indiv_precision"].mean()),
            "indiv_rec_mean": float(summary_df["indiv_recall"].mean()),
            "indiv_mcc_mean": float(summary_df["indiv_mcc"].mean()),
            "shuffle_f1_mean": float(summary_df["shuffle_f1"].mean()),
            "shuffle_f1_std": float(summary_df["shuffle_f1"].std(ddof=1)),
        }
        pd.DataFrame([stats]).to_excel(writer, sheet_name="SummaryStats", index=False)

    logging.info(f"✅ Saved summary: {summary_path}")

if __name__ == "__main__":
    main()
