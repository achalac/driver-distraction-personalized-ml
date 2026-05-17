"""
SLP GROUP MODEL (LOPO) - IEEE-reviewer safe + faster + visible progress + FULL SUMMARY METRICS

✅ LOPO participant split (no leakage)
✅ Window-level dataset (5s) with mean features (transparent)
✅ Blockwise window validation split (last 10% windows per file)
✅ Balanced training windows to avoid collapse
✅ Threshold tuned ONLY on validation
✅ Metrics (REAL): Accuracy, Balanced Acc, Precision, Recall, F1, Macro-F1, MCC
✅ Confusion matrices (counts + normalized)
✅ Shuffle negative control (lighter but valid)
✅ ALL outputs in Excel + loss plot PNG
✅ FAST: caches window datasets per file to disk, reused across folds
✅ Logs progress clearly

Outputs:
  <project_root>/OUTPUT_SLP_GROUP_LOPO_FAST/
    - SLP_GROUP_LOPO_excl_XXX.xlsx (one per fold)
    - LOSS_LOPO_excl_XXX.png (one per fold)
    - SUMMARY_SLP_GROUP_LOPO.xlsx (includes ALL metrics per participant + SummaryStats)

NOTE:
- Works on CPU-only machines too.
- If you later fix TensorFlow GPU, you can increase BATCH (8192–32768).
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

# ----------------------------
# Window dataset cache
# ----------------------------
def window_cache_path(cache_dir, file_id, fs, window_sec, n_features):
    safe = file_id.replace(".csv", "")
    return os.path.join(cache_dir, f"{safe}_fs{fs}_w{window_sec}_f{n_features}.npz")

def build_window_dataset_from_csv(csv_path, file_id, participant_id, label,
                                  fs=128, window_sec=5, n_features=16):
    df = pd.read_csv(csv_path, usecols=range(n_features))

    # your cleaning
    if "GSR" in df.columns:
        df = df[df["GSR"] >= 0]

    df = df.reset_index(drop=True)
    df = df.fillna(df.median(numeric_only=True))

    win_len = int(fs * window_sec)
    n = len(df)

    X_list, y_list, start_list = [], [], []
    feature_cols = list(df.columns[:n_features])

    # non-overlapping windows
    for start in range(0, n - win_len + 1, win_len):
        w = df.iloc[start:start + win_len][feature_cols]
        X_list.append(w.mean(axis=0).values.astype(np.float32))
        y_list.append(int(label))
        start_list.append(int(start))

    Xw = np.asarray(X_list, dtype=np.float32)
    yw = np.asarray(y_list, dtype=int)

    meta = {
        "file_id": file_id,
        "participant_id": int(participant_id),
        "label": int(label),
        "window_start_sample": np.asarray(start_list, dtype=np.int32),
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

    Xw, yw, meta = build_window_dataset_from_csv(
        csv_path, file_id, participant_id, label,
        fs=fs, window_sec=window_sec, n_features=n_features
    )
    np.savez_compressed(npz_path, Xw=Xw, yw=yw, meta=meta)
    return Xw, yw, meta

def concat_windows(items):
    Xs, ys, metas = [], [], []
    for it in items:
        if it is None:
            continue
        Xw, yw, meta = it
        if len(yw) == 0:
            continue

        Xs.append(Xw)
        ys.append(yw)

        metas.append(pd.DataFrame({
            "file_id": [meta["file_id"]] * len(yw),
            "participant_id": [meta["participant_id"]] * len(yw),
            "label": [meta["label"]] * len(yw),
            "window_start_sample": meta["window_start_sample"],
        }))

    if not Xs:
        return np.empty((0, 16), dtype=np.float32), np.empty((0,), dtype=int), pd.DataFrame()

    X = np.vstack(Xs).astype(np.float32)
    y = np.concatenate(ys).astype(int)
    meta = pd.concat(metas, axis=0, ignore_index=True)
    return X, y, meta

# ----------------------------
# Splits / balancing / threshold
# ----------------------------
def blockwise_val_mask(meta, val_fraction=0.1):
    mask = np.zeros(len(meta), dtype=bool)
    for file_id, g in meta.groupby("file_id", sort=False):
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
# Model
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
# Metrics / plots / excel
# ----------------------------
def cm_normalized(cm):
    cm = cm.astype(float)
    denom = cm.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return cm / denom

def dist_str(y):
    u, c = np.unique(y, return_counts=True)
    return str({int(k): int(v) for k, v in zip(u, c)})

def save_loss_plot(out_folder, fold_id, history):
    png_path = os.path.join(out_folder, f"LOSS_LOPO_excl_{fold_id:03d}.png")
    plt.figure()
    plt.plot(history.history.get("loss", []), label="train_loss")
    if "val_loss" in history.history:
        plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss (LOPO excl {fold_id:03d})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()
    return png_path

def save_fold_excel(out_folder, fold_id, settings, metrics_main, cm_counts, cm_norm,
                    metrics_shuffle, distributions, history_df=None):
    os.makedirs(out_folder, exist_ok=True)
    xlsx_path = os.path.join(out_folder, f"SLP_GROUP_LOPO_excl_{fold_id:03d}.xlsx")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame([settings]).to_excel(writer, sheet_name="Settings", index=False)
        pd.DataFrame([metrics_main]).to_excel(writer, sheet_name="WindowMetrics", index=False)
        pd.DataFrame([distributions]).to_excel(writer, sheet_name="Distributions", index=False)
        pd.DataFrame(cm_counts, index=["True0", "True1"], columns=["Pred0", "Pred1"]).to_excel(writer, sheet_name="CM_Counts")
        pd.DataFrame(cm_norm, index=["True0", "True1"], columns=["Pred0", "Pred1"]).to_excel(writer, sheet_name="CM_Normalized")
        pd.DataFrame([metrics_shuffle]).to_excel(writer, sheet_name="ShuffleControl", index=False)
        if history_df is not None:
            history_df.to_excel(writer, sheet_name="TrainHistory", index=False)

    logging.info(f"Saved fold Excel: {xlsx_path}")

# ----------------------------
# MAIN
# ----------------------------
def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_folder = os.path.join(project_root, "data", "ALLDATAFINALPAS")

    out_folder = os.path.join(project_root, "OUTPUT_SLP_GROUP_LOPO_FAST")
    cache_dir = os.path.join(project_root, "CACHE_WINDOWS_SLP")
    os.makedirs(out_folder, exist_ok=True)

    # Settings
    FS = 128
    WINDOW_SEC = 5
    N_FEATURES = 16

    LR = 0.001
    L2 = 0.001
    UNITS = 128

    EPOCHS = 50
    BATCH = 2048         # CPU-friendly; if GPU works, try 8192–32768
    VAL_FRAC = 0.10
    PATIENCE = 6
    SHUFFLE_EPOCHS = 10  # lighter shuffle control

    # master files
    all_files = []
    for i in range(1, 51):
        all_files.append(f"GAC{str(i).zfill(3)}_Normal_F.csv")
        all_files.append(f"GAC{str(i).zfill(3)}_Load_F.csv")

    summary_rows = []

    for exclude_pid in range(1, 51):
        t0 = time.time()
        logging.info(f"\n=== LOPO fold: TEST participant {exclude_pid:03d} ===")

        train_files = [f for f in all_files if parse_participant_id(f) != exclude_pid]
        test_files  = [f for f in all_files if parse_participant_id(f) == exclude_pid]

        logging.info("Loading/Building cached window datasets...")
        train_items = [load_or_build_cached_windows(data_folder, f, cache_dir, FS, WINDOW_SEC, N_FEATURES) for f in train_files]
        test_items  = [load_or_build_cached_windows(data_folder, f, cache_dir, FS, WINDOW_SEC, N_FEATURES) for f in test_files]

        Xw_train, yw_train, meta_train = concat_windows(train_items)
        Xw_test,  yw_test,  meta_test  = concat_windows(test_items)

        logging.info(f"Windows loaded: train={len(yw_train):,} ({dist_str(yw_train)}), test={len(yw_test):,} ({dist_str(yw_test)})")

        if len(yw_test) == 0 or len(yw_train) == 0:
            logging.warning("No windows found. Skipping fold.")
            continue

        # Scale on TRAIN only
        scaler = StandardScaler()
        Xw_train = scaler.fit_transform(Xw_train)
        Xw_test  = scaler.transform(Xw_test)

        # Blockwise window validation
        val_mask = blockwise_val_mask(meta_train, val_fraction=VAL_FRAC)
        X_tr, y_tr = Xw_train[~val_mask], yw_train[~val_mask]
        X_val, y_val = Xw_train[val_mask], yw_train[val_mask]

        # Balance training windows
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

        loss_png = save_loss_plot(out_folder, exclude_pid, hist)
        logging.info(f"Saved loss plot: {loss_png}")

        # Threshold tuning on validation
        val_prob = model.predict(X_val, verbose=0).reshape(-1)
        best_thr, best_val_f1 = choose_threshold_on_val(y_val, val_prob)
        logging.info(f"Best threshold={best_thr:.2f} (val F1={best_val_f1:.3f})")

        # Test metrics (publishable)
        test_prob = model.predict(Xw_test, verbose=0).reshape(-1)
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

        # Shuffle negative control
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
        test_prob_s = model_s.predict(Xw_test, verbose=0).reshape(-1)
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
            "exclude_participant": exclude_pid,
            "fs": FS,
            "window_sec": WINDOW_SEC,
            "lr": LR,
            "l2": L2,
            "units": UNITS,
            "epochs": EPOCHS,
            "batch": BATCH,
            "val_frac_windows": VAL_FRAC,
            "early_stopping_patience": PATIENCE,
            "shuffle_epochs": SHUFFLE_EPOCHS,
            "cache_dir": cache_dir,
        }

        save_fold_excel(out_folder, exclude_pid, settings, metrics_main, cm, cmn,
                        metrics_shuffle, distributions, history_df=hist_df)

        # ✅ Summary row includes ALL metrics
        summary_rows.append({
            "participant": exclude_pid,

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
            f"✅ Fold {exclude_pid:03d} done in {elapsed/60:.1f} min | "
            f"REAL F1={metrics_main['f1']:.3f} MCC={metrics_main['mcc']:.3f} | "
            f"SHUFFLE F1={metrics_shuffle['f1']:.3f}"
        )

        tf.keras.backend.clear_session()
        gc.collect()

    # ----------------------------
    # SUMMARY EXCEL (per participant + mean±std for ALL metrics)
    # ----------------------------
    if not summary_rows:
        logging.error("No folds completed.")
        return

    summary_df = pd.DataFrame(summary_rows).sort_values("participant")
    summary_path = os.path.join(out_folder, "SUMMARY_SLP_GROUP_LOPO.xlsx")

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

    logging.info(f"✅ Saved summary (with ALL metrics): {summary_path}")

if __name__ == "__main__":
    main()
