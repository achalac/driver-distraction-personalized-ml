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

LSTM GROUP MODEL (LOPO) 
Outputs:
  <project_root>/OUTPUT_LSTM_GROUP_LOPO_BEST/
Cache:
  <project_root>/CACHE_WINDOWS_LSTM/

Requirements:
  pip install openpyxl
  tensorflow, numpy, pandas, scikit-learn, matplotlib

NOTE (important):
- If your TF GPU list is empty, it will run on CPU (still fine, but slower).
- This code is optimized for CPU friendliness via DOWNSAMPLE and moderate model size.
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
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------
# Reproducibility + TF setup
# ----------------------------
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# (Optional) reduce TF spam
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# (Optional) GPU memory growth (won't error if no GPU)
try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

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
# Cache helpers
# ----------------------------
def window_cache_path(cache_dir, file_id, fs, window_sec, n_features, downsample):
    safe = file_id.replace(".csv", "")
    return os.path.join(
        cache_dir,
        f"{safe}_fs{fs}_w{window_sec}_f{n_features}_ds{downsample}.npz"
    )


def ensure_numeric_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    # force numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # replace inf with nan
    df = df.replace([np.inf, -np.inf], np.nan)
    # fill NaN with median
    df = df.fillna(df.median(numeric_only=True))
    return df


def build_window_sequences_from_csv(
    csv_path: str,
    file_id: str,
    participant_id: int,
    label: int,
    fs: int = 128,
    window_sec: int = 5,
    n_features: int = 16,
    downsample: int = 4
):
    """
    Returns:
      Xw: [n_windows, timesteps, n_features] float32
      yw: [n_windows] int
      meta: dict with arrays for window_start_sample
    """
    df = pd.read_csv(csv_path, usecols=range(n_features))

    # Cleaning consistent with your pipeline
    if "GSR" in df.columns:
        df = df[df["GSR"] >= 0]

    df = df.reset_index(drop=True)
    df = ensure_numeric_and_clean(df)

    win_len = int(fs * window_sec)
    step = win_len  # non-overlapping

    # downsample: take every `downsample` sample inside window
    ds = int(max(1, downsample))
    timesteps = win_len // ds

    X_list = []
    start_list = []

    feat_cols = list(df.columns[:n_features])
    n = len(df)

    for start in range(0, n - win_len + 1, step):
        window = df.iloc[start:start + win_len][feat_cols].to_numpy(dtype=np.float32, copy=False)
        # downsample inside window
        window = window[::ds]
        # ensure exact timesteps length (in case win_len not divisible)
        window = window[:timesteps]
        if window.shape[0] != timesteps:
            continue
        X_list.append(window)
        start_list.append(int(start))

    Xw = np.asarray(X_list, dtype=np.float32)
    yw = np.full((len(Xw),), int(label), dtype=int)

    meta = {
        "file_id": file_id,
        "participant_id": int(participant_id),
        "label": int(label),
        "window_start_sample": np.asarray(start_list, dtype=np.int32),
        "timesteps": int(timesteps),
        "n_features": int(n_features),
    }
    return Xw, yw, meta


def load_or_build_cached_windows(
    data_folder: str,
    file_id: str,
    cache_dir: str,
    fs: int,
    window_sec: int,
    n_features: int,
    downsample: int
):
    os.makedirs(cache_dir, exist_ok=True)

    pid = parse_participant_id(file_id)
    label = parse_label(file_id)
    npz_path = window_cache_path(cache_dir, file_id, fs, window_sec, n_features, downsample)

    if os.path.exists(npz_path):
        z = np.load(npz_path, allow_pickle=True)
        return z["Xw"], z["yw"], z["meta"].item()

    csv_path = os.path.join(data_folder, file_id)
    if not os.path.exists(csv_path):
        logging.warning(f"Missing file: {csv_path}")
        return None

    Xw, yw, meta = build_window_sequences_from_csv(
        csv_path=csv_path,
        file_id=file_id,
        participant_id=pid,
        label=label,
        fs=fs,
        window_sec=window_sec,
        n_features=n_features,
        downsample=downsample
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
        mdf = pd.DataFrame({
            "file_id": [meta["file_id"]] * len(yw),
            "participant_id": [meta["participant_id"]] * len(yw),
            "label": [meta["label"]] * len(yw),
            "window_start_sample": meta["window_start_sample"]
        })
        metas.append(mdf)

    if not Xs:
        return np.empty((0, 1, 1), dtype=np.float32), np.empty((0,), dtype=int), pd.DataFrame()

    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.concatenate(ys).astype(int)
    meta = pd.concat(metas, axis=0, ignore_index=True)
    return X, y, meta


# ----------------------------
# Splits / balancing / threshold
# ----------------------------
def blockwise_val_mask(meta: pd.DataFrame, val_fraction: float = 0.1) -> np.ndarray:
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
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        pred = (y_prob >= t).astype(int)
        f1v = f1_score(y_true, pred, zero_division=0)
        if f1v > best_f1:
            best_f1 = f1v
            best_t = float(t)
    return best_t, float(best_f1)


# ----------------------------
# Sequence scaling (no leakage)
# ----------------------------
def fit_sequence_scaler(X_seq: np.ndarray) -> StandardScaler:
    """
    Fit scaler on TRAIN only, using all timesteps stacked:
      X_seq: [N, T, F] -> reshape to [N*T, F]
    """
    N, T, F = X_seq.shape
    flat = X_seq.reshape(N * T, F)
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler


def transform_sequence_scaler(X_seq: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    N, T, F = X_seq.shape
    flat = X_seq.reshape(N * T, F)
    flat2 = scaler.transform(flat).astype(np.float32)
    return flat2.reshape(N, T, F)


# ----------------------------
# LSTM model
# ----------------------------
def build_lstm_model(timesteps: int, n_features: int, lr=0.001, units=64, dropout=0.2):
    model = Sequential([
        Input(shape=(timesteps, n_features)),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy")
    return model


# ----------------------------
# Metrics + plots + excel
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
    png_path = os.path.join(out_folder, f"LOSS_LSTM_LOPO_excl_{fold_id:03d}.png")
    plt.figure()
    plt.plot(history.history.get("loss", []), label="train_loss")
    if "val_loss" in history.history:
        plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"LSTM Loss (LOPO excl {fold_id:03d})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()
    return png_path


def save_fold_excel(out_folder, fold_id, settings, metrics_main, cm_counts, cm_norm,
                    metrics_shuffle, distributions, history_df):
    os.makedirs(out_folder, exist_ok=True)
    xlsx_path = os.path.join(out_folder, f"LSTM_GROUP_LOPO_excl_{fold_id:03d}.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame([settings]).to_excel(writer, sheet_name="Settings", index=False)
        pd.DataFrame([metrics_main]).to_excel(writer, sheet_name="WindowMetrics", index=False)
        pd.DataFrame([distributions]).to_excel(writer, sheet_name="Distributions", index=False)
        pd.DataFrame(cm_counts, index=["True0", "True1"], columns=["Pred0", "Pred1"]).to_excel(writer, sheet_name="CM_Counts")
        pd.DataFrame(cm_norm, index=["True0", "True1"], columns=["Pred0", "Pred1"]).to_excel(writer, sheet_name="CM_Normalized")
        pd.DataFrame([metrics_shuffle]).to_excel(writer, sheet_name="ShuffleControl", index=False)
        history_df.to_excel(writer, sheet_name="TrainHistory", index=False)
    logging.info(f"Saved fold Excel: {xlsx_path}")


# ----------------------------
# MAIN
# ----------------------------
def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_folder = os.path.join(project_root, "data", "ALLDATAFINALPAS")

    out_folder = os.path.join(project_root, "OUTPUT_LSTM_GROUP_LOPO_BEST")
    cache_dir = os.path.join(project_root, "CACHE_WINDOWS_LSTM")
    os.makedirs(out_folder, exist_ok=True)

    # --- Settings (CPU/GPU friendly) ---
    FS = 128
    WINDOW_SEC = 5
    N_FEATURES = 16
    DOWNSAMPLE = 4  # ✅ speed-up: 128Hz->32Hz, timesteps = 640/4=160

    # Model/training
    LR = 0.001
    LSTM_UNITS = 64
    DROPOUT = 0.2

    EPOCHS = 25              # LSTM is heavier; 25 + early stopping is enough baseline
    BATCH = 256              # moderate batch for sequences
    VAL_FRAC = 0.10
    PATIENCE = 5

    # Shuffle control (lighter)
    SHUFFLE_EPOCHS = 8

    # Master file list
    all_files = []
    for i in range(1, 51):
        all_files.append(f"GAC{i:03d}_Normal_F.csv")
        all_files.append(f"GAC{i:03d}_Load_F.csv")

    summary_rows = []
    skipped_rows = []

    for exclude_pid in range(1, 51):
        t0 = time.time()
        logging.info(f"\n=== LSTM LOPO fold: TEST participant {exclude_pid:03d} ===")

        train_files = [f for f in all_files if parse_participant_id(f) != exclude_pid]
        test_files  = [f for f in all_files if parse_participant_id(f) == exclude_pid]

        # Load/build cached windows
        logging.info("Loading/Building cached LSTM windows...")
        train_items = [
            load_or_build_cached_windows(data_folder, f, cache_dir, FS, WINDOW_SEC, N_FEATURES, DOWNSAMPLE)
            for f in train_files
        ]
        test_items = [
            load_or_build_cached_windows(data_folder, f, cache_dir, FS, WINDOW_SEC, N_FEATURES, DOWNSAMPLE)
            for f in test_files
        ]

        Xw_train, yw_train, meta_train = concat_windows(train_items)
        Xw_test,  yw_test,  meta_test  = concat_windows(test_items)

        if len(yw_train) == 0 or len(yw_test) == 0:
            skipped_rows.append({"participant": exclude_pid, "reason": "no_windows"})
            logging.warning("No windows found. Skipping fold.")
            continue

        timesteps = Xw_train.shape[1]
        n_features = Xw_train.shape[2]

        logging.info(
            f"Windows: train={len(yw_train):,} ({dist_str(yw_train)}), "
            f"test={len(yw_test):,} ({dist_str(yw_test)}), "
            f"seq_shape=({timesteps},{n_features})"
        )

        # Blockwise val split (by file)
        val_mask = blockwise_val_mask(meta_train, val_fraction=VAL_FRAC)
        X_tr, y_tr = Xw_train[~val_mask], yw_train[~val_mask]
        X_val, y_val = Xw_train[val_mask], yw_train[val_mask]

        # Balance training windows
        X_trb, y_trb = balance_windows(X_tr, y_tr, seed=SEED)

        if len(np.unique(y_trb)) < 2:
            skipped_rows.append({"participant": exclude_pid, "reason": "one_class_in_train"})
            logging.warning("Only one class in balanced train set. Skipping fold.")
            continue

        # Scale using TRAIN ONLY (stacked timesteps)
        scaler = fit_sequence_scaler(X_trb)
        X_trb = transform_sequence_scaler(X_trb, scaler)
        X_val = transform_sequence_scaler(X_val, scaler)
        Xw_test = transform_sequence_scaler(Xw_test, scaler)

        # Train REAL model
        model = build_lstm_model(
            timesteps=timesteps,
            n_features=n_features,
            lr=LR,
            units=LSTM_UNITS,
            dropout=DROPOUT
        )
        early = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)

        logging.info("Training REAL LSTM model...")
        hist = model.fit(
            X_trb, y_trb,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH,
            callbacks=[early],
            verbose=1  # show epoch progress
        )
        hist_df = pd.DataFrame(hist.history)
        loss_png = save_loss_plot(out_folder, exclude_pid, hist)
        logging.info(f"Saved loss plot: {loss_png}")

        # Threshold tuning on validation
        val_prob = model.predict(X_val, verbose=0).reshape(-1)
        best_thr, best_val_f1 = choose_threshold_on_val(y_val, val_prob)
        logging.info(f"Best threshold={best_thr:.2f} (val F1={best_val_f1:.3f})")

        # Test (publish)
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
            "train_windows_train": int((~val_mask).sum()),
            "train_windows_val": int(val_mask.sum()),
            "train_label_dist_all": dist_str(yw_train),
            "train_label_dist_train": dist_str(y_tr),
            "train_label_dist_train_balanced": dist_str(y_trb),
            "val_label_dist": dist_str(y_val),
            "test_label_dist": dist_str(yw_test),
            "test_pred_dist": dist_str(y_pred),
        }

        # Shuffle control (lighter)
        y_tr_shuf = y_trb.copy()
        np.random.shuffle(y_tr_shuf)

        model_s = build_lstm_model(
            timesteps=timesteps,
            n_features=n_features,
            lr=LR,
            units=LSTM_UNITS,
            dropout=DROPOUT
        )
        logging.info("Training SHUFFLE control LSTM model...")
        model_s.fit(
            X_trb, y_tr_shuf,
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
            "n_features": N_FEATURES,
            "downsample": DOWNSAMPLE,
            "timesteps": timesteps,
            "val_frac_windows": VAL_FRAC,
            "lr": LR,
            "lstm_units": LSTM_UNITS,
            "dropout": DROPOUT,
            "epochs": EPOCHS,
            "batch": BATCH,
            "early_patience": PATIENCE,
            "shuffle_epochs": SHUFFLE_EPOCHS,
            "cache_dir": cache_dir
        }

        save_fold_excel(
            out_folder, exclude_pid,
            settings, metrics_main,
            cm, cmn,
            metrics_shuffle, distributions,
            history_df=hist_df
        )

        # ✅ Summary row includes ALL metrics you asked for
        summary_rows.append({
            "participant": exclude_pid,

            "group_accuracy": metrics_main["accuracy"],
            "group_balanced_accuracy": metrics_main["balanced_accuracy"],
            "group_precision": metrics_main["precision"],
            "group_recall": metrics_main["recall"],
            "group_f1": metrics_main["f1"],
            "group_macro_f1": metrics_main["macro_f1"],
            "group_mcc": metrics_main["mcc"],

            "threshold_used": metrics_main["threshold_used"],
            "val_f1_at_threshold": metrics_main["val_f1_at_threshold"],
            "n_test_windows": metrics_main["n_test_windows"],

            "shuffle_accuracy": metrics_shuffle["accuracy"],
            "shuffle_balanced_accuracy": metrics_shuffle["balanced_accuracy"],
            "shuffle_precision": metrics_shuffle["precision"],
            "shuffle_recall": metrics_shuffle["recall"],
            "shuffle_f1": metrics_shuffle["f1"],
            "shuffle_macro_f1": metrics_shuffle["macro_f1"],
            "shuffle_mcc": metrics_shuffle["mcc"],
        })

        elapsed = time.time() - t0
        logging.info(
            f"✅ Fold {exclude_pid:03d} done in {elapsed/60:.1f} min | "
            f"REAL F1={metrics_main['f1']:.3f} | Acc={metrics_main['accuracy']:.3f} | "
            f"MCC={metrics_main['mcc']:.3f} | SHUFFLE F1={metrics_shuffle['f1']:.3f}"
        )

        tf.keras.backend.clear_session()
        gc.collect()

    # ----------------------------
    # Summary Excel
    # ----------------------------
    summary_path = os.path.join(out_folder, "SUMMARY_LSTM_GROUP_LOPO.xlsx")
    summary_df = pd.DataFrame(summary_rows).sort_values("participant") if summary_rows else pd.DataFrame()
    skipped_df = pd.DataFrame(skipped_rows).sort_values("participant") if skipped_rows else pd.DataFrame(columns=["participant", "reason"])

    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        skipped_df.to_excel(writer, sheet_name="SkippedFolds", index=False)

        stats = {
            "n_completed": int(len(summary_df)),
            "n_skipped": int(len(skipped_df)),

            "group_f1_mean": float(summary_df["group_f1"].mean()) if len(summary_df) else np.nan,
            "group_f1_std": float(summary_df["group_f1"].std(ddof=1)) if len(summary_df) else np.nan,
            "group_acc_mean": float(summary_df["group_accuracy"].mean()) if len(summary_df) else np.nan,
            "group_mcc_mean": float(summary_df["group_mcc"].mean()) if len(summary_df) else np.nan,

            "shuffle_f1_mean": float(summary_df["shuffle_f1"].mean()) if len(summary_df) else np.nan,
            "shuffle_f1_std": float(summary_df["shuffle_f1"].std(ddof=1)) if len(summary_df) else np.nan,
        }
        pd.DataFrame([stats]).to_excel(writer, sheet_name="SummaryStats", index=False)

    logging.info(f"\n✅ Saved summary Excel: {summary_path}")
    logging.info(f"Completed={len(summary_df)} | Skipped={len(skipped_df)}")


if __name__ == "__main__":
    main()
