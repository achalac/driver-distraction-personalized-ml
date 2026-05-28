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

TRANSFORMER GROUP MODEL (LOPO) - 
Outputs:
  <project_root>/OUTPUT_TRANSFORMER_GROUP_LOPO_FIXED/
  <project_root>/CACHE_WINDOWS_TRANSFORMER/
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

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN

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
# Utilities
# ----------------------------
def dist_str(y):
    u, c = np.unique(y, return_counts=True)
    return str({int(k): int(v) for k, v in zip(u, c)})

def cm_normalized(cm):
    cm = cm.astype(float)
    denom = cm.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return cm / denom

def ensure_finite_seq(X: np.ndarray) -> np.ndarray:
    """
    X: [N, T, F] float32
    Replace NaN/Inf with per-feature median (computed over N*T).
    """
    X = X.astype(np.float32, copy=False)
    X[~np.isfinite(X)] = np.nan

    if X.ndim != 3:
        raise ValueError(f"Expected X shape [N,T,F], got {X.shape}")

    N, T, F = X.shape
    flat = X.reshape(N * T, F)

    med = np.nanmedian(flat, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)

    inds = np.where(np.isnan(flat))
    if len(inds[0]) > 0:
        flat[inds] = med[inds[1]]

    flat = flat.reshape(N, T, F)

    if not np.isfinite(flat).all():
        raise ValueError("X still contains NaN/Inf after cleaning.")
    return flat

def balance_windows_seq(X, y, seed=SEED):
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
# Cache: build/load window sequences per CSV
# ----------------------------
def cache_path(cache_dir, file_id, fs, window_sec, n_features):
    safe = file_id.replace(".csv", "")
    return os.path.join(cache_dir, f"{safe}_fs{fs}_w{window_sec}_f{n_features}.npz")

def build_windows_from_csv(csv_path, file_id, pid, label, fs=128, window_sec=5, n_features=16):
    df = pd.read_csv(csv_path, usecols=range(n_features))

    if "GSR" in df.columns:
        df = df[df["GSR"] >= 0]

    df = df.reset_index(drop=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(df.median(numeric_only=True))

    win_len = int(fs * window_sec)
    n = len(df)
    feature_cols = list(df.columns[:n_features])

    X_list, y_list, start_list = [], [], []
    for start in range(0, n - win_len + 1, win_len):
        w = df.iloc[start:start + win_len][feature_cols].values.astype(np.float32)
        X_list.append(w)                 # [T,F]
        y_list.append(int(label))        # file label
        start_list.append(int(start))

    Xw = np.asarray(X_list, dtype=np.float32)   # [Nwin, T, F]
    yw = np.asarray(y_list, dtype=int)

    meta = {
        "file_id": file_id,
        "participant_id": int(pid),
        "label": int(label),
        "window_start_sample": np.asarray(start_list, dtype=np.int32)
    }
    return Xw, yw, meta

def load_or_build_cached(data_folder, file_id, cache_dir, fs=128, window_sec=5, n_features=16):
    os.makedirs(cache_dir, exist_ok=True)
    pid = parse_participant_id(file_id)
    label = parse_label(file_id)

    npz = cache_path(cache_dir, file_id, fs, window_sec, n_features)
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=True)
        return z["Xw"], z["yw"], z["meta"].item()

    csv_path = os.path.join(data_folder, file_id)
    if not os.path.exists(csv_path):
        logging.warning(f"Missing file: {csv_path}")
        return None

    Xw, yw, meta = build_windows_from_csv(csv_path, file_id, pid, label, fs, window_sec, n_features)
    if len(yw) > 0:
        Xw = ensure_finite_seq(Xw)

    np.savez_compressed(npz, Xw=Xw, yw=yw, meta=meta)
    return Xw, yw, meta

def concat_cached(items):
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
            "window_start_sample": meta["window_start_sample"]
        }))

    if not Xs:
        return np.empty((0, 1, 16), dtype=np.float32), np.empty((0,), dtype=int), pd.DataFrame()

    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.concatenate(ys).astype(int)
    meta = pd.concat(metas, axis=0, ignore_index=True)
    return X, y, meta

def blockwise_val_mask(meta_df, val_fraction=0.1):
    mask = np.zeros(len(meta_df), dtype=bool)
    for file_id, g in meta_df.groupby("file_id", sort=False):
        idx = g.index.to_numpy()
        cut = int((1.0 - val_fraction) * len(idx))
        mask[idx[cut:]] = True
    return mask

# ----------------------------
# Scaling sequences safely
# ----------------------------
def fit_seq_scaler(X_train_seq):
    N, T, F = X_train_seq.shape
    flat = X_train_seq.reshape(N * T, F)
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler

def transform_seq_scaler(X_seq, scaler):
    N, T, F = X_seq.shape
    flat = X_seq.reshape(N * T, F)
    flat2 = scaler.transform(flat)
    return flat2.reshape(N, T, F).astype(np.float32)

# ----------------------------
# Transformer model (stable)
# ----------------------------
def transformer_encoder_block(x, d_model=32, num_heads=2, ff_dim=64, dropout=0.1):
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout)(x1, x1)
    x2 = layers.Add()([x, attn])

    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    ff = layers.Dense(ff_dim, activation="relu")(x3)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(x.shape[-1])(ff)
    out = layers.Add()([x2, ff])
    return out

def build_transformer(input_timesteps, input_features,
                      d_model=32, num_heads=2, ff_dim=64,
                      num_blocks=1, dropout=0.1, lr=1e-4):
    inp = layers.Input(shape=(input_timesteps, input_features))
    x = layers.Dense(d_model)(inp)

    for _ in range(num_blocks):
        x = transformer_encoder_block(x, d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out)
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    model.compile(optimizer=opt, loss="binary_crossentropy")
    return model

# ----------------------------
# Plot + Excel saving
# ----------------------------
def save_loss_plot(out_folder, fold_id, history):
    png = os.path.join(out_folder, f"LOSS_TRANSFORMER_LOPO_excl_{fold_id:03d}.png")
    plt.figure()
    plt.plot(history.history.get("loss", []), label="train_loss")
    if "val_loss" in history.history:
        plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Transformer Loss (LOPO excl {fold_id:03d})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png, dpi=200)
    plt.close()
    return png

def save_fold_excel(out_folder, fold_id, settings, metrics_main, cm_counts, cm_norm,
                    metrics_shuffle, distributions, history_df):
    xlsx = os.path.join(out_folder, f"TRANSFORMER_GROUP_LOPO_excl_{fold_id:03d}.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        pd.DataFrame([settings]).to_excel(writer, sheet_name="Settings", index=False)
        pd.DataFrame([metrics_main]).to_excel(writer, sheet_name="WindowMetrics", index=False)
        pd.DataFrame([distributions]).to_excel(writer, sheet_name="Distributions", index=False)
        pd.DataFrame(cm_counts, index=["True0", "True1"], columns=["Pred0", "Pred1"]).to_excel(writer, sheet_name="CM_Counts")
        pd.DataFrame(cm_norm, index=["True0", "True1"], columns=["Pred0", "Pred1"]).to_excel(writer, sheet_name="CM_Normalized")
        pd.DataFrame([metrics_shuffle]).to_excel(writer, sheet_name="ShuffleControl", index=False)
        history_df.to_excel(writer, sheet_name="TrainHistory", index=False)
    logging.info(f"Saved fold Excel: {xlsx}")

# ----------------------------
# MAIN
# ----------------------------
def main():
    # GPU check (informative)
    logging.info(f"TF version: {tf.__version__}")
    logging.info(f"GPUs: {tf.config.list_physical_devices('GPU')}")
    if len(tf.config.list_physical_devices("GPU")) == 0:
        logging.warning("No GPU detected by TensorFlow. Transformer will run on CPU and be slow.")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_folder = os.path.join(project_root, "data", "ALLDATAFINALPAS")

    out_folder = os.path.join(project_root, "OUTPUT_TRANSFORMER_GROUP_LOPO_FIXED")
    cache_dir = os.path.join(project_root, "CACHE_WINDOWS_TRANSFORMER")

    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # ----------------------------
    # RESUME SETTINGS
    # ----------------------------
    START_FOLD = 25                 # <-- resume from participant 25
    END_FOLD = 50                   # <-- last participant
    SKIP_IF_EXCEL_EXISTS = True     # <-- auto-skip folds already done

    # --- Dataset settings ---
    FS = 128
    WINDOW_SEC = 5
    N_FEATURES = 16
    WIN_LEN = FS * WINDOW_SEC

    # --- Training settings ---
    EPOCHS = 20
    SHUFFLE_EPOCHS = 8
    BATCH = 128
    VAL_FRAC = 0.10
    PATIENCE = 4

    # Transformer size (small, stable)
    D_MODEL = 32
    NUM_HEADS = 2
    FF_DIM = 64
    NUM_BLOCKS = 1
    DROPOUT = 0.1
    LR = 1e-4

    # Build file list
    all_files = []
    for i in range(1, 51):
        all_files.append(f"GAC{i:03d}_Normal_F.csv")
        all_files.append(f"GAC{i:03d}_Load_F.csv")

    summary_rows = []

    # Loop only the range you want
    for exclude_pid in range(START_FOLD, END_FOLD + 1):
        fold_excel = os.path.join(out_folder, f"TRANSFORMER_GROUP_LOPO_excl_{exclude_pid:03d}.xlsx")
        if SKIP_IF_EXCEL_EXISTS and os.path.exists(fold_excel):
            logging.info(f"Skipping fold {exclude_pid:03d} (already exists): {fold_excel}")
            continue

        t0 = time.time()
        logging.info(f"\n=== TRANSFORMER LOPO fold: TEST participant {exclude_pid:03d} ===")

        train_files = [f for f in all_files if parse_participant_id(f) != exclude_pid]
        test_files  = [f for f in all_files if parse_participant_id(f) == exclude_pid]

        logging.info("Loading/Building cached window sequences...")
        train_items = [load_or_build_cached(data_folder, f, cache_dir, FS, WINDOW_SEC, N_FEATURES) for f in train_files]
        test_items  = [load_or_build_cached(data_folder, f, cache_dir, FS, WINDOW_SEC, N_FEATURES) for f in test_files]

        X_train, y_train, meta_train = concat_cached(train_items)
        X_test,  y_test,  meta_test  = concat_cached(test_items)

        logging.info(f"Windows: train={len(y_train):,} ({dist_str(y_train)}), test={len(y_test):,} ({dist_str(y_test)})")

        if len(y_train) == 0 or len(y_test) == 0:
            logging.warning("No windows found. Skipping fold.")
            continue

        X_train = ensure_finite_seq(X_train)
        X_test  = ensure_finite_seq(X_test)

        val_mask = blockwise_val_mask(meta_train, val_fraction=VAL_FRAC)
        X_tr, y_tr = X_train[~val_mask], y_train[~val_mask]
        X_val, y_val = X_train[val_mask], y_train[val_mask]

        logging.info(f"Split: X_tr={X_tr.shape}, X_val={X_val.shape}, val_dist={dist_str(y_val)}")

        X_tr_bal, y_tr_bal = balance_windows_seq(X_tr, y_tr, seed=SEED)
        logging.info(f"Balanced train: {X_tr_bal.shape}, dist={dist_str(y_tr_bal)}")

        scaler = fit_seq_scaler(X_tr_bal)
        X_tr_bal = transform_seq_scaler(X_tr_bal, scaler)
        X_val    = transform_seq_scaler(X_val, scaler)
        X_test_s = transform_seq_scaler(X_test, scaler)

        X_tr_bal = ensure_finite_seq(X_tr_bal)
        X_val    = ensure_finite_seq(X_val)
        X_test_s = ensure_finite_seq(X_test_s)

        model = build_transformer(
            input_timesteps=WIN_LEN,
            input_features=N_FEATURES,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            ff_dim=FF_DIM,
            num_blocks=NUM_BLOCKS,
            dropout=DROPOUT,
            lr=LR
        )

        early = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
        nan_kill = TerminateOnNaN()

        logging.info("Training REAL Transformer model...")
        hist = model.fit(
            X_tr_bal, y_tr_bal,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH,
            callbacks=[early, nan_kill],
            verbose=1
        )
        hist_df = pd.DataFrame(hist.history)

        loss_png = save_loss_plot(out_folder, exclude_pid, hist)
        logging.info(f"Saved loss plot: {loss_png}")

        val_prob = model.predict(X_val, verbose=0).reshape(-1)
        best_thr, best_val_f1 = choose_threshold_on_val(y_val, val_prob)
        logging.info(f"Best threshold={best_thr:.2f} (val F1={best_val_f1:.3f})")

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
            "train_windows_total": int(len(y_train)),
            "train_windows_val": int(val_mask.sum()),
            "train_windows_train": int((~val_mask).sum()),
            "train_label_dist_all": dist_str(y_train),
            "train_label_dist_train": dist_str(y_tr),
            "train_label_dist_train_balanced": dist_str(y_tr_bal),
            "val_label_dist": dist_str(y_val),
            "test_label_dist": dist_str(y_test),
            "test_pred_dist": dist_str(y_pred),
        }

        logging.info("Training SHUFFLE control model...")
        y_tr_shuf = y_tr_bal.copy()
        np.random.shuffle(y_tr_shuf)

        model_s = build_transformer(
            input_timesteps=WIN_LEN,
            input_features=N_FEATURES,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            ff_dim=FF_DIM,
            num_blocks=NUM_BLOCKS,
            dropout=DROPOUT,
            lr=LR
        )
        model_s.fit(
            X_tr_bal, y_tr_shuf,
            validation_data=(X_val, y_val),
            epochs=SHUFFLE_EPOCHS,
            batch_size=BATCH,
            callbacks=[early, nan_kill],
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
            "exclude_participant": exclude_pid,
            "fs": FS,
            "window_sec": WINDOW_SEC,
            "window_len_samples": WIN_LEN,
            "n_features": N_FEATURES,
            "transformer_d_model": D_MODEL,
            "transformer_heads": NUM_HEADS,
            "transformer_ff_dim": FF_DIM,
            "transformer_blocks": NUM_BLOCKS,
            "dropout": DROPOUT,
            "lr": LR,
            "epochs": EPOCHS,
            "batch": BATCH,
            "val_frac_windows": VAL_FRAC,
            "early_patience": PATIENCE,
            "shuffle_epochs": SHUFFLE_EPOCHS,
            "cache_dir": cache_dir,
        }

        save_fold_excel(out_folder, exclude_pid, settings, metrics_main, cm, cmn,
                        metrics_shuffle, distributions, hist_df)

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

            "shuffle_accuracy": metrics_shuffle["accuracy"],
            "shuffle_balanced_accuracy": metrics_shuffle["balanced_accuracy"],
            "shuffle_precision": metrics_shuffle["precision"],
            "shuffle_recall": metrics_shuffle["recall"],
            "shuffle_f1": metrics_shuffle["f1"],
            "shuffle_macro_f1": metrics_shuffle["macro_f1"],
            "shuffle_mcc": metrics_shuffle["mcc"],

            "n_test_windows": metrics_main["n_test_windows"],
            "test_label_dist": distributions["test_label_dist"],
            "test_pred_dist": distributions["test_pred_dist"],
        })

        elapsed = time.time() - t0
        logging.info(
            f" Fold {exclude_pid:03d} done in {elapsed/60:.1f} min | "
            f"REAL F1={metrics_main['f1']:.3f} | REAL MCC={metrics_main['mcc']:.3f} | "
            f"SHUFFLE F1={metrics_shuffle['f1']:.3f}"
        )

        tf.keras.backend.clear_session()
        gc.collect()

    # ----------------------------
    # Summary Excel across folds (only for folds run in THIS session)
    # ----------------------------
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("participant")
        summary_path = os.path.join(out_folder, f"SUMMARY_TRANSFORMER_GROUP_LOPO_{START_FOLD:03d}_to_{END_FOLD:03d}.xlsx")

        with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            stats = {
                "n_folds_in_this_run": int(len(summary_df)),
                "group_f1_mean": float(summary_df["group_f1"].mean()),
                "group_f1_std": float(summary_df["group_f1"].std(ddof=1)) if len(summary_df) > 1 else 0.0,
                "group_mcc_mean": float(summary_df["group_mcc"].mean()),
                "group_accuracy_mean": float(summary_df["group_accuracy"].mean()),
                "group_precision_mean": float(summary_df["group_precision"].mean()),
                "group_recall_mean": float(summary_df["group_recall"].mean()),
                "shuffle_f1_mean": float(summary_df["shuffle_f1"].mean()),
                "shuffle_mcc_mean": float(summary_df["shuffle_mcc"].mean()),
            }
            pd.DataFrame([stats]).to_excel(writer, sheet_name="SummaryStats", index=False)

        logging.info(f" Saved partial-run summary: {summary_path}")
    else:
        logging.info("No new folds were run (everything already existed).")

if __name__ == "__main__":
    main()
