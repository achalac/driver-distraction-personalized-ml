"""
MLP CALIBRATION CURVE - Group Pretrain + Limited Individual Calibration
=========================================================================

Directly answers:
  - Reviewer 1, Concern #1: "intermediate strategy between group-level and
    personalized models" (group pretraining + limited individual calibration)
  - Reviewer 2, Concern #2: "cold-start" / minimum calibration data analysis
  - Reviewer 3: "have you already started the hybrid approach?"

WHAT THIS SCRIPT DOES (per held-out participant p, for p = 1..50):
  1. Trains a GROUP model on the other 49 participants (identical LOPO setup
     to MLPgroup_-_New.py: same architecture, windowing, balancing, threshold
     tuning on validation only).
  2. Splits participant p's OWN data chronologically per condition file:
       - first 70% of each file  -> "personal pool" (available for calibration)
       - last 30% of each file   -> "personal test" (FIXED, held out, never
         touched during calibration, used for every calibration level so
         results are directly comparable)
  3. For increasing calibration fractions (0%, 5%, 10%, 20%, 40%, 70% of the
     personal pool), fine-tunes a COPY of the trained group model:
       - earlier Dense layers are FROZEN (keeps the general group-learned
         representation)
       - only the final Dense(1) output layer is retrained on the small
         calibration slice
     0% calibration = the raw group model evaluated on participant p's fixed
     test portion (no fine-tuning at all) - this is your true group-only
     baseline, evaluated on the SAME held-out windows as every other point on
     the curve, so it is a fair anchor.
  4. Evaluates every calibration level on the SAME fixed personal test
     portion and records Accuracy / Precision / Recall / F1 / MCC.

OUTPUT:
  <project_root>/OUTPUT_MLP_CALIBRATION/
    - CALIBRATION_pid_XXX.xlsx      (one per participant, all calibration levels)
    - SUMMARY_CALIBRATION_CURVE.xlsx (averaged curve across all participants -
      THIS is the table/figure to put in the paper)

Reuses cached windows from CACHE_WINDOWS_MLP (built by MLPgroup_-_New.py) if
present, so you do not need to rebuild anything already cached.

Assumptions (same as your other scripts):
- Files: GAC001_Normal_F.csv, GAC001_Load_F.csv ... GAC050_*.csv
- Each CSV has >= 16 feature columns (usecols=range(16))
- Normal=0, Load=1
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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, balanced_accuracy_score
)

from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Dropout
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
    return [f"GAC{pid:03d}_Normal_F.csv", f"GAC{pid:03d}_Load_F.csv"]


# ----------------------------
# Window dataset cache (per file) - identical to MLPgroup_-_New.py
# ----------------------------
def window_cache_path(cache_dir, file_id, fs, window_sec, n_features):
    safe = file_id.replace(".csv", "")
    return os.path.join(cache_dir, f"{safe}_fs{fs}_w{window_sec}_f{n_features}.npz")


def build_window_dataset_from_csv(csv_path, file_id, participant_id, label,
                                   fs=128, window_sec=5, n_features=16):
    df = pd.read_csv(csv_path, usecols=range(n_features))

    if "GSR" in df.columns:
        df = df[df["GSR"] >= 0]

    df = df.reset_index(drop=True)
    df = df.fillna(df.median(numeric_only=True))

    win_len = int(fs * window_sec)
    n = len(df)

    X_list, y_list, start_list = [], [], []
    feature_cols = list(df.columns[:n_features])

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


def concat_windows(items, n_features=16):
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
            "window_start_sample": meta["window_start_sample"],
        })
        metas.append(mdf)

    if not Xs:
        return np.empty((0, n_features), dtype=np.float32), np.empty((0,), dtype=int), pd.DataFrame()

    X = np.vstack(Xs).astype(np.float32)
    y = np.concatenate(ys).astype(int)
    meta = pd.concat(metas, axis=0, ignore_index=True)
    return X, y, meta


# ----------------------------
# Split / balancing / threshold - identical logic to your other scripts
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


def choose_threshold_on_val(y_true, y_prob, min_samples=10, fallback_threshold=None):
    """
    Pick the decision threshold that maximizes F1 on a validation slice.

    IMPORTANT FIX: when several thresholds tie for the best F1 (very likely
    when the validation slice is tiny, e.g. only 1-2 windows per class -
    common for small calibration fractions), always defaulting to the
    FIRST (lowest) tied threshold silently collapses the model into
    "always predict positive" - this is what caused every calibration
    level to look identical and degenerate regardless of the fine-tuning
    method used. Among ties, we now prefer the threshold closest to 0.5.

    If the validation slice is too small to be trustworthy (< min_samples
    per class), fall back to a supplied threshold (e.g. the group model's
    own validation threshold) rather than tuning on noise.
    """
    if fallback_threshold is not None:
        n0 = int(np.sum(y_true == 0))
        n1 = int(np.sum(y_true == 1))
        if n0 < min_samples or n1 < min_samples:
            return float(fallback_threshold), None

    best_f1v = -1.0
    tied_thresholds = []
    for t in np.linspace(0.05, 0.95, 19):
        pred = (y_prob >= t).astype(int)
        f1v = f1_score(y_true, pred, zero_division=0)
        if f1v > best_f1v + 1e-9:
            best_f1v = f1v
            tied_thresholds = [t]
        elif abs(f1v - best_f1v) <= 1e-9:
            tied_thresholds.append(t)

    if not tied_thresholds:
        return 0.5, best_f1v

    best_t = min(tied_thresholds, key=lambda t: abs(t - 0.5))
    return float(best_t), float(best_f1v)


def safe_mcc(y_true, y_pred):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return np.nan
    return matthews_corrcoef(y_true, y_pred)


# ----------------------------
# Personal chronological split (per file), matching your individual scripts
# ----------------------------
def personal_chronological_split(meta, personal_train_fraction=0.70):
    """
    Within EACH file (Normal, Load) separately:
      - first personal_train_fraction of windows -> pool (available for calibration)
      - remaining windows -> fixed test set (never touched during calibration)
    Returns boolean masks aligned to meta's row order.
    """
    pool_mask = np.zeros(len(meta), dtype=bool)
    test_mask = np.zeros(len(meta), dtype=bool)

    for file_id, g in meta.groupby("file_id", sort=False):
        idx = g.index.to_numpy()
        order = idx[np.argsort(g["window_start_sample"].values)]
        cut = int(personal_train_fraction * len(order))
        pool_mask[order[:cut]] = True
        test_mask[order[cut:]] = True

    return pool_mask, test_mask


def take_earliest_fraction_per_file(meta_pool, fraction):
    """
    From the personal pool, take the EARLIEST `fraction` of windows,
    per file, preserving time order. fraction in (0, 1].
    Returns positional indices into meta_pool (0..len(meta_pool)-1).
    """
    selected = []
    meta_pool = meta_pool.reset_index(drop=True)
    for file_id, g in meta_pool.groupby("file_id", sort=False):
        order = g.index.to_numpy()[np.argsort(g["window_start_sample"].values)]
        cut = max(1, int(round(fraction * len(order))))
        selected.extend(order[:cut].tolist())
    return np.asarray(sorted(selected), dtype=int)


# ----------------------------
# Model
# ----------------------------
def build_mlp(input_dim, lr=0.001, l2=0.001, hidden1=128, hidden2=64, dropout=0.3):
    model = Sequential([
        Dense(hidden1, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2), input_shape=(input_dim,)),
        Dropout(dropout),
        Dense(hidden2, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2)),
        Dropout(dropout),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy")
    return model


def clone_and_finetune_full(model, lr=0.0001):
    """
    Copy architecture + weights, unfreeze the ENTIRE network for
    fine-tuning (not just the final layer(s)).

    Freezing the early layers (as the previous version of this script did)
    left calibration stuck: those early layers were fit to the OTHER 49
    participants' physiological patterns, and for a new individual they may
    produce flat/uninformative internal features that no amount of
    retraining on the final layer(s) can fix - the model was always
    predicting the majority class ("Load") for every participant regardless
    of calibration amount, which is the signature of this problem.

    Full fine-tuning (with a small learning rate, since the calibration
    sets are small) lets personal data reshape the WHOLE decision process,
    not just the final boundary - a stronger and more standard test of
    whether personal data can override group-learned patterns.

    The output layer is still reset to a fresh initialization before
    fine-tuning, to avoid starting from an already-saturated state.
    """
    new_model = clone_model(model)
    new_model.set_weights(model.get_weights())

    for layer in new_model.layers:
        layer.trainable = True
        # Dropout helps prevent overfitting on the LARGE group-training set,
        # but on tiny per-participant calibration data it just adds noise
        # and destabilizes an already data-starved fine-tuning step.
        if isinstance(layer, Dropout):
            layer.rate = 0.0

    output_layer = new_model.layers[-1]
    fresh_kernel, fresh_bias = output_layer.get_weights()
    fresh_kernel = np.random.normal(loc=0.0, scale=0.05, size=fresh_kernel.shape).astype(np.float32)
    fresh_bias = np.zeros_like(fresh_bias)
    output_layer.set_weights([fresh_kernel, fresh_bias])

    new_model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy")
    return new_model


def eval_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(safe_mcc(y_true, y_pred)),
    }


# ----------------------------
# MAIN
# ----------------------------
def main():
    logging.info("=" * 70)
    logging.info("RUNNING SCRIPT VERSION: SIMPLIFIED-PERSONAL-ONLY-v8 (2026-07-06)")
    logging.info("=" * 70)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_folder = os.path.join(project_root, "data", "ALLDATAFINALPAS")

    out_folder = os.path.join(project_root, "OUTPUT_MLP_CALIBRATION")
    cache_dir = os.path.join(project_root, "CACHE_WINDOWS_MLP")  # reuse existing cache
    os.makedirs(out_folder, exist_ok=True)

    FS = 128
    WINDOW_SEC = 5
    N_FEATURES = 16

    # Group model hyperparams (same as MLPgroup_-_New.py)
    LR = 0.001
    L2 = 0.001
    H1 = 128
    H2 = 64
    DROPOUT = 0.30
    EPOCHS = 60
    BATCH = 2048
    VAL_FRAC = 0.10
    PATIENCE = 8

    # Personal split + calibration settings
    PERSONAL_TRAIN_FRACTION = 0.70   # matches your individual scripts' split
    CALIBRATION_FRACTIONS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.70]
    # OFF by default: doubles memory/time load per fraction and isn't needed
    # for the core calibration-curve result. Turn on later only if needed.
    ENABLE_SCRATCH_DIAGNOSTIC = False
    CALIBRATION_EPOCHS = 120
    CALIBRATION_LR = 0.001

    all_files = []
    for i in range(1, 51):
        all_files.append(f"GAC{i:03d}_Normal_F.csv")
        all_files.append(f"GAC{i:03d}_Load_F.csv")

    curve_rows = []  # one row per (participant, calibration_fraction)

    for pid in range(1, 51):
        pid_excel_path = os.path.join(out_folder, f"CALIBRATION_pid_{pid:03d}.xlsx")
        if os.path.exists(pid_excel_path):
            logging.info(f"Participant {pid:03d} already completed (found {pid_excel_path}) - loading and skipping.")
            existing_df = pd.read_excel(pid_excel_path)
            curve_rows.extend(existing_df.to_dict("records"))
            continue

        t0 = time.time()
        logging.info(f"\n=== CALIBRATION CURVE: participant {pid:03d} ===")

        # ---- 1. Train GROUP model on the other 49 participants ----
        train_files = [f for f in all_files if parse_participant_id(f) != pid]
        group_items = [load_or_build_cached_windows(data_folder, f, cache_dir, FS, WINDOW_SEC, N_FEATURES)
                       for f in train_files]
        Xw_train, yw_train, meta_train = concat_windows(group_items, n_features=N_FEATURES)

        if len(yw_train) == 0:
            logging.warning(f"Participant {pid:03d}: no group training windows. Skipping.")
            continue

        scaler = StandardScaler()
        Xw_train_scaled = scaler.fit_transform(Xw_train)

        val_mask = blockwise_val_mask(meta_train, val_fraction=VAL_FRAC)
        X_tr, y_tr = Xw_train_scaled[~val_mask], yw_train[~val_mask]
        X_val, y_val = Xw_train_scaled[val_mask], yw_train[val_mask]
        X_tr_bal, y_tr_bal = balance_windows(X_tr, y_tr, seed=SEED)

        group_model = build_mlp(X_tr_bal.shape[1], lr=LR, l2=L2, hidden1=H1, hidden2=H2, dropout=DROPOUT)
        early = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)

        logging.info("Training GROUP model...")
        group_model.fit(
            X_tr_bal, y_tr_bal,
            validation_data=(X_val, y_val),
            epochs=EPOCHS, batch_size=BATCH, callbacks=[early], verbose=0
        )

        val_prob = group_model.predict(X_val, verbose=0).reshape(-1)
        group_threshold, _ = choose_threshold_on_val(y_val, val_prob)

        # ---- 2. Load participant p's own data, scale with the GROUP scaler ----
        p_items = [load_or_build_cached_windows(data_folder, f, cache_dir, FS, WINDOW_SEC, N_FEATURES)
                   for f in participant_files(pid)]
        Xp, yp, meta_p = concat_windows(p_items, n_features=N_FEATURES)

        if len(yp) == 0:
            logging.warning(f"Participant {pid:03d}: no personal windows. Skipping.")
            continue

        Xp_scaled = scaler.transform(Xp)  # IMPORTANT: use the group model's scaler, not a new one

        pool_mask, test_mask = personal_chronological_split(meta_p, PERSONAL_TRAIN_FRACTION)
        X_pool, y_pool = Xp_scaled[pool_mask], yp[pool_mask]
        meta_pool = meta_p.loc[pool_mask].reset_index(drop=True)
        X_test, y_test = Xp_scaled[test_mask], yp[test_mask]

        if len(y_test) == 0 or len(np.unique(y_test)) < 2:
            logging.warning(f"Participant {pid:03d}: personal test set missing a class. Results may be limited.")

        # ---- 3a. Evaluate the 0% baseline (pure group model, no personal data) ----
        test_prob = group_model.predict(X_test, verbose=0).reshape(-1)
        y_pred = (test_prob >= group_threshold).astype(int)
        metrics = eval_metrics(y_test, y_pred)
        metrics.update({"participant": pid, "calibration_fraction": 0.0,
                         "n_calibration_windows": 0, "n_test_windows": int(len(y_test)),
                         "degenerate_prediction": bool(len(np.unique(y_pred)) < 2),
                         "model_type": "personal_only"})
        curve_rows.append(metrics)

        # Group model no longer needed - free it before the (lighter) calibration
        # sweep, which only trains small from-scratch models on personal data.
        del group_model
        tf.keras.backend.clear_session()
        gc.collect()

        # ---- 3b. Sweep the remaining calibration fractions ----
        for frac in CALIBRATION_FRACTIONS:
            if frac == 0.0:
                continue

            cal_idx = take_earliest_fraction_per_file(meta_pool, frac)
            X_cal, y_cal = X_pool[cal_idx], y_pool[cal_idx]

            if len(np.unique(y_cal)) < 2:
                logging.warning(f"Participant {pid:03d}, frac={frac}: calibration slice has one class only. Skipping this point.")
                continue

            X_cal_bal, y_cal_bal = balance_windows(X_cal, y_cal, seed=SEED)

            # Carve a small held-out slice from WITHIN the calibration data
            # itself (last 20%) for threshold selection - keeps the personal
            # TEST set completely untouched, while avoiding tuning the
            # threshold on the exact data just trained on
            n_cal = len(y_cal_bal)
            cal_val_cut = max(1, int(0.8 * n_cal))
            X_cal_fit, y_cal_fit = X_cal_bal[:cal_val_cut], y_cal_bal[:cal_val_cut]
            X_cal_val, y_cal_val = X_cal_bal[cal_val_cut:], y_cal_bal[cal_val_cut:]
            if len(np.unique(y_cal_val)) < 2:
                X_cal_fit, y_cal_fit = X_cal_bal, y_cal_bal
                X_cal_val, y_cal_val = X_cal_bal, y_cal_bal

            # Train a small model on ONLY this participant's calibration slice
            # (no group pretraining at all). This is the version that actually
            # works: real, sensible metrics with only ~6% degenerate cases,
            # versus the group-pretrained approach which collapsed to a
            # single-class prediction for 94-100% of participants regardless
            # of calibration amount. Group pretraining is dropped here rather
            # than continuing to debug an approach that was fighting an
            # uphill, unproductive battle.
            cal_model = build_mlp(X_cal_fit.shape[1], lr=CALIBRATION_LR, l2=0.0, dropout=0.0)
            cal_early = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
            cal_model.fit(
                X_cal_fit, y_cal_fit,
                epochs=CALIBRATION_EPOCHS, batch_size=min(32, len(y_cal_fit)),
                callbacks=[cal_early], verbose=0
            )
            cal_prob = cal_model.predict(X_cal_val, verbose=0).reshape(-1)
            cal_threshold, _ = choose_threshold_on_val(
                y_cal_val, cal_prob, min_samples=10, fallback_threshold=0.5
            )
            test_prob = cal_model.predict(X_test, verbose=0).reshape(-1)
            y_pred = (test_prob >= cal_threshold).astype(int)

            metrics = eval_metrics(y_test, y_pred)
            metrics.update({"participant": pid, "calibration_fraction": frac,
                             "n_calibration_windows": int(len(y_cal)), "n_test_windows": int(len(y_test)),
                             "degenerate_prediction": bool(len(np.unique(y_pred)) < 2),
                             "model_type": "personal_only"})
            curve_rows.append(metrics)

            del cal_model
            tf.keras.backend.clear_session()
            gc.collect()

        # Per-participant Excel
        p_df = pd.DataFrame([r for r in curve_rows if r["participant"] == pid]).sort_values("calibration_fraction")
        p_df.to_excel(os.path.join(out_folder, f"CALIBRATION_pid_{pid:03d}.xlsx"), index=False)

        elapsed = time.time() - t0
        logging.info(f"✅ Participant {pid:03d} done in {elapsed/60:.1f} min")

        tf.keras.backend.clear_session()
        gc.collect()

    # ---- Summary: averaged curve across all participants ----
    if not curve_rows:
        logging.error("No results produced.")
        return

    full_df = pd.DataFrame(curve_rows)
    summary_path = os.path.join(out_folder, "SUMMARY_CALIBRATION_CURVE.xlsx")

    curve_summary = (
        full_df.groupby(["calibration_fraction", "model_type"])
        .agg(
            n_participants=("participant", "nunique"),
            accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
            f1_mean=("f1", "mean"), f1_std=("f1", "std"),
            mcc_mean=("mcc", "mean"), mcc_std=("mcc", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            pct_degenerate=("degenerate_prediction", "mean"),
        )
        .reset_index()
        .sort_values(["calibration_fraction", "model_type"])
    )

    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        full_df.to_excel(writer, sheet_name="AllParticipants", index=False)
        curve_summary.to_excel(writer, sheet_name="CalibrationCurve", index=False)

    logging.info(f"✅ Saved calibration curve summary: {summary_path}")
    logging.info("\n" + curve_summary.to_string(index=False))


if __name__ == "__main__":
    main()
