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

NAIVE BAYES GROUP MODEL (LOPO) 



Outputs:
  <project_root>/OUTPUT_NB_GROUP_LOPO/
  <project_root>/CACHE_WINDOWS_NB_GROUP/
"""

import os
import re
import gc
import time
import logging
import random
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, balanced_accuracy_score
)

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42
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
# Cache (per file window features)
# ----------------------------
def window_cache_path(cache_dir, file_id, fs, window_sec, n_features):
    safe = file_id.replace(".csv", "")
    return os.path.join(cache_dir, f"{safe}_fs{fs}_w{window_sec}_f{n_features}.npz")

def build_window_features_from_csv(csv_path, file_id, participant_id, label,
                                   fs=128, window_sec=5, n_features=16):
    df = pd.read_csv(csv_path, usecols=range(n_features))

    # your cleaning
    if "GSR" in df.columns:
        df = df[df["GSR"] >= 0]

    df = df.reset_index(drop=True)

    # IMPORTANT: fill numeric NaNs, but if a column is all-NaN, median is NaN -> handled later by imputer
    df = df.fillna(df.median(numeric_only=True))

    win_len = int(fs * window_sec)
    n = len(df)
    feat_cols = list(df.columns[:n_features])

    X_list, y_list, start_list = [], [], []
    for start in range(0, n - win_len + 1, win_len):
        w = df.iloc[start:start + win_len][feat_cols]
        xw = w.mean(axis=0).values.astype(np.float32)

        X_list.append(xw)
        y_list.append(int(label))
        start_list.append(int(start))

    Xw = np.asarray(X_list, dtype=np.float32)
    yw = np.asarray(y_list, dtype=np.int32)

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
    pid = parse_participant_id(file_id)
    lab = parse_label(file_id)
    npz_path = window_cache_path(cache_dir, file_id, fs, window_sec, n_features)

    if os.path.exists(npz_path):
        z = np.load(npz_path, allow_pickle=True)
        return z["Xw"], z["yw"], z["meta"].item()

    csv_path = os.path.join(data_folder, file_id)
    if not os.path.exists(csv_path):
        logging.warning(f"Missing file: {csv_path}")
        return None

    Xw, yw, meta = build_window_features_from_csv(
        csv_path, file_id, pid, lab, fs=fs, window_sec=window_sec, n_features=n_features
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
        return np.empty((0, 16), dtype=np.float32), np.empty((0,), dtype=np.int32), pd.DataFrame()

    X = np.vstack(Xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int32)
    meta_df = pd.concat(metas, axis=0, ignore_index=True)
    return X, y, meta_df

# ----------------------------
# Splits / balancing / threshold
# ----------------------------
def blockwise_val_mask(meta_df, val_fraction=0.10):
    mask = np.zeros(len(meta_df), dtype=bool)
    for file_id, g in meta_df.groupby("file_id", sort=False):
        idx = g.index.to_numpy()
        # ensure time order by window_start_sample
        idx_sorted = idx[np.argsort(g["window_start_sample"].values)]
        cut = int((1.0 - val_fraction) * len(idx_sorted))
        mask[idx_sorted[cut:]] = True
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
# Robust cleaning (fix your crash)
# ----------------------------
def drop_nonfinite(X, y, meta_df, tag=""):
    finite_mask = np.isfinite(X).all(axis=1)
    dropped = int((~finite_mask).sum())
    if dropped > 0:
        logging.warning(f"{tag} Dropping {dropped} windows with NaN/inf features.")
    X2 = X[finite_mask]
    y2 = y[finite_mask]
    meta2 = meta_df.iloc[np.where(finite_mask)[0]].reset_index(drop=True)
    return X2, y2, meta2

def dist_str(y):
    u, c = np.unique(y, return_counts=True)
    return str({int(k): int(v) for k, v in zip(u, c)})

def cm_normalized(cm):
    cm = cm.astype(float)
    denom = cm.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return cm / denom

# ----------------------------
# Excel saving
# ----------------------------
def save_fold_excel(out_folder, fold_id, settings, metrics_main, cm_counts, cm_norm, metrics_shuffle, distributions):
    os.makedirs(out_folder, exist_ok=True)
    xlsx_path = os.path.join(out_folder, f"NB_GROUP_LOPO_excl_{fold_id:03d}.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame([settings]).to_excel(writer, sheet_name="Settings", index=False)
        pd.DataFrame([metrics_main]).to_excel(writer, sheet_name="WindowMetrics", index=False)
        pd.DataFrame([distributions]).to_excel(writer, sheet_name="Distributions", index=False)
        pd.DataFrame(cm_counts, index=["True0", "True1"], columns=["Pred0", "Pred1"]).to_excel(writer, sheet_name="CM_Counts")
        pd.DataFrame(cm_norm, index=["True0", "True1"], columns=["Pred0", "Pred1"]).to_excel(writer, sheet_name="CM_Normalized")
        pd.DataFrame([metrics_shuffle]).to_excel(writer, sheet_name="ShuffleControl", index=False)
    logging.info(f"Saved fold Excel: {xlsx_path}")

# ----------------------------
# MAIN
# ----------------------------
def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_folder = os.path.join(project_root, "data", "ALLDATAFINALPAS")

    out_folder = os.path.join(project_root, "OUTPUT_NB_GROUP_LOPO")
    cache_dir  = os.path.join(project_root, "CACHE_WINDOWS_NB_GROUP")
    os.makedirs(out_folder, exist_ok=True)

    FS = 128
    WINDOW_SEC = 5
    N_FEATURES = 16
    VAL_FRAC = 0.10

    # Master list
    all_files = []
    for i in range(1, 51):
        all_files.append(f"GAC{str(i).zfill(3)}_Normal_F.csv")
        all_files.append(f"GAC{str(i).zfill(3)}_Load_F.csv")

    summary_rows = []

    # Pipeline = median imputer + scaler + GaussianNB (this prevents NaN crash)
    def make_model():
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("nb", GaussianNB())
        ])

    for exclude_pid in range(1, 51):
        t0 = time.time()
        logging.info(f"\n=== NB LOPO fold: TEST participant {exclude_pid:03d} ===")

        train_files = [f for f in all_files if parse_participant_id(f) != exclude_pid]
        test_files  = [f for f in all_files if parse_participant_id(f) == exclude_pid]

        logging.info("Loading/Building cached window features...")
        train_items = [load_or_build_cached_windows(data_folder, f, cache_dir, FS, WINDOW_SEC, N_FEATURES) for f in train_files]
        test_items  = [load_or_build_cached_windows(data_folder, f, cache_dir, FS, WINDOW_SEC, N_FEATURES) for f in test_files]

        X_train, y_train, meta_train = concat_windows(train_items)
        X_test,  y_test,  meta_test  = concat_windows(test_items)

        logging.info(f"Windows: train={len(y_train):,} ({dist_str(y_train)}), test={len(y_test):,} ({dist_str(y_test)})")

        if len(y_train) == 0 or len(y_test) == 0:
            logging.warning("No windows found. Skipping fold.")
            continue

        # Drop non-finite windows BEFORE splits (extra safety)
        X_train, y_train, meta_train = drop_nonfinite(X_train, y_train, meta_train, tag="[TRAIN]")
        X_test,  y_test,  meta_test  = drop_nonfinite(X_test,  y_test,  meta_test,  tag="[TEST]")

        val_mask = blockwise_val_mask(meta_train, val_fraction=VAL_FRAC)
        X_tr, y_tr = X_train[~val_mask], y_train[~val_mask]
        X_val, y_val = X_train[val_mask], y_train[val_mask]

        # Balance training windows
        X_tr_bal, y_tr_bal = balance_windows(X_tr, y_tr, seed=SEED)

        # If after balancing only one class exists, skip (rare but possible)
        if len(np.unique(y_tr_bal)) < 2:
            logging.warning("Balanced training has only 1 class. Skipping fold.")
            continue

        logging.info(f"Split: X_tr={X_tr.shape}, X_val={X_val.shape}, balanced_train={X_tr_bal.shape}, val_dist={dist_str(y_val)}")

        model = make_model()
        model.fit(X_tr_bal, y_tr_bal)

        # threshold tune on validation
        val_prob = model.predict_proba(X_val)[:, 1]
        best_thr, best_val_f1 = choose_threshold_on_val(y_val, val_prob)
        logging.info(f"Best threshold={best_thr:.2f} (val F1={best_val_f1:.3f})")

        # REAL test
        test_prob = model.predict_proba(X_test)[:, 1]
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
            "train_label_dist_all": dist_str(y_train),
            "train_label_dist_train": dist_str(y_tr),
            "train_label_dist_train_balanced": dist_str(y_tr_bal),
            "val_label_dist": dist_str(y_val),
            "test_label_dist": dist_str(y_test),
            "test_pred_dist": dist_str(y_pred),
        }

        # SHUFFLE control
        y_tr_shuf = y_tr_bal.copy()
        np.random.shuffle(y_tr_shuf)

        model_s = make_model()
        model_s.fit(X_tr_bal, y_tr_shuf)

        test_prob_s = model_s.predict_proba(X_test)[:, 1]
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
            "n_features": N_FEATURES,
            "val_frac_windows": VAL_FRAC,
            "pipeline": "SimpleImputer(median)+StandardScaler+GaussianNB",
            "threshold_tuned_on": "validation_only",
            "cache_dir": cache_dir,
        }

        save_fold_excel(out_folder, exclude_pid, settings, metrics_main, cm, cmn, metrics_shuffle, distributions)

        # Summary row includes ALL key metrics (what you want in your final sheet)
        summary_rows.append({
            "participant": exclude_pid,
            "group_accuracy": metrics_main["accuracy"],
            "group_precision": metrics_main["precision"],
            "group_recall": metrics_main["recall"],
            "group_f1": metrics_main["f1"],
            "group_macro_f1": metrics_main["macro_f1"],
            "group_balanced_accuracy": metrics_main["balanced_accuracy"],
            "group_mcc": metrics_main["mcc"],
            "threshold": metrics_main["threshold_used"],
            "shuffle_accuracy": metrics_shuffle["accuracy"],
            "shuffle_f1": metrics_shuffle["f1"],
            "shuffle_mcc": metrics_shuffle["mcc"],
            "n_test_windows": metrics_main["n_test_windows"],
            "test_label_dist": distributions["test_label_dist"],
            "test_pred_dist": distributions["test_pred_dist"],
        })

        elapsed = time.time() - t0
        logging.info(f"✅ Fold {exclude_pid:03d} done in {elapsed/60:.1f} min | REAL F1={metrics_main['f1']:.3f} | SHUFFLE F1={metrics_shuffle['f1']:.3f}")

        gc.collect()

    if not summary_rows:
        logging.error("No folds completed. Check data.")
        return

    summary_df = pd.DataFrame(summary_rows).sort_values("participant")
    summary_path = os.path.join(out_folder, "SUMMARY_NB_GROUP_LOPO.xlsx")

    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        stats = {
            "n_participants": int(len(summary_df)),
            "group_f1_mean": float(summary_df["group_f1"].mean()),
            "group_f1_std": float(summary_df["group_f1"].std(ddof=1)),
            "group_acc_mean": float(summary_df["group_accuracy"].mean()),
            "group_prec_mean": float(summary_df["group_precision"].mean()),
            "group_rec_mean": float(summary_df["group_recall"].mean()),
            "group_mcc_mean": float(summary_df["group_mcc"].mean()),
            "shuffle_f1_mean": float(summary_df["shuffle_f1"].mean()),
            "shuffle_f1_std": float(summary_df["shuffle_f1"].std(ddof=1)),
        }
        pd.DataFrame([stats]).to_excel(writer, sheet_name="SummaryStats", index=False)

    logging.info(f"✅ Saved summary: {summary_path}")

if __name__ == "__main__":
    main()
