"""
SIGNAL PREPROCESSING - addresses IEEE Access reviewer comments R1#5 and R2#3
==============================================================================

Reviewer 1 (#5) and Reviewer 2 (#3) both asked for concrete preprocessing
details: filter parameters, artifact-rejection criteria, rejected-data
percentage. Previously, raw signals were used with only z-score
normalization at the modeling stage - no filtering or artifact rejection
was actually performed. This script does that properly, and produces the
exact numbers needed to answer both reviewer comments honestly.

WHAT THIS SCRIPT DOES, per participant, per condition file (Normal, Load):

1. EEG (14 channels):
   - Bandpass filter 1-40 Hz (Butterworth, 4th order, zero-phase via filtfilt)
     - removes DC drift / slow baseline wander below 1 Hz
     - removes high-frequency EMG/muscle and electrical noise above 40 Hz
   - Notch filter at 50 Hz (mains interference - set NOTCH_FREQ=60 if your
     data was collected somewhere with 60 Hz mains, e.g. the US)
   - Amplitude-based artifact rejection: after windowing, any 5-second
     window where a channel's range exceeds a PARTICIPANT-SPECIFIC threshold
     (median +/- 4x MAD of that channel, for that participant) is flagged
     and rejected. Participant-specific (not a fixed uV cutoff) because the
     raw Emotiv units in this dataset are not calibrated uV.

2. GSR:
   - Reject/interpolate implausible sample-to-sample jumps (derivative-based
     outlier detection), since these upstream CSVs show forward-filled
     values from a slower native sampling rate - genuine phasic responses
     rise/fall smoothly, whereas isolated single-sample spikes are likely
     motion/contact artifacts.

3. HR:
   - Reject physiologically implausible values (outside 40-200 BPM) and
     replace via forward/backward fill.

4. Windows failing amplitude-based EEG rejection are DROPPED entirely
   (not interpolated), consistent with standard EEG artifact-rejection
   practice - a rejected window contributes no data downstream.

OUTPUT:
  <project_root>/data/ALLDATAFINALPAS_CLEANED/
    - GACxxx_Condition_F.csv  (same column structure as raw input, minus
      rejected rows - so all 12 existing model scripts work unchanged,
      only the data_folder path needs to change)
  <project_root>/PREPROCESSING_REPORT/
    - preprocessing_summary.xlsx  (per-participant + overall % windows
      rejected, % samples affected by GSR/HR correction - THESE ARE THE
      NUMBERS TO REPORT IN THE PAPER for R1#5 / R2#3)

INTEGRATION WITH YOUR EXISTING SCRIPTS:
  In each of your 12 model scripts, change:
      data_folder = os.path.join(project_root, "data", "ALLDATAFINALPAS")
  to:
      data_folder = os.path.join(project_root, "data", "ALLDATAFINALPAS_CLEANED")
  and DELETE your existing CACHE_WINDOWS_* folders before re-running, since
  the cached windows were built from the old, unfiltered data.
"""

import os
import re
import logging
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ----------------------------
# Configuration
# ----------------------------
FS = 128                # EEG sampling rate (Hz)
WINDOW_SEC = 5
N_FEATURES = 16          # 14 EEG + GSR + HR (matches your existing scripts)
N_EEG_CHANNELS = 14

BANDPASS_LOW = 1.0        # Hz
BANDPASS_HIGH = 40.0      # Hz
NOTCH_FREQ = 50.0         # Hz - CHANGE TO 60.0 if collected in a 60Hz-mains country
NOTCH_Q = 30.0            # notch filter quality factor

EEG_ARTIFACT_MAD_MULTIPLIER = 4.0   # participant-specific rejection threshold
HR_MIN_BPM, HR_MAX_BPM = 40.0, 200.0
GSR_JUMP_MAD_MULTIPLIER = 6.0       # for derivative-based GSR spike detection

PARTICIPANT_RE = re.compile(r"GAC(\d{3})_")


def parse_participant_id(filename: str) -> int:
    m = PARTICIPANT_RE.search(filename)
    return int(m.group(1))


def participant_files(pid: int):
    return [f"GAC{pid:03d}_Normal_F.csv", f"GAC{pid:03d}_Load_F.csv"]


# ----------------------------
# Filters
# ----------------------------
def bandpass_filter(signal, fs, low, high, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def notch_filter(signal, fs, freq, q):
    nyq = fs / 2.0
    b, a = iirnotch(freq / nyq, q)
    return filtfilt(b, a, signal)


def clean_eeg_channel(signal, fs):
    filtered = bandpass_filter(signal, fs, BANDPASS_LOW, BANDPASS_HIGH)
    filtered = notch_filter(filtered, fs, NOTCH_FREQ, NOTCH_Q)
    return filtered


# ----------------------------
# GSR / HR cleaning
# ----------------------------
def clean_gsr(gsr):
    gsr = gsr.copy()
    diffs = np.abs(np.diff(gsr, prepend=gsr[0]))
    med = np.median(diffs)
    mad = np.median(np.abs(diffs - med)) + 1e-9
    threshold = med + GSR_JUMP_MAD_MULTIPLIER * mad
    bad = diffs > threshold
    n_bad = int(bad.sum())
    if n_bad > 0:
        gsr_series = pd.Series(gsr)
        gsr_series[bad] = np.nan
        gsr_series = gsr_series.interpolate(limit_direction="both")
        gsr = gsr_series.values
    return gsr, n_bad


def clean_hr(hr):
    hr = hr.copy()
    bad = (hr < HR_MIN_BPM) | (hr > HR_MAX_BPM)
    n_bad = int(bad.sum())
    if n_bad > 0:
        hr_series = pd.Series(hr)
        hr_series[bad] = np.nan
        hr_series = hr_series.interpolate(limit_direction="both")
        hr = hr_series.values
    return hr, n_bad


# ----------------------------
# EEG artifact rejection (per 5-second window, participant-specific threshold)
# ----------------------------
def compute_eeg_reject_mask(eeg_matrix, fs, window_sec, mad_multiplier):
    """
    eeg_matrix: shape (n_samples, n_eeg_channels), ALREADY bandpass+notch filtered.
    Returns a boolean mask, one entry per window, True = REJECT.
    Threshold is participant-specific: computed from this file's own
    per-channel window-range distribution (median + k*MAD).
    """
    win_len = fs * window_sec
    n_samples = eeg_matrix.shape[0]
    n_windows = n_samples // win_len

    window_ranges = np.zeros((n_windows, eeg_matrix.shape[1]))
    for w in range(n_windows):
        seg = eeg_matrix[w * win_len:(w + 1) * win_len, :]
        window_ranges[w, :] = seg.max(axis=0) - seg.min(axis=0)

    reject_mask = np.zeros(n_windows, dtype=bool)
    for ch in range(eeg_matrix.shape[1]):
        col = window_ranges[:, ch]
        med = np.median(col)
        mad = np.median(np.abs(col - med)) + 1e-9
        threshold = med + mad_multiplier * mad
        reject_mask |= (col > threshold)

    return reject_mask, win_len, n_windows


# ----------------------------
# MAIN per-file cleaning
# ----------------------------
def clean_one_file(csv_path, n_features=N_FEATURES):
    df = pd.read_csv(csv_path)  # read ALL columns - extra columns (e.g. INCIDENTS) must survive
    columns = list(df.columns)

    eeg_cols = columns[:N_EEG_CHANNELS]
    gsr_col = "GSR" if "GSR" in columns else columns[N_EEG_CHANNELS]
    hr_col = "HR" if "HR" in columns else columns[N_EEG_CHANNELS + 1]

    df = df.fillna(df.median(numeric_only=True))

    # --- EEG: bandpass + notch filter every channel ---
    eeg_filtered = np.zeros((len(df), N_EEG_CHANNELS))
    for i, col in enumerate(eeg_cols):
        eeg_filtered[:, i] = clean_eeg_channel(df[col].values.astype(float), FS)

    # --- GSR / HR cleaning ---
    gsr_clean, n_gsr_fixed = clean_gsr(df[gsr_col].values.astype(float))
    hr_clean, n_hr_fixed = clean_hr(df[hr_col].values.astype(float))

    # --- EEG artifact rejection (participant/file-specific threshold) ---
    reject_mask, win_len, n_windows = compute_eeg_reject_mask(
        eeg_filtered, FS, WINDOW_SEC, EEG_ARTIFACT_MAD_MULTIPLIER
    )

    # Build cleaned, row-aligned output, dropping rejected windows entirely
    keep_sample_mask = np.ones(len(df), dtype=bool)
    for w in range(n_windows):
        if reject_mask[w]:
            keep_sample_mask[w * win_len:(w + 1) * win_len] = False
    # Any leftover samples beyond the last full window (rare remainder) - keep as-is

    out = pd.DataFrame(eeg_filtered, columns=eeg_cols)
    out[gsr_col] = gsr_clean
    out[hr_col] = hr_clean
    # carry through any remaining original columns (e.g. INCIDENTS) unchanged
    for col in columns[n_features:]:
        out[col] = df[col].values if col in df.columns else np.nan

    out_cleaned = out.loc[keep_sample_mask].reset_index(drop=True)

    stats = {
        "n_windows_total": int(n_windows),
        "n_windows_rejected": int(reject_mask.sum()),
        "pct_windows_rejected": float(reject_mask.mean() * 100),
        "n_gsr_samples_corrected": int(n_gsr_fixed),
        "n_hr_samples_corrected": int(n_hr_fixed),
        "n_samples_total": int(len(df)),
    }
    return out_cleaned, stats


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_folder = os.path.join(project_root, "data", "ALLDATAFINALPAS")
    clean_folder = os.path.join(project_root, "data", "ALLDATAFINALPAS_CLEANED")
    report_folder = os.path.join(project_root, "PREPROCESSING_REPORT")
    os.makedirs(clean_folder, exist_ok=True)
    os.makedirs(report_folder, exist_ok=True)

    all_files = []
    for i in range(1, 51):
        all_files.extend(participant_files(i))

    all_stats = []
    for fname in all_files:
        raw_path = os.path.join(raw_folder, fname)
        if not os.path.exists(raw_path):
            logging.warning(f"Missing raw file, skipping: {fname}")
            continue

        out_path = os.path.join(clean_folder, fname)
        stats_cache_path = out_path + ".stats.json"

        if os.path.exists(out_path) and os.path.exists(stats_cache_path):
            logging.info(f"{fname} already cleaned - loading saved stats and skipping.")
            import json
            with open(stats_cache_path, "r") as f:
                stats = json.load(f)
            all_stats.append(stats)
            continue

        try:
            logging.info(f"Cleaning {fname} ...")
            cleaned_df, stats = clean_one_file(raw_path)

            cleaned_df.to_csv(out_path, index=False)

            stats["file_id"] = fname
            stats["participant_id"] = parse_participant_id(fname)
            stats["condition"] = "Normal" if "Normal" in fname else "Load"
            all_stats.append(stats)

            import json
            with open(stats_cache_path, "w") as f:
                json.dump(stats, f)

        except Exception as e:
            logging.error(f"FAILED on {fname}: {e}. Skipping this file - rerun the script to retry it.")
            continue

        import gc
        gc.collect()

    stats_df = pd.DataFrame(all_stats)
    summary_path = os.path.join(report_folder, "preprocessing_summary.xlsx")

    overall = pd.DataFrame([{
        "n_files": len(stats_df),
        "mean_pct_windows_rejected": stats_df["pct_windows_rejected"].mean(),
        "sd_pct_windows_rejected": stats_df["pct_windows_rejected"].std(),
        "min_pct_windows_rejected": stats_df["pct_windows_rejected"].min(),
        "max_pct_windows_rejected": stats_df["pct_windows_rejected"].max(),
        "total_gsr_samples_corrected": stats_df["n_gsr_samples_corrected"].sum(),
        "total_hr_samples_corrected": stats_df["n_hr_samples_corrected"].sum(),
    }])

    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        stats_df.to_excel(writer, sheet_name="PerFile", index=False)
        overall.to_excel(writer, sheet_name="Overall", index=False)

    logging.info(f"Done. Cleaned data: {clean_folder}")
    logging.info(f"Report: {summary_path}")
    logging.info("\n" + overall.to_string(index=False))


if __name__ == "__main__":
    main()
