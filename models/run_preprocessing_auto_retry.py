"""
AUTO-RETRY WRAPPER for preprocess_signals.py (pure Python version)
====================================================================

This automatically re-runs preprocess_signals.py over and over until
ALL 100 files (50 participants x 2 conditions) are actually done - no
need to manually type the command again, and no .bat file confusion.

HOW TO USE:
1. Put this file in the SAME "scripts" folder as preprocess_signals.py
2. Run it exactly like you'd run any other script:
       python run_preprocessing_auto_retry.py
3. Walk away - it keeps retrying by itself until finished, or until it
   hits the maximum attempt limit below, whichever comes first.

Since preprocess_signals.py already saves progress per file (each
cleaned CSV + a small ".stats.json" marker), every retry picks up
exactly where the last one stopped - no wasted work.
"""

import os
import subprocess
import sys
import time

MAX_ATTEMPTS = 30
TARGET_FILE_COUNT = 100
WAIT_BETWEEN_RETRIES_SEC = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESS_SCRIPT = os.path.join(SCRIPT_DIR, "preprocess_signals.py")
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CLEANED_FOLDER = os.path.join(PROJECT_ROOT, "data", "ALLDATAFINALPAS_CLEANED")


def count_completed_files():
    if not os.path.exists(CLEANED_FOLDER):
        return 0
    return len([f for f in os.listdir(CLEANED_FOLDER) if f.endswith(".csv")])


def main():
    if not os.path.exists(PREPROCESS_SCRIPT):
        print(f"ERROR: Could not find preprocess_signals.py in {SCRIPT_DIR}")
        print("Make sure this wrapper is in the SAME folder as preprocess_signals.py")
        sys.exit(1)

    attempt = 0
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        print("\n" + "=" * 70)
        print(f"  ATTEMPT {attempt} of {MAX_ATTEMPTS}  -  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70 + "\n")

        result = subprocess.run([sys.executable, PREPROCESS_SCRIPT])

        completed = count_completed_files()
        print(f"\nFiles completed so far: {completed} / {TARGET_FILE_COUNT}")

        if completed >= TARGET_FILE_COUNT:
            print("\n" + "=" * 70)
            print("  ALL FILES COMPLETE - DONE!")
            print("=" * 70)
            return

        if result.returncode == 0 and completed < TARGET_FILE_COUNT:
            # Script finished cleanly but somehow fewer files than expected
            # (e.g. some raw files genuinely missing) - no point retrying forever
            print("\nScript completed without crashing, but file count is below target.")
            print("This likely means some raw source files are missing rather than a crash.")
            print("Check the log above for any 'Missing raw file' warnings.")
            return

        print(f"\nStopped early (crash or interruption). Retrying in {WAIT_BETWEEN_RETRIES_SEC} seconds...")
        time.sleep(WAIT_BETWEEN_RETRIES_SEC)

    print("\n" + "=" * 70)
    print(f"  Reached maximum attempt limit ({MAX_ATTEMPTS}) without finishing.")
    print("  Something is stopping this repeatedly at roughly the same point.")
    print("  Please check:")
    print("    - Windows Defender / antivirus real-time scanning")
    print("    - Power settings (sleep / screen timeout) - keep the laptop plugged in and awake")
    print("    - Available disk space")
    print("=" * 70)


if __name__ == "__main__":
    main()
