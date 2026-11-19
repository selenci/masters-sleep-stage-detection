import glob
import gc
import os
import csv
import time
import signal
import mne
import yasa
import sklearn
import numpy as np
import pandas as pd
from collections import defaultdict

# ---------------------------------------------------------------------------
# Diagnostics: print library versions once, up front. If yasa's bundled
# pickled models were trained on a different sklearn version than what's
# installed, that mismatch alone can cause slow or wrong behavior.
# ---------------------------------------------------------------------------
print(f"yasa version: {yasa.__version__}", flush=True)
print(f"scikit-learn version: {sklearn.__version__}", flush=True)
print(f"mne version: {mne.__version__}", flush=True)


def age_group_for_sleep(days):
    if days < 2 * 365:
        return 3    # infant
    elif days < 12 * 365:
        return 2    # child
    elif days < 18 * 365:
        return 1    # teen
    else:
        return 0    # adult


GROUP_NAMES = {3: "infant", 2: "child", 1: "teen", 0: "adult"}

# Per-file hard timeout in seconds. If a single file exceeds this, it's
# killed and logged rather than eating the whole job's time budget.
PER_FILE_TIMEOUT_SECONDS = 20 * 60  # 20 minutes


class FileTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise FileTimeout("File processing exceeded per-file timeout")


signal.signal(signal.SIGALRM, _timeout_handler)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
filenames = glob.glob('../Data Downloader/Data/*.edf')
filenames = [filename.split('/')[-1].split('.edf')[0] for filename in filenames]
df_ages = pd.read_csv('../Data/PROCESSED_NCH/SLEEP_STUDY.csv')

results_path = "results.csv"
write_header = not os.path.exists(results_path)
if write_header:
    with open(results_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "filename", "age_years", "age_group", "age_group_name",
            "accuracy", "n_epochs", "sfreq_hz", "load_seconds",
            "sls1_seconds", "sls2_seconds", "total_seconds", "status"
        ])

accuracies = []
group_accuracies = defaultdict(list)

# --- Epoch-count tracking, per age group and overall ---
group_epoch_counts = defaultdict(int)
total_epochs = 0

print(f"Found {len(filenames)} files to process. Timeout per file: {PER_FILE_TIMEOUT_SECONDS}s", flush=True)

for filename in filenames:
    t_file_start = time.time()
    signal.alarm(PER_FILE_TIMEOUT_SECONDS)

    load_seconds = None
    sls1_seconds = None
    sls2_seconds = None
    sfreq = None
    n_epochs = None
    status = "ok"

    try:
        age_row = df_ages.loc[
            (df_ages['STUDY_PAT_ID'].astype(str) == filename.split('_')[0]) &
            (df_ages['SLEEP_STUDY_ID'].astype(str) == filename.split('_')[1])
        ]
        age_days = age_row['AGE_AT_SLEEP_STUDY_DAYS'].values[0]
        age = age_days / 365.25
        group = age_group_for_sleep(age_days)
        print(f"\n=== {filename} | age {age:.2f}y | {GROUP_NAMES[group]} ===", flush=True)

        # --- Load only the channel YASA needs, before preloading full data ---
        t0 = time.time()
        raw = mne.io.read_raw_edf(f'../Data Downloader/Data/{filename}.edf', preload=False)
        raw.pick(['EEG C3-M2'])
        raw.load_data()
        load_seconds = time.time() - t0
        sfreq = raw.info['sfreq']
        print(f"Loaded in {load_seconds:.1f}s | sfreq={sfreq} Hz", flush=True)

        # A sampling rate that doesn't divide cleanly into YASA's target
        # (100 Hz) can occasionally push MNE's FFT-based resampling into a
        # very slow path. Flagging it here so it shows up in the log.
        if sfreq % 100 != 0:
            print(f"NOTE: sfreq={sfreq} is not a clean multiple of 100 Hz "
                  f"-- resampling inside YASA may be slower than usual for this file.", flush=True)

        df = pd.read_csv(f'../Data Downloader/Data/{filename}.tsv', sep='\t')
        df = df.loc[df['description'].isin(
            ['Sleep stage W', "Sleep stage ?", 'Sleep stage N1',
             'Sleep stage N2', 'Sleep stage N3', 'Sleep stage R']
        )]
        df['description'] = df['description'].map({
            'Sleep stage W': "W", 'Sleep stage N1': "N1", 'Sleep stage N2': "N2",
            'Sleep stage N3': "N3", 'Sleep stage R': "R"
        })

        tmin = df['onset'].min()
        tmax = df['onset'].max()
        if tmin is not None and tmax is not None:
            print(f"Cropping to [{tmin}, {tmax}]", flush=True)
            raw = raw.crop(tmin=tmin, tmax=tmax)
            df = df.loc[(df['onset'] >= tmin) & (df['onset'] < tmax)]

        # --- Run YASA, timing each call separately ---
        t1 = time.time()
        sls1 = yasa.SleepStaging(raw, eeg_name="EEG C3-M2", metadata=dict(age=age, male=True))
        y_pred = sls1.predict()
        sls1_seconds = time.time() - t1
        print(f"sls1 (male=True) took {sls1_seconds:.1f}s", flush=True)

        t2 = time.time()
        sls2 = yasa.SleepStaging(raw, eeg_name="EEG C3-M2", metadata=dict(age=age, male=False))
        y_pred2 = sls2.predict()
        sls2_seconds = time.time() - t2
        print(f"sls2 (male=False) took {sls2_seconds:.1f}s", flush=True)

        y_act = df['description'].to_numpy()

        # YASA predicts one label per 30s epoch; the annotation array can
        # occasionally be a different length (e.g. a trailing partial epoch
        # at a recording boundary). Truncate to the overlapping length so
        # n_epochs reflects exactly what was compared, and so the equality
        # check below can't silently misbehave on mismatched lengths.
        n_epochs = min(len(y_pred), len(y_pred2), len(y_act))
        if len(y_pred) != len(y_act) or len(y_pred2) != len(y_act):
            print(f"NOTE: epoch count mismatch (y_pred={len(y_pred)}, "
                  f"y_pred2={len(y_pred2)}, y_act={len(y_act)}) -- "
                  f"comparing only the overlapping {n_epochs} epochs.", flush=True)

        y_pred_c = y_pred[:n_epochs]
        y_pred2_c = y_pred2[:n_epochs]
        y_act_c = y_act[:n_epochs]

        acc = np.mean(y_pred_c == y_act_c)
        acc2 = np.mean(y_pred2_c == y_act_c)
        best_acc = max(acc, acc2) * 100

        accuracies.append(best_acc)
        group_accuracies[group].append(best_acc)
        group_epoch_counts[group] += n_epochs
        total_epochs += n_epochs

        signal.alarm(0)  # cancel timeout, file finished normally

    except FileTimeout:
        status = "timeout"
        age, group = None, None
        best_acc = None
        print(f"TIMEOUT: {filename} exceeded {PER_FILE_TIMEOUT_SECONDS}s, skipping.", flush=True)

    except Exception as e:
        status = f"error: {e}"
        age, group = None, None
        best_acc = None
        print(f'Error processing {filename}: {e}', flush=True)

    finally:
        signal.alarm(0)
        total_seconds = time.time() - t_file_start

        with open(results_path, "a", newline="") as f:
            csv.writer(f).writerow([
                filename,
                age if age is not None else "",
                group if group is not None else "",
                GROUP_NAMES.get(group, "") if group is not None else "",
                best_acc if best_acc is not None else "",
                n_epochs if n_epochs is not None else "",
                sfreq if sfreq is not None else "",
                round(load_seconds, 1) if load_seconds is not None else "",
                round(sls1_seconds, 1) if sls1_seconds is not None else "",
                round(sls2_seconds, 1) if sls2_seconds is not None else "",
                round(total_seconds, 1),
                status,
            ])

        # Clean up to keep memory from accumulating across iterations
        for var in ("raw", "sls1", "sls2", "y_pred", "y_pred2", "df"):
            if var in locals():
                del locals()[var]
        gc.collect()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
if accuracies:
    print(f'\nAverage accuracy across {len(accuracies)} successfully processed recordings: '
          f'{np.mean(accuracies):.2f} % ± {np.std(accuracies):.2f} %', flush=True)
    print(accuracies, flush=True)
else:
    print("\nNo recordings were successfully processed.", flush=True)

print(f"\nTotal epochs evaluated across all recordings: {total_epochs}", flush=True)

print("\nAccuracy by age group:", flush=True)
for group in sorted(group_accuracies.keys()):
    accs = group_accuracies[group]
    n_ep = group_epoch_counts[group]
    print(f'{GROUP_NAMES[group]} (n={len(accs)} recordings, {n_ep} epochs): '
          f'{np.mean(accs):.2f} % ± {np.std(accs):.2f} %', flush=True)

print(f"\nFull per-file results written to {results_path}", flush=True)
