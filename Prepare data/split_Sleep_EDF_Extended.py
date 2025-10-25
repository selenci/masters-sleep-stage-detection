import mne
import pandas as pd
import numpy as np
import os
import sys
import scipy.signal

SUBFOLDER = 'sleep-telemetry'  # or 'sleep-cassette'
FILE_BASE_NAME = sys.argv[1]
HYPNOGRAM = sys.argv[2]
TARGET_SFREQ = 128


def get_spectrogram(raw):
    segment_data = raw.get_data()

    spectrograms = []
    for channel_data in segment_data:
        f, t, Zxx = scipy.signal.stft(
            channel_data,
            fs=TARGET_SFREQ,
            nperseg=2 * TARGET_SFREQ,
            noverlap=TARGET_SFREQ
        )
        
        f_mask = (f < 50)
        Zxx = Zxx[f_mask, :]

        spectrogram = np.log1p(np.abs(Zxx))
        spectrograms.append(spectrogram)
    return np.stack(spectrograms, axis=0).astype(np.float32)

edf_file = f"../Data/Sleep-EDF Database Expanded/{SUBFOLDER}/{FILE_BASE_NAME}.edf"
raw = mne.io.read_raw_edf(edf_file, include=['EEG Fpz-Cz', 'EEG Pz-Oz'], preload=True)

sfreq = raw.info['sfreq']
print(f"Sampling frequency: {sfreq} Hz")
print(raw.ch_names)

raw.filter(l_freq=0.3, h_freq=30)
raw.resample(sfreq=TARGET_SFREQ)

st_file = f"../Data/Sleep-EDF Database Expanded/{SUBFOLDER}/{HYPNOGRAM}.edf"
annot = mne.read_annotations(st_file)
df = annot.to_data_frame()

print(df['description'].unique())

stage_map = {
    "Sleep stage W": "W",
    "Sleep stage 1": "S1",
    "Sleep stage 2": "S2",
    "Sleep stage 3": "S3",
    "Sleep stage 4": "S4",
    "Sleep stage R": "R",
}
df["description"] = df["description"].map(stage_map).fillna(df["description"])
first_onset = df['onset'].iloc[0]
df['onset'] = (df['onset'] - first_onset).dt.total_seconds()

print(df)



file_counter = 0
for _, r in df.iterrows():

    if r['description'] not in ['W', 'S1', 'S2', 'S3', 'S4', 'R']:
        print(f"Skipping invalid sleep stage: {r['description']}")
        continue

    start_time = r['onset']
    end_time = r['onset'] + 30  # 30 seconds duration

    while end_time <= r['onset'] + r['duration']:
        segment = raw.copy().crop(tmin=start_time, tmax=end_time)

        if segment.times.size / TARGET_SFREQ < 29.9:
            print(f"Skipping segment {file_counter:04d} due to short duration.")
            start_time = end_time
            end_time = start_time + 30
            continue

        spectrogram = get_spectrogram(segment)

        out_dir = f"../Data/PROCESSED_SLEEP_EDF/{FILE_BASE_NAME}"
        os.makedirs(out_dir, exist_ok=True)

        np.save(f"../Data/PROCESSED_SLEEP_EDF/{FILE_BASE_NAME}/{file_counter:04d}.npy", spectrogram)

        # Save the corresponding label
        with open(f"../Data/PROCESSED_SLEEP_EDF/{FILE_BASE_NAME}/{file_counter:04d}-stage.txt", "w") as f:
            f.write(r['description'])


        file_counter += 1

        start_time = end_time
        end_time = start_time + 30