import pandas as pd
import mne
from datetime import datetime, timedelta
import io
import os
import scipy.signal
import numpy as np
import sys

TARGET_SFREQ = 128
FILE_BASE_NAME = sys.argv[1]

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

edf_file = f"../Data/Haaglanden Medisch Centrum sleep staging database/recordings/{FILE_BASE_NAME}.edf"
raw = mne.io.read_raw_edf(edf_file, preload=True)

sfreq = raw.info['sfreq']
print(f"Sampling frequency: {sfreq} Hz")
print(raw.ch_names)

channels_of_interest = ['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2']
raw.pick(channels_of_interest)
raw.filter(l_freq=0.3, h_freq=30)
raw.resample(sfreq=TARGET_SFREQ)

st_file = f"../Data/Haaglanden Medisch Centrum sleep staging database/recordings/{FILE_BASE_NAME}_sleepscoring.txt"
df = pd.read_csv(st_file, skipinitialspace=True)

print(df)

onsets = df["Recording onset"]
durations = df["Duration"]
descriptions = df["Annotation"].tolist()


stage_map = {
    "Sleep stage W": "W",
    "Sleep stage N1": "N1",
    "Sleep stage N2": "N2",
    "Sleep stage N3": "N3",
    "Sleep stage R": "R"
}
df["Annotation"] = df["Annotation"].map(stage_map).fillna(df["Annotation"])



file_counter = 0
for _, r in df.iterrows():
    start_time = r['Recording onset']
    end_time = r['Recording onset'] + 30  # 30 seconds duration

    if r['Annotation'] not in ['W', 'N1', 'N2', 'N3', 'R']:
        print(f"Skipping invalid sleep stage: {r['Annotation']}")
        continue

    while end_time <= r['Recording onset'] + r['Duration']:
        segment = raw.copy().crop(tmin=start_time, tmax=end_time)

        if segment.times.size / TARGET_SFREQ < 29.9:
            print(f"Skipping segment {file_counter:04d} due to short duration.")
            start_time = end_time
            end_time = start_time + 30
            continue

        spectrogram = get_spectrogram(segment)

        out_dir = f"../Data/PROCESSED_HAAGLANDEN/{FILE_BASE_NAME}"
        os.makedirs(out_dir, exist_ok=True)

        np.save(f"../Data/PROCESSED_HAAGLANDEN/{FILE_BASE_NAME}/{file_counter:04d}.npy", spectrogram)

        # Save the corresponding label
        with open(f"../Data/PROCESSED_HAAGLANDEN/{FILE_BASE_NAME}/{file_counter:04d}-stage.txt", "w") as f:
            f.write(r['Annotation'])


        file_counter += 1

        start_time = end_time
        end_time = start_time + 30