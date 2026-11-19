import pandas as pd
import mne
from datetime import datetime, timedelta
import io
import os
import scipy.signal
import numpy as np
import sys

FILE_BASE_NAME = sys.argv[1]
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


edf_file = f"../Data/NCH Sleep DataBank/{FILE_BASE_NAME}.edf"
tsv_file = f"../Data/NCH Sleep DataBank/{FILE_BASE_NAME}.tsv"
raw = mne.io.read_raw_edf(edf_file, preload=True)
print(raw.ch_names)
raw = raw.pick(['EEG F3-M2', 'EEG F4-M1', 'EEG C3-M2', 'EEG C4-M1', 'EEG O1-M2', 'EEG O2-M1'])
raw.filter(l_freq=0.3, h_freq=30)
raw.resample(sfreq=TARGET_SFREQ)

df = pd.read_csv(tsv_file, sep="\t")
df = df.loc[df['description'].isin(['Sleep stage N1', 'Sleep stage N2', 'Sleep stage N3', 'Sleep stage R', 'Sleep stage W'])]# 
df['description'] = df['description'].map({
    'Sleep stage N1': 'N1',
    'Sleep stage N2': 'N2',
    'Sleep stage N3': 'N3',
    'Sleep stage R': 'R',
    'Sleep stage W': 'W'
})

file_counter = 0
for idx, row in df.iterrows():
    if row['description'] not in ['W', 'N1', 'N2', 'N3', 'R']:
        print(f"Skipping invalid sleep stage: {row['description']}")
        continue

    start = row['onset']
    duration = row['duration']
    description = row['description']

    if duration != 30:
        print("BE CAREFUL SOLVE THIS", duration)
        continue

    segment = raw.copy().crop(tmin=start, tmax=start+duration)
    spectrogram = get_spectrogram(segment)

    out_dir = f"../Data/PROCESSED_NCH/{FILE_BASE_NAME}"
    os.makedirs(out_dir, exist_ok=True)

    np.save(f"../Data/PROCESSED_NCH/{FILE_BASE_NAME}/{file_counter:04d}.npy", spectrogram)
    with open(f"../Data/PROCESSED_NCH/{FILE_BASE_NAME}/{file_counter:04d}-stage.txt", "w") as f:
        f.write(description)
    
    file_counter += 1


