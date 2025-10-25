import pandas as pd
import mne
from datetime import datetime, timedelta
import io
import os
import scipy.signal
import numpy as np
import sys

TARGET_SFREQ = 128

PAIRS = [
    ('F4-Ref', 'C4-Ref'),
    ('C4-Ref', 'P4-Ref'),
    ('P4-Ref', 'O2-Ref'),
    ('F3-Ref', 'C3-Ref'),
    ('C3-Ref', 'P3-Ref'),
    ('P3-Ref', 'O1-Ref'),
]

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

FILE_BASE_NAME = sys.argv[1]
directory_folder = f"..\Data\Adult High-Density & Wearable EEG\{FILE_BASE_NAME}"

edf_file = None
txt_file = None

for fname in os.listdir(directory_folder):
    if fname.lower().endswith(".edf"):
        edf_file = os.path.join(directory_folder, fname)
    elif fname.lower().endswith(".txt"):
        txt_file = os.path.join(directory_folder, fname)



raw = mne.io.read_raw_edf(edf_file, preload=False)
print(raw.ch_names)


raw = mne.io.read_raw_edf(edf_file, include=list(set([ch for tup in PAIRS for ch in tup])), preload=True)
for ch_a, ch_b in PAIRS:
    raw = mne.set_bipolar_reference(
        raw,
        anode=ch_a,
        cathode=ch_b,
        ch_name=f"{ch_a}-{ch_b}",
        drop_refs=False
    )

if raw.info['sfreq'] < TARGET_SFREQ:
    raise ValueError(f"Sampling frequency is lower than target frequency ({TARGET_SFREQ} Hz)")

raw = raw.pick([f"{a}-{b}" for a, b in PAIRS])
raw.filter(l_freq=0.3, h_freq=30)
raw.resample(sfreq=TARGET_SFREQ)


df = pd.read_csv(txt_file, sep="\t", header=None)
df.columns = ['SleepStage', 'Onset', 'Duration']
print(df['SleepStage'].unique())

file_counter = 0
for idx, row in df.iterrows():
    if row['SleepStage'] not in ['W', 'N1', 'N2', 'N3', 'R']:
        print(f"Skipping invalid sleep stage: {row['SleepStage']}")
        continue

    start = row['Onset']
    duration = row['Duration']
    description = row['SleepStage']

    if duration != 30:
        print("BE CAREFUL SOLVE THIS", duration)
        continue

    segment = raw.copy().crop(tmin=start, tmax=start+duration)
    spectrogram = get_spectrogram(segment)

    out_dir = f"../Data/PROCESSED_AHD_2/{FILE_BASE_NAME}"
    os.makedirs(out_dir, exist_ok=True)

    np.save(f"../Data/PROCESSED_AHD_2/{FILE_BASE_NAME}/{file_counter:04d}.npy", spectrogram)
    with open(f"../Data/PROCESSED_AHD_2/{FILE_BASE_NAME}/{file_counter:04d}-stage.txt", "w") as f:
        f.write(description)
    
    file_counter += 1



