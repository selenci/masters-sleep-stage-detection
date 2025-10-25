import mne
import pandas as pd
import numpy as np
import os
import sys
import scipy.signal

TARGET_SFREQ = 128

import numpy as np
import mne

def print_stats(raw, prefix=""):
    # compute on all EEG channels (skip non-EEG if present)
    picks = mne.pick_types(raw.info, eeg=True, meg=False)
    data, times = raw[picks, :]
    std_uv = data.std(axis=1) * 1e6
    maxabs_uv = np.abs(data).max(axis=1) * 1e6
    print(f"{prefix}Std per channel (µV):", np.round(std_uv, 2))
    print(f"{prefix}Max abs per channel (µV):", np.round(maxabs_uv, 2))

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
edf_file = f"../Data/Dataset of EEG recordings containing HFO markings for 30 pediatric patients with epilepsy/sub-{FILE_BASE_NAME}/ses-01/eeg/sub-{FILE_BASE_NAME}_ses-01_task-hfo_eeg.edf"
raw = mne.io.read_raw_edf(edf_file, preload=True)
raw._data /= 1e6
print_stats(raw, prefix="Before: ")

sfreq = raw.info['sfreq']
print(f"Sampling frequency: {sfreq} Hz")
print(f"Available channels: {raw.ch_names}")
PAIRS = [
    ('F4', 'C4'),
    ('C4', 'P4'),
    ('P4', 'O2'),
    ('F3', 'C3'),
    ('C3', 'P3'),
    ('P3', 'O1'),
]

for ch_a, ch_b in PAIRS:
    raw = mne.set_bipolar_reference(
        raw,
        anode=ch_a,
        cathode=ch_b,
        ch_name=f"{ch_a}-{ch_b}",
        drop_refs=False
    )

raw = raw.pick([f"{a}-{b}" for a, b in PAIRS])
raw.filter(l_freq=0.3, h_freq=30)
raw.resample(sfreq=TARGET_SFREQ)

print_stats(raw, prefix="After: ")

st_file = f"../Data/Dataset of EEG recordings containing HFO markings for 30 pediatric patients with epilepsy/derivatives/sub-{FILE_BASE_NAME}/ses-01/eeg/DataIntervals.tsv"
df = pd.read_csv(st_file, sep="\t")

df["onsets"] = df["StartInd"] / sfreq
df["durations"] = (df["EndInd"] - df["StartInd"]) / sfreq
df["descriptions"] = df["SleepStage"].tolist()

stage_map = {
    "Awake": "W",
    "REM": "R"
}
df["descriptions"] = df["SleepStage"].map(stage_map).fillna(df["SleepStage"])
print(df['descriptions'].unique())

file_counter = 0
for _, r in df.iterrows():
    start_time = r['onsets']
    end_time = r['onsets'] + 30  # 30 seconds duration

    while end_time <= r['onsets'] + r['durations']:
        segment = raw.copy().crop(tmin=start_time, tmax=end_time)

        if segment.times.size / TARGET_SFREQ < 29.9:
            print(f"Skipping segment {file_counter:04d} due to short duration.")
            start_time = end_time
            end_time = start_time + 30
            continue

        spectrogram = get_spectrogram(segment)

        out_dir = f"../Data/PROCESSED_HFO_2/{FILE_BASE_NAME}"
        os.makedirs(out_dir, exist_ok=True)

        np.save(f"../Data/PROCESSED_HFO_2/{FILE_BASE_NAME}/{file_counter:04d}.npy", spectrogram)

        # Save the corresponding label
        with open(f"../Data/PROCESSED_HFO_2/{FILE_BASE_NAME}/{file_counter:04d}-stage.txt", "w") as f:
            f.write(r['descriptions'])


        file_counter += 1

        start_time = end_time
        end_time = start_time + 30