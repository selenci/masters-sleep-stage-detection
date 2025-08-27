import mne
import pandas as pd
import os
import sys

FILE_BASE_NAME = sys.argv[1]
edf_file = f"../Data/Dataset of EEG recordings containing HFO markings for 30 pediatric patients with epilepsy/sub-{FILE_BASE_NAME}/ses-01/eeg/sub-{FILE_BASE_NAME}_ses-01_task-hfo_eeg.edf"
raw = mne.io.read_raw_edf(edf_file, preload=True)
raw._data /= 1e6

sfreq = raw.info['sfreq']
print(f"Sampling frequency: {sfreq} Hz")
print(f"Available channels: {raw.ch_names}")
pairs = [
    ('F4', 'A1'),
    ('C4', 'A1'),
    ('C3', 'A2'),
    ('O1', 'A2'),
    ('O2', 'A1'),
    ('F3', 'A2'),
]

for ch_a, ch_b in pairs:
    raw = mne.set_bipolar_reference(
        raw,
        anode=ch_a,
        cathode=ch_b,
        ch_name=f"{ch_a}-{ch_b}",
        drop_refs=False
    )

raw = raw.pick([f"{a}-{b}" for a, b in pairs])
raw.filter(l_freq=0.3, h_freq=30)

st_file = f"../Data/Dataset of EEG recordings containing HFO markings for 30 pediatric patients with epilepsy/derivatives/sub-{FILE_BASE_NAME}/ses-01/eeg/DataIntervals.tsv"
df = pd.read_csv(st_file, sep="\t")
print(df.head())

df["onsets"] = df["StartInd"] / sfreq
df["durations"] = (df["EndInd"] - df["StartInd"]) / sfreq
df["descriptions"] = df["SleepStage"].tolist()


file_counter = 0
for _, r in df.iterrows():
    start_time = r['onsets']
    end_time = r['onsets'] + 30  # 30 seconds duration

    while end_time <= r['onsets'] + r['durations']:
        segment = raw.copy().crop(tmin=start_time, tmax=end_time)

        out_dir = f"../Data/PROCESSED_HFO/{FILE_BASE_NAME}"
        os.makedirs(out_dir, exist_ok=True)

        segment.save(f"../Data/PROCESSED_HFO/{FILE_BASE_NAME}/{file_counter:02d}.fif", overwrite=True)

        # Save the corresponding label
        with open(f"../Data/PROCESSED_HFO/{FILE_BASE_NAME}/{file_counter:02d}-stage.txt", "w") as f:
            f.write(r['descriptions'])


        file_counter += 1

        start_time = end_time
        end_time = start_time + 30