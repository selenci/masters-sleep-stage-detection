import mne
import pandas as pd

FILE_BASE_NAME = '04'
edf_file = f"../Data/Dataset of EEG recordings containing HFO markings for 30 pediatric patients with epilepsy/sub-{FILE_BASE_NAME}/ses-01/eeg/sub-{FILE_BASE_NAME}_ses-01_task-hfo_eeg.edf"
raw = mne.io.read_raw_edf(edf_file, preload=True)
raw._data /= 1e6

sfreq = raw.info['sfreq']
print(f"Sampling frequency: {sfreq} Hz")

pairs = [
    ('F4', 'A1'),
    ('C4', 'A1'),
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

onsets = df["StartInd"] / sfreq
durations = (df["EndInd"] - df["StartInd"]) / sfreq
descriptions = df["SleepStage"].tolist()

annot = mne.Annotations(
    onset=onsets,
    duration=durations,
    description=descriptions
)
raw.set_annotations(annot)
print(raw.get_data().max(), raw.get_data().min())
raw.plot(block=True, scalings="auto")
