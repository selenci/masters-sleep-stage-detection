import mne
import pandas as pd

FILE_BASE_NAME = '01'
pairs = [
    ('F4', 'TP9'),
    ('C4', 'TP9'),
    ('O2', 'TP9'),
    ('F3', 'TP10'),
]
edf_file = f"../Data/Adult High-Density & Wearable EEG/EPCTL{FILE_BASE_NAME}/EPCTL{FILE_BASE_NAME} - fixed.edf"
raw = mne.io.read_raw_edf(edf_file, include=list(set([ch for tup in pairs for ch in tup])), preload=True)

sfreq = raw.info['sfreq']
print(f"Sampling frequency: {sfreq} Hz")
print(raw.ch_names)


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

st_file = f"../Data/Adult High-Density & Wearable EEG/EPCTL{FILE_BASE_NAME}/test1.txt"
df = pd.read_csv(st_file, sep="\t", header=None)
df.columns = ['SleepStage', 'Onset', 'Duration']

print(df.head())

onsets = df["Onset"]
durations = df["Duration"]
descriptions = df["SleepStage"].tolist()

annot = mne.Annotations(
    onset=onsets,
    duration=durations,
    description=descriptions
)
raw.set_annotations(annot)
print(raw.get_data().max(), raw.get_data().min())
raw.plot(block=True, scalings="auto")

from draw2stages import draw_specs

draw_specs(raw, 'N1', 'N3', ch_idx=1)
