import mne
import pandas as pd

FILE_BASE_NAME = 'SN001'

edf_file = f"../Data/Haaglanden Medisch Centrum sleep staging database/recordings/{FILE_BASE_NAME}.edf"
raw = mne.io.read_raw_edf(edf_file, preload=True)

sfreq = raw.info['sfreq']
print(f"Sampling frequency: {sfreq} Hz")
print(raw.ch_names)

channels_of_interest = ['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2']
raw.pick(channels_of_interest)
raw.filter(l_freq=0.3, h_freq=35)

st_file = f"../Data/Haaglanden Medisch Centrum sleep staging database/recordings/{FILE_BASE_NAME}_sleepscoring.txt"
df = pd.read_csv(st_file, skipinitialspace=True)

print(df)

onsets = df["Recording onset"]
durations = df["Duration"]
descriptions = df["Annotation"].tolist()

annot = mne.Annotations(
    onset=onsets,
    duration=durations,
    description=descriptions
)
raw.set_annotations(annot)
print(raw.get_data().max(), raw.get_data().min())
raw.plot(block=True, scalings='auto')

from draw2stages import draw_specs
draw_specs(raw, 'Sleep stage N1', 'Sleep stage N2', ch_idx=1)