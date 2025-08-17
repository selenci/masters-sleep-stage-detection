import mne
import pandas as pd

SUBFOLDER = 'sleep-telemetry'
FILE_BASE_NAME = 'ST7011J'

edf_file = f"../Data/Sleep-EDF Database Expanded/{SUBFOLDER}/{FILE_BASE_NAME}0-PSG.edf"
raw = mne.io.read_raw_edf(edf_file, preload=True)

sfreq = raw.info['sfreq']
print(f"Sampling frequency: {sfreq} Hz")
print(raw.ch_names)

raw.filter(l_freq=0.3, h_freq=30)

st_file = f"../Data/Sleep-EDF Database Expanded/{SUBFOLDER}/{FILE_BASE_NAME}P-Hypnogram.edf"
annot = mne.read_annotations(st_file)
print(annot)

raw.set_annotations(annot)
print(raw.get_data().max(), raw.get_data().min())
raw.plot(block=True, scalings="auto")


from draw2stages import draw_specs

draw_specs(raw, 'Sleep stage 1', 'Sleep stage 4', ch_idx=0)