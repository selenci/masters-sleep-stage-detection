import mne
import pandas as pd

FILE_BASE_NAME = 'sub-NORB00001'
pairs = [
    ('F4', 'Pg1'),  # was F4–A1
    ('C4', 'Pg1'),  # was C4–A1
    ('O2', 'Pg1'),  # was O2–A1
    ('F3', 'Pg2'),  # was F3–A2
]
edf_file = f"../Data/Normative longitudinal EEG recordings during sleep stage II in the first year of age/{FILE_BASE_NAME}/ses-1/eeg/{FILE_BASE_NAME}_ses-1_task-EEG_eeg.edf"
raw = mne.io.read_raw_edf(edf_file, preload=True)

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

raw.plot(block=True, scalings='auto')
