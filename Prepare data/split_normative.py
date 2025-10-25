import pandas as pd
import mne
from datetime import datetime, timedelta
import io
import os
import scipy.signal
import numpy as np
import sys

FILE_BASE_NAME = sys.argv[1]
PAIRS = [
    ('F4', 'C4'),
    ('C4', 'P4'),
    ('P4', 'O2'),
    ('F3', 'C3'),
    ('C3', 'P3'),
    ('P3', 'O1'),
]
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

edf_file = f"../Data/Normative longitudinal EEG recordings during sleep stage II in the first year of age/{FILE_BASE_NAME}/ses-1/eeg/{FILE_BASE_NAME}_ses-1_task-EEG_eeg.edf"
raw = mne.io.read_raw_edf(edf_file, preload=True)

sfreq = raw.info['sfreq']
print(f"Sampling frequency: {sfreq} Hz")
print(raw.ch_names)



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


start = 0
end_time = start + 30
length = raw.n_times / raw.info['sfreq']
file_counter = 0
while end_time < length:
    segment = raw.copy().crop(tmin=start, tmax=end_time, include_tmax=False)
    spectrogram = get_spectrogram(segment)
    out_dir = f"../Data/PROCESSED_NORMATIVE/{FILE_BASE_NAME}"
    os.makedirs(out_dir, exist_ok=True)


    np.save(f"../Data/PROCESSED_NORMATIVE/{FILE_BASE_NAME}/{file_counter:04d}.npy", spectrogram)
    with open(f"../Data/PROCESSED_NORMATIVE/{FILE_BASE_NAME}/{file_counter:04d}-stage.txt", "w") as f:
        f.write('II')

    start += 30
    end_time = start + 30
    file_counter += 1