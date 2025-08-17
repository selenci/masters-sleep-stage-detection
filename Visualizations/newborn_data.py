from scipy.io import loadmat
import numpy as np
import mne
FILE_BASE_NAME = 'eeg_36'

data = loadmat('../Data/Newborn sleep EEG data/eeg_36.mat')

pacient = 12

signal_ch1 = np.asarray(data['D2'][pacient, 0]).ravel().astype(float)  # C3-T3
signal_ch2 = np.asarray(data['D2'][pacient, 1]).ravel().astype(float)  # C4-T4

max_len = max(len(signal_ch1), len(signal_ch2))
print((0, max_len - len(signal_ch2)))
sig1_pad = np.pad(signal_ch1, (0, max_len - len(signal_ch1)), constant_values=np.nan)
sig2_pad = np.pad(signal_ch2, (0, max_len - len(signal_ch2)), constant_values=np.nan)
signals = np.vstack([sig1_pad, sig2_pad])
signals /= 1e6

info = mne.create_info(
    ch_names=['C3-T3', 'C4-T4'],       # adjust as needed
    sfreq=100.0,              # <-- sampling rate from publication
    ch_types=['eeg', 'eeg']
)
raw = mne.io.RawArray(signals, info)
raw.plot(duration=120, block=True, scalings='auto')
