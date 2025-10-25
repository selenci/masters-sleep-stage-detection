import pandas as pd
import mne
from datetime import datetime, timedelta
import io
import os
import scipy.signal
import numpy as np
import sys

FILE_BASE_NAME = sys.argv[1]
directory_folder = f"..\Data\Adult High-Density & Wearable EEG\{FILE_BASE_NAME}"

for fname in os.listdir(directory_folder):
    if fname.lower().endswith(".edf"):
        edf_file = os.path.join(directory_folder, fname)
    elif fname.lower().endswith(".txt"):
        txt_file = os.path.join(directory_folder, fname)



raw = mne.io.read_raw_edf(edf_file, preload=False)
print(raw.ch_names)