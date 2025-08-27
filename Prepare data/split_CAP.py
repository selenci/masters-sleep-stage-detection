import pandas as pd
import mne
from datetime import datetime, timedelta
import io
import os
import scipy.signal
import numpy as np
import sys

TARGET_SFREQ = 256

def get_spectrogram(raw):
        segment_data = raw.get_data()

        spectrograms = []
        for channel_data in segment_data:
            f, t, Zxx = scipy.signal.stft(
                channel_data,
                fs=TARGET_SFREQ,
                nperseg=256,
                noverlap=192
            )
            
            f_mask = (f < 50)
            Zxx = Zxx[f_mask, :]

            spectrogram = np.log1p(np.abs(Zxx))
            spectrograms.append(spectrogram)
        return np.stack(spectrograms, axis=0).astype(np.float32)

def cut_beginning_rows(file_name):
    output = ""
    add = False
    with open(file_name) as f:
        for line in f:
            if line.strip().startswith("Sleep Stage"):
                add = True
            if add:
                output += line
    return io.StringIO(output)


FILE_BASE_NAME = sys.argv[1]
# --- Load EDF ---
edf_file = f"../Data/cap-sleep-database-1.0.0/{FILE_BASE_NAME}.edf"
eeg_channels = ['F4-C4', 'C4-P4', 'P4-O2', 'F3-C3', 'C3-P3', 'P3-O1']
raw = mne.io.read_raw_edf(edf_file, preload=True, include=eeg_channels)
print(f"Sampling frequency: {raw.info['sfreq']} Hz")

if raw.info['sfreq'] < TARGET_SFREQ:
    raise ValueError(f"Sampling frequency is lower than target frequency ({TARGET_SFREQ} Hz)")

raw.filter(l_freq=0.3, h_freq=30)
raw.resample(sfreq=TARGET_SFREQ)

start_time = raw.info['meas_date'].replace(tzinfo=None)
end_time = start_time + timedelta(seconds=raw.times[-1])
print(f"Data start time: {start_time}")
print(f"Data end time:   {end_time}")

st_file = f"../Data/cap-sleep-database-1.0.0/{FILE_BASE_NAME}.txt"
df = pd.read_csv(cut_beginning_rows(st_file), sep="\t")
onsets = []
durations = []
descriptions = []

for _, row in df.iterrows():
    event_clock = datetime.strptime(row["Time [hh:mm:ss]"], "%H:%M:%S").time()
    event_datetime = datetime.combine(start_time.date(), event_clock)
    
    # If clock time < start clock time → it's past midnight → add a day
    if event_clock < start_time.time():
        event_datetime += timedelta(days=1)
    
    onset_sec = (event_datetime - start_time).total_seconds()
    
    onsets.append(onset_sec)
    durations.append(row["Duration[s]"])
    descriptions.append(row["Sleep Stage"])

file_counter = 0
for i in range(len(onsets)):
    start_time = onsets[i]
    end_time = start_time + 30  # 30 seconds duration

    while end_time <= onsets[i] + durations[i]:
        segment = raw.copy().crop(tmin=start_time, tmax=end_time)

        if segment.times.size / TARGET_SFREQ < 29.9:
            print(f"Skipping segment {file_counter:04d} due to short duration.")
            start_time = end_time
            end_time = start_time + 30
            continue

        spectrogram = get_spectrogram(segment)

        out_dir = f"../Data/PROCESSED_CAP/{FILE_BASE_NAME}"
        os.makedirs(out_dir, exist_ok=True)

        np.save(f"../Data/PROCESSED_CAP/{FILE_BASE_NAME}/{file_counter:04d}.npy", spectrogram)

        # Save the corresponding label
        with open(f"../Data/PROCESSED_CAP/{FILE_BASE_NAME}/{file_counter:04d}-stage.txt", "w") as f:
            f.write(descriptions[i])


        file_counter += 1

        start_time = end_time
        end_time = start_time + 30
