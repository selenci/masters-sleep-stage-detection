import pandas as pd
import mne
from datetime import datetime, timedelta
import io

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


FILE_BASE_NAME = 'n5'
# --- Load EDF ---
edf_file = f"../Data/cap-sleep-database-1.0.0/{FILE_BASE_NAME}.edf"
raw = mne.io.read_raw_edf(edf_file, preload=True)
print(f"Sampling frequency: {raw.info['sfreq']} Hz")

channels_of_interest = ['Fp2-F4', 'F3-C3', 'C4-A1', 'ROC-LOC']
raw.pick(channels_of_interest)


raw.filter(l_freq=0.3, h_freq=30)

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

annot = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
raw.set_annotations(annot)

raw.plot(duration=120, block=True, scalings='auto')



from draw2stages import draw_specs
draw_specs(raw, 'R', 'S4', ch_idx=1)