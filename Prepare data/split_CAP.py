import pandas as pd
import mne
from datetime import datetime, timedelta
import io
import os
import scipy.signal
import numpy as np
import sys
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
CHANNEL_SUBSTITUTIONS = {
    # --- RIGHT HEMISPHERE (AASM) ---
    "F4-A1": [
        # Direct synonyms and naming variations
        "F4-M1", "F4-A2", "F4-M2",  # other mastoid refs
        "F4 A1", "F4 M1", "F4 A2", "F4 M2",
        "F4A1", "F4M1", "F4A2", "F4M2",
        "EEG F4-A1", "EEG F4-M1", "EEG F4-A2", "EEG F4-M2",
        # Broader frontal-right substitutions
        "Fp2-F4", "F4-C4", "F4-T4", "Fz-F4",
        # Cross-hemisphere backups
        "F3-A2", "F3-M2", "F3-C3",
    ],

    "C4-A1": [
        "C4-M1", "C4-A2", "C4-M2",
        "C4 A1", "C4 M1", "C4 A2", "C4 M2",
        "C4A1", "C4M1", "C4A2", "C4M2",
        "EEG C4-A1", "EEG C4-M1", "EEG C4-A2", "EEG C4-M2",
        "F4-C4", "C4-P4", "Cz-C4",
        "C3-A2", "C3-M2", "C3-P3",
    ],

    "O2-A1": [
        "O2-M1", "O2-A2", "O2-M2",
        "O2 A1", "O2 M1", "O2 A2", "O2 M2",
        "O2A1", "O2M1", "O2A2", "O2M2",
        "EEG O2-A1", "EEG O2-M1", "EEG O2-A2", "EEG O2-M2",
        # Nearby or similar occipital derivations
        "P4-O2", "Pz-O2", "T6-O2",
        # Cross-hemisphere fallbacks
        "O1-A2", "O1-M2", "O1-P3",
    ],

    # --- LEFT HEMISPHERE (AASM) ---
    "F3-A2": [
        "F3-M2", "F3-A1", "F3-M1",
        "F3 A2", "F3 M2", "F3 A1", "F3 M1",
        "F3A2", "F3M2", "F3A1", "F3M1",
        "EEG F3-A2", "EEG F3-M2", "EEG F3-A1", "EEG F3-M1",
        "Fp1-F3", "F3-C3", "F3-T3", "Fz-F3",
        "F4-A1", "F4-M1", "F4-C4",
    ],

    "C3-A2": [
        "C3-M2", "C3-A1", "C3-M1",
        "C3 A2", "C3 M2", "C3 A1", "C3 M1",
        "C3A2", "C3M2", "C3A1", "C3M1",
        "EEG C3-A2", "EEG C3-M2", "EEG C3-A1", "EEG C3-M1",
        "F3-C3", "C3-P3", "Cz-C3",
        "C4-A1", "C4-M1", "C4-P4",
    ],

    "O1-A2": [
        "O1-M2", "O1-A1", "O1-M1",
        "O1 A2", "O1 M2", "O1 A1", "O1 M1",
        "O1A2", "O1M2", "O1A1", "O1M1",
        "EEG O1-A2", "EEG O1-M2", "EEG O1-A1", "EEG O1-M1",
        "P3-O1", "Pz-O1", "T5-O1",
        "O2-A1", "O2-M1", "O2-P4",
    ],
}


TARGET_CHANNELS = list(CHANNEL_SUBSTITUTIONS.keys())

def get_available_channels_raw(raw, target_channels, substitutions):
    """
    Picks, creates, or substitutes target EEG channels to ensure consistency.

    This function follows a hierarchy:
    1. Tries to find the exact target channel (e.g., 'F4-C4').
    2. If not found, it tries to create it by subtracting its constituent monopolar 
       channels (e.g., finds 'F4' and 'C4' and computes 'F4' - 'C4').
    3. If monopolar channels aren't available, it searches the substitution dictionary.
    4. It returns a new MNE Raw object with consistently named channels or None if no
       channels could be processed.
    """
    final_channels_data = []
    final_ch_names = []
    # Use a lower-case version for case-insensitive matching
    available_ch_lower = [ch.lower() for ch in raw.ch_names]
    used_chs = set()

    for target_ch in target_channels:
        found = False

        # 1. Try to find the target channel directly (case-insensitive)
        if target_ch.lower() in available_ch_lower:
            used_chs.add(target_ch.lower())
            idx = available_ch_lower.index(target_ch.lower())
            original_ch_name = raw.ch_names[idx]
            channel_data, _ = raw[original_ch_name]
            final_channels_data.append(channel_data[0])
            final_ch_names.append(target_ch)  # Use standard name
            continue

        # 2. If not found, try to create it from monopolar channels
        try:
            anode, cathode = target_ch.split('-')
            if anode.lower() in available_ch_lower and cathode.lower() in available_ch_lower:
                used_chs.add(anode.lower() + "-" + cathode.lower())
                anode_idx = available_ch_lower.index(anode.lower())
                cathode_idx = available_ch_lower.index(cathode.lower())
                anode_name = raw.ch_names[anode_idx]
                cathode_name = raw.ch_names[cathode_idx]

                print(f"→ Creating '{target_ch}' from '{anode_name}' - '{cathode_name}'")
                anode_data, _ = raw[anode_name]
                cathode_data, _ = raw[cathode_name]
                bipolar_data = anode_data[0] - cathode_data[0]
                
                final_channels_data.append(bipolar_data)
                final_ch_names.append(target_ch)
                continue
        except ValueError:
            # Cannot split channel name, so it's not a bipolar channel we can create
            pass

        # 3. If it cannot be created, look for substitutes
        for sub in substitutions.get(target_ch, []):
            if sub.lower() in available_ch_lower:
                idx = available_ch_lower.index(sub.lower())
                sub_name = raw.ch_names[idx]

                # Avoid reusing channels already assigned to other targets
                if sub.lower() in used_chs:
                    continue

                used_chs.add(sub.lower())

                print(f"→ Substituting with {sub_name}")
                channel_data, _ = raw[sub_name]
                final_channels_data.append(channel_data[0])
                final_ch_names.append(target_ch) # Use standard name for consistency
                found = True
                break
        
        if not found:
             print(f"→ FAILED: No substitute found for {target_ch}. Available channels: {raw.ch_names}")

    # If no channels were found at all, we cannot proceed
    if not final_ch_names:
        print(f"Warning: No target EEG channels could be found or created for {FILE_BASE_NAME}.")
        return None

    # Create a new Raw object with the collected data and standardized names
    info = mne.create_info(ch_names=final_ch_names, sfreq=raw.info['sfreq'], ch_types='eeg')
    info.set_meas_date(raw.info['meas_date'])

    new_raw = mne.io.RawArray(np.array(final_channels_data), info)

    return new_raw

raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
raw = get_available_channels_raw(raw, TARGET_CHANNELS, CHANNEL_SUBSTITUTIONS)

if raw is None:
    print(f"Skipping {FILE_BASE_NAME} due to lack of channels.")
    sys.exit(0)

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

        out_dir = f"../Data/AASM/PROCESSED_CAP/{FILE_BASE_NAME}"
        os.makedirs(out_dir, exist_ok=True)

        np.save(f"../Data/AASM/PROCESSED_CAP/{FILE_BASE_NAME}/{file_counter:04d}.npy", spectrogram)

        # Save the corresponding label
        with open(f"../Data/AASM/PROCESSED_CAP/{FILE_BASE_NAME}/{file_counter:04d}-stage.txt", "w") as f:
            f.write(descriptions[i])


        file_counter += 1

        start_time = end_time
        end_time = start_time + 30
