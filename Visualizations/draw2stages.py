import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
from scipy.signal import spectrogram

def get_random_stage_clip(raw, stage_label, clip_len=30):
    stage_annot = [ann for ann in raw.annotations if ann['description'] == stage_label]
    if not stage_annot:
        raise ValueError(f"No annotations for stage {stage_label}")
    
    ann = random.choice(stage_annot)
    start = ann['onset']
    end = start + ann['duration'] - clip_len
    if end <= start:  # if duration < clip_len, just take from onset
        start_time = start
    else:
        start_time = random.uniform(start, end)
    stop_time = start_time + clip_len
    return raw.copy().crop(tmin=start_time, tmax=stop_time)

def plot_spectrogram(clip, title, ch_idx=0, ax=None, log_scale=False):
    sfreq = clip.info['sfreq']
    data = clip.get_data(picks=ch_idx)[0] * 1e6
    
    f, t, Sxx = spectrogram(data, fs=sfreq, nperseg=int(sfreq*2), noverlap=int(sfreq*1))
    
    if ax is None:
        ax = plt.gca()
        
    im = ax.pcolormesh(t, f, 10*np.log10(Sxx), shading="auto")
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim(0, 30)

    if log_scale:
        ax.set_yscale('log')
        from matplotlib.ticker import ScalarFormatter
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_yticks([1, 5, 10, 20, 30])
        ax.set_ylim(0.5, 30) # Adjust ylim for log scale
    
    return im

def draw_specs(raw, t1, t2, ch_idx=0):
    clip_N1 = get_random_stage_clip(raw, t1, clip_len=30)
    clip_N3 = get_random_stage_clip(raw, t2, clip_len=30)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot both spectrograms
    im1 = plot_spectrogram(clip_N1, f'{t1} Stage - Spectrogram', ch_idx, ax1, True)
    im2 = plot_spectrogram(clip_N3, f'{t2} Stage - Spectrogram', ch_idx, ax2, True)
    
    vmin = min(im1.get_clim()[0], im2.get_clim()[0])
    vmax = max(im1.get_clim()[1], im2.get_clim()[1])
    
    # Apply same scale to both
    im1.set_clim(vmin, vmax)
    im2.set_clim(vmin, vmax)
    
    # Add shared colorbar
    plt.tight_layout()
    cbar = fig.colorbar(im1, ax=[ax1, ax2], shrink=0.8, aspect=30)
    cbar.set_label('Power/Frequency (dB/Hz)')
    
    plt.show()