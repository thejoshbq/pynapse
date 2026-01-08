#!/usr/bin/env python3
"""
Pipeline Comparison Script
Traces and compares intermediate processing steps between the new and legacy pipelines
"""

import os
import numpy as np
import scipy.io as sio

# Test on a single sample to compare processing
test_day = "0 EarlyAcq"
test_animal = "IG-19"
test_fov = "FOV2"

basedir = "data"
indir = os.path.join(basedir, test_day, test_animal, test_fov)

print("="*80)
print(f"Comparing pipelines for: {test_day}/{test_animal}/{test_fov}")
print("="*80)

# Get files
tempfiles = os.listdir(indir)
npyfiles = [f for f in tempfiles if f.endswith('.npy') and 'extractedsignals_raw' in f]
matfiles = [f for f in tempfiles if f.endswith('.mat') and 'popevents' not in f and 'alignedevents' not in f]

if len(npyfiles) > 1:
    npyfile = [f for f in npyfiles if 'part2' not in f and 'part3' not in f and 'part4' not in f][0]
    matfile = [f for f in matfiles if 'part2' not in f and 'part3' not in f and 'part4' not in f][0]
else:
    npyfile = npyfiles[0]
    matfile = matfiles[0]

print(f"\nFiles:")
print(f"  Signal: {npyfile}")
print(f"  Events: {matfile}")

# Load raw data
signals_raw = np.squeeze(np.load(os.path.join(indir, npyfile)))
behaviordata = sio.loadmat(os.path.join(indir, matfile))
eventlog = np.squeeze(behaviordata['eventlog'])

print(f"\nRaw Data:")
print(f"  Signals shape: {signals_raw.shape}")
print(f"  Event log shape: {eventlog.shape}")
print(f"  Num neurons: {signals_raw.shape[0]}")
print(f"  Num raw frames: {signals_raw.shape[1]}")

# =============================================================================
# LEGACY PIPELINE PROCESSING
# =============================================================================
print("\n" + "="*80)
print("LEGACY PIPELINE PROCESSING")
print("="*80)

# Configuration
frameaveraging = 4
framerate = 30
timebetweenframes = 33.333333
averagedframerate = framerate / frameaveraging
separation_requirement = 1000
mineventstoanalyze = 3

# Extract events
activelever = eventlog[eventlog[:, 0] == 22, 1]
activelevertimeout = eventlog[eventlog[:, 0] == 222, 1]

print(f"\nEvent extraction:")
print(f"  Active lever presses: {len(activelever)}")
print(f"  Active lever timeouts: {len(activelevertimeout)}")

# Overlap deletion
temp = np.sort(np.hstack((activelever, activelevertimeout)))
print(f"  Combined events (before overlap deletion): {len(temp)}")
temp = np.delete(temp, np.argwhere(np.ediff1d(temp) < separation_requirement) + 1)
activeleverall_legacy = temp
print(f"  Combined events (after overlap deletion, 1000ms): {len(activeleverall_legacy)}")

# Frame timestamp correction
frame_ts_raw = eventlog[eventlog[:, 0] == 9, 1]
print(f"\nFrame timestamp processing:")
print(f"  Raw frame timestamps: {len(frame_ts_raw)}")

def fix_any_dropped_frames_legacy(frame_timestamps, timebetweenframes=33.333333):
    first_frame = np.array([0])
    last_frame = np.array([int(np.max(frame_timestamps) + (500 * timebetweenframes))])
    frame_index_temp = np.concatenate((first_frame, frame_timestamps, last_frame))
    frames_missed = []
    
    for i in range(len(frame_index_temp) - 1):
        numframes_missed = int(np.round((frame_index_temp[i+1] - frame_index_temp[i]) / timebetweenframes) - 1)
        if numframes_missed > 0:
            for j in range(numframes_missed):
                frame_missed = np.array([frame_index_temp[i] + (int(timebetweenframes * (j + 1)))])
                frames_missed = np.concatenate((frames_missed, frame_missed))
    
    corrected_frame_index = np.array(sorted(np.concatenate((frame_index_temp, frames_missed))))
    return corrected_frame_index

frame_timestamps_legacy = fix_any_dropped_frames_legacy(frame_ts_raw, timebetweenframes)
print(f"  After dropped frame correction: {len(frame_timestamps_legacy)}")

frame_timestamps_legacy = frame_timestamps_legacy[::frameaveraging]
print(f"  After frame averaging (every {frameaveraging}): {len(frame_timestamps_legacy)}")

# Signal alignment
signals_legacy = signals_raw.copy()
if signals_legacy.shape[1] > frame_timestamps_legacy.shape[0]:
    signals_legacy = signals_legacy[:, :frame_timestamps_legacy.shape[0] - 1]
    print(f"  Signals trimmed to: {signals_legacy.shape}")

# Normalization
signals_legacy_mean_norm = signals_legacy / np.nanmean(signals_legacy, axis=1)[:, None]
print(f"\nNormalization (Legacy):")
print(f"  After mean normalization: shape={signals_legacy_mean_norm.shape}")
print(f"  Sample neuron mean before z-score: {np.nanmean(signals_legacy_mean_norm[0]):.6f}")
print(f"  Sample neuron std before z-score: {np.nanstd(signals_legacy_mean_norm[0]):.6f}")

# Z-score
signals_legacy_zscore = signals_legacy_mean_norm.copy()
for neuron in range(signals_legacy_zscore.shape[0]):
    mean = np.nanmean(signals_legacy_zscore[neuron])
    std = np.nanstd(signals_legacy_zscore[neuron])
    if std > 0:
        signals_legacy_zscore[neuron] = (signals_legacy_zscore[neuron] - mean) / std

print(f"  After z-score: shape={signals_legacy_zscore.shape}")
print(f"  Sample neuron mean after z-score: {np.nanmean(signals_legacy_zscore[0]):.6f}")
print(f"  Sample neuron std after z-score: {np.nanstd(signals_legacy_zscore[0]):.6f}")

# Event window extraction
pre_window_size = int(10 * averagedframerate)  # 75 frames
window_size = int((pre_window_size * 2) + (1.6 * averagedframerate))  # 162 frames
post_window_size = window_size - pre_window_size  # 87 frames

print(f"\nEvent window extraction (Legacy):")
print(f"  Pre-window: {pre_window_size} frames")
print(f"  Post-window: {post_window_size} frames")
print(f"  Total window: {window_size} frames")

def framenumberforevent(event, frame_timestamps):
    framenumber = np.nan * np.zeros(event.shape)
    for ie, e in enumerate(event):
        if np.isnan(e):
            framenumber[ie] = np.nan
        else:
            temp = np.nonzero(frame_timestamps <= e)[0]
            if temp.shape[0] > 0:
                framenumber[ie] = np.nonzero(frame_timestamps <= e)[0][-1]
            else:
                framenumber[ie] = 0
    return framenumber

signalsT_legacy = signals_legacy_zscore.T
framenumberfor_eventofinterest = np.squeeze(framenumberforevent(activeleverall_legacy, frame_timestamps_legacy))
numtrials = framenumberfor_eventofinterest.shape[0]
alignedevents_legacy = np.nan * np.zeros([numtrials, window_size, signals_legacy_zscore.shape[0]])
valid_trials = []

for i in range(numtrials):
    eventindex = framenumberfor_eventofinterest[i]
    if (np.isfinite(eventindex) and 
        eventindex > pre_window_size and 
        eventindex < signalsT_legacy.shape[0] - post_window_size):
        eventindex = int(eventindex)
        alignedevents_legacy[i, :, :] = signalsT_legacy[eventindex - pre_window_size:eventindex + post_window_size, :]
        valid_trials.append(i)

alignedevents_legacy = alignedevents_legacy[valid_trials, :, :]
print(f"  Valid trials: {len(valid_trials)} out of {numtrials}")
print(f"  Aligned events shape: {alignedevents_legacy.shape} (trials x frames x neurons)")

alignedevents_legacy = np.swapaxes(alignedevents_legacy, 0, 2)
print(f"  After swapaxes: {alignedevents_legacy.shape} (neurons x frames x trials)")

popevents_legacy = np.nanmean(alignedevents_legacy, axis=2)
print(f"  After mean over trials: {popevents_legacy.shape} (neurons x frames)")

# Baseline subtraction
baselinefirstframe = 0
baselinelastframe = int(3 * averagedframerate)  # ~22.5 frames
baseline_legacy = np.mean(popevents_legacy[:, baselinefirstframe:baselinelastframe], axis=1)
popevents_legacy_bs = popevents_legacy - baseline_legacy[:, None]
print(f"\nBaseline subtraction (Legacy):")
print(f"  Baseline window: {baselinefirstframe} to {baselinelastframe} frames")
print(f"  Final output shape: {popevents_legacy_bs.shape}")
print(f"  Sample baseline values (first 3 neurons): {baseline_legacy[:3]}")

# =============================================================================
# NEW PIPELINE PROCESSING
# =============================================================================
print("\n" + "="*80)
print("NEW PIPELINE PROCESSING")
print("="*80)

from pynapse.config.events import LEGACY_HER
from pynapse.core.sample import Sample
from pynapse.analysis.preprocessing.continuous.normalization import LegacyNormalize
from pynapse.analysis.peri_event import SampleEventTensor

mat_files = [os.path.join(indir, matfile)]
npy_files = [os.path.join(indir, npyfile)]

sample = Sample(
    event_data=mat_files,
    signal_data=npy_files,
    name=f"{test_day}/{test_animal}/{test_fov}",
    fps=30,
    frame_averaging=4,
    frame_correction=True,
    correction_file="../data/empty.mat",
    event_dict=LEGACY_HER
)

print(f"\nSample created:")
print(f"  Neurons: {sample.num_neurons}")
print(f"  Frames: {sample.num_frames}")
print(f"  Events: {sample.num_events}")

# Get frame timestamps (new pipeline)
frame_ts_new = sample._get_frame_timestamps()
print(f"\nFrame timestamp processing (New):")
print(f"  Final frame timestamps: {len(frame_ts_new)}")

# Get signals before preprocessing
signals_new_raw = sample.get_signals()
print(f"\nSignals (New):")
print(f"  Raw signals shape: {signals_new_raw.shape}")

# Apply preprocessing
preprocessor = LegacyNormalize(z_score=True)
signals_new_preprocessed = preprocessor.apply(signals_new_raw)
print(f"\nNormalization (New):")
print(f"  After LegacyNormalize: shape={signals_new_preprocessed.shape}")
print(f"  Sample neuron mean: {np.nanmean(signals_new_preprocessed[0]):.6f}")
print(f"  Sample neuron std: {np.nanstd(signals_new_preprocessed[0]):.6f}")

# Event extraction
df = sample.get_dataframe()
event_ts_ms = []
for eid in [22, 222]:
    event_ts_ms.extend(df[df["code"] == eid]["t1"].values)
event_ts_ms = np.sort(np.array(event_ts_ms, dtype=np.float64))

print(f"\nEvent extraction (New):")
print(f"  Combined events (before buffer): {len(event_ts_ms)}")

# Buffer deletion
buffer_ms = 1000
diffs_ms = np.diff(event_ts_ms)
close_later_idx = np.where(diffs_ms < buffer_ms)[0] + 1
event_ts_ms_filtered = np.delete(event_ts_ms, close_later_idx)
print(f"  Combined events (after buffer {buffer_ms}ms): {len(event_ts_ms_filtered)}")

# Event window extraction (new)
tensor = SampleEventTensor(
    sample,
    event_id=[22, 222],
    pre_event=10,
    post_event=11.6,
    buffer_ms=1000,
    min_trials=3,
    pre_window_preprocessor=LegacyNormalize(),
    post_window_preprocessor=None,
)

event_windows_new = tensor.get_event_windows()
print(f"\nEvent window extraction (New):")
print(f"  Event windows shape: {event_windows_new.shape} (trials x neurons x frames)")

# Mean over trials
popevents_new = np.nanmean(event_windows_new, axis=0)
print(f"  After mean over trials: {popevents_new.shape} (neurons x frames)")

# Baseline subtraction
frame_duration_ms = 1000 / sample.effective_fps
baseline_frames_new = int(3000 / frame_duration_ms)
baseline_new = np.mean(popevents_new[:, 0:baseline_frames_new], axis=1)
popevents_new_bs = popevents_new - baseline_new[:, None]
print(f"\nBaseline subtraction (New):")
print(f"  Frame duration: {frame_duration_ms} ms")
print(f"  Baseline frames: {baseline_frames_new}")
print(f"  Final output shape: {popevents_new_bs.shape}")
print(f"  Sample baseline values (first 3 neurons): {baseline_new[:3]}")

# =============================================================================
# COMPARISON
# =============================================================================
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\nShape comparison:")
print(f"  Legacy: {popevents_legacy_bs.shape}")
print(f"  New:    {popevents_new_bs.shape}")

print(f"\nNumber of events after filtering:")
print(f"  Legacy: {len(activeleverall_legacy)}")
print(f"  New:    {len(event_ts_ms_filtered)}")

print(f"\nNumber of valid trials:")
print(f"  Legacy: {len(valid_trials)}")
print(f"  New:    {event_windows_new.shape[0]}")

print(f"\nFrame timestamps comparison:")
print(f"  Legacy timestamps: {len(frame_timestamps_legacy)}")
print(f"  New timestamps:    {len(frame_ts_new)}")

if len(frame_timestamps_legacy) == len(frame_ts_new):
    print(f"  Timestamps match in length")
    max_diff = np.max(np.abs(frame_timestamps_legacy - frame_ts_new))
    print(f"  Max difference in timestamps: {max_diff:.6f} ms")
else:
    print(f"  Timestamps DO NOT match in length!")

print(f"\nData value comparison (first neuron, first 5 frames):")
print(f"  Legacy: {popevents_legacy_bs[0, :5]}")
print(f"  New:    {popevents_new_bs[0, :5]}")

if popevents_legacy_bs.shape == popevents_new_bs.shape:
    diff = np.abs(popevents_legacy_bs - popevents_new_bs)
    print(f"\nDifference statistics:")
    print(f"  Mean absolute difference: {np.nanmean(diff):.6f}")
    print(f"  Max absolute difference:  {np.nanmax(diff):.6f}")
    print(f"  Correlation: {np.corrcoef(popevents_legacy_bs.flatten(), popevents_new_bs.flatten())[0, 1]:.6f}")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
