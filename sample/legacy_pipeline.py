#!/usr/bin/env python3
"""
Legacy Pipeline - Refactored from sample_pipeline.ipynb

This script processes 2-photon imaging data with behavioral event alignment.
It matches the original notebook logic but runs without cached files.

Author: Refactored from original notebook
Date: 2026-01-08
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters for the pipeline"""
    # Analysis parameters
    alphalevel = 0.05
    bh_correction = False
    mineventstoanalyze = 3
    
    # Overlap deletion
    delete_overlaps = True
    separation_requirement = 1000  # milliseconds
    
    # Frame parameters
    frameaveraging = 4
    framerate = 30
    timebetweenframes = 33.333333
    averagedframerate = framerate / frameaveraging
    
    # Z-score preprocessing
    z_score_data = True
    
    # Window parameters (in frames)
    pre_window_size = int(10 * averagedframerate)  # 10 seconds before event
    window_size = int((pre_window_size * 2) + (1.6 * averagedframerate))
    post_window_size = window_size - pre_window_size
    
    # Baseline and analysis windows
    baselinefirstframe = 0
    baselinelastframe = int(3 * averagedframerate)
    infusionframe = int(pre_window_size + (1.6 * averagedframerate))
    aucfirstframe = int(pre_window_size - (5 * averagedframerate))
    auclastframe = int(pre_window_size + (5 * averagedframerate))
    
    # Event of interest
    eventofinterest = 'activeleverall'
    
    # Data directory
    basedir = '../data'


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def fix_any_dropped_frames(frame_timestamps, timebetweenframes=33.333333):
    """
    Fix dropped frames by interpolating missing timestamps.
    
    Args:
        frame_timestamps: Array of frame timestamps
        timebetweenframes: Expected time between frames in ms
    
    Returns:
        Corrected frame timestamp array
    """
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


def framenumberforevent(event, frame_timestamps):
    """
    Find frame indices for each event timestamp.
    
    Args:
        event: Array of event timestamps
        frame_timestamps: Array of frame timestamps
    
    Returns:
        Array of frame indices corresponding to events
    """
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


def iterate_dirs(basedir):
    """
    Iterate through data directory structure: day/animal/fov
    
    Args:
        basedir: Base directory path
    
    Yields:
        Tuples of (basedir, day, animal, fov)
    """
    days = sorted([d for d in os.listdir(basedir) 
                   if os.path.isdir(os.path.join(basedir, d)) 
                   and not d.startswith('.')])
    
    for day in days:
        day_path = os.path.join(basedir, day)
        animals = sorted([a for a in os.listdir(day_path) 
                         if os.path.isdir(os.path.join(day_path, a))])
        
        for animal in animals:
            animal_path = os.path.join(day_path, animal)
            fovs = sorted([f for f in os.listdir(animal_path) 
                          if os.path.isdir(os.path.join(animal_path, f))])
            
            for fov in fovs:
                yield basedir, day, animal, fov


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def analyze_single_session(indir, config):
    tempfiles = os.listdir(indir)
    npyfiles = [f for f in tempfiles if f.endswith('.npy') and 'extractedsignals_raw' in f]
    matfiles = [f for f in tempfiles if f.endswith('.mat') and 'popevents' not in f and 'alignedevents' not in f]
    if len(npyfiles) == 0 or len(matfiles) == 0:
        print(f"  Skipping {indir}: No data files found")
        return None
    if len(npyfiles) > 1:
        npyfile = [f for f in npyfiles if 'part2' not in f and 'part3' not in f and 'part4' not in f][0]
        matfile = [f for f in matfiles if 'part2' not in f and 'part3' not in f and 'part4' not in f][0]
    else:
        npyfile = npyfiles[0]
        matfile = matfiles[0]
    signals = np.squeeze(np.load(os.path.join(indir, npyfile)))
    numrois = signals.shape[0]
    behaviordata = sio.loadmat(os.path.join(indir, matfile))
    eventlog = np.squeeze(behaviordata['eventlog'])
    activelever = eventlog[eventlog[:, 0] == 22, 1]
    activelevertimeout = eventlog[eventlog[:, 0] == 222, 1]
    inactivelever = eventlog[eventlog[:, 0] == 21, 1]
    inactivelevertimeout = eventlog[eventlog[:, 0] == 212, 1]
    cues = eventlog[eventlog[:, 0] == 7, 1]
    infusions = eventlog[eventlog[:, 0] == 4, 1]
    if config.delete_overlaps:
        if 'active' in config.eventofinterest:
            temp = np.sort(np.hstack((activelever, activelevertimeout)))
            temp = np.delete(temp, np.argwhere(np.ediff1d(temp) < config.separation_requirement) + 1)
            if config.eventofinterest == 'activelever':
                activelever = np.array([t for t in temp if t in activelever])
            elif config.eventofinterest == 'activelevertimeout':
                activelevertimeout = np.array([t for t in temp if t in activelevertimeout])
            elif config.eventofinterest == 'activeleverall':
                activeleverall = temp
        elif 'inactive' in config.eventofinterest:
            temp = np.sort(np.hstack((inactivelever, inactivelevertimeout)))
            temp = np.delete(temp, np.argwhere(np.ediff1d(temp) < config.separation_requirement) + 1)
            if config.eventofinterest == 'inactivelever':
                inactivelever = np.array([t for t in temp if t in inactivelever])
            elif config.eventofinterest == 'inactivelevertimeout':
                inactivelevertimeout = np.array([t for t in temp if t in inactivelevertimeout])
            elif config.eventofinterest == 'inactiveleverall':
                inactiveleverall = temp
    else:
        activeleverall = np.sort(np.hstack((activelever, activelevertimeout)))
        inactiveleverall = np.sort(np.hstack((inactivelever, inactivelevertimeout)))
    if 'activeleverall' not in locals():
        activeleverall = np.sort(np.hstack((activelever, activelevertimeout)))
    if 'inactiveleverall' not in locals():
        inactiveleverall = np.sort(np.hstack((inactivelever, inactivelevertimeout)))
    event_map = {
        'activelever': activelever,
        'activelevertimeout': activelevertimeout,
        'inactivelever': inactivelever,
        'inactivelevertimeout': inactivelevertimeout,
        'cues': cues,
        'infusions': infusions,
        'activeleverall': activeleverall,
        'inactiveleverall': inactiveleverall
    }
    events = event_map.get(config.eventofinterest, activeleverall)
    if len(events) < config.mineventstoanalyze:
        print(f"  Skipping {indir}: Only {len(events)} events (need {config.mineventstoanalyze})")
        return None
    frame_ts_raw = eventlog[eventlog[:, 0] == 9, 1]
    if len(frame_ts_raw) == 0:
        print(f"  Skipping {indir}: No frame timestamps in eventlog")
        return None
    frame_timestamps = fix_any_dropped_frames(frame_ts_raw, config.timebetweenframes)
    frame_timestamps = frame_timestamps[::config.frameaveraging]  # Apply frame averaging
    if signals.shape[1] > frame_timestamps.shape[0]:
        signals = signals[:, :frame_timestamps.shape[0] - 1]
    signals = signals / np.nanmean(signals, axis=1)[:, None]
    if config.z_score_data:
        for neuron in range(signals.shape[0]):
            mean = np.nanmean(signals[neuron])
            std = np.nanstd(signals[neuron])
            if std > 0:
                signals[neuron] = (signals[neuron] - mean) / std
    signalsT = signals.T
    framenumberfor_eventofinterest = np.squeeze(framenumberforevent(events, frame_timestamps))
    numtrials = framenumberfor_eventofinterest.shape[0]
    alignedevents = np.nan * np.zeros([numtrials, config.window_size, numrois])
    valid_trials = []
    for i in range(numtrials):
        eventindex = framenumberfor_eventofinterest[i]
        if (np.isfinite(eventindex) and 
            eventindex > config.pre_window_size and 
            eventindex < signalsT.shape[0] - config.post_window_size):
            eventindex = int(eventindex)
            alignedevents[i, :, :] = signalsT[eventindex - config.pre_window_size:eventindex + config.post_window_size, :]
            valid_trials.append(i)
    alignedevents = alignedevents[valid_trials, :, :]
    if alignedevents.shape[0] == 0:
        print(f"  Skipping {indir}: No valid trial windows")
        return None
    alignedevents = np.swapaxes(alignedevents, 0, 2)
    popevents = np.nanmean(alignedevents, axis=2)
    return popevents, alignedevents, events, events


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    config = Config()
    print("=" * 80)
    print("Legacy Pipeline - Event-Aligned Neural Data Processing")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Base directory: {config.basedir}")
    print(f"  Event of interest: {config.eventofinterest}")
    print(f"  Window size: {config.window_size} frames ({config.window_size/config.averagedframerate:.1f}s)")
    print(f"  Pre-event: {config.pre_window_size} frames ({config.pre_window_size/config.averagedframerate:.1f}s)")
    print(f"  Post-event: {config.post_window_size} frames ({config.post_window_size/config.averagedframerate:.1f}s)")
    print(f"  Min events: {config.mineventstoanalyze}")
    print(f"  Z-score: {config.z_score_data}")
    print(f"  Delete overlaps: {config.delete_overlaps} ({config.separation_requirement}ms)")
    print()
    popevents_day = {}
    popevents_fov = {}
    print("Processing sessions...")
    session_count = 0
    for basedir, day, animal, fov in iterate_dirs(config.basedir):
        if day not in popevents_fov:
            popevents_day[day] = []
            popevents_fov[day] = {}
            print(f"\n{day}")
        indir = os.path.join(basedir, day, animal, fov)
        session_count += 1
        if session_count % 10 == 0:
            print(f"  ... processed {session_count} sessions so far")
        result = analyze_single_session(indir, config)
        if result is not None:
            popevents, alignedevents, events, _ = result
            if animal not in popevents_fov[day]:
                popevents_fov[day][animal] = {}
            popevents_fov[day][animal][fov] = popevents
            print(f"  {animal}/{fov}: {popevents.shape[0]} neurons, {alignedevents.shape[2]} trials")
    print("\n" + "=" * 80)
    print("Applying baseline subtraction...")
    print("=" * 80)
    for day in popevents_fov:
        for animal in popevents_fov[day]:
            for fov in popevents_fov[day][animal]:
                data = popevents_fov[day][animal][fov]
                baseline = np.mean(data[:, config.baselinefirstframe:config.baselinelastframe], axis=1)
                popevents_fov[day][animal][fov] = data - baseline[:, None]
                popevents_day[day].append(popevents_fov[day][animal][fov])
    for day in popevents_day:
        if len(popevents_day[day]) > 0:
            popevents_day[day] = np.vstack(popevents_day[day])
            
            baseline = np.mean(popevents_day[day][:, config.baselinefirstframe:config.baselinelastframe], axis=1)
            popevents_day[day] = popevents_day[day] - baseline[:, None]
            print(f"{day}: {popevents_day[day].shape[0]} neurons")
    print("\n" + "=" * 80)
    print("Generating visualization...")
    print("=" * 80)
    days = sorted(popevents_day.keys())
    n_days = len(days)
    if n_days == 0:
        print("No data to plot!")
        return
    fig, axes = plt.subplots(2, n_days, figsize=(4 * n_days, 8))
    if n_days == 1:
        axes = axes.reshape(-1, 1)
    time_axis = np.arange(config.window_size) / config.averagedframerate - config.pre_window_size / config.averagedframerate
    for i, day in enumerate(days):
        data = popevents_day[day]
        im = axes[0, i].imshow(data, aspect='auto', cmap='RdYlGn',
                               vmin=-0.4, vmax=0.4, interpolation='nearest')
        axes[0, i].axvline(config.pre_window_size, color='black', linestyle='--', linewidth=2)
        axes[0, i].set_title(f"{day}\n{data.shape[0]} neurons")
        axes[0, i].set_ylabel("Neurons")
        axes[0, i].set_xlabel("Time (frames)")
        mean_trace = np.nanmean(data, axis=0)
        axes[1, i].plot(time_axis, mean_trace, linewidth=2)
        axes[1, i].axvline(0, color='black', linestyle='--', linewidth=2)
        axes[1, i].axhline(0, color='gray', linestyle=':', linewidth=1)
        axes[1, i].set_ylim(-0.4, 0.4)
        axes[1, i].set_xlabel("Time (s)")
        axes[1, i].set_ylabel("Mean Activity")
        axes[1, i].grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(config.basedir, 'legacy_pipeline_output.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to: {output_path}")
    plt.close()
    print("\nProcessing complete!")
if __name__ == "__main__":
    main()
