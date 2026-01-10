# main.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu

import os
from pynapse.config.events import LEGACY_HER
from pynapse.analysis.preprocessing.epoch.pipelines import *
import matplotlib.pyplot as plt
from pynapse.analysis.peri_event import *


def analysis(basedir: str):
    sample_names = [s for s in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, s))]
    samples = []
    for s in sample_names:
        sample_dir = os.path.join(basedir, s)
        FOVs = [f for f in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, f))]
        for f in FOVs:
            FOV_dir = os.path.join(sample_dir, f)
            mat_files = [os.path.join(FOV_dir, m) for m in os.listdir(FOV_dir) if m.endswith(".mat") and "popevents" not in m]
            npy_files = [os.path.join(FOV_dir, n) for n in os.listdir(FOV_dir) if n.endswith(".npy") and "extracted" in n]
            try:
                sample = Sample(
                    event_data=mat_files,
                    signal_data=npy_files,
                    name=s,
                    fps=30,
                    frame_averaging=4,
                    frame_correction=False,
                    correction_file=None,
                    event_dict=LEGACY_HER
                )
                samples.append(sample)
            except Exception as e:
                print(f"  Sample {s} failed ({e}); skipping")
                continue
    if len(samples) == 0:
        return None
    population = Population(name=basedir, samples=samples)
    population_tensor = PopulationEventTensor(
        population,
        event_id=[22],
        pre_event=10,
        post_event=11.6,
        min_trials=3,
        buffer_ms=1000,
        trace_preprocess=None,
        window_preprocess=OTIS_PIPE,
    )
    event_windows = population_tensor.get_event_windows()
    if len(event_windows) == 0:
        return None
    processed_neurons = []
    for sample_tensor in event_windows:
        if sample_tensor.shape[0] == 0:  # no valid trials
            continue
        mean_per_sample = np.nanmean(sample_tensor, axis=0)
        processed_neurons.append(mean_per_sample)
    if not processed_neurons:
        return None
    all_neurons = np.vstack(processed_neurons)
    return all_neurons

if __name__ == "__main__":
    basedir = "data/"
    print("="*80)
    print("Starting Analysis Pipeline")
    print("="*80)
    print("\nProcessing populations...\n")
    population_data = {}
    for population in sorted(os.listdir(basedir)):
        if not os.path.isdir(os.path.join(basedir, population)):
            continue
        if population.startswith('.'):
            continue
        print(f"{population}")
        result = analysis(os.path.join(basedir, population))
        if result is not None:
            population_data[population] = result
        else:
            print(f"  -> No valid data\n")
    if not population_data:
        print("No valid populations with complete data found.")
    else:
        print("="*80)
        print("Generating visualization...")
        print("="*80)
        n_pops = len(population_data)
        # plt.style.use('dark_background')
        fig, axes = plt.subplots(2, n_pops, figsize=(4 * n_pops, 8))
        if n_pops == 1:
            axes = axes.reshape(-1, 1)
        for i, (pop_name, data) in enumerate(population_data.items()):
            event_frame = 75  # press at frame 75 (10 s pre @ 7.5 Hz)
            post_start = event_frame
            post_end = event_frame + int(8 * 7.5)  # e.g., first ~8 s post-press (adjust to focus)
            post_slice = slice(post_start, post_end)
            peak_frames = np.argmax(data[:, post_slice], axis=1)  # relative to slice
            peak_times_s = peak_frames / 7.5  # optional: convert to seconds for logging
            # amplitudes = np.max(data[:, post_slice], axis=1)
            # sort_idx = np.argsort(-amplitudes)  # negative for descending
            # means = np.nanmean(data[:, post_slice], axis=1)
            # sort_idx = np.argsort(-means)
            # Argsort: earliest peaks first
            sort_idx = np.argsort(peak_frames)
            sorted_data = data[sort_idx]
            cmap = plt.get_cmap('vanimo').reversed()
            im = axes[0, i].imshow(sorted_data, aspect='auto', cmap=cmap,
                                   vmin=-0.4, vmax=0.4, interpolation='nearest')
            axes[0, i].axvline(75, color='black', linestyle='--', linewidth=2)
            axes[0, i].set_title(f"{pop_name}\n{data.shape[0]} neurons")
            axes[0, i].set_ylabel("Neurons")
            axes[0, i].set_xlabel("Frames")
            mean_trace = np.nanmean(data, axis=0)
            time_axis = (np.arange(data.shape[1]) - 75) / 7.5
            axes[1, i].plot(time_axis, mean_trace, linewidth=2)
            axes[1, i].axvline(0, color='black', linestyle='--', linewidth=2)
            axes[1, i].axhline(0, color='gray', linestyle=':', linewidth=1)
            axes[1, i].set_ylim(-0.4, 0.4)
            axes[1, i].set_xlabel("Time (s)")
            axes[1, i].set_ylabel("Mean Activity")
            axes[1, i].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        print("\nProcessing complete!")

