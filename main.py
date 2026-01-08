# main.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu

import os
from pynapse.config.events import LEGACY_HER
from pynapse.analysis.preprocessing.epoch.pipelines import *
from pynapse.analysis.preprocessing.continuous.normalization import *
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
    frame_duration_ms = 1000 / population.get_samples()[0].effective_fps
    population_tensor = PopulationEventTensor(
        population,
        event_id=[22, 222],
        pre_event=10,
        post_event=11.6,
        min_trials=3,
        buffer_ms=1000,
        pre_window_preprocessor=LegacyNormalize(),
        post_window_preprocessor=None,
    )
    event_windows = population_tensor.get_event_windows()
    if len(event_windows) == 0:
        return None
    baseline_frames = int(3000 / frame_duration_ms)
    processed_samples = []
    for sample_event_window_set in event_windows:
        if sample_event_window_set.shape[0] == 0:
            continue
        mean_per_sample = np.nanmean(sample_event_window_set, axis=0)
        baseline = np.mean(mean_per_sample[:, 0:baseline_frames], axis=1)
        mean_per_sample_bs = mean_per_sample - baseline[:, None]
        if mean_per_sample_bs.shape[0] == 0:
            continue
        processed_samples.append(mean_per_sample_bs)
    all_neurons = np.vstack(processed_samples)
    baseline = np.mean(all_neurons[:, 0:baseline_frames], axis=1)
    all_neurons_bs = all_neurons - baseline[:, None]
    return all_neurons_bs


if __name__ == "__main__":
    basedir = "data/"
    print("="*80)
    print("New Pipeline - Matching Legacy Logic")
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
            print(f"  -> {result.shape[0]} neurons, {result.shape[1]} frames\n")
        else:
            print(f"  -> No valid data\n")
    if not population_data:
        print("No valid populations with complete data found.")
    else:
        print("="*80)
        print("Generating visualization...")
        print("="*80)
        n_pops = len(population_data)
        fig, axes = plt.subplots(2, n_pops, figsize=(4 * n_pops, 8))
        if n_pops == 1:
            axes = axes.reshape(-1, 1)
        for i, (pop_name, data) in enumerate(population_data.items()):
            im = axes[0, i].imshow(data, aspect='auto', cmap='RdYlGn',
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

