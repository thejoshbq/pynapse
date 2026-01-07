# test_analysis.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu

import os
from pynapse.config.events import LEGACY_HER
from pynapse.analysis.preprocessing.epoch.pipelines import *
from pynapse.analysis.preprocessing.continuous.normalization import *
import matplotlib.pyplot as plt
from pynapse.analysis.peri_event import *


def test_analysis(basedir: str):
    import pathlib
    correction_path = str(pathlib.Path(__file__).parent.parent.parent / "data" / "empty.mat")
    sample_names = [s for s in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, s))]
    samples = []
    for s in sample_names:
        sample_dir = os.path.join(basedir, s)
        FOVs = [f for f in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, f))]
        for f in FOVs:
            FOV_dir = os.path.join(sample_dir, f)
            mat_files = [os.path.join(FOV_dir, m) for m in os.listdir(FOV_dir) if m.endswith(".mat")]
            npy_files = [os.path.join(FOV_dir, n) for n in os.listdir(FOV_dir) if n.endswith(".npy") and "extracted" in n]
            try:
                sample = Sample(
                    event_data=mat_files,
                    signal_data=npy_files,
                    name=s,
                    fps=30,
                    frame_averaging=4,
                    frame_correction=True,
                    correction_file=correction_path,
                    event_dict=LEGACY_HER
                )
                samples.append(sample)
            except Exception as e:
                print(f"Sample {s} failed ({e}); skipping")
                continue
    population = Population(name=basedir, samples=samples)
    frame_duration_ms = 1000 / population.get_samples()[0].effective_fps
    population_tensor = PopulationEventTensor(
        population,
        event_id=22,
        pre_event=10,
        post_event=11.6,
        min_trials=3,
        buffer_ms=1000,
        pre_window_preprocessor=LegacyNormalize(),
        post_window_preprocessor=BaselineSubtraction(method='mean', window_ms=(0, 3000), frame_duration_ms=frame_duration_ms),
    )
    event_windows = population_tensor.get_event_windows()
    return event_windows


if __name__ == "__main__":
    import pathlib
    basedir = str(pathlib.Path(__file__).parent.parent.parent / "data")
    mean_windows = {}
    pipe = OTIS_PIPE
    for population in sorted(os.listdir(basedir)):
        mean_windows[population] = {}
        if os.path.isdir(os.path.join(basedir, population)):
            processed_population_mean_windows = []
            unprocessed_population_mean_windows = []
            population_event_windows = test_analysis(os.path.join(basedir, population))
            if not population_event_windows or len(population_event_windows) == 0:
                print(f"Population {population}: No valid event windows found; skipping")
                continue
            for sample_event_window_set in population_event_windows:
                processed = np.nanmean(pipe(sample_event_window_set), axis=(0, 1))
                unprocessed = np.nanmean(sample_event_window_set, axis=(0, 1))
                processed_population_mean_windows.append(processed)
                unprocessed_population_mean_windows.append(unprocessed)
            if processed_population_mean_windows and unprocessed_population_mean_windows:
                mean_windows[population]["processed"] = np.nanmean(processed_population_mean_windows, axis=0)
                mean_windows[population]["unprocessed"] = np.nanmean(unprocessed_population_mean_windows, axis=0)
            else:
                print(f"Population {population}: No mean windows calculated; skipping")
                continue
    mean_windows = {k: v for k, v in mean_windows.items() if "processed" in v and "unprocessed" in v}
    if not mean_windows:
        print("No valid populations with complete data found.")
    else:
        fig, axs = plt.subplots(2, len(mean_windows), figsize=(16, 6))
        for i, (key, data) in enumerate(mean_windows.items()):
            axs[0, i].plot(data["unprocessed"])
            axs[0, i].set_title(key)
            axs[1, i].plot(data["processed"])
        plt.tight_layout()
        plt.show()

