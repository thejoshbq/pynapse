# test_analysis.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu

from brew.analysis.peri_event import *
import os
from brew.config.events import LEGACY_HER
from brew.config.pipelines import *
import matplotlib.pyplot as plt
import numpy as np

def test_analysis(name: str, basedir: str):
    sample_names = [s for s in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, s))]
    windows = []
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
                    correction_file=r"../../data/empty.mat",
                    event_dict=LEGACY_HER
                )
                matrix = EventMatrix(sample, 22, 10, 11.6, True)
                raw = matrix.get_event_windows()
                pipe = OTIS_PIPE
                windows.append(pipe(raw))
            except Exception as e:
                print(f"Sample {s} failed ({e}); skipping")
                continue
    return windows

if __name__ == "__main__":
    data = test_analysis("Early Acquisition", r"../../data/1 MidAcq")
    means = []
    for w in data:
        mean = np.nanmean(w, axis=(0, 1))
        means.append(mean)
    total_mean = np.nanmean(np.stack(means), axis=0)
    plt.plot(total_mean, linewidth=2, color="r")
    plt.show()