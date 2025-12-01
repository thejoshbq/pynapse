# test_analysis.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu

from brew.analysis.peri_event import *
import os
from brew.config.events import LEGACY_HER
from brew.config.pipelines import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


def test_analysis(basedir: str):
    sample_names = [s for s in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, s))]
    windows = []
    raw_windows = []
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
                    correction_file=r"./data/empty.mat",
                    event_dict=LEGACY_HER
                )
                matrix = EventMatrix(
                    sample,
                    event_id=22,
                    pre_event=10,
                    post_event=11.6,
                    downsample=True,
                    min_events=3
                )
                raw = matrix.get_event_windows()
                pipe = LEGACY_PIPE
                processed = pipe(raw)
                windows.append(processed)
                raw_windows.append(raw)
            except Exception as e:
                print(f"Sample {s} failed ({e}); skipping")
                continue
    return windows, raw_windows


if __name__ == "__main__":
    plt.style.use('default')
    sns.set_context("paper", font_scale=1.6)
    sns.set_style("whitegrid")

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    day_labels = ["EarlyAcq", "MidAcq", "LateAcq", "EarlyExt", "LastExt", "CueRein", "DrugRein", "TMTRein"]
    event_frame = 75
    time_vec = np.linspace(-10, 11.6, 162)

    all_raw_traces = []
    all_raw_sems = []
    all_processed_traces = []
    all_processed_sems = []
    all_n = []
    stats_out = []

    for day_idx, day_folder in enumerate(
            sorted([d for d in os.listdir("./data") if os.path.isdir(f"./data/{d}")])):
        if not any(l in day_folder for l in day_labels):
            continue
        path = f"./data/{day_folder}"
        try:
            windows, raw_windows = test_analysis(path)
            if not windows:
                continue
        except Exception as e:
            print(e)
            continue

        raw_neuron_traces = [np.nanmean(w, axis=0) for w in raw_windows]
        raw_full_matrix = np.vstack(raw_neuron_traces)

        processed_neuron_traces = [np.nanmean(w, axis=0) for w in windows]
        processed_full_matrix = np.vstack(processed_neuron_traces)

        post_resp = np.mean(processed_full_matrix[:, event_frame + 5:event_frame + 40], axis=1)
        sort_idx = np.argsort(-post_resp)

        raw_sorted_matrix = raw_full_matrix[sort_idx]
        processed_sorted_matrix = processed_full_matrix[sort_idx]

        raw_trace = np.mean(raw_sorted_matrix, axis=0)
        raw_sem = stats.sem(raw_sorted_matrix, axis=0)
        processed_trace = np.mean(processed_sorted_matrix, axis=0)
        processed_sem = stats.sem(processed_sorted_matrix, axis=0)

        all_raw_traces.append(raw_trace)
        all_raw_sems.append(raw_sem)
        all_processed_traces.append(processed_trace)
        all_processed_sems.append(processed_sem)
        all_n.append(processed_sorted_matrix.shape[0])

        pre = processed_trace[event_frame - 30:event_frame]
        post = processed_trace[event_frame:event_frame + 30]
        delta = np.mean(post) - np.mean(pre)
        d = delta / np.sqrt((np.std(pre) ** 2 + np.std(post) ** 2) / 2)
        _, p = stats.ttest_rel(post, pre)  # Paired t-test (assumes dependent samples)
        p = p if not np.isnan(p) else 1.0
        stats_out.append({
            'day': day_labels[day_idx],
            'delta': delta,
            'd': d,
            'p': p,
            'sig': '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        })

    fig, axes = plt.subplots(2, 8, figsize=(16, 8), sharex=True)
    axes = axes.flatten()

    for i, (raw_trace, raw_sem, n) in enumerate(zip(all_raw_traces, all_raw_sems, all_n)):
        ax = axes[i]
        ax.fill_between(time_vec, raw_trace - raw_sem, raw_trace + raw_sem,
                        color=colors[i], alpha=0.3)
        ax.plot(time_vec, raw_trace, color=colors[i], lw=3, label=f"n={n:,}")
        ax.axvline(0, color='black', lw=2, linestyle='--')
        ax.axhline(0, color='gray', lw=1, linestyle=':')
        ax.set_title(day_labels[i], fontsize=16, pad=15)
        ax.set_ylim(.85, 1.12)

    for i, (processed_trace, processed_sem, n) in enumerate(zip(all_processed_traces, all_processed_sems, all_n)):
        ax = axes[8 + i]
        ax.fill_between(time_vec, processed_trace - processed_sem, processed_trace + processed_sem,
                        color=colors[i], alpha=0.3)
        ax.plot(time_vec, processed_trace, color=colors[i], lw=3, label=f"n={n:,}")
        ax.axvline(0, color='black', lw=2, linestyle='--')
        ax.axhline(0, color='gray', lw=1, linestyle=':')
        ax.set_ylim(-0.15, 0.12)

    axes[0].set_ylabel("Raw Fluorescence (a.u.)", fontsize=14)
    axes[8].set_ylabel("ΔF/F (z-scored)", fontsize=14)

    plt.suptitle("PFC Lever Press Responses — Heroin Self-Administration\n(Raw vs. Processed)", fontsize=20, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    print("Stats (Processed Data):")
    for s in stats_out:
        print(f"{s['day']:10} Δ = {s['delta']:+.4f}, d = {s['d']:+.3f}, p = {s['p']:.3f} {s['sig']}")