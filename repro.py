import os
import numpy as np
from pynapse.core.sample import Sample
from pynapse.core.population import Population
from pynapse.analysis.peri_event import PopulationEventTensor
from pynapse.config.events import LEGACY_HER

def reproduce():
    basedir = "data/2 LateAcq"
    if not os.path.exists(basedir):
        print(f"Directory {basedir} does not exist")
        return

    sample_names = [s for s in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, s))]
    print(f"Samples found: {sample_names}")
    samples = []
    for s in sample_names:
        sample_dir = os.path.join(basedir, s)
        FOVs = [f for f in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, f))]
        for f in FOVs:
            FOV_dir = os.path.join(sample_dir, f)
            mat_files = [os.path.join(FOV_dir, m) for m in os.listdir(FOV_dir) if m.endswith(".mat")]
            npy_files = [os.path.join(FOV_dir, n) for n in os.listdir(FOV_dir) if n.endswith(".npy") and "extracted" in n]
            
            print(f"Sample {s}, FOV {f}: mat_files={len(mat_files)}, npy_files={len(npy_files)}")
            
            try:
                sample = Sample(
                    event_data=mat_files,
                    signal_data=npy_files,
                    name=s,
                    fps=30,
                    frame_averaging=4,
                    frame_correction=True,
                    correction_file="data/empty.mat",
                    event_dict=LEGACY_HER
                )
                samples.append(sample)
                print(f"Successfully loaded sample {s}")
                df = sample.get_dataframe()
                print(f"Events: {df['code'].value_counts().to_dict()}")
            except Exception as e:
                print(f"Sample {s} failed ({e}); skipping")
                import traceback
                traceback.print_exc()
                continue
    
    if not samples:
        print("No samples loaded")
        return

    population = Population(name=basedir, samples=samples)
    print(f"Population created with {len(samples)} samples")
    
    population_tensor = PopulationEventTensor(
        population,
        event_id=22,
        pre_event=10,
        post_event=11.6,
        min_trials=3,
        buffer_ms=1000,
    )
    event_windows = population_tensor.get_event_windows()
    print(f"Number of event windows: {len(event_windows)}")
    for i, win in enumerate(event_windows):
        print(f"Sample {i} windows shape: {win.shape}")

if __name__ == "__main__":
    reproduce()
