# test_core.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu

from brew.core.project import *
from brew.analysis.peri_event import *
import os
from brew.config.events import LEGACY_HER

def test_core(basedir: str):
    population_names = [p for p in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, p))]
    populations = []
    for p in population_names:
        population_dir = os.path.join(basedir, p)
        sample_names = [s for s in os.listdir(population_dir) if os.path.isdir(os.path.join(population_dir, s))]
        samples = []
        for s in sample_names:
            sample_dir = os.path.join(population_dir, s)
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
                    samples.append(sample)
                except Exception as e:
                    print(f"Sample {s} failed ({e}); skipping")
                    continue
        population = Population(name=p, samples=samples)
        populations.append(population)
    p = Project(
        name="Test Project",
        populations=populations,
        authors=["Author 1", "Author 2"],
        description="This is a test project."
    )
    return p

if __name__ == "__main__":
    test_project = test_core(r"../../data")
    print(test_project)
    for pop in test_project.get_populations():
        print(pop)
        for sam in pop.get_samples():
            print(sam)