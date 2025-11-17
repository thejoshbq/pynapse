from roast import *
from ground import *
from bean import *
from lot import *
from farm import *
import os

if __name__ == "__main__":
    basedir = r"../../data"
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
                event_dict = {
                    22: "active_lever",
                    222: "active_lever_timeout",
                    21: "inactive_lever",
                    212: "inactive_lever_timeout",
                    7: "cue",
                    4: "infusion",
                }
                sample = Sample(
                    event_data=mat_files,
                    signal_data=npy_files,
                    name=s,
                    fps=30,
                    frame_averaging=4,
                    frame_correction=True,
                    correction_file=r"../../data/empty.mat",
                    event_dict=event_dict
                )
                samples.append(sample)
                print("========== SAMPLE ==========")
                print(sample)
                print()
        population = Population(name=p, samples=samples)
        populations.append(population)
        print("========== POPULATION ==========")
        print(population)
        print()
    project = Project(
        name="Test Project",
        populations=populations,
        authors=["Author 1", "Author 2"],
        description="This is a test project."
    )
    print(project)