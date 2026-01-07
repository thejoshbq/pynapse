import os
import pathlib
from pynapse.core.sample import Sample
from pynapse.core.population import Population
from pynapse.config.events import LEGACY_HER

basedir = str(pathlib.Path(__file__).parent / "data")
correction_path = str(pathlib.Path(__file__).parent / "data" / "empty.mat")

# Test just one population
test_pop = os.path.join(basedir, "0 EarlyAcq")
sample_names = [s for s in os.listdir(test_pop) if os.path.isdir(os.path.join(test_pop, s))]

print(f"Found {len(sample_names)} samples: {sample_names}")

for s in sample_names[:3]:  # Just test first 3
    sample_dir = os.path.join(test_pop, s)
    FOVs = [f for f in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, f))]
    print(f"\nSample {s} has FOVs: {FOVs}")
    
    for f in FOVs[:1]:  # Just test first FOV
        FOV_dir = os.path.join(sample_dir, f)
        mat_files = [os.path.join(FOV_dir, m) for m in os.listdir(FOV_dir) if m.endswith(".mat")]
        npy_files = [os.path.join(FOV_dir, n) for n in os.listdir(FOV_dir) if n.endswith(".npy") and "extracted" in n]
        
        print(f"  FOV {f}:")
        print(f"    Mat files: {len(mat_files)}")
        print(f"    Npy files: {len(npy_files)}")
        
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
            
            print(f"    Sample created successfully!")
            print(f"    Total events: {sample.num_events}")
            print(f"    Event ID 2 count: {sample.get_num_events(2)}")
            
            # Check the dataframe
            df = sample.get_dataframe()
            print(f"    Events with code 2: {len(df[df['code'] == 2])}")
            print(f"    Unique event codes: {sorted(df['code'].unique())}")
            
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
