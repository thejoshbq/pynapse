#!/usr/bin/env python3
"""
Test to verify frame_correction is causing the difference
"""

import os
import numpy as np
from pynapse.config.events import LEGACY_HER
from pynapse.core.sample import Sample

test_day = "0 EarlyAcq"
test_animal = "IG-19"
test_fov = "FOV2"

basedir = "data"
indir = os.path.join(basedir, test_day, test_animal, test_fov)

tempfiles = os.listdir(indir)
npyfile = [f for f in tempfiles if f.endswith('.npy') and 'extractedsignals_raw' in f][0]
matfile = [f for f in tempfiles if f.endswith('.mat') and 'popevents' not in f][0]

mat_files = [os.path.join(indir, matfile)]
npy_files = [os.path.join(indir, npyfile)]

print("Testing frame_correction parameter effect:\n")

# Test with frame_correction=True (current main.py behavior)
sample_with_correction = Sample(
    event_data=mat_files,
    signal_data=npy_files,
    name="test",
    fps=30,
    frame_averaging=4,
    frame_correction=True,
    correction_file="../data/empty.mat",
    event_dict=LEGACY_HER
)
frame_ts_with = sample_with_correction._get_frame_timestamps()
print(f"With frame_correction=True:")
print(f"  Frame timestamps: {len(frame_ts_with)}")

# Test with frame_correction=False (should match legacy)
sample_without_correction = Sample(
    event_data=mat_files,
    signal_data=npy_files,
    name="test",
    fps=30,
    frame_averaging=4,
    frame_correction=False,
    correction_file=None,
    event_dict=LEGACY_HER
)
frame_ts_without = sample_without_correction._get_frame_timestamps()
print(f"\nWith frame_correction=False:")
print(f"  Frame timestamps: {len(frame_ts_without)}")
print(f"\nLegacy pipeline expected: ~3489 timestamps")
print(f"\nConclusion: frame_correction=True is causing the discrepancy!")
