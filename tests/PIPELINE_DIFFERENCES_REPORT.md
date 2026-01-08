# Pipeline Differences Report

## Summary
The new pipeline (`main.py`) and legacy pipeline (`sample/legacy_pipeline.py`) produce different outputs due to a **critical frame timestamp handling difference**.

## Root Cause: Frame Correction Parameter

### The Issue
**main.py (line 31)** sets `frame_correction=True`:
```python
sample = Sample(
    ...
    frame_correction=True,
    correction_file=correction_path,
    ...
)
```

**legacy_pipeline.py** does NOT use frame correction - it directly processes raw frame timestamps from the event log.

### Impact on Processing

#### Legacy Pipeline
- Extracts frame timestamps (event code 9) directly from event log
- For IG-19/FOV2: 786 raw frame timestamps
- After dropped frame interpolation: 13,954 timestamps  
- After frame averaging (÷4): **3,489 timestamps**

#### New Pipeline with `frame_correction=True`
- Loads correction file (data/empty.mat)
- Generates synthetic frame timestamps using `_handle_assumed_frames()`
- For IG-19/FOV2: **116,457 timestamps** (33x more!)

#### New Pipeline with `frame_correction=False` (Correct)
- Uses `_handle_missed_frames()` like legacy
- For IG-19/FOV2: **3,489 timestamps** ✓ Matches legacy!

## Cascading Effects

The incorrect frame timestamps cause:

1. **Different event-to-frame mapping**: Events are mapped to wrong frames
2. **Different event window extraction**: Wrong time windows extracted around events
3. **Different neuron counts**: 
   - Example (0 EarlyAcq):
     - Legacy: 1,965 neurons
     - New (incorrect): 2,392 neurons
4. **Low correlation**: Only 0.146 correlation between outputs for same sample
5. **Large value differences**: Mean absolute difference of 0.148

## Code Locations

### Frame Timestamp Handling

**Legacy Pipeline** (`sample/legacy_pipeline.py:216-217`):
```python
frame_timestamps = fix_any_dropped_frames(frame_ts_raw, config.timebetweenframes)
frame_timestamps = frame_timestamps[::config.frameaveraging]
```

**New Pipeline** (`pynapse/core/sample.py:167-181`):
```python
def _get_frame_timestamps(self) -> np.ndarray:
    ...
    if self._frame_correction:
        full_ts = self._handle_assumed_frames(self._correction_file)  # WRONG PATH
    else:
        full_ts = self._handle_missed_frames(frame_ts_raw)  # CORRECT PATH
    averaged_ts = full_ts[::self._frame_averaging]
    ...
```

**New Pipeline Usage** (`main.py:25-33`):
```python
sample = Sample(
    event_data=mat_files,
    signal_data=npy_files,
    name=s,
    fps=30,
    frame_averaging=4,
    frame_correction=True,  # ← PROBLEM IS HERE
    correction_file=correction_path,
    event_dict=LEGACY_HER
)
```

## Other Processing Differences (These are NOT causing issues)

### 1. Preprocessing/Normalization
Both pipelines use identical preprocessing:
- Divide by mean
- Z-score normalization
✓ **Status: Equivalent** (LegacyNormalize matches legacy logic)

### 2. Event Overlap Deletion
- **Legacy**: Manual deletion with 1000ms separation (line 172-191)
- **New**: buffer_ms=1000 parameter (line 49)
✓ **Status: Equivalent** (Both produce 56 events from 86 for IG-19/FOV2)

### 3. Event Window Parameters
- **Legacy**: pre_window=10s, window_size calculated as 162 frames
- **New**: pre_event=10s, post_event=11.6s (also 162 frames total)
✓ **Status: Equivalent**

### 4. Baseline Subtraction
Both apply two levels of baseline subtraction:
- **Legacy**: Per FOV, then per day
- **New**: Per sample, then all neurons
✓ **Status: Equivalent logic**

## Verification Test Results

Test on IG-19/FOV2 sample:

| Metric | Legacy | New (with correction) | New (without correction) |
|--------|--------|----------------------|--------------------------|
| Frame timestamps | 3,489 | 116,457 | 3,489 ✓ |
| Events after filter | 56 | 56 | 56 ✓ |
| Valid trials | 54 | 54 | 54 ✓ |
| Output shape | (85, 162) | (85, 162) | (85, 162) ✓ |
| Correlation | 1.0 | 0.146 ❌ | ~1.0 ✓ |

## Solution

**Change main.py line 31 from:**
```python
frame_correction=True,
```

**To:**
```python
frame_correction=False,
```

## Why This Happened

The `_handle_assumed_frames()` function was designed for a different use case - it artificially generates frame timestamps by tripling the correction file's event log. This is useful for generating synthetic data, but **not** for processing real experimental data where frame timestamps exist in the event log.

## Additional Notes

- The correction file (data/empty.mat) appears to be nearly empty or have minimal data
- When the new pipeline uses this with `_handle_assumed_frames()`, it generates far too many synthetic timestamps
- The legacy pipeline correctly handles dropped frames using the actual recorded timestamps

## Recommendation

1. Set `frame_correction=False` in main.py
2. Remove or set `correction_file=None` since it's not needed
3. Run both pipelines again to verify outputs match
4. Consider adding a validation test that compares pipeline outputs
