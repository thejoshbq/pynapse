# Logic Differences Between Jupyter Notebook and pynapse Package

## Summary

The primary discrepancy between your pynapse package and the original Jupyter notebook stems from **the order of operations** in data preprocessing. The notebook normalizes and z-scores the FULL TRACE before extracting event windows, while your package extracts windows first and then applies preprocessing.

## Detailed Breakdown

### Jupyter Notebook Processing Order

```
1. Load raw signals (neurons × frames)
2. Normalize by mean: signals /= np.nanmean(signals, axis=1)[:, None]
3. Z-score ENTIRE trace (lines 290-294):
   for neuron in range(signals.shape[0]):
       mean = np.nanmean(signals[neuron])
       std = np.nanstd(signals[neuron])
       signals[neuron] = (signals[neuron] - mean) / std
4. Extract event windows from preprocessed trace
5. Average across trials → popevents (neurons × window_size)
6. Subtract baseline mean (lines 517-518, 521-522):
   baseline = np.mean(popevents[:, 0:22], axis=1)
   popevents = popevents - baseline[:, None]
7. For plotting: divide by baseline std (lines 658-661)
```

**Key parameters:**
- `frameaveraging = 4`
- `framerate = 30` → `averagedframerate = 7.5`
- `pre_window_size = int(10*7.5) = 75` frames
- `window_size = int((75*2)+(1.6*7.5)) = 162` frames
- `baselinefirstframe = 0`
- `baselinelastframe = int(3*7.5) = 22` frames
- Event happens at frame 75

### pynapse Package Processing Order (Original)

```
1. Load raw signals
2. Extract event windows FIRST
3. Apply OTIS_PIPE to windows:
   - DFOverF: (F - F0_8th_percentile) / F0_8th_percentile
   - ZScore using baseline window
   - GaussianSmoothing
4. Average across neurons and trials
```

## Why This Matters

The notebook's approach of z-scoring the FULL TRACE means:
- The mean and standard deviation are calculated from the **entire recording session**
- After windowing, the baseline may not be centered at 0
- An additional baseline subtraction step is needed to center each window

The package's approach of windowing first means:
- Each window is processed independently
- The baseline is determined only from that window
- No need for post-windowing baseline subtraction

These are **fundamentally different transformations** that produce different results.

## The Fix

I've created three new modules to match the notebook logic:

### 1. `LegacyNormalize` (preprocessing/legacy_normalize.py)

Matches the notebook's normalization:
```python
# Step 1: Divide by mean
signals /= np.nanmean(signals, axis=1)[:, None]

# Step 2: Z-score full trace
for neuron in range(signals.shape[0]):
    mean = np.nanmean(signals[neuron])
    std = np.nanstd(signals[neuron])
    signals[neuron] = (signals[neuron] - mean) / std
```

### 2. `WindowBaselineSubtraction` (preprocessing/window_baseline_subtraction.py)

Subtracts baseline mean after windowing:
```python
baseline = np.mean(signals[:, 0:22], axis=1, keepdims=True)
signals = signals - baseline
```

### 3. `LegacyPopulationEventTensor` (analysis/legacy_peri_event.py)

Applies preprocessing in the correct order:
```python
1. Apply pre-windowing preprocessing to full trace
2. Extract event windows from preprocessed trace
3. Apply post-windowing preprocessing (baseline subtraction)
```

## How to Use

Run the new test file that uses legacy preprocessing:

```bash
cd pynapse/analysis
python -m brew.analysis.test_analysis_legacy
```

Or use in your code:

```python
from brew.analysis.legacy_peri_event import LegacyPopulationEventTensor
from brew.analysis.preprocessing import LegacyNormalize, WindowBaselineSubtraction

# Create preprocessors
pre_window_proc = LegacyNormalize(z_score=True)
post_window_proc = WindowBaselineSubtraction(baseline_frames=(0, 22))

# Extract windows with legacy logic
population_tensor = LegacyPopulationEventTensor(
    population,
    event_id=22,
    pre_event=10,
    post_event=11.6,
    min_trials=3,
    buffer_ms=1000,
    pre_window_preprocessor=pre_window_proc,
    post_window_preprocessor=post_window_proc
)

event_windows = population_tensor.get_event_windows()
```

## Which Approach is More Accurate?

Both approaches are **methodologically valid** but serve different purposes:

### Notebook Approach (Full Trace Preprocessing)
**Advantages:**
- Normalizes activity across the entire session
- Accounts for slow drifts in baseline fluorescence
- Useful for comparing activity patterns within the same recording

**Disadvantages:**
- Less interpretable baseline (not necessarily at 0)
- Harder to compare across sessions
- Requires additional baseline correction

### Modern Approach (Window Preprocessing)
**Advantages:**
- Each trial is independent
- Baseline is always centered at 0
- Easier to compare across sessions and animals
- More commonly used in recent literature

**Disadvantages:**
- May lose information about session-wide trends
- Could miss slow dynamics

## Recommendation

For **reproducing the notebook results**, use the legacy approach with `test_analysis_legacy.py`.

For **new analyses**, I recommend:
1. Using the modern approach (original `test_analysis.py`)
2. Document clearly which preprocessing was used
3. Consider if session-wide normalization is important for your scientific question

If you need session-wide comparisons, the legacy approach makes sense. If you're comparing event-locked responses across animals/conditions, the modern approach is better.
