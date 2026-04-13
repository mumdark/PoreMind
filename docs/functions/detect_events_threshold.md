# `detect_events_threshold`
**Module:** `events.py`

Threshold-based event detection on residual signal.

## Parameters
- `signal` (`np.ndarray`): Signal array.
- `baseline` (`np.ndarray`): Baseline array.
- `sampling_rate_hz` (`float`): Sampling rate.
- `sigma_k` (`float`): Threshold multiplier.
- `min_duration_s` (`float`): Minimum event duration (s).

## Returns
- `list[Event]`
