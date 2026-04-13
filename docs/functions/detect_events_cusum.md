# `detect_events_cusum`
**Module:** `events.py`

CUSUM-based event detection on standardized residuals.

## Parameters
- `signal` (`np.ndarray`): Signal array.
- `baseline` (`np.ndarray`): Baseline array.
- `sampling_rate_hz` (`float`): Sampling rate.
- `drift` (`float`): CUSUM drift term.
- `threshold` (`float`): CUSUM trigger threshold.
- `min_duration_s` (`float`): Minimum duration (s).

## Returns
- `list[Event]`
