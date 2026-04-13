# `detect_events_pelt`
**Module:** `events.py`

PELT change-point segmentation based event detection.

## Parameters
- `signal` (`np.ndarray`): Signal array.
- `baseline` (`np.ndarray`): Baseline array.
- `sampling_rate_hz` (`float`): Sampling rate.
- `model` (`str`): Ruptures cost model.
- `penalty` (`float`): PELT penalty.
- `sigma_k` (`float`): Residual threshold multiplier.
- `min_duration_s` (`float`): Minimum duration (s).

## Returns
- `list[Event]`
