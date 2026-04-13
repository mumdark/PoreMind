# `detect_events_hmm`
**Module:** `events.py`

Gaussian HMM event detection selecting low-mean state as event state.

## Parameters
- `signal` (`np.ndarray`): Signal array.
- `baseline` (`np.ndarray`): Baseline array.
- `sampling_rate_hz` (`float`): Sampling rate.
- `n_components` (`int`): Number of HMM states.
- `covariance_type` (`str`): HMM covariance type.
- `n_iter` (`int`): Max EM iterations.
- `min_duration_s` (`float`): Minimum duration (s).

## Returns
- `list[Event]`
