# `MultiSampleAnalysis.detect_events_simple`
**Module:** `workflow.py`

Run event detection only on a selected time window for quick method selection and parameter tuning.

## Parameters
- `detect_method` (`str`): Detection method: threshold, zscore_threshold, cusum, pelt, hmm.
- `detect_params` (`dict[str, Any] | None`): Method-specific parameters. Uses defaults when None.
- `baseline_method` (`str`): Baseline estimator method.
- `baseline_params` (`dict[str, Any] | None`): Baseline parameters.
- `sample_id` (`str | None`): Target trace id. None means run for all loaded traces.
- `current` (`str`): `denoise` or `raw` signal for detection.
- `start_ms` (`float`): Window start time in milliseconds. Default `0.0`.
- `end_ms` (`float`): Window end time in milliseconds. Default `1000.0`.

## Returns
- `dict[str, list[Event]]`: Detected events keyed by trace id. Event indices are mapped back to the original full trace indices.

## Side effects
- Writes results to `analysis.detect_events_simple_object` (primary store for plotting with `event_current_simple`).
- Also keeps backward-compatible mirror in `analysis.simple_events`.
