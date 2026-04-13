# `MultiSampleAnalysis.detect_events`
**Module:** `workflow.py`

Runs baseline estimation and event detection for each denoised trace.
For fast local tuning on a bounded time window, see `detect_events_simple`.

## Parameters
- `detect_method` (`str`): Detection method: threshold, zscore_threshold, cusum, pelt, hmm.
- `detect_params` (`dict[str, Any] | None`): Method-specific detection parameters. If None, defaults are used.
- `detect_direction` (`str`): Event polarity, `down` (default) or `up`.
- `baseline_method` (`str`): Baseline estimator method.
- `baseline_params` (`dict[str, Any] | None`): Baseline parameters. For `rolling_quantile`, defaults to `{"window": 10000, "q": 0.5}`. For `global_baseline`, pass `{"baseline": xx}`.

## Returns
- `Self (MultiSampleAnalysis).`
