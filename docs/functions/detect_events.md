# `MultiSampleAnalysis.detect_events`
**Module:** `workflow.py`

Runs baseline estimation and event detection for each denoised trace.
For fast local tuning on a bounded time window, see `detect_events_simple`.

## Parameters
- `detect_method` (`str`): Detection method: threshold, zscore_threshold, cusum, pelt, hmm.
- `detect_params` (`dict[str, Any] | None`): Method-specific detection parameters. If None, defaults are used.
- `baseline_method` (`str`): Baseline estimator method.
- `baseline_params` (`dict[str, Any] | None`): Baseline method parameters.

## Returns
- `Self (MultiSampleAnalysis).`
