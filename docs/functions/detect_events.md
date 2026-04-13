# `MultiSampleAnalysis.detect_events`
**Module:** `workflow.py`

Runs baseline estimation and event detection for each denoised trace.

## Parameters
- `detect_method` (`str`): Detection method: threshold, zscore_threshold, cusum, pelt, hmm.
- `detect_params` (`dict[str, Any] | None`): Method-specific detection parameters. If None, defaults are used.
- `baseline_method` (`str`): Baseline estimator method.
- `baseline_params` (`dict[str, Any] | None`): Baseline method parameters.

## Returns
- `Self (MultiSampleAnalysis).`
