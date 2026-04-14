# `MultiSampleAnalysis.detect_events`
**Module:** `workflow.py`

Runs baseline estimation and event detection for each denoised trace.
For fast local tuning on a bounded time window, see `detect_events_simple`.

## Parameters
- `detect_method` (`str`): Detection method: threshold, zscore_threshold, cusum, pelt, hmm.
- `detect_params` (`dict[str, Any] | None`): Method-specific detection parameters. If None, defaults are used. Methods using noise scale support `noise_method` (`mad` default, or `std`).
- `detect_direction` (`str`): Event polarity, `down` (default) or `up`.
- `exclude_current` (`bool`): Whether to exclude current outside a range when computing baseline/noise statistics. Default `True`.
- `exclude_current_params` (`dict[str, Any] | None`): Current filter bounds for statistics. Use `{"min": xx, "max": yy}`. Defaults by direction: `up -> {"min": None, "max": 0}`, `down -> {"min": 0, "max": None}`.
- `merge_event` (`bool`): Whether to merge adjacent events (default `False`).
- `merge_event_params` (`dict[str, Any] | None`): Merge parameters. Use `{"merge_gap_ms": xx}` to merge events when the gap is within `xx` ms.
- `baseline_method` (`str`): Baseline estimator method.
- `baseline_params` (`dict[str, Any] | None`): Baseline parameters. For `rolling_quantile`, defaults to `{"window": 10000, "q": 0.5}`. For `global_quantile`, pass `{"q": xx}` (default `0.5`, i.e., global median).

## Returns
- `Self (MultiSampleAnalysis).`

## Notes
- Displays per-sample progress with a progress bar when `tqdm` is available.
- If filtered effective points are `<=1`, detection raises `ValueError` directly (no fallback).
