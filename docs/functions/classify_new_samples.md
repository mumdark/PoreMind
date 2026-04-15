# `MultiSampleAnalysis.classify_new_samples`
**Module:** `workflow.py`

Reuses fitted workflow/model to classify events from new samples.

## Parameters
- `new_sample_paths` (`dict[str, str | Path]`): Input file mapping for new samples.
- `reader` (`str | None`): Optional reader override.
- `reader_kwargs` (`dict[str, Any] | None`): Optional reader kwargs override.

## Returns
- `tuple[MultiSampleAnalysis, pandas.DataFrame]`:
  - First value: analysis object for new samples (contains traces/events/features and plotting accessor `pl`/`plot`).
  - Second value: per-event prediction table (`pred_df`).
