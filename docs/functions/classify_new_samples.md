# `MultiSampleAnalysis.classify_new_samples`
**Module:** `workflow.py`

Reuses fitted workflow/model to classify events from new samples.

## Parameters
- `new_sample_paths` (`dict[str, str | Path]`): Input file mapping for new samples.
- `reader` (`str | None`): Optional reader override.
- `reader_kwargs` (`dict[str, Any] | None`): Optional reader kwargs override.

## Returns
- `pandas.DataFrame with per-event predictions.`
