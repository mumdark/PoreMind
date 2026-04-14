# `MultiSampleAnalysis.extract_features`
**Module:** `workflow.py`

Builds event-level feature table including baseline, morphology and statistical descriptors.

## Parameters
- `custom_feature_fns` (`dict[str, Callable] | None`): Optional custom feature functions; each receives event segment and returns dict.
- `max_event_per_sample` (`int | None`): Maximum number of events to extract per sample (take the first N events). `None` means extract all events.

## Returns
- `pandas.DataFrame (feature table).`

## Notes
- Shows sample-level and event-level progress bars automatically when `tqdm` is available in the environment.
