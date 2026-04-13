# `MultiSampleAnalysis.extract_features`
**Module:** `workflow.py`

Builds event-level feature table including baseline, morphology and statistical descriptors.

## Parameters
- `custom_feature_fns` (`dict[str, Callable] | None`): Optional custom feature functions; each receives event segment and returns dict.

## Returns
- `pandas.DataFrame (feature table).`
