# `MultiSampleAnalysis.classify_new_samples`
**Module:** `workflow.py`

Reuses fitted workflow/model to classify events from new samples.

## Parameters
- `new_sample_paths` (`dict[str, str | Path]`): Input file mapping for new samples.
- `reader` (`str | None`): Optional reader override.
- `reader_kwargs` (`dict[str, Any] | None`): Optional reader kwargs override.
- `custom_feature_fns` (`dict[str, Callable] | None`): Optional custom feature functions forwarded to `extract_features`.
- `model` (`str | Any | None`): Optional trained model selector/object.  
  - For ML models, pass a model name from `build_best_model` candidates (or model object).  
  - For DL models, pass `model_name` used by `build_DL_model` (e.g. `"1D-CNN"`).

## Returns
- `tuple[MultiSampleAnalysis, pandas.DataFrame]`:
  - First value: analysis object for new samples (contains traces/events/features and plotting accessor `pl`/`plot`).
  - Second value: per-event prediction table (`pred_df`):  
    - ML path: `pred_label`, `pred_proba_<class>`  
    - DL path: `pred_label_<model_name>`, `pred_proba_<class>_<model_name>`
