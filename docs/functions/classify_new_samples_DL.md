# `MultiSampleAnalysis.classify_new_samples_DL`
**Module:** `workflow.py`

Classify new samples with the DL model built by `build_DL_model`.

## Parameters
- `new_sample_paths` (`dict[str, str | Path]`): Input file mapping for new samples.
- `reader` (`str | None`): Optional reader override.
- `reader_kwargs` (`dict[str, Any] | None`): Optional reader kwargs override.
- `custom_feature_fns` (`dict[str, Callable] | None`): Optional custom features for new samples.

## Returns
- `tuple[MultiSampleAnalysis, pandas.DataFrame]`:
  - New analysis object for the new samples.
  - Prediction dataframe with DL-model-suffixed label/probability columns.
