# `MultiSampleAnalysis.build_DL_model`
**Module:** `workflow.py`

Train a deep-learning event classifier (default 1D-CNN feature extractor + classification head) using filtered events.

## Parameters
- `device` (`str`): Preferred device (`cuda` by default; falls back to CPU if CUDA unavailable).
- `model` (`Any | None`): Optional custom feature extractor model.
- `model_name` (`str`): Name key used in `model_cv_results` and output columns.
- `feature_cols` (`list[str] | None`): Optional statistical features concatenated with CNN features.
- `interp_length` (`int`): Interpolation length of event current segments.
- `expand` (`int`): Left/right expansion points around each event.
- `scale` (`str | None`): Segment scaling mode (`mad`, `minmax`, `none`).
- `batch_size` / `learning_rate` / `epoch`: Training hyper-parameters.
- `early_stop_patience` (`int`): Early-stopping patience.
- `cv` (`int`): CV folds.
- `label_col` (`str`): Label column.

## Returns
- `dict` DL package (`self.DL_model_package`) including trained model, weights, class mapping, cv results.
