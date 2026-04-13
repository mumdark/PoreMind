# `MultiSampleAnalysis.build_best_model`
**Module:** `workflow.py`

Evaluates multiple candidate models with CV, computes weighted train/test metrics, and fits best model on full data.

## Parameters
- `models` (`dict[str, Any] | None`): Optional model dictionary; defaults include 10 sklearn models.
- `label_col` (`str`): Label column in feature table.
- `cv` (`int`): Number of CV folds (default 10).
- `scoring` (`str`): Selection metric: accuracy/f1/recall variants.
- `exclude_noise` (`bool`): Whether to exclude rows marked as noise.

## Returns
- `dict model package including best model, feature columns, scores, and cv_results.`
