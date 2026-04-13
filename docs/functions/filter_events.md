# `MultiSampleAnalysis.filter_events`
**Module:** `workflow.py`

Flags noisy events using blockade_gmm (default) or alternative anomaly detectors.

## Parameters
- `method` (`str`): Filtering method: blockade_gmm (default), isolation_forest, lof.
- `contamination` (`float`): Contamination ratio for isolation/LOF methods.
- `feature_cols` (`list[str] | None`): Feature subset for anomaly detectors.
- `blockade_col` (`str`): Blockade feature column used by blockade_gmm.
- `dwell_col` (`str`): Dwell-time feature column used by blockade_gmm visualization.
- `rm_index` (`np.ndarray | None`): Optional boolean mask to preselect rows for GMM fitting.
- `n_components` (`int`): Gaussian mixture component count.
- `prior_mean` (`float | None`): Optional prior mean used to choose target GMM component.
- `visualize` (`bool`): Whether to visualize density/scatter during blockade_gmm.

## Returns
- `pandas.DataFrame with is_noise/quality_tag columns.`
