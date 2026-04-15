# `MultiSampleAnalysis.filter_events`
**Module:** `workflow.py`

Flags noisy events using blockade_gmm (default) or alternative anomaly detectors.

## Parameters
- `method` (`str`): Filtering method. Default is `blockade_gmm`; supported: `blockade_gmm`, `peak_detection`, `isolation_forest`, `lof`.
- `parameters` (`dict[str, Any] | None`): Method-specific parameters.  
  Defaults per method:
  - `blockade_gmm` / `peak_detection`: `{"n_components": 2, "prior_mean": None, "blockade_col": "blockade_ratio", "dwell_col": "duration_s", "rm_index": None, "visualize": False}`
  - `isolation_forest`: `{"contamination": 0.05, "feature_cols": ["duration_s", "blockade_ratio", "segment_skew", "segment_kurt"]}`
  - `lof`: `{"contamination": 0.05, "feature_cols": ["duration_s", "blockade_ratio", "segment_skew", "segment_kurt"]}`
- `blockage_lim` (`tuple[float, float]`): Blockade-ratio hard-threshold range applied first; only events within this range enter the chosen anomaly-filtering method. Default `(0.1, 1.0)`.

## Returns
- `None`.

## Notes
- Filtering is performed **per sample** (grouped by `sample_id`; fallback to `trace_id`), not on all samples pooled together.
- `self.feature_df` is updated in-place to include `is_noise` and `quality_tag`.
- `self.filtered_df` stores only events tagged as `valid`.
