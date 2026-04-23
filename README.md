# PoreMind

[English](./README.md) | [中文](./README.zh.md)

API-oriented single-molecule nanopore analysis toolkit supporting a stepwise multi-sample workflow:

1. Signal denoising (multiple methods)
2. Event detection (multiple methods)
3. Event feature extraction (built-in + custom)
4. Abnormal event filtering (`noise` labeling)
5. Multi-model 10-fold comparison and best-model selection
6. Event-level classification for new samples


## Local Web UI (Gradio Blocks)

Run:

```bash
poremind-ui
# or
python -m ui.app
```

The Web UI supports interactive preprocessing, event detection tuning, feature analysis, model training, prediction, and reproducible export.

## Documentation

- Method framework (English): `docs/nanopore_single_molecule_framework.md`
- Chinese guide: `README.zh.md`

## Installation

```bash
conda create -n poremind python=3.10 -y
conda activate poremind
pip install -e . # -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
pip install pyabf
```

## Quick Usage (ABF Input)

```python
from poremind import create_analysis_object

sample_paths = {
    "std_A_01": "std_A_01.abf",
    "std_B_01": "std_B_01.abf",
}
sample_to_group = {
    "std_A_01": "A",
    "std_B_01": "B",
}

analysis = create_analysis_object(
    sample_paths,
    sample_to_group=sample_to_group,
    reader="abf",
).load()

analysis.denoise()  # default: butterworth_filtfilt
analysis.detect_events(detect_method="threshold")  # options: threshold / zscore_threshold / cusum / pelt / hmm
# optional direction: detect_direction="down" (default) or "up"
# optional baseline: baseline_method="global_quantile", baseline_params={"q": 0.5}
# optional event merge: merge_event=True, merge_event_params={"merge_gap_ms": 0.2}
# optional current filtering: exclude_current=True, exclude_current_params={"min": 0, "max": None}

# quick parameter tuning in a local time window (default 0-1000 ms)
simple_events = analysis.detect_events_simple(
    sample_id=None,
    current="denoise",
    start_ms=0.0,
    end_ms=1000.0,
    detect_method="threshold",
)

# plotting module (pl): default 0-1 ms
analysis.pl.current(sample_id=None, current="denoise", start_ms=0.0, end_ms=1.0, width=10, height=3)
analysis.plot.event_current_simple(sample_id=None, current="denoise", start_event=1, end_event=5, ylim=None)
analysis.plot.event_current(sample_id=None, current="denoise", start_event=1, end_event=5, ylim=None)

features = analysis.extract_features(max_event_per_sample=None)  # limit top-N events per sample; None means all
analysis.filter_events(method="blockade_gmm", parameters={"n_components": 2, "prior_mean": None}, blockage_lim=(0.1, 1.0))
analysis.do_pca(feature_cols=["duration_s", "blockade_ratio"], data="filtered")
analysis.do_tsne(feature_cols=["duration_s", "blockade_ratio"], data="filtered")
# analysis.do_umap(feature_cols=["duration_s", "blockade_ratio"], data="filtered")  # requires umap-learn
best_pkg = analysis.build_best_model(cv=10, scoring="accuracy")
analysis.pl.model_cm(model_name=best_pkg["best_model"], split="test")
analysis.pl.model_metric_bar(metric="accuracy", split="test")
analysis.pl.plot_2d(data="filtered", value="label")
analysis.pl.plot_3d(data="filtered", value="label")
analysis.pl.stacked_bar(group_col="sample_id", value_col="label", data="filtered")
analysis.pl.box_significance(group_col="label", value_col="blockade_ratio", data="filtered", method="ttest")
dl_pkg = analysis.build_DL_model(model_name="1D-CNN", device="cuda", epoch=30, batch_size=64)
analysis.pl.plot_fold_loss(model_name="1D-CNN", type="train")
new_analysis, pred = analysis.classify_new_samples(
    {"unknown_01": "unknown_01.abf"},
    reader="abf",
    custom_feature_fns={"seg": lambda x: {"ptp": float(x.max() - x.min())}},
    model="Random Forest",
)
# same interface supports DL models:
new_analysis_dl, pred_dl = analysis.classify_new_samples({"unknown_01": "unknown_01.abf"}, reader="abf", model="1D-CNN")
# new_analysis can be used directly for visualization: new_analysis.plot.event_current(...) / new_analysis.pl.plot_2d(...)
# pred contains pred_label and per-class probability columns (pred_proba_<class>)
```

## Notes

- ABF mode traverses all channels and sweeps in each file and outputs `channel` and `sweep` columns in the event table.
- Default denoising method is `butterworth_filtfilt` (zero-phase filtering, no phase delay), requiring `scipy`.
- Event detection supports `threshold`, `zscore_threshold`, `cusum`, `pelt`, and `hmm` with default parameters.
- Event direction supports `detect_direction="down"` (default) or `detect_direction="up"`.
- Baseline supports `baseline_method="global_quantile"` with `baseline_params={"q": xx}` (default `q=0.5`, global median).
- Event merging is supported via `merge_event=True` + `merge_event_params={"merge_gap_ms": xx}` to merge adjacent events within `xx` ms.
- Default `min_duration_s=0`; `rolling_quantile` default is `window=10000, q=0.5`.
- Default noise scale estimation is `noise_method="mad"` (switchable to `std`).
- By default, `exclude_current=True`: for `up`, range is `(-inf, 0)`; for `down`, range is `(0, +inf)`; if valid points after filtering are `<=1`, an error is raised.
- In `extract_features`, `delta_i` and `blockade_ratio` are direction-normalized according to `detect_direction` (`up` uses sign-adjusted expansion).
- `filter_events` is configured by `method + parameters`; default is `blockade_gmm` with `n_components=2, prior_mean=None`.
- `isolation_forest` / `lof` default features: `duration_s`, `blockade_ratio`, `segment_skew`, `segment_kurt`.
- Default `blockage_lim` is `(0.1, 1.0)` and is applied as a hard threshold before model-based filtering; `filter_events` adds `quality_tag` to `analysis.feature_df`, and `analysis.filtered_df` keeps only `valid` events.
- `detect_events_simple` is provided for preliminary method selection and parameter tuning in local windows.
- Results from `detect_events_simple` are stored in `analysis.detect_events_simple_object` (compatible with `analysis.simple_events`).
- `detect_events` / `detect_events_simple` display per-sample progress bars when `tqdm` is installed.
- Default modeling candidates include RF / LR / SVM / MLP / ElasticNet / Lasso / Decision Tree / LDA / AdaBoost / Gaussian Naive Bayes.
- `analysis.plot` is an alias of `analysis.pl`; supports `analysis.plot.event_current_simple` / `analysis.plot.event_current` for event-range waveform visualization (red dashed lines mark event boundaries).
- Also supports `analysis.pl.plot_2d` / `analysis.pl.plot_3d` (and compatibility aliases `getattr(analysis.pl, "2d_plot")` / `getattr(analysis.pl, "3d_plot")`) for 2D/3D feature visualization.

Full step-by-step notebook: `notebooks/step_by_step_analysis.ipynb`


## Citation & License

- Please cite this repository and the exact version/commit in publications.
- License intent: MIT-style permissive use (including commercial and non-commercial) with attribution.
- If you require non-commercial-only terms, use a different license policy because MIT does not restrict commercial usage.

## contact
[mudark: Email](mailto:ningyan1212@gmail.com)
