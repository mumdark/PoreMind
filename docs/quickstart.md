# Quick Start (Notebook-aligned)

This page follows the same step order as:

- `notebooks/step_by_step_analysis.ipynb`

## Step 0: Data init

```python
from poremind import create_analysis_object

sample_paths = {
    'A8': 'example_data\\A8_20240628_0009.abf',
    'A16': 'example_data\\A16_20240417_0006.abf'
}
sample_to_group = {
    'A8': 'A8',
    'A16': 'A16',
}
analysis = create_analysis_object(sample_paths, sample_to_group=sample_to_group, reader='abf').load()
```

## Step 1: Denoise

```python
analysis.denoise()
analysis.preview_signal(next(iter(analysis.traces.keys())), start_s=0.0, end_s=0.002).head()
```

## Step 2: Event detection

```python
analysis.detect_events_simple(
    detect_method='threshold',
    sample_id='A8__ch0_sw0',
    current='denoise'
)

analysis = analysis.detect_events(
    detect_method='threshold'
)
```

## Step 3: Feature extract

```python
def custom_shape_features(seg):
    import numpy as np
    return {
        'p2p': float(np.max(seg) - np.min(seg)),
        'energy': float(np.sum(seg ** 2))
    }

analysis.extract_features(
    custom_feature_fns={
        'custom': custom_shape_features
    }
)
```

## Step 4: Events filter

```python
analysis.filter_events(
    method='blockade_gmm',
    parameters={'n_components': 2, 'prior_mean': None},
    blockage_lim=(0.1, 1.0)
)
```

## Step 5: Model training

```python
feature_cols = ["duration_s", "blockade_ratio", "segment_std", "segment_skew", "segment_kurt"]
best_pkg = analysis.build_best_model(cv=10, scoring='accuracy', feature_cols=feature_cols)
analysis.pl.model_cm(model_name=best_pkg['best_model'], split='test')
analysis.pl.model_metric_bar(metric='accuracy', split='test')
```

## Step 5.1: Model training (neural network)

```python
dl_pkg = analysis.build_DL_model(
    model=None,
    model_name='1D-CNN',
    feature_cols=feature_cols,
    cv=10,
)
analysis.pl.plot_fold_loss(model_name='1D-CNN')
```

## Step 6: New sample classify

```python
new_paths = {'unknown_01': 'example_data\\A16_20240410_0012.abf'}
new_analysis, pred = analysis.classify_new_samples(
    new_paths,
    reader='abf',
    model=best_pkg['best_model']
)
```

---

For full runnable cells and all plotting examples, open the original notebook directly:

- `notebooks/step_by_step_analysis.ipynb`
