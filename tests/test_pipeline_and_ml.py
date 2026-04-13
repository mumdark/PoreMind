from pathlib import Path

import numpy as np
import pandas as pd

from poremind import __version__, create_analysis_object


def _make_trace(path: Path, seed: int):
    rng = np.random.default_rng(seed)
    sr = 10000
    t = np.arange(0, 1.0, 1 / sr)
    sig = rng.normal(0, 0.3, size=t.shape)
    sig[2000:2060] -= 4.0
    sig[6500:6580] -= 5.0
    pd.DataFrame({"time": t, "current": sig}).to_csv(path, index=False)


def test_object_workflow_end_to_end(tmp_path: Path):
    assert __version__

    s1 = tmp_path / "A1.csv"
    s2 = tmp_path / "B1.csv"
    new_s = tmp_path / "U1.csv"
    _make_trace(s1, 42)
    _make_trace(s2, 7)
    _make_trace(new_s, 100)

    analysis = create_analysis_object(
        sample_paths={"A1": s1, "B1": s2},
        sample_to_group={"A1": "A", "B1": "B"},
        reader="csv",
    ).load()

    assert hasattr(analysis, "pl")
    analysis.denoise(method="moving_average", window=5)
    preview = analysis.preview_signal("A1", start_s=0.0, end_s=0.2)
    assert not preview.empty

    defaults = analysis._default_detect_params("cusum")
    assert {"drift", "threshold", "min_duration_s"}.issubset(defaults.keys())

    analysis.detect_events(
        detect_method="threshold",
        detect_params={"sigma_k": 3.5, "min_duration_s": 0.001},
        baseline_method="rolling_quantile",
        baseline_params={"window": 201, "q": 0.5},
    )

    feat_df = analysis.extract_features()
    assert len(feat_df) > 0
    assert {"left_baseline", "right_baseline", "blockade_ratio", "channel", "sweep", "sample_id", "segment_skew", "segment_kurt", "peak_factor"}.issubset(feat_df.columns)

    filtered = analysis.filter_events(method="isolation_forest", contamination=0.1)
    assert "quality_tag" in filtered.columns

    pkg = analysis.build_best_model(cv=2, scoring="accuracy")
    assert "best_model" in pkg
    assert pkg["best_model"] in analysis.model_cv_results
    try:
        _ = analysis.pl.model_metric_bar(metric="accuracy", split="test")
        _ = analysis.pl.model_cm(model_name=pkg["best_model"], split="test")
    except ImportError:
        pass

    pred = analysis.classify_new_samples({"U1": new_s}, reader="csv")
    assert "pred_label" in pred.columns
