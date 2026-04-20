from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poremind import __version__, create_analysis_object


def _make_trace(path: Path, seed: int):
    rng = np.random.default_rng(seed)
    sr = 10000
    t = np.arange(0, 1.0, 1 / sr)
    sig = rng.normal(0, 0.3, size=t.shape)
    sig[2000:2060] -= 4.0
    sig[6500:6580] -= 5.0
    pd.DataFrame({"time": t, "current": sig}).to_csv(path, index=False)


def _make_trace_up(path: Path, seed: int):
    rng = np.random.default_rng(seed)
    sr = 10000
    t = np.arange(0, 1.0, 1 / sr)
    sig = rng.normal(-5.0, 0.2, size=t.shape)
    sig[2200:2280] += 2.5
    sig[7000:7080] += 2.0
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
        exclude_current=False,
    )

    feat_df = analysis.extract_features()
    assert len(feat_df) > 0
    assert {"left_baseline", "right_baseline", "blockade_ratio", "channel", "sweep", "sample_id", "segment_skew", "segment_kurt", "peak_factor"}.issubset(feat_df.columns)
    feat_limited = analysis.extract_features(max_event_per_sample=1)
    assert feat_limited.groupby("trace_id").size().max() <= 1
    with pytest.raises(ValueError):
        analysis.extract_features(max_event_per_sample=0)
    feat_df = analysis.extract_features()
    assert len(feat_df) > 0

    analysis.filter_events(
        method="blockade_gmm",
        parameters={"prior_mean": {"A1": 0.2, "B1": 0.2}},
    )
    assert analysis.feature_df is not None
    filtered = analysis.feature_df
    assert "quality_tag" in filtered.columns
    assert analysis.filtered_df is not None
    assert set(analysis.filtered_df["quality_tag"].unique()).issubset({"valid"})
    expected_noise = pd.Series(False, index=analysis.feature_df.index)
    for sample_key, idx in analysis.feature_df.groupby("sample_id").groups.items():
        sub_df = analysis.feature_df.loc[idx].copy()
        valid_mask = analysis._blockade_gmm_mask(
            sub_df,
            rm_index=None,
            blockade_col="blockade_ratio",
            dwell_col="duration_s",
            n_components=2,
            visualize=False,
            prior_mean=0.2,
        )
        expected_noise.loc[idx] = ~valid_mask
    in_lim = (analysis.feature_df["blockade_ratio"].to_numpy(dtype=float) >= 0.1) & (
        analysis.feature_df["blockade_ratio"].to_numpy(dtype=float) <= 1.0
    )
    expected_quality = np.where((~expected_noise.to_numpy()) & in_lim, "valid", "noise")
    assert np.array_equal(filtered["quality_tag"].to_numpy(), expected_quality)
    analysis.filter_events(method="isolation_forest")
    assert analysis.feature_df is not None
    assert "quality_tag" in analysis.feature_df.columns
    analysis.filter_events(method="lof")
    assert analysis.feature_df is not None
    assert "quality_tag" in analysis.feature_df.columns
    analysis.filter_events(
        method="blockade_gmm",
        parameters={"prior_mean": {"A1": 0.2, "B1": 0.2}},
        blockage_lim=(0.0, 2.0),
    )
    assert analysis.filtered_df is not None
    if len(analysis.filtered_df) == 0:
        analysis.filtered_df = analysis.feature_df.copy()
        analysis.filtered_df["is_noise"] = False
        analysis.filtered_df["quality_tag"] = "valid"
    pca_df = analysis.do_pca(feature_cols=["duration_s", "blockade_ratio"], data="filtered")
    assert {"PC1", "PC2"}.issubset(pca_df.columns)
    tsne_df = analysis.do_tsne(feature_cols=["duration_s", "blockade_ratio"], data="filtered", perplexity=5.0, n_iter=300)
    assert {"TSNE1", "TSNE2"}.issubset(tsne_df.columns)
    try:
        umap_df = analysis.do_umap(feature_cols=["duration_s", "blockade_ratio"], data="filtered", n_neighbors=5)
        assert {"UMAP1", "UMAP2"}.issubset(umap_df.columns)
    except ImportError:
        pass

    simple_events = analysis.detect_events_simple(
        sample_id="A1",
        current="denoise",
        start_ms=0.0,
        end_ms=300.0,
        detect_method="threshold",
        detect_params={"sigma_k": 3.0, "min_duration_s": 0.001},
    )
    assert "A1" in simple_events
    assert analysis.detect_events_simple_object == simple_events
    _ = analysis.detect_events_simple(
        sample_id="A1",
        current="raw",
        start_ms=0.0,
        end_ms=300.0,
        detect_method="threshold",
        detect_direction="up",
        baseline_method="global_quantile",
        baseline_params={"q": 0.5},
        detect_params={"sigma_k": 3.0, "min_duration_s": 0.0, "noise_method": "std"},
        merge_event=True,
        merge_event_params={"merge_gap_ms": 0.2},
    )
    _ = analysis.detect_events_simple(
        sample_id="A1",
        current="raw",
        detect_method="threshold",
        exclude_current=False,
    )
    with pytest.raises(ValueError):
        analysis.detect_events_simple(
            sample_id="A1",
            current="raw",
            detect_method="threshold",
            exclude_current=True,
            exclude_current_params={"min": 1e9, "max": None},
        )

    pkg = analysis.build_best_model(cv=2, scoring="accuracy")
    assert "best_model" in pkg
    assert pkg["best_model"] in analysis.model_cv_results
    assert {"duration_s", "blockade_ratio", "segment_std", "segment_skew", "segment_kurt"} == set(pkg["feature_cols"])
    assert "all_samples_feature_pred" in pkg
    assert "best_model_pred" in pkg["all_samples_feature_pred"].columns
    try:
        _ = analysis.pl.model_metric_bar(metric="accuracy", split="test")
        _ = analysis.pl.model_cm(model_name=pkg["best_model"], split="test")
        _ = analysis.pl.plot_2d(data="filtered", value="label")
        _ = analysis.pl.plot_3d(data="filtered", value="label")
        _ = analysis.pl.stacked_bar(group_col="sample_id", value_col="label", data="filtered")
        _ = analysis.pl.box_significance(group_col="label", value_col="blockade_ratio", data="filtered", method="ttest")
        _ = analysis.plot.event_current_simple(sample_id="A1", start_event=1, end_event=2, ylim=(-10, 10))
        _ = analysis.plot.event_current(sample_id="A1", start_event=1, end_event=2, ylim=(-10, 10))
    except ImportError:
        pass

    try:
        import torch  # noqa: F401

        dl_pkg = analysis.build_DL_model(
            model_name="1D-CNN",
            cv=2,
            epoch=2,
            early_stop_patience=1,
            batch_size=16,
            device="cpu",
        )
        assert "model_state_dict" in dl_pkg
        assert analysis.filtered_df is not None
        assert any(c.startswith("pred_label_1D-CNN") for c in analysis.filtered_df.columns)
        _ = analysis.pl.plot_fold_loss(model_name="1D-CNN", type="train")
        other_dl, pred_dl = analysis.classify_new_samples({"U1": new_s}, reader="csv", model="1D-CNN")
        assert other_dl.feature_df is not None
        assert any(c.startswith("pred_label_1D-CNN") for c in pred_dl.columns)
    except ImportError:
        pass

    other_analysis, pred = analysis.classify_new_samples(
        {"U1": new_s},
        reader="csv",
        custom_feature_fns={"seg": lambda x: {"ptp": float(np.max(x) - np.min(x))}},
    )
    assert hasattr(other_analysis, "pl")
    assert len(other_analysis.events) > 0
    assert "pred_label" in pred.columns
    proba_cols = [c for c in pred.columns if c.startswith("pred_proba_")]
    assert len(proba_cols) >= 2
    assert "seg_ptp" in pred.columns
    assert other_analysis.feature_df is not None
    assert "pred_label" in other_analysis.feature_df.columns
    for c in proba_cols:
        assert c in other_analysis.feature_df.columns
    try:
        _ = other_analysis.plot.event_current_label(
            sample_id="U1",
            lable_col="pred_label",
            start_event=1,
            end_event=2,
            label_offset=3.0,
            ylim=(-10, 10),
        )
    except ImportError:
        pass


def test_extract_features_up_direction_delta_and_blockade(tmp_path: Path):
    s1 = tmp_path / "UP1.csv"
    _make_trace_up(s1, 123)

    analysis = create_analysis_object(
        sample_paths={"UP1": s1},
        sample_to_group={"UP1": "UP"},
        reader="csv",
    ).load()

    analysis.denoise(method="moving_average", window=5)
    analysis.detect_events(
        detect_method="threshold",
        detect_direction="up",
        detect_params={"sigma_k": 3.0, "min_duration_s": 0.001, "noise_method": "std"},
        baseline_method="global_quantile",
        baseline_params={"q": 0.5},
        exclude_current=False,
    )
    feat_df = analysis.extract_features()
    assert len(feat_df) > 0

    r0 = feat_df.iloc[0]
    expected_delta = -float(r0["global_baseline"]) - (-float(r0["segment_mean"]))
    expected_blockade = expected_delta / (-float(r0["global_baseline"]) + 1e-12)
    assert np.isclose(float(r0["delta_i"]), expected_delta, atol=1e-9)
    assert np.isclose(float(r0["blockade_ratio"]), expected_blockade, atol=1e-9)
