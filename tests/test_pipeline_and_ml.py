from pathlib import Path

import numpy as np
import pandas as pd

from poremind.ml import LabeledDataset, predict_events, train_event_classifier
from poremind.pipeline import AnalysisConfig, analyze_abf_to_event_df


def test_csv_pipeline_and_ml(tmp_path: Path):
    sr = 10000
    t = np.arange(0, 1.0, 1 / sr)
    signal = np.random.normal(0, 0.3, size=t.shape)

    signal[2000:2050] -= 4.0
    signal[6000:6080] -= 5.0

    csv_path = tmp_path / "trace.csv"
    pd.DataFrame({"time": t, "current": signal}).to_csv(csv_path, index=False)

    cfg = AnalysisConfig(reader="csv", detect_params={"sigma_k": 3.5, "min_duration_s": 0.001})
    df = analyze_abf_to_event_df(csv_path, config=cfg)

    assert len(df) >= 2
    assert {"start_time_s", "end_time_s", "delta_i", "snr"}.issubset(df.columns)

    labeled = df.copy()
    labeled["label"] = np.where(labeled["delta_i"] > labeled["delta_i"].median(), "A", "B")
    pkg = train_event_classifier(LabeledDataset(labeled), model_name="random_forest", model_params={"n_estimators": 10, "random_state": 0})
    pred = predict_events(pkg, df)

    assert "pred_label" in pred.columns
