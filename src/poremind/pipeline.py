from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .baseline import estimate_baseline
from .events import detect_events_threshold
from .features import events_to_dataframe
from .io import read_abf, read_csv
from .preprocess import preprocess_signal


@dataclass
class AnalysisConfig:
    reader: Literal["abf", "csv"] = "abf"
    preprocess_method: str = "drift_corrected_moving_average"
    preprocess_params: dict[str, Any] = field(default_factory=lambda: {"drift_window": 1001, "smooth_window": 5})
    baseline_method: str = "rolling_quantile"
    baseline_params: dict[str, Any] = field(default_factory=lambda: {"window": 501, "q": 0.5})
    detect_params: dict[str, Any] = field(default_factory=lambda: {"sigma_k": 5.0, "min_duration_s": 2e-4})


def analyze_abf_to_event_df(path: str | Path, config: AnalysisConfig | None = None, **reader_kwargs: Any) -> pd.DataFrame:
    config = config or AnalysisConfig()

    if config.reader == "abf":
        trace = read_abf(path, **reader_kwargs)
    elif config.reader == "csv":
        trace = read_csv(path, **reader_kwargs)
    else:
        raise ValueError("reader must be 'abf' or 'csv'")

    pre = preprocess_signal(trace.current, method=config.preprocess_method, **config.preprocess_params)
    baseline = estimate_baseline(pre, method=config.baseline_method, **config.baseline_params)
    events = detect_events_threshold(pre, baseline, trace.sampling_rate_hz, **config.detect_params)
    df = events_to_dataframe(events, trace.time, pre)

    df["source"] = trace.source
    df["sampling_rate_hz"] = trace.sampling_rate_hz
    return df
