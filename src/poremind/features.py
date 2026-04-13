from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .events import Event


def events_to_dataframe(events: Iterable[Event], time: np.ndarray, signal: np.ndarray) -> pd.DataFrame:
    rows = []
    for idx, e in enumerate(events):
        segment = signal[e.start_idx:e.end_idx]
        rows.append(
            {
                "event_id": idx,
                "start_idx": e.start_idx,
                "end_idx": e.end_idx,
                "start_time_s": float(time[e.start_idx]),
                "end_time_s": float(time[e.end_idx - 1]),
                "duration_s": e.dwell_time_s,
                "baseline_local": e.baseline_local,
                "delta_i": e.delta_i,
                "snr": e.snr,
                "segment_mean": float(np.mean(segment)),
                "segment_std": float(np.std(segment)),
                "segment_min": float(np.min(segment)),
            }
        )
    return pd.DataFrame(rows)


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = {"event_id", "label"}
    return [c for c in df.columns if c not in blocked and pd.api.types.is_numeric_dtype(df[c])]
