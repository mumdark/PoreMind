from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Event:
    start_idx: int
    end_idx: int
    baseline_local: float
    delta_i: float
    dwell_time_s: float
    snr: float


def detect_events_threshold(
    signal: np.ndarray,
    baseline: np.ndarray,
    sampling_rate_hz: float,
    sigma_k: float = 5.0,
    min_duration_s: float = 0.0002,
) -> List[Event]:
    residual = signal - baseline
    sigma = float(np.std(residual)) + 1e-12
    threshold = -sigma_k * sigma
    is_event = residual < threshold

    min_samples = max(1, int(min_duration_s * sampling_rate_hz))
    events: List[Event] = []
    i = 0
    n = len(signal)

    while i < n:
        if not is_event[i]:
            i += 1
            continue
        start = i
        while i < n and is_event[i]:
            i += 1
        end = i

        if (end - start) < min_samples:
            continue

        seg_signal = signal[start:end]
        seg_base = baseline[start:end]
        delta_i = float(np.mean(seg_base - seg_signal))
        local_base = float(np.mean(seg_base))
        dwell = (end - start) / sampling_rate_hz
        snr = delta_i / sigma
        events.append(Event(start, end, local_base, delta_i, dwell, snr))

    return events
