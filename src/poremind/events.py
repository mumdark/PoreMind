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


def _mask_to_events(mask: np.ndarray, baseline: np.ndarray, signal: np.ndarray, sampling_rate_hz: float, min_duration_s: float) -> List[Event]:
    min_samples = max(1, int(min_duration_s * sampling_rate_hz))
    sigma = float(np.std(signal - baseline)) + 1e-12
    events: List[Event] = []
    i = 0
    n = len(signal)

    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i < n and mask[i]:
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
    mask = residual < threshold
    return _mask_to_events(mask, baseline, signal, sampling_rate_hz, min_duration_s)


def detect_events_cusum(
    signal: np.ndarray,
    baseline: np.ndarray,
    sampling_rate_hz: float,
    drift: float = 0.02,
    threshold: float = 8.0,
    min_duration_s: float = 0.0002,
) -> List[Event]:
    """One-sided CUSUM on standardized residual for blockade-like (negative) events."""
    residual = signal - baseline
    sigma = float(np.std(residual)) + 1e-12
    z = residual / sigma

    s_neg = np.zeros_like(z)
    for i in range(1, len(z)):
        s_neg[i] = min(0.0, s_neg[i - 1] + z[i] + drift)
    mask = s_neg < -abs(threshold)
    return _mask_to_events(mask, baseline, signal, sampling_rate_hz, min_duration_s)


def detect_events_pelt(
    signal: np.ndarray,
    baseline: np.ndarray,
    sampling_rate_hz: float,
    model: str = "l2",
    penalty: float = 8.0,
    sigma_k: float = 3.0,
    min_duration_s: float = 0.0002,
) -> List[Event]:
    """PELT change-point segmentation + residual thresholding within segments."""
    try:
        import ruptures as rpt  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("detect_events_pelt requires ruptures. Install with `pip install ruptures`.") from exc

    residual = signal - baseline
    algo = rpt.Pelt(model=model).fit(residual)
    bkps = algo.predict(pen=penalty)

    mask = np.zeros_like(signal, dtype=bool)
    start = 0
    sigma = float(np.std(residual)) + 1e-12
    thr = -abs(sigma_k) * sigma
    for end in bkps:
        seg = residual[start:end]
        if len(seg):
            mask[start:end] = seg < thr
        start = end
    return _mask_to_events(mask, baseline, signal, sampling_rate_hz, min_duration_s)


def detect_events_hmm(
    signal: np.ndarray,
    baseline: np.ndarray,
    sampling_rate_hz: float,
    n_components: int = 2,
    covariance_type: str = "diag",
    n_iter: int = 200,
    min_duration_s: float = 0.0002,
) -> List[Event]:
    """Gaussian HMM on residual; state with lower mean is treated as event state."""
    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("detect_events_hmm requires hmmlearn. Install with `pip install hmmlearn`.") from exc

    residual = signal - baseline
    X = residual.reshape(-1, 1)
    hmm = GaussianHMM(n_components=n_components, covariance_type=covariance_type, n_iter=n_iter, random_state=42)
    states = hmm.fit_predict(X)
    means = hmm.means_.reshape(-1)
    event_state = int(np.argmin(means))
    mask = states == event_state
    return _mask_to_events(mask, baseline, signal, sampling_rate_hz, min_duration_s)
