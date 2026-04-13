from __future__ import annotations

import numpy as np


def moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return x.copy()
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def median_filter(x: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return x.copy()
    half = window // 2
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + half + 1)
        out[i] = np.median(x[lo:hi])
    return out


def remove_slow_drift(x: np.ndarray, window: int = 1001) -> np.ndarray:
    baseline = moving_average(x, window=window)
    return x - baseline + np.median(baseline)


def preprocess_signal(x: np.ndarray, method: str = "moving_average", **kwargs) -> np.ndarray:
    methods = {
        "none": lambda sig, **_: sig,
        "moving_average": moving_average,
        "median": median_filter,
        "drift_corrected_moving_average": lambda sig, **k: moving_average(remove_slow_drift(sig, window=k.get("drift_window", 1001)), window=k.get("smooth_window", 5)),
    }
    if method not in methods:
        raise ValueError(f"unsupported preprocess method: {method}; options={list(methods)}")
    return methods[method](x, **kwargs)
