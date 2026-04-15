from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import AdaBoostClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from .baseline import estimate_baseline
from .events import (
    Event,
    detect_events_cusum,
    detect_events_hmm,
    detect_events_pelt,
    detect_events_threshold,
)
from .features import select_feature_columns
from .io import Trace, read_abf, read_abf_all, read_csv
from .pl import PlotAccessor
from .preprocess import preprocess_signal

FeatureFn = Callable[[np.ndarray], dict[str, float]]


@dataclass
class MultiSampleAnalysis:
    sample_paths: dict[str, str | Path]
    sample_to_group: dict[str, str] | None = None
    reader: str = "abf"
    reader_kwargs: dict[str, Any] = field(default_factory=dict)

    traces: dict[str, Trace] = field(default_factory=dict)
    denoised: dict[str, np.ndarray] = field(default_factory=dict)
    baselines: dict[str, np.ndarray] = field(default_factory=dict)
    events: dict[str, list[Event]] = field(default_factory=dict)
    simple_events: dict[str, list[Event]] = field(default_factory=dict)
    detect_events_simple_object: dict[str, list[Event]] = field(default_factory=dict)
    feature_df: pd.DataFrame | None = None
    filtered_df: pd.DataFrame | None = None
    best_model_package: dict[str, Any] | None = None
    model_cv_results: dict[str, Any] = field(default_factory=dict)

    preprocess_state: dict[str, Any] = field(default_factory=dict)
    detect_state: dict[str, Any] = field(default_factory=dict)
    simple_detect_state: dict[str, Any] = field(default_factory=dict)
    feature_state: dict[str, Any] = field(default_factory=dict)
    trace_to_sample: dict[str, str] = field(default_factory=dict)
    pl: PlotAccessor = field(init=False)
    plot: PlotAccessor = field(init=False)

    def __post_init__(self) -> None:
        self.pl = PlotAccessor(self)
        self.plot = self.pl

    def load(self) -> "MultiSampleAnalysis":
        self.traces = {}
        self.trace_to_sample = {}
        for sid, path in self.sample_paths.items():
            if self.reader == "abf":
                # 默认遍历 ABF 全部 channel + sweep
                if self.reader_kwargs:
                    trace = read_abf(path, **self.reader_kwargs)
                    key = f"{sid}__ch{trace.channel}_sw{trace.sweep}"
                    self.traces[key] = trace
                    self.trace_to_sample[key] = sid
                else:
                    for trace in read_abf_all(path):
                        key = f"{sid}__ch{trace.channel}_sw{trace.sweep}"
                        self.traces[key] = trace
                        self.trace_to_sample[key] = sid
            else:
                trace = read_csv(path, **self.reader_kwargs)
                self.traces[sid] = trace
                self.trace_to_sample[sid] = sid
        return self

    # Step 1: denoise + preview/visualize
    def denoise(self, method: str = "butterworth_filtfilt", **kwargs: Any) -> "MultiSampleAnalysis":
        if not self.traces:
            self.load()
        self.preprocess_state = {"method": method, **kwargs}
        self.denoised = {sid: preprocess_signal(tr.current, method=method, **kwargs) for sid, tr in self.traces.items()}
        return self

    def preview_signal(self, sample_id: str, start_s: float = 0.0, end_s: float | None = None, max_points: int = 5000) -> pd.DataFrame:
        trace = self.traces[sample_id]
        sig = self.denoised.get(sample_id, trace.current)
        t = trace.time
        end_s = float(t[-1]) if end_s is None else end_s
        mask = (t >= start_s) & (t <= end_s)
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return pd.DataFrame(columns=["time", "current"])
        if len(idx) > max_points:
            step = max(1, len(idx) // max_points)
            idx = idx[::step]
        return pd.DataFrame({"time": t[idx], "current": sig[idx]})

    def visualize_signal(self, sample_id: str, start_s: float = 0.0, end_s: float | None = None, max_points: int = 5000):
        df = self.preview_signal(sample_id, start_s=start_s, end_s=end_s, max_points=max_points)
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            raise ImportError("visualize_signal requires matplotlib") from exc
        ax = df.plot(x="time", y="current", figsize=(10, 3), title=f"Signal preview: {sample_id}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Current")
        plt.tight_layout()
        return ax

    # Step 2: event detection (multi-method)
    def detect_events(
        self,
        detect_method: str = "threshold",
        detect_params: dict[str, Any] | None = None,
        baseline_method: str = "rolling_quantile",
        baseline_params: dict[str, Any] | None = None,
        detect_direction: str = "down",
        merge_event: bool = False,
        merge_event_params: dict[str, Any] | None = None,
        exclude_current: bool = True,
        exclude_current_params: dict[str, Any] | None = None,
    ) -> "MultiSampleAnalysis":
        if not self.denoised:
            self.denoise()
        if detect_params is None:
            detect_params = self._default_detect_params(detect_method)
        baseline_params = baseline_params or {"window": 10000, "q": 0.5}
        self.detect_state = {
            "detect_method": detect_method,
            "detect_params": detect_params,
            "baseline_method": baseline_method,
            "baseline_params": baseline_params,
            "detect_direction": detect_direction,
            "merge_event": merge_event,
            "merge_event_params": merge_event_params or {"merge_gap_ms": 0.0},
            "exclude_current": exclude_current,
            "exclude_current_params": exclude_current_params,
        }

        self.baselines = {}
        self.events = {}
        trace_items = list(self.traces.items())
        iterator = trace_items
        try:
            from tqdm.auto import tqdm  # type: ignore

            iterator = tqdm(trace_items, desc="detect_events", unit="sample")
        except Exception:
            iterator = trace_items

        for sid, tr in iterator:
            if hasattr(iterator, "set_postfix_str"):
                iterator.set_postfix_str(str(sid))
            sig = self.denoised[sid]
            stats_mask = self._build_stats_mask(
                signal=sig,
                detect_direction=detect_direction,
                exclude_current=exclude_current,
                exclude_current_params=exclude_current_params,
            )
            baseline = self._estimate_baseline(sig, method=baseline_method, baseline_params=baseline_params, stats_mask=stats_mask)
            self.baselines[sid] = baseline
            evts = self._detect_events_by_method(
                sig,
                baseline,
                tr.sampling_rate_hz,
                detect_method=detect_method,
                detect_params=detect_params,
                detect_direction=detect_direction,
                stats_mask=stats_mask,
            )
            if merge_event:
                evts = self._merge_nearby_events(
                    evts,
                    signal=sig,
                    baseline=baseline,
                    sampling_rate_hz=tr.sampling_rate_hz,
                    merge_gap_ms=float((merge_event_params or {"merge_gap_ms": 0.0}).get("merge_gap_ms", 0.0)),
                )
            self.events[sid] = evts
        return self

    @staticmethod
    def _build_stats_mask(
        signal: np.ndarray,
        detect_direction: str,
        exclude_current: bool,
        exclude_current_params: dict[str, Any] | None,
    ) -> np.ndarray:
        if not exclude_current:
            return np.ones(len(signal), dtype=bool)
        params = (exclude_current_params or {}).copy()
        if "min" not in params and "max" not in params:
            if detect_direction == "up":
                params = {"min": None, "max": 0.0}
            else:
                params = {"min": 0.0, "max": None}

        lo = params.get("min")
        hi = params.get("max")
        mask = np.ones(len(signal), dtype=bool)
        if lo is not None:
            mask &= signal > float(lo)
        if hi is not None:
            mask &= signal < float(hi)
        if int(mask.sum()) <= 1:
            raise ValueError("effective points for baseline/noise statistics must be > 1 after exclude_current filtering")
        return mask

    @staticmethod
    def _estimate_baseline(signal: np.ndarray, method: str, baseline_params: dict[str, Any], stats_mask: np.ndarray) -> np.ndarray:
        if method == "global_quantile":
            q = float(baseline_params.get("q", 0.5))
            valid = signal[stats_mask]
            if len(valid) <= 1:
                raise ValueError("effective points for global_quantile baseline must be > 1 after exclude_current filtering")
            val = float(np.quantile(valid, q))
            return np.full_like(signal, val, dtype=float)
        if method == "rolling_quantile":
            if bool(np.all(stats_mask)):
                return estimate_baseline(signal, method="rolling_quantile", **baseline_params)
            window = int(baseline_params.get("window", 10000))
            q = float(baseline_params.get("q", 0.5))
            if window <= 1:
                valid = signal[stats_mask]
                if len(valid) <= 1:
                    raise ValueError("effective points for rolling_quantile baseline must be > 1 after exclude_current filtering")
                val = float(np.quantile(valid, q))
                return np.full_like(signal, val, dtype=float)
            half = window // 2
            out = np.empty_like(signal, dtype=float)
            for i in range(len(signal)):
                lo = max(0, i - half)
                hi = min(len(signal), i + half + 1)
                local_mask = stats_mask[lo:hi]
                if int(local_mask.sum()) <= 1:
                    raise ValueError("rolling_quantile window has <=1 effective points after exclude_current filtering")
                out[i] = np.quantile(signal[lo:hi][local_mask], q)
            return out
        if method == "global_median":
            valid = signal[stats_mask]
            if len(valid) <= 1:
                raise ValueError("effective points for global_median baseline must be > 1 after exclude_current filtering")
            return np.full_like(signal, float(np.median(valid)), dtype=float)
        return estimate_baseline(signal, method=method, **baseline_params)

    @staticmethod
    def _merge_nearby_events(
        events: list[Event],
        signal: np.ndarray,
        baseline: np.ndarray,
        sampling_rate_hz: float,
        merge_gap_ms: float,
    ) -> list[Event]:
        if len(events) <= 1:
            return events
        sigma = float(np.std(signal - baseline)) + 1e-12
        res = baseline - signal
        cumsum_res = np.concatenate(([0.0], np.cumsum(res, dtype=float)))
        cumsum_base = np.concatenate(([0.0], np.cumsum(baseline, dtype=float)))
        gap_samples = max(0, int((merge_gap_ms / 1000.0) * sampling_rate_hz))
        merged: list[Event] = []
        cur_start = events[0].start_idx
        cur_end = events[0].end_idx
        for e in events[1:]:
            if e.start_idx - cur_end <= gap_samples:
                cur_end = max(cur_end, e.end_idx)
            else:
                merged.append(
                    MultiSampleAnalysis._build_event(
                        cur_start,
                        cur_end,
                        signal,
                        baseline,
                        sampling_rate_hz,
                        sigma=sigma,
                        cumsum_res=cumsum_res,
                        cumsum_base=cumsum_base,
                    )
                )
                cur_start, cur_end = e.start_idx, e.end_idx
        merged.append(
            MultiSampleAnalysis._build_event(
                cur_start,
                cur_end,
                signal,
                baseline,
                sampling_rate_hz,
                sigma=sigma,
                cumsum_res=cumsum_res,
                cumsum_base=cumsum_base,
            )
        )
        return merged

    @staticmethod
    def _build_event(
        start_idx: int,
        end_idx: int,
        signal: np.ndarray,
        baseline: np.ndarray,
        sr: float,
        sigma: float | None = None,
        cumsum_res: np.ndarray | None = None,
        cumsum_base: np.ndarray | None = None,
    ) -> Event:
        n = max(1, end_idx - start_idx)
        if cumsum_res is not None and cumsum_base is not None:
            delta_i = float((cumsum_res[end_idx] - cumsum_res[start_idx]) / n)
            baseline_local = float((cumsum_base[end_idx] - cumsum_base[start_idx]) / n)
        else:
            seg_signal = signal[start_idx:end_idx]
            seg_base = baseline[start_idx:end_idx]
            delta_i = float(np.mean(seg_base - seg_signal))
            baseline_local = float(np.mean(seg_base))
        dwell = (end_idx - start_idx) / sr
        if sigma is None:
            sigma = float(np.std(signal - baseline)) + 1e-12
        snr = delta_i / sigma
        return Event(start_idx=start_idx, end_idx=end_idx, baseline_local=baseline_local, delta_i=delta_i, dwell_time_s=dwell, snr=snr)

    @staticmethod
    def _noise_scale(residual: np.ndarray, method: str = "mad", stats_mask: np.ndarray | None = None) -> float:
        if stats_mask is not None:
            residual = residual[stats_mask]
        if len(residual) <= 1:
            raise ValueError("effective points for noise estimation must be > 1 after exclude_current filtering")
        method = method.lower()
        if method == "mad":
            med = float(np.median(residual))
            mad = float(np.median(np.abs(residual - med)))
            return 1.4826 * mad + 1e-12
        if method == "std":
            return float(np.std(residual)) + 1e-12
        raise ValueError("noise_method must be 'mad' or 'std'")

    @staticmethod
    def _detect_events_by_method(
        signal: np.ndarray,
        baseline: np.ndarray,
        sampling_rate_hz: float,
        detect_method: str,
        detect_params: dict[str, Any],
        detect_direction: str = "down",
        stats_mask: np.ndarray | None = None,
    ) -> list[Event]:
        if detect_direction not in {"down", "up"}:
            raise ValueError("detect_direction must be 'down' or 'up'")
        work_signal = signal if detect_direction == "down" else -signal
        work_baseline = baseline if detect_direction == "down" else -baseline

        if detect_method == "threshold":
            if detect_direction == "down":
                return detect_events_threshold(signal, baseline, sampling_rate_hz, stats_mask=stats_mask, **detect_params)
            sigma_k = float(detect_params.get("sigma_k", 5.0))
            min_duration_s = float(detect_params.get("min_duration_s", 0.0))
            noise_method = str(detect_params.get("noise_method", "mad"))
            residual = signal - baseline
            sigma = MultiSampleAnalysis._noise_scale(residual, method=noise_method, stats_mask=stats_mask)
            mask = residual > sigma_k * sigma
            return MultiSampleAnalysis._mask_to_events(mask, baseline, signal, sampling_rate_hz, min_duration_s=min_duration_s)
        if detect_method == "zscore_threshold":
            residual = signal - baseline
            noise_method = str(detect_params.get("noise_method", "mad"))
            z = residual / MultiSampleAnalysis._noise_scale(residual, method=noise_method, stats_mask=stats_mask)
            z_thr = float(detect_params.get("z", 4.0))
            min_duration_s = float(detect_params.get("min_duration_s", 0.0))
            mask = z < -z_thr if detect_direction == "down" else z > z_thr
            return MultiSampleAnalysis._mask_to_events(mask, baseline, signal, sampling_rate_hz, min_duration_s=min_duration_s)
        if detect_method == "cusum":
            return detect_events_cusum(work_signal, work_baseline, sampling_rate_hz, stats_mask=stats_mask, **detect_params)
        if detect_method == "pelt":
            return detect_events_pelt(work_signal, work_baseline, sampling_rate_hz, stats_mask=stats_mask, **detect_params)
        if detect_method == "hmm":
            return detect_events_hmm(work_signal, work_baseline, sampling_rate_hz, **detect_params)
        raise ValueError("unsupported detect method")

    def detect_events_simple(
        self,
        detect_method: str = "threshold",
        detect_params: dict[str, Any] | None = None,
        baseline_method: str = "rolling_quantile",
        baseline_params: dict[str, Any] | None = None,
        sample_id: str | None = None,
        current: str = "denoise",
        start_ms: float = 0.0,
        end_ms: float = 1000.0,
        detect_direction: str = "down",
        merge_event: bool = False,
        merge_event_params: dict[str, Any] | None = None,
        exclude_current: bool = True,
        exclude_current_params: dict[str, Any] | None = None,
    ) -> dict[str, list[Event]]:
        if not self.traces:
            self.load()
        if current not in {"denoise", "raw"}:
            raise ValueError("current must be 'denoise' or 'raw'")
        if current == "denoise" and not self.denoised:
            self.denoise()
        if detect_params is None:
            detect_params = self._default_detect_params(detect_method)
        baseline_params = baseline_params or {"window": 10000, "q": 0.5}

        self.simple_detect_state = {
            "detect_method": detect_method,
            "detect_params": detect_params,
            "baseline_method": baseline_method,
            "baseline_params": baseline_params,
            "sample_id": sample_id,
            "current": current,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "detect_direction": detect_direction,
            "merge_event": merge_event,
            "merge_event_params": merge_event_params or {"merge_gap_ms": 0.0},
            "exclude_current": exclude_current,
            "exclude_current_params": exclude_current_params,
        }

        target_ids = [sample_id] if sample_id is not None else list(self.traces.keys())
        out: dict[str, list[Event]] = {}
        iterator = target_ids
        try:
            from tqdm.auto import tqdm  # type: ignore

            iterator = tqdm(target_ids, desc="detect_events_simple", unit="sample")
        except Exception:
            iterator = target_ids

        for sid in iterator:
            if hasattr(iterator, "set_postfix_str"):
                iterator.set_postfix_str(str(sid))
            tr = self.traces[sid]
            sig_full = self.denoised[sid] if current == "denoise" else tr.current
            t_ms = tr.time * 1000.0
            idx = np.flatnonzero((t_ms >= start_ms) & (t_ms <= end_ms))
            if len(idx) == 0:
                out[sid] = []
                continue
            lo, hi = int(idx[0]), int(idx[-1]) + 1
            sig = sig_full[lo:hi]
            stats_mask = self._build_stats_mask(
                signal=sig,
                detect_direction=detect_direction,
                exclude_current=exclude_current,
                exclude_current_params=exclude_current_params,
            )
            baseline = self._estimate_baseline(sig, method=baseline_method, baseline_params=baseline_params, stats_mask=stats_mask)
            evts_local = self._detect_events_by_method(
                sig,
                baseline,
                tr.sampling_rate_hz,
                detect_method=detect_method,
                detect_params=detect_params,
                detect_direction=detect_direction,
                stats_mask=stats_mask,
            )
            if merge_event:
                evts_local = self._merge_nearby_events(
                    evts_local,
                    signal=sig,
                    baseline=baseline,
                    sampling_rate_hz=tr.sampling_rate_hz,
                    merge_gap_ms=float((merge_event_params or {"merge_gap_ms": 0.0}).get("merge_gap_ms", 0.0)),
                )
            out[sid] = [
                Event(
                    start_idx=e.start_idx + lo,
                    end_idx=e.end_idx + lo,
                    baseline_local=e.baseline_local,
                    delta_i=e.delta_i,
                    dwell_time_s=e.dwell_time_s,
                    snr=e.snr,
                )
                for e in evts_local
            ]
        self.simple_events = out
        self.detect_events_simple_object = out
        return out

    @staticmethod
    def _default_detect_params(detect_method: str) -> dict[str, Any]:
        defaults = {
            "threshold": {"sigma_k": 5.0, "min_duration_s": 0.0, "noise_method": "mad"},
            "zscore_threshold": {"z": 4.0, "min_duration_s": 0.0, "noise_method": "mad"},
            "cusum": {"drift": 0.02, "threshold": 8.0, "min_duration_s": 0.0, "noise_method": "mad"},
            "pelt": {"model": "l2", "penalty": 8.0, "sigma_k": 3.0, "min_duration_s": 0.0, "noise_method": "mad"},
            "hmm": {"n_components": 2, "covariance_type": "diag", "n_iter": 200, "min_duration_s": 0.0},
        }
        if detect_method not in defaults:
            raise ValueError(f"unsupported detect method: {detect_method}")
        return defaults[detect_method].copy()

    @staticmethod
    def _mask_to_events(mask: np.ndarray, baseline: np.ndarray, signal: np.ndarray, sr: float, min_duration_s: float) -> list[Event]:
        min_samples = max(1, int(min_duration_s * sr))
        sigma = float(np.std(signal - baseline)) + 1e-12
        if len(mask) == 0:
            return []

        m = mask.astype(np.int8, copy=False)
        d = np.diff(m)
        starts = np.flatnonzero(d == 1) + 1
        ends = np.flatnonzero(d == -1) + 1
        if bool(mask[0]):
            starts = np.r_[0, starts]
        if bool(mask[-1]):
            ends = np.r_[ends, len(mask)]

        if len(starts) == 0:
            return []

        dur = ends - starts
        valid = dur >= min_samples
        starts = starts[valid]
        ends = ends[valid]
        if len(starts) == 0:
            return []

        res = baseline - signal
        cumsum_res = np.concatenate(([0.0], np.cumsum(res, dtype=float)))
        cumsum_base = np.concatenate(([0.0], np.cumsum(baseline, dtype=float)))

        out: list[Event] = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            n = e - s
            delta_i = float((cumsum_res[e] - cumsum_res[s]) / n)
            baseline_local = float((cumsum_base[e] - cumsum_base[s]) / n)
            out.append(Event(s, e, baseline_local, delta_i, n / sr, delta_i / sigma))
        return out

    # Step 3: feature extraction (built-in + custom)
    def extract_features(
        self,
        custom_feature_fns: dict[str, FeatureFn] | None = None,
        max_event_per_sample: int | None = None,
    ) -> pd.DataFrame:
        if not self.events:
            self.detect_events()
        custom_feature_fns = custom_feature_fns or {}
        if max_event_per_sample is not None and max_event_per_sample <= 0:
            raise ValueError("max_event_per_sample must be > 0 or None")
        rows: list[dict[str, Any]] = []

        sample_items = list(self.events.items())
        sample_iterator = sample_items
        detect_direction = str(self.detect_state.get("detect_direction", "down")).lower()
        if detect_direction not in {"down", "up"}:
            detect_direction = "down"
        try:
            from tqdm.auto import tqdm  # type: ignore

            sample_iterator = tqdm(sample_items, desc="extract_features(samples)", unit="sample")
        except Exception:
            sample_iterator = sample_items

        for sid, evts in sample_iterator:
            if hasattr(sample_iterator, "set_postfix_str"):
                sample_iterator.set_postfix_str(str(sid))
            tr = self.traces[sid]
            source_sample_id = self.trace_to_sample.get(sid, sid)
            sig = self.denoised[sid]
            base = self.baselines[sid]
            global_base = float(np.median(base))
            sig_len = len(sig)
            evts_use = evts if max_event_per_sample is None else evts[:max_event_per_sample]

            # prefix sums for fast window means (left/right baseline)
            cumsum_sig = np.concatenate(([0.0], np.cumsum(sig, dtype=float)))

            def _window_mean(lo: int, hi: int, fallback_idx: int) -> float:
                if hi <= lo:
                    return float(base[fallback_idx])
                return float((cumsum_sig[hi] - cumsum_sig[lo]) / (hi - lo))

            event_iterator = evts_use
            try:
                from tqdm.auto import tqdm  # type: ignore

                event_iterator = tqdm(
                    evts_use,
                    desc=f"extract_features(events:{sid})",
                    unit="event",
                    leave=False,
                )
            except Exception:
                event_iterator = evts_use

            for i, e in enumerate(event_iterator):
                if hasattr(event_iterator, "set_postfix_str"):
                    event_iterator.set_postfix_str(f"event={i + 1}")
                seg = sig[e.start_idx:e.end_idx]
                seg_mean = float(np.mean(seg))
                seg_std = float(np.std(seg))
                seg_min = float(np.min(seg))
                seg_max = float(np.max(seg))
                seg_abs_max = float(np.max(np.abs(seg)))

                left_lo = max(0, e.start_idx - 50)
                left_hi = e.start_idx
                right_lo = e.end_idx
                right_hi = min(sig_len, e.end_idx + 50)
                left_base = _window_mean(left_lo, left_hi, e.start_idx)
                right_base = _window_mean(right_lo, right_hi, e.end_idx - 1)

                if detect_direction == "up":
                    delta_i_feature = float((-global_base) - (-seg_mean))
                    blockade_ratio = float(delta_i_feature / ((-global_base) + 1e-12))
                else:
                    delta_i_feature = float(global_base - seg_mean)
                    blockade_ratio = float(delta_i_feature / (global_base + 1e-12))
                centered = seg - seg_mean
                denom = seg_std + 1e-12
                skew = float(np.mean((centered / denom) ** 3))
                kurt = float(np.mean((centered / denom) ** 4))
                rms = float(np.sqrt(np.mean(seg ** 2))) + 1e-12
                peak_factor = float(seg_abs_max / rms)
                row = {
                    "trace_id": sid,
                    "sample_id": source_sample_id,
                    "channel": tr.channel,
                    "sweep": tr.sweep,
                    "event_id": i,
                    "start_idx": e.start_idx,
                    "end_idx": e.end_idx,
                    "start_time_s": float(tr.time[e.start_idx]),
                    "end_time_s": float(tr.time[e.end_idx - 1]),
                    "duration_s": e.dwell_time_s,
                    "delta_i": delta_i_feature,
                    "snr": e.snr,
                    "left_baseline": left_base,
                    "right_baseline": right_base,
                    "global_baseline": global_base,
                    "blockade_ratio": blockade_ratio,
                    "segment_mean": seg_mean,
                    "segment_std": seg_std,
                    "segment_min": seg_min,
                    "segment_max": seg_max,
                    "segment_skew": skew,
                    "segment_kurt": kurt,
                    "peak_factor": peak_factor,
                }
                for name, fn in custom_feature_fns.items():
                    feats = fn(seg)
                    row.update({f"{name}_{k}": v for k, v in feats.items()})
                if self.sample_to_group and source_sample_id in self.sample_to_group:
                    row["label"] = self.sample_to_group[source_sample_id]
                rows.append(row)

        self.feature_state = {"custom_features": list(custom_feature_fns)}
        self.feature_df = pd.DataFrame(rows)
        return self.feature_df

    @staticmethod
    def _blockade_gmm_mask(
        df: pd.DataFrame,
        rm_index: np.ndarray | None = None,
        blockade_col: str = "blockade_ratio",
        dwell_col: str = "duration_s",
        n_components: int = 2,
        visualize: bool = False,
        prior_mean: float | None = None,
    ) -> np.ndarray:
        if blockade_col not in df.columns:
            raise ValueError(f"missing blockade column: {blockade_col}")
        if dwell_col not in df.columns:
            raise ValueError(f"missing dwell column: {dwell_col}")

        X = df.copy()
        if rm_index is None:
            rm_index = np.ones(len(X), dtype=bool)

        data = X.loc[rm_index, blockade_col].to_numpy(dtype=float)
        if len(data) < 5:
            return rm_index

        # KDE curve is kept only for optional visualization.
        grid = np.linspace(np.nanmin(data), np.nanmax(data), 256)
        bw = max(1e-6, np.std(data) * 0.2)
        density = np.sum(np.exp(-0.5 * ((grid[:, None] - data[None, :]) / bw) ** 2), axis=1)

        if visualize:
            try:
                import matplotlib.pyplot as plt
            except Exception as exc:  # pragma: no cover
                raise ImportError("blockade_gmm visualization requires matplotlib") from exc
            fig, axs = plt.subplots(2, 1, sharex=False, gridspec_kw={"height_ratios": [1, 3]}, figsize=(4, 5))
            axs[0].plot(grid, density)
            axs[0].set_ylabel("Density")
            axs[1].scatter(X[blockade_col], np.log10(X[dwell_col].to_numpy(dtype=float) + 1e-12), c=rm_index, alpha=0.5, s=4)
            axs[1].set_xlabel(blockade_col)
            axs[1].set_ylabel(f"log10({dwell_col})")
            plt.tight_layout()

        data2 = X.loc[rm_index, blockade_col].to_numpy(dtype=float).reshape(-1, 1)
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(data2)
        labels = gmm.predict(data2)

        means = gmm.means_.flatten()
        covariances = np.sqrt(gmm.covariances_).flatten()
        valid_indices = np.where((means >= 0) & (means <= 1.2))[0]
        if len(valid_indices) == 0:
            valid_indices = np.arange(len(means))

        if prior_mean is None:
            if len(valid_indices) > 1:
                target = valid_indices[np.argmin(covariances[valid_indices])]
            else:
                target = valid_indices[0]
        else:
            if len(valid_indices) > 1:
                target = valid_indices[np.argmin(np.abs(means[valid_indices] - prior_mean))]
            else:
                target = valid_indices[0]

        local = rm_index.copy()
        local_idx = np.where(rm_index)[0]
        local[local_idx] = labels == target
        return local

    # Step 4: noise/outlier flagging
    def filter_events(
        self,
        method: str = "blockade_gmm",
        parameters: dict[str, Any] | None = None,
        blockage_lim: tuple[float, float] = (0.1, 1.0),
    ) -> None:
        if self.feature_df is None:
            self.extract_features()
        assert self.feature_df is not None
        df = self.feature_df.copy()

        method = method.lower()
        default_params: dict[str, dict[str, Any]] = {
            "blockade_gmm": {
                "blockade_col": "blockade_ratio",
                "dwell_col": "duration_s",
                "rm_index": None,
                "n_components": 2,
                "prior_mean": None,
                "visualize": False,
            },
            "peak_detection": {
                "blockade_col": "blockade_ratio",
                "dwell_col": "duration_s",
                "rm_index": None,
                "n_components": 2,
                "prior_mean": None,
                "visualize": False,
            },
            "isolation_forest": {
                "contamination": 0.05,
                "feature_cols": ["duration_s", "blockade_ratio", "segment_skew", "segment_kurt"],
            },
            "lof": {
                "contamination": 0.05,
                "feature_cols": ["duration_s", "blockade_ratio", "segment_skew", "segment_kurt"],
            },
        }
        if method not in default_params:
            raise ValueError("unsupported outlier method")
        cfg = default_params[method].copy()
        if parameters:
            cfg.update(parameters)

        if len(df) == 0:
            df["is_noise"] = []
            df["quality_tag"] = []
            self.feature_df = df
            self.filtered_df = df.copy()
            return

        group_col = "sample_id" if "sample_id" in df.columns else ("trace_id" if "trace_id" in df.columns else None)
        if group_col is None:
            group_slices: list[tuple[str, pd.Index]] = [("__all__", df.index)]
        else:
            group_slices = [(str(sample_key), idx) for sample_key, idx in df.groupby(group_col).groups.items()]

        rm_series = None
        rm_index = cfg.get("rm_index")
        if rm_index is not None:
            rm_index = np.asarray(rm_index, dtype=bool)
            if len(rm_index) != len(df):
                raise ValueError("rm_index length must match feature dataframe rows")
            rm_series = pd.Series(rm_index, index=df.index)

        blockade_col = str(cfg.get("blockade_col", "blockade_ratio"))
        if blockade_col not in df.columns:
            raise ValueError(f"missing blockade column for blockage_lim filtering: {blockade_col}")
        lo, hi = float(blockage_lim[0]), float(blockage_lim[1])
        in_blockage_lim = (df[blockade_col].to_numpy(dtype=float) >= lo) & (df[blockade_col].to_numpy(dtype=float) <= hi)
        in_lim_series = pd.Series(in_blockage_lim, index=df.index)

        # Hard threshold first: out-of-range events are fixed as noise.
        df["is_noise"] = ~in_blockage_lim
        for sample_key, idx in group_slices:
            idx_in = idx[in_lim_series.loc[idx].to_numpy(dtype=bool)]
            if len(idx_in) == 0:
                continue
            sub_df = df.loc[idx_in].copy()
            sub_rm_index = None if rm_series is None else rm_series.loc[idx_in].to_numpy(dtype=bool)
            sample_prior_mean: float | None
            prior_mean_cfg = cfg.get("prior_mean")
            if isinstance(prior_mean_cfg, dict):
                sample_prior_mean = prior_mean_cfg.get(sample_key)
            else:
                sample_prior_mean = prior_mean_cfg

            if method in {"blockade_gmm", "peak_detection"}:
                valid_mask = self._blockade_gmm_mask(
                    sub_df,
                    rm_index=sub_rm_index,
                    blockade_col=str(cfg["blockade_col"]),
                    dwell_col=str(cfg["dwell_col"]),
                    n_components=int(cfg["n_components"]),
                    visualize=bool(cfg["visualize"]),
                    prior_mean=sample_prior_mean,
                )
                df.loc[idx_in, "is_noise"] = ~valid_mask
            else:
                local_feature_cols = list(cfg.get("feature_cols") or select_feature_columns(sub_df))
                X = sub_df[local_feature_cols].fillna(0.0)
                if method == "isolation_forest":
                    detector = IsolationForest(contamination=float(cfg["contamination"]), random_state=42)
                    pred = detector.fit_predict(X)
                elif method == "lof":
                    detector = LocalOutlierFactor(contamination=float(cfg["contamination"]))
                    pred = detector.fit_predict(X)
                else:
                    raise ValueError("unsupported outlier method")
                df.loc[idx_in, "is_noise"] = pred == -1

        df["quality_tag"] = np.where(~df["is_noise"], "valid", "noise")
        self.feature_df = df
        self.filtered_df = df[df["quality_tag"] == "valid"].copy()
        return

    # Step 5: model selection (10-fold)
    @staticmethod
    def _metric_config(y: pd.Series) -> dict[str, Any]:
        classes = sorted(pd.unique(y))
        if len(classes) == 2:
            return {"average": "binary", "pos_label": classes[-1], "mode": "binary"}
        return {"average": "macro", "pos_label": None, "mode": "macro"}

    @staticmethod
    def _score_value(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray, cfg: dict[str, Any]) -> float:
        if metric_name == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        if metric_name == "f1":
            if cfg["average"] == "binary":
                return float(f1_score(y_true, y_pred, average="binary", pos_label=cfg["pos_label"], zero_division=0))
            return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if metric_name == "recall":
            if cfg["average"] == "binary":
                return float(recall_score(y_true, y_pred, average="binary", pos_label=cfg["pos_label"], zero_division=0))
            return float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        raise ValueError("unsupported metric name")

    def _evaluate_model_cv(self, est: Any, X: pd.DataFrame, y: pd.Series, cv: int) -> dict[str, Any]:
        cfg = self._metric_config(y)
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        folds: list[dict[str, Any]] = []

        all_labels = sorted(pd.unique(y))
        zero_cm = np.zeros((len(all_labels), len(all_labels)), dtype=int)

        for fold_id, (tr_idx, te_idx) in enumerate(splitter.split(X, y), start=1):
            model = clone(est)
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
            try:
                model.fit(X_tr, y_tr)
                pred_tr = model.predict(X_tr)
                pred_te = model.predict(X_te)

                fold = {
                    "fold": fold_id,
                    "train_n": int(len(y_tr)),
                    "test_n": int(len(y_te)),
                    "train_f1": self._score_value("f1", y_tr.to_numpy(), pred_tr, cfg),
                    "train_accuracy": self._score_value("accuracy", y_tr.to_numpy(), pred_tr, cfg),
                    "train_recall": self._score_value("recall", y_tr.to_numpy(), pred_tr, cfg),
                    "test_f1": self._score_value("f1", y_te.to_numpy(), pred_te, cfg),
                    "test_accuracy": self._score_value("accuracy", y_te.to_numpy(), pred_te, cfg),
                    "test_recall": self._score_value("recall", y_te.to_numpy(), pred_te, cfg),
                    "train_cm": confusion_matrix(y_tr, pred_tr, labels=all_labels),
                    "test_cm": confusion_matrix(y_te, pred_te, labels=all_labels),
                    "fit_error": None,
                }
            except Exception as exc:
                fold = {
                    "fold": fold_id,
                    "train_n": int(len(y_tr)),
                    "test_n": int(len(y_te)),
                    "train_f1": np.nan,
                    "train_accuracy": np.nan,
                    "train_recall": np.nan,
                    "test_f1": np.nan,
                    "test_accuracy": np.nan,
                    "test_recall": np.nan,
                    "train_cm": zero_cm.copy(),
                    "test_cm": zero_cm.copy(),
                    "fit_error": str(exc),
                }
            folds.append(fold)

        train_total = sum(f["train_n"] for f in folds)
        test_total = sum(f["test_n"] for f in folds)

        def wavg(key: str, split_total: int, n_key: str) -> float:
            items = [(f[key], f[n_key]) for f in folds if not np.isnan(f[key])]
            if not items:
                return float("nan")
            denom = max(1, sum(w for _, w in items))
            return float(sum(v * w for v, w in items) / denom)

        agg = {
            "train_n_total": train_total,
            "test_n_total": test_total,
            "train_f1_weighted": wavg("train_f1", train_total, "train_n"),
            "train_accuracy_weighted": wavg("train_accuracy", train_total, "train_n"),
            "train_recall_weighted": wavg("train_recall", train_total, "train_n"),
            "test_f1_weighted": wavg("test_f1", test_total, "test_n"),
            "test_accuracy_weighted": wavg("test_accuracy", test_total, "test_n"),
            "test_recall_weighted": wavg("test_recall", test_total, "test_n"),
            "average_mode": cfg["mode"],
        }
        return {"folds": folds, "aggregate": agg, "labels": all_labels}

    def build_best_model(
        self,
        models: dict[str, Any] | None = None,
        label_col: str = "label",
        feature_cols: list[str] | None = None,
        cv: int = 10,
        scoring: str = "accuracy",
        exclude_noise: bool = True,
    ) -> dict[str, Any]:
        if self.filtered_df is None:
            self.filter_events()
        assert self.filtered_df is not None
        df = self.filtered_df
        if exclude_noise and "is_noise" in df.columns:
            df = df[~df["is_noise"]].copy()
        if label_col not in df.columns:
            raise ValueError("label column missing, provide sample_to_group at init or add labels manually")

        default_feature_cols = ["duration_s", "blockade_ratio", "segment_std", "segment_skew", "segment_kurt"]
        feature_cols = list(feature_cols) if feature_cols is not None else default_feature_cols
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"feature columns missing in dataframe: {missing}")
        X = df[feature_cols].fillna(0.0)
        y = df[label_col]

        models = models or {
            "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42, n_jobs=-1),
            "SVM": SVC(probability=True, random_state=42),
            "Neural Network": MLPClassifier(max_iter=500, random_state=42),
            "Elastic Net": LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=500, random_state=42, n_jobs=-1),
            "Lasso": LogisticRegression(penalty="l1", solver="saga", max_iter=500, random_state=42, n_jobs=-1),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "LDA": LDA(),
            "AdaBoost": AdaBoostClassifier(random_state=42),
            "Naive Bayes": GaussianNB(),
        }

        scoring_key_map = {
            "f1": "test_f1_weighted",
            "f1_macro": "test_f1_weighted",
            "accuracy": "test_accuracy_weighted",
            "accuracy_macro": "test_accuracy_weighted",
            "recall": "test_recall_weighted",
            "recall_macro": "test_recall_weighted",
        }
        score_key = scoring_key_map.get(scoring, "test_accuracy_weighted")

        self.model_cv_results = {}
        scores: dict[str, float] = {}
        best_name = None
        best_score = -np.inf
        best_estimator = None

        for name, est in models.items():
            cv_result = self._evaluate_model_cv(est, X, y, cv=cv)
            self.model_cv_results[name] = cv_result
            s = float(cv_result["aggregate"][score_key])
            scores[name] = s
            if not np.isnan(s) and s > best_score:
                best_score = s
                best_name = name
                best_estimator = clone(est)

        if best_estimator is None or best_name is None:
            raise RuntimeError("all candidate models failed during CV; check labels/class balance")
        best_estimator.fit(X, y)
        all_feature_df = self.feature_df.copy() if self.feature_df is not None else df.copy()
        all_X = all_feature_df[feature_cols].fillna(0.0)
        all_pred = best_estimator.predict(all_X)
        summary_cols = []
        for c in ["trace_id", "sample_id", label_col]:
            if c in all_feature_df.columns:
                summary_cols.append(c)
        all_samples_feature_pred = all_feature_df[summary_cols + feature_cols].copy()
        all_samples_feature_pred["best_model_pred"] = all_pred
        self.best_model_package = {
            "model": best_estimator,
            "feature_cols": feature_cols,
            "scores": scores,
            "best_model": best_name,
            "cv_results": self.model_cv_results,
            "all_samples_feature_pred": all_samples_feature_pred,
            "preprocess_state": self.preprocess_state,
            "detect_state": self.detect_state,
            "feature_state": self.feature_state,
        }
        return self.best_model_package

    # Step 6: reuse current pipeline + best model on new samples
    def classify_new_samples(
        self,
        new_sample_paths: dict[str, str | Path],
        reader: str | None = None,
        reader_kwargs: dict[str, Any] | None = None,
        custom_feature_fns: dict[str, FeatureFn] | None = None,
    ) -> tuple["MultiSampleAnalysis", pd.DataFrame]:
        if self.best_model_package is None:
            self.build_best_model()
        assert self.best_model_package is not None

        other = MultiSampleAnalysis(
            sample_paths=new_sample_paths,
            sample_to_group=None,
            reader=reader or self.reader,
            reader_kwargs=reader_kwargs or self.reader_kwargs,
        )
        preprocess_kwargs = self.preprocess_state.copy()
        if "kwargs" in preprocess_kwargs and isinstance(preprocess_kwargs["kwargs"], dict):
            nested = preprocess_kwargs.pop("kwargs")
            preprocess_kwargs.update(nested)

        other.load().denoise(**preprocess_kwargs).detect_events(**self.detect_state)
        features = other.extract_features(custom_feature_fns=custom_feature_fns)

        X = features[self.best_model_package["feature_cols"]].fillna(0.0)
        pred = self.best_model_package["model"].predict(X)
        out = features.copy()
        out["pred_label"] = pred
        model = self.best_model_package["model"]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            classes = getattr(model, "classes_", np.arange(proba.shape[1]))
            for i, cls in enumerate(classes):
                out[f"pred_proba_{cls}"] = proba[:, i]
        other.feature_df = out.copy()
        return other, out


def create_analysis_object(
    sample_paths: dict[str, str | Path],
    sample_to_group: dict[str, str] | None = None,
    reader: str = "abf",
    reader_kwargs: dict[str, Any] | None = None,
) -> MultiSampleAnalysis:
    """Factory for building a multi-sample, step-by-step analysis object."""
    return MultiSampleAnalysis(sample_paths=sample_paths, sample_to_group=sample_to_group, reader=reader, reader_kwargs=reader_kwargs or {})
