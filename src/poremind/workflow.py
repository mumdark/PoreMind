from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    feature_df: pd.DataFrame | None = None
    filtered_df: pd.DataFrame | None = None
    best_model_package: dict[str, Any] | None = None

    preprocess_state: dict[str, Any] = field(default_factory=dict)
    detect_state: dict[str, Any] = field(default_factory=dict)
    feature_state: dict[str, Any] = field(default_factory=dict)
    trace_to_sample: dict[str, str] = field(default_factory=dict)
    pl: PlotAccessor = field(init=False)

    def __post_init__(self) -> None:
        self.pl = PlotAccessor(self)

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
    ) -> "MultiSampleAnalysis":
        if not self.denoised:
            self.denoise()
        if detect_params is None:
            detect_params = self._default_detect_params(detect_method)
        baseline_params = baseline_params or {"window": 501, "q": 0.5}
        self.detect_state = {
            "detect_method": detect_method,
            "detect_params": detect_params,
            "baseline_method": baseline_method,
            "baseline_params": baseline_params,
        }

        self.baselines = {}
        self.events = {}
        for sid, tr in self.traces.items():
            sig = self.denoised[sid]
            baseline = estimate_baseline(sig, method=baseline_method, **baseline_params)
            self.baselines[sid] = baseline
            if detect_method == "threshold":
                evts = detect_events_threshold(sig, baseline, tr.sampling_rate_hz, **detect_params)
            elif detect_method == "zscore_threshold":
                z = (sig - baseline) / (np.std(sig - baseline) + 1e-12)
                thr = -float(detect_params.get("z", 4.0))
                mask = z < thr
                evts = self._mask_to_events(mask, baseline, sig, tr.sampling_rate_hz, min_duration_s=float(detect_params.get("min_duration_s", 2e-4)))
            elif detect_method == "cusum":
                evts = detect_events_cusum(sig, baseline, tr.sampling_rate_hz, **detect_params)
            elif detect_method == "pelt":
                evts = detect_events_pelt(sig, baseline, tr.sampling_rate_hz, **detect_params)
            elif detect_method == "hmm":
                evts = detect_events_hmm(sig, baseline, tr.sampling_rate_hz, **detect_params)
            else:
                raise ValueError("unsupported detect method")
            self.events[sid] = evts
        return self

    @staticmethod
    def _default_detect_params(detect_method: str) -> dict[str, Any]:
        defaults = {
            "threshold": {"sigma_k": 5.0, "min_duration_s": 2e-4},
            "zscore_threshold": {"z": 4.0, "min_duration_s": 2e-4},
            "cusum": {"drift": 0.02, "threshold": 8.0, "min_duration_s": 2e-4},
            "pelt": {"model": "l2", "penalty": 8.0, "sigma_k": 3.0, "min_duration_s": 2e-4},
            "hmm": {"n_components": 2, "covariance_type": "diag", "n_iter": 200, "min_duration_s": 2e-4},
        }
        if detect_method not in defaults:
            raise ValueError(f"unsupported detect method: {detect_method}")
        return defaults[detect_method].copy()

    @staticmethod
    def _mask_to_events(mask: np.ndarray, baseline: np.ndarray, signal: np.ndarray, sr: float, min_duration_s: float) -> list[Event]:
        min_samples = max(1, int(min_duration_s * sr))
        out: list[Event] = []
        i = 0
        n = len(mask)
        sigma = float(np.std(signal - baseline)) + 1e-12
        while i < n:
            if not mask[i]:
                i += 1
                continue
            s = i
            while i < n and mask[i]:
                i += 1
            e = i
            if e - s < min_samples:
                continue
            seg_signal = signal[s:e]
            seg_base = baseline[s:e]
            delta_i = float(np.mean(seg_base - seg_signal))
            out.append(Event(s, e, float(np.mean(seg_base)), delta_i, (e - s) / sr, delta_i / sigma))
        return out

    # Step 3: feature extraction (built-in + custom)
    def extract_features(self, custom_feature_fns: dict[str, FeatureFn] | None = None) -> pd.DataFrame:
        if not self.events:
            self.detect_events()
        custom_feature_fns = custom_feature_fns or {}
        rows: list[dict[str, Any]] = []

        for sid, evts in self.events.items():
            tr = self.traces[sid]
            source_sample_id = self.trace_to_sample.get(sid, sid)
            sig = self.denoised[sid]
            base = self.baselines[sid]
            for i, e in enumerate(evts):
                seg = sig[e.start_idx:e.end_idx]
                left = sig[max(0, e.start_idx - 50):e.start_idx]
                right = sig[e.end_idx:min(len(sig), e.end_idx + 50)]
                left_base = float(np.mean(left)) if len(left) else float(base[e.start_idx])
                right_base = float(np.mean(right)) if len(right) else float(base[e.end_idx - 1])
                global_base = float(np.median(base))
                blockade_ratio = float(e.delta_i / (abs(global_base) + 1e-12))
                seg_mean = float(np.mean(seg))
                seg_std = float(np.std(seg))
                centered = seg - seg_mean
                denom = float(np.std(seg)) + 1e-12
                skew = float(np.mean((centered / denom) ** 3))
                kurt = float(np.mean((centered / denom) ** 4))
                rms = float(np.sqrt(np.mean(seg ** 2))) + 1e-12
                peak_factor = float(np.max(np.abs(seg)) / rms)
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
                    "delta_i": e.delta_i,
                    "snr": e.snr,
                    "left_baseline": left_base,
                    "right_baseline": right_base,
                    "global_baseline": global_base,
                    "blockade_ratio": blockade_ratio,
                    "segment_mean": seg_mean,
                    "segment_std": seg_std,
                    "segment_min": float(np.min(seg)),
                    "segment_max": float(np.max(seg)),
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

    # Step 4: noise/outlier flagging
    def filter_events(self, method: str = "isolation_forest", contamination: float = 0.05, feature_cols: list[str] | None = None) -> pd.DataFrame:
        if self.feature_df is None:
            self.extract_features()
        assert self.feature_df is not None
        df = self.feature_df.copy()
        feature_cols = feature_cols or select_feature_columns(df)
        X = df[feature_cols].fillna(0.0)

        if method == "isolation_forest":
            detector = IsolationForest(contamination=contamination, random_state=42)
            pred = detector.fit_predict(X)
        elif method == "lof":
            detector = LocalOutlierFactor(contamination=contamination)
            pred = detector.fit_predict(X)
        else:
            raise ValueError("unsupported outlier method")

        df["is_noise"] = pred == -1
        df["quality_tag"] = np.where(df["is_noise"], "noise", "valid")
        self.filtered_df = df
        return df

    # Step 5: model selection (10-fold)
    def build_best_model(
        self,
        models: dict[str, Any] | None = None,
        label_col: str = "label",
        cv: int = 10,
        scoring: str = "f1_macro",
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

        feature_cols = [c for c in select_feature_columns(df) if c not in {"is_noise"}]
        X = df[feature_cols].fillna(0.0)
        y = df[label_col]

        models = models or {
            "logreg": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]),
            "rf": RandomForestClassifier(n_estimators=400, random_state=42),
            "svm": Pipeline([("scaler", StandardScaler()), ("model", SVC(probability=True))]),
        }

        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores: dict[str, float] = {}
        best_name = None
        best_score = -np.inf
        best_estimator = None

        for name, est in models.items():
            s = float(np.mean(cross_val_score(est, X, y, cv=splitter, scoring=scoring)))
            scores[name] = s
            if s > best_score:
                best_score = s
                best_name = name
                best_estimator = clone(est)

        assert best_estimator is not None and best_name is not None
        best_estimator.fit(X, y)
        self.best_model_package = {
            "model": best_estimator,
            "feature_cols": feature_cols,
            "scores": scores,
            "best_model": best_name,
            "preprocess_state": self.preprocess_state,
            "detect_state": self.detect_state,
            "feature_state": self.feature_state,
        }
        return self.best_model_package

    # Step 6: reuse current pipeline + best model on new samples
    def classify_new_samples(self, new_sample_paths: dict[str, str | Path], reader: str | None = None, reader_kwargs: dict[str, Any] | None = None) -> pd.DataFrame:
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
        features = other.extract_features()

        X = features[self.best_model_package["feature_cols"]].fillna(0.0)
        pred = self.best_model_package["model"].predict(X)
        out = features.copy()
        out["pred_label"] = pred
        model = self.best_model_package["model"]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            out["pred_score_max"] = np.max(proba, axis=1)
        return out


def create_analysis_object(
    sample_paths: dict[str, str | Path],
    sample_to_group: dict[str, str] | None = None,
    reader: str = "abf",
    reader_kwargs: dict[str, Any] | None = None,
) -> MultiSampleAnalysis:
    """Factory for building a multi-sample, step-by-step analysis object."""
    return MultiSampleAnalysis(sample_paths=sample_paths, sample_to_group=sample_to_group, reader=reader, reader_kwargs=reader_kwargs or {})
