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
from scipy.signal import find_peaks
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
            evts = self._detect_events_by_method(sig, baseline, tr.sampling_rate_hz, detect_method=detect_method, detect_params=detect_params)
            self.events[sid] = evts
        return self

    @staticmethod
    def _detect_events_by_method(signal: np.ndarray, baseline: np.ndarray, sampling_rate_hz: float, detect_method: str, detect_params: dict[str, Any]) -> list[Event]:
        if detect_method == "threshold":
            return detect_events_threshold(signal, baseline, sampling_rate_hz, **detect_params)
        if detect_method == "zscore_threshold":
            z = (signal - baseline) / (np.std(signal - baseline) + 1e-12)
            thr = -float(detect_params.get("z", 4.0))
            mask = z < thr
            return MultiSampleAnalysis._mask_to_events(mask, baseline, signal, sampling_rate_hz, min_duration_s=float(detect_params.get("min_duration_s", 2e-4)))
        if detect_method == "cusum":
            return detect_events_cusum(signal, baseline, sampling_rate_hz, **detect_params)
        if detect_method == "pelt":
            return detect_events_pelt(signal, baseline, sampling_rate_hz, **detect_params)
        if detect_method == "hmm":
            return detect_events_hmm(signal, baseline, sampling_rate_hz, **detect_params)
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
    ) -> dict[str, list[Event]]:
        if not self.traces:
            self.load()
        if current not in {"denoise", "raw"}:
            raise ValueError("current must be 'denoise' or 'raw'")
        if current == "denoise" and not self.denoised:
            self.denoise()
        if detect_params is None:
            detect_params = self._default_detect_params(detect_method)
        baseline_params = baseline_params or {"window": 501, "q": 0.5}

        self.simple_detect_state = {
            "detect_method": detect_method,
            "detect_params": detect_params,
            "baseline_method": baseline_method,
            "baseline_params": baseline_params,
            "sample_id": sample_id,
            "current": current,
            "start_ms": start_ms,
            "end_ms": end_ms,
        }

        target_ids = [sample_id] if sample_id is not None else list(self.traces.keys())
        out: dict[str, list[Event]] = {}
        for sid in target_ids:
            tr = self.traces[sid]
            sig_full = self.denoised[sid] if current == "denoise" else tr.current
            t_ms = tr.time * 1000.0
            idx = np.flatnonzero((t_ms >= start_ms) & (t_ms <= end_ms))
            if len(idx) == 0:
                out[sid] = []
                continue
            lo, hi = int(idx[0]), int(idx[-1]) + 1
            sig = sig_full[lo:hi]
            baseline = estimate_baseline(sig, method=baseline_method, **baseline_params)
            evts_local = self._detect_events_by_method(sig, baseline, tr.sampling_rate_hz, detect_method=detect_method, detect_params=detect_params)
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
        return out

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

        # KDE-based peak counting (no seaborn dependency)
        grid = np.linspace(np.nanmin(data), np.nanmax(data), 256)
        bw = max(1e-6, np.std(data) * 0.2)
        density = np.sum(np.exp(-0.5 * ((grid[:, None] - data[None, :]) / bw) ** 2), axis=1)
        peaks, _ = find_peaks(density)

        if visualize:
            try:
                import matplotlib.pyplot as plt
            except Exception as exc:  # pragma: no cover
                raise ImportError("blockade_gmm visualization requires matplotlib") from exc
            fig, axs = plt.subplots(2, 1, sharex=False, gridspec_kw={"height_ratios": [1, 3]}, figsize=(4, 5))
            axs[0].plot(grid, density)
            if len(peaks):
                axs[0].plot(grid[peaks], density[peaks], "ro")
            axs[0].set_ylabel("Density")
            axs[1].scatter(X[blockade_col], np.log10(X[dwell_col].to_numpy(dtype=float) + 1e-12), c=rm_index, alpha=0.5, s=4)
            axs[1].set_xlabel(blockade_col)
            axs[1].set_ylabel(f"log10({dwell_col})")
            plt.tight_layout()

        if len(peaks) > 1:
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

        return rm_index

    # Step 4: noise/outlier flagging
    def filter_events(
        self,
        method: str = "blockade_gmm",
        contamination: float = 0.05,
        feature_cols: list[str] | None = None,
        blockade_col: str = "blockade_ratio",
        dwell_col: str = "duration_s",
        rm_index: np.ndarray | None = None,
        n_components: int = 2,
        prior_mean: float | None = None,
        visualize: bool = False,
    ) -> pd.DataFrame:
        if self.feature_df is None:
            self.extract_features()
        assert self.feature_df is not None
        df = self.feature_df.copy()

        if method in {"blockade_gmm", "peak_detection"}:
            valid_mask = self._blockade_gmm_mask(
                df,
                rm_index=rm_index,
                blockade_col=blockade_col,
                dwell_col=dwell_col,
                n_components=n_components,
                visualize=visualize,
                prior_mean=prior_mean,
            )
            df["is_noise"] = ~valid_mask
        else:
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
        return {"folds": folds, "aggregate": agg}

    def build_best_model(
        self,
        models: dict[str, Any] | None = None,
        label_col: str = "label",
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

        feature_cols = [c for c in select_feature_columns(df) if c not in {"is_noise"}]
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
        self.best_model_package = {
            "model": best_estimator,
            "feature_cols": feature_cols,
            "scores": scores,
            "best_model": best_name,
            "cv_results": self.model_cv_results,
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
