from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from poremind.features import select_feature_columns
from poremind.workflow import MultiSampleAnalysis, create_analysis_object

from .session import UIAnalysisSession


class AnalysisController:
    """Application service layer that isolates UI from core algorithm internals."""

    def __init__(self, session: UIAnalysisSession | None = None) -> None:
        self.session = session or UIAnalysisSession()

    def load_samples(
        self,
        sample_paths: dict[str, str],
        sample_to_group: dict[str, str] | None = None,
        reader: str = "abf",
        reader_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.session.sample_paths = sample_paths.copy()
        self.session.sample_to_group = (sample_to_group or {}).copy()
        analysis = create_analysis_object(
            sample_paths=sample_paths,
            sample_to_group=sample_to_group,
            reader=reader,
            reader_kwargs=reader_kwargs or {},
        )
        analysis.load()
        self.session.analysis = analysis

        samples = []
        for trace_id, tr in analysis.traces.items():
            samples.append(
                {
                    "trace_id": trace_id,
                    "source": str(tr.source),
                    "channel": int(tr.channel),
                    "sweep": int(tr.sweep),
                    "points": int(len(tr.current)),
                    "duration_s": float(tr.time[-1]) if len(tr.time) else 0.0,
                    "sampling_rate_hz": float(tr.sampling_rate_hz),
                }
            )
        df = pd.DataFrame(samples)
        self.session.outputs["samples"] = df
        return {
            "analysis": analysis,
            "sample_df": df,
            "summary": {
                "n_samples": int(len(sample_paths)),
                "n_traces": int(len(analysis.traces)),
                "reader": reader,
            },
        }

    def run_denoise(self, method: str = "butterworth_filtfilt", **kwargs: Any) -> dict[str, Any]:
        analysis = self._require_analysis()
        analysis.denoise(method=method, **kwargs)
        self.session.preprocess_params = {"method": method, **kwargs}
        self.session.outputs["denoise"] = {"method": method, "params": kwargs}
        return self.session.outputs["denoise"]

    def run_detect(
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
        stage: str = "global",
        sample_id: str | None = None,
        current: str = "denoise",
        start_ms: float = 0.0,
        end_ms: float = 1000.0,
    ) -> dict[str, Any]:
        analysis = self._require_analysis()
        detect_params = detect_params or analysis._default_detect_params(detect_method)
        baseline_params = baseline_params or {"window": 10000, "q": 0.5}

        if stage == "preview":
            simple = analysis.detect_events_simple(
                detect_method=detect_method,
                detect_params=detect_params,
                baseline_method=baseline_method,
                baseline_params=baseline_params,
                sample_id=sample_id,
                current=current,
                start_ms=start_ms,
                end_ms=end_ms,
                detect_direction=detect_direction,
                merge_event=merge_event,
                merge_event_params=merge_event_params,
                exclude_current=exclude_current,
                exclude_current_params=exclude_current_params,
            )
            out = {"stage": "preview", "event_counts": {k: len(v) for k, v in simple.items()}}
        else:
            analysis.detect_events(
                detect_method=detect_method,
                detect_params=detect_params,
                baseline_method=baseline_method,
                baseline_params=baseline_params,
                detect_direction=detect_direction,
                merge_event=merge_event,
                merge_event_params=merge_event_params,
                exclude_current=exclude_current,
                exclude_current_params=exclude_current_params,
            )
            out = {"stage": "global", "event_counts": {k: len(v) for k, v in analysis.events.items()}}

        self.session.detect_params = {
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
        self.session.outputs["detect"] = out
        return out

    def extract_features(self, max_event_per_sample: int | None = None) -> pd.DataFrame:
        analysis = self._require_analysis()
        df = analysis.extract_features(max_event_per_sample=max_event_per_sample)
        self.session.feature_params = {"max_event_per_sample": max_event_per_sample}
        self.session.outputs["feature_df"] = df
        return df

    def filter_events(self, method: str = "blockade_gmm", parameters: dict[str, Any] | None = None) -> pd.DataFrame:
        analysis = self._require_analysis()
        analysis.filter_events(method=method, parameters=parameters)
        self.session.filter_params = {"method": method, "parameters": parameters or {}}
        assert analysis.filtered_df is not None
        self.session.outputs["filtered_df"] = analysis.filtered_df
        return analysis.filtered_df

    def do_dimensionality_reduction(
        self,
        method: str,
        feature_cols: list[str] | None = None,
        data: str = "filtered",
        **kwargs: Any,
    ) -> pd.DataFrame:
        analysis = self._require_analysis()
        if method == "pca":
            return analysis.do_pca(feature_cols=feature_cols, data=data, **kwargs)
        if method == "tsne":
            return analysis.do_tsne(feature_cols=feature_cols, data=data, **kwargs)
        if method == "umap":
            return analysis.do_umap(feature_cols=feature_cols, data=data, **kwargs)
        raise ValueError("method must be one of: pca, tsne, umap")

    def train_model(
        self,
        label_col: str = "label",
        feature_cols: list[str] | None = None,
        cv: int = 5,
        scoring: str = "accuracy",
        exclude_noise: bool = True,
    ) -> dict[str, Any]:
        analysis = self._require_analysis()
        package = analysis.build_best_model(
            label_col=label_col,
            feature_cols=feature_cols,
            cv=cv,
            scoring=scoring,
            exclude_noise=exclude_noise,
        )
        best_name = str(package["best_model"])
        cv_result = analysis.model_cv_results.get(best_name, {})
        agg = cv_result.get("aggregate", {})
        self.session.model_params = {
            "label_col": label_col,
            "feature_cols": feature_cols,
            "cv": cv,
            "scoring": scoring,
            "exclude_noise": exclude_noise,
        }
        self.session.outputs["model"] = {
            "best_model": best_name,
            "scores": package["scores"],
            "aggregate": agg,
            "feature_cols": package["feature_cols"],
        }
        return self.session.outputs["model"]

    def predict_new(
        self,
        new_sample_paths: dict[str, str],
        reader: str | None = None,
        reader_kwargs: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        analysis = self._require_analysis()
        _, pred = analysis.classify_new_samples(
            new_sample_paths=new_sample_paths,
            reader=reader,
            reader_kwargs=reader_kwargs,
        )
        self.session.outputs["pred_df"] = pred
        return pred

    def export_tables(self, output_dir: str | Path) -> dict[str, str]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        exported: dict[str, str] = {}
        for name in ["feature_df", "filtered_df", "pred_df"]:
            df = self.session.outputs.get(name)
            if isinstance(df, pd.DataFrame):
                p = out_dir / f"{name}.csv"
                df.to_csv(p, index=False)
                exported[name] = str(p)
        return exported

    def export_params_json(self, path: str | Path) -> str:
        payload = {
            "sample_paths": self.session.sample_paths,
            "sample_to_group": self.session.sample_to_group,
            "preprocess_params": self.session.preprocess_params,
            "detect_params": self.session.detect_params,
            "feature_params": self.session.feature_params,
            "filter_params": self.session.filter_params,
            "model_params": self.session.model_params,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(p)

    def export_analysis_script(self, path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        script = f'''from poremind.workflow import create_analysis_object

sample_paths = {self.session.sample_paths!r}
sample_to_group = {self.session.sample_to_group!r}

analysis = create_analysis_object(
    sample_paths=sample_paths,
    sample_to_group=sample_to_group,
    reader={self._reader()!r},
)
analysis.load()
analysis.denoise(**{self.session.preprocess_params!r})
analysis.detect_events(**{self.session.detect_params!r})
analysis.extract_features(**{self.session.feature_params!r})
analysis.filter_events(**{self.session.filter_params!r})
analysis.build_best_model(**{self.session.model_params!r})
'''
        p.write_text(script, encoding="utf-8")
        return str(p)

    def suggest_feature_columns(self) -> list[str]:
        analysis = self._require_analysis()
        if analysis.filtered_df is not None:
            return select_feature_columns(analysis.filtered_df)
        if analysis.feature_df is not None:
            return select_feature_columns(analysis.feature_df)
        return []

    def _require_analysis(self) -> MultiSampleAnalysis:
        if self.session.analysis is None:
            raise ValueError("Please load samples first.")
        return self.session.analysis

    def _reader(self) -> str:
        analysis = self._require_analysis()
        return analysis.reader
