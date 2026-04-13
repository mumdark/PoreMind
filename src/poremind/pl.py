from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .workflow import MultiSampleAnalysis


@dataclass
class PlotAccessor:
    analysis: "MultiSampleAnalysis"

    def current(
        self,
        sample_id: str | None = None,
        current: str = "denoise",
        start_ms: float = 0.0,
        end_ms: float = 1.0,
        width: float = 10.0,
        height: float = 3.0,
    ):
        """Plot raw/denoised current in a time range (ms)."""
        if not self.analysis.traces:
            self.analysis.load()

        if sample_id is None:
            sample_id = next(iter(self.analysis.traces))

        trace = self.analysis.traces[sample_id]
        if current == "denoise":
            if not self.analysis.denoised:
                self.analysis.denoise()
            y = self.analysis.denoised[sample_id]
        elif current == "raw":
            y = trace.current
        else:
            raise ValueError("current must be 'denoise' or 'raw'")

        t_ms = trace.time * 1000.0
        mask = (t_ms >= start_ms) & (t_ms <= end_ms)

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            raise ImportError("analysis.pl.current requires matplotlib") from exc

        fig, ax = plt.subplots(figsize=(width, height))
        ax.plot(t_ms[mask], y[mask], lw=0.8)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Current")
        ax.set_title(f"{sample_id} | {current} | {start_ms}-{end_ms} ms")
        plt.tight_layout()
        return ax

    def model_cm(self, model_name: str, split: str = "test", width: float = 5.5, height: float = 4.5):
        """Visualize aggregated 10-fold confusion matrix for train or test split."""
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if model_name not in self.analysis.model_cv_results:
            raise ValueError("model_name not found, run build_best_model first")

        folds = self.analysis.model_cv_results[model_name]["folds"]
        key = f"{split}_cm"
        cm = np.sum([f[key] for f in folds], axis=0)

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            raise ImportError("analysis.pl.model_cm requires matplotlib") from exc

        fig, ax = plt.subplots(figsize=(width, height))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{model_name} | {split} confusion matrix (10-fold sum)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        return ax

    def model_metric_bar(self, metric: str = "accuracy", split: str = "test", width: float = 8.0, height: float = 4.0):
        """Bar plot for model weighted metrics; default macro-weighted accuracy."""
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        metric = metric.lower()
        key_map = {
            "f1": f"{split}_f1_weighted",
            "accuracy": f"{split}_accuracy_weighted",
            "recall": f"{split}_recall_weighted",
        }
        if metric not in key_map:
            raise ValueError("metric must be one of: f1, accuracy, recall")

        key = key_map[metric]
        names = []
        vals = []
        for model_name, result in self.analysis.model_cv_results.items():
            names.append(model_name)
            vals.append(float(result["aggregate"][key]))

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            raise ImportError("analysis.pl.model_metric_bar requires matplotlib") from exc

        fig, ax = plt.subplots(figsize=(width, height))
        ax.bar(names, vals)
        ax.set_ylabel(key)
        ax.set_title(f"Model comparison | {key}")
        ax.tick_params(axis="x", rotation=35)
        plt.tight_layout()
        return ax
