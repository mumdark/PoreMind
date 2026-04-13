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

    def _pick_df(self, data: str = "filtered"):
        if data == "filtered":
            if self.analysis.filtered_df is not None:
                return self.analysis.filtered_df
            if self.analysis.feature_df is not None:
                return self.analysis.feature_df
            return self.analysis.extract_features()
        if data == "full":
            if self.analysis.feature_df is not None:
                return self.analysis.feature_df
            return self.analysis.extract_features()
        raise ValueError("data must be 'filtered' or 'full'")

    @staticmethod
    def _maybe_log2(v: np.ndarray, enabled: bool) -> np.ndarray:
        if not enabled:
            return v
        return np.log2(np.abs(v) + 1e-12)

    def plot_2d(
        self,
        x: str = "blockade_ratio",
        y: str = "duration_s",
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        x_log2: bool = False,
        y_log2: bool = True,
        data: str = "filtered",
        value: str | None = None,
        width: float = 6.0,
        height: float = 4.5,
    ):
        df = self._pick_df(data=data).copy()
        if x not in df.columns or y not in df.columns:
            raise ValueError("x/y column not found in dataframe")

        xv = self._maybe_log2(df[x].to_numpy(dtype=float), x_log2)
        yv = self._maybe_log2(df[y].to_numpy(dtype=float), y_log2)

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            raise ImportError("analysis.pl.plot_2d requires matplotlib") from exc

        fig, ax = plt.subplots(figsize=(width, height))
        if value is None or value not in df.columns:
            ax.scatter(xv, yv, s=6, alpha=0.55)
        else:
            col = df[value]
            if col.dtype.kind in "biufc" and col.nunique() > 12:
                sc = ax.scatter(xv, yv, c=col.to_numpy(dtype=float), cmap="viridis", s=6, alpha=0.6)
                fig.colorbar(sc, ax=ax, label=value)
            else:
                cats = col.astype(str)
                uniq = sorted(cats.unique())
                cmap = plt.get_cmap("tab20", len(uniq))
                for i, u in enumerate(uniq):
                    m = cats == u
                    ax.scatter(xv[m], yv[m], s=6, alpha=0.6, color=cmap(i), label=u)
                ax.legend(markerscale=3, fontsize=8, loc="best")

        ax.set_xlabel(f"{x}{' (log2)' if x_log2 else ''}")
        ax.set_ylabel(f"{y}{' (log2)' if y_log2 else ''}")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title("2D feature scatter")
        plt.tight_layout()
        return ax

    def plot_3d(
        self,
        x: str = "blockade_ratio",
        y: str = "duration_s",
        z: str = "segment_std",
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        zlim: tuple[float, float] | None = None,
        x_log2: bool = False,
        y_log2: bool = True,
        z_log2: bool = False,
        data: str = "filtered",
        value: str | None = None,
        width: float = 7.0,
        height: float = 5.0,
    ):
        df = self._pick_df(data=data).copy()
        for c in (x, y, z):
            if c not in df.columns:
                raise ValueError(f"column not found: {c}")

        xv = self._maybe_log2(df[x].to_numpy(dtype=float), x_log2)
        yv = self._maybe_log2(df[y].to_numpy(dtype=float), y_log2)
        zv = self._maybe_log2(df[z].to_numpy(dtype=float), z_log2)

        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise ImportError("analysis.pl.plot_3d requires matplotlib") from exc

        fig = plt.figure(figsize=(width, height))
        ax = fig.add_subplot(111, projection="3d")

        if value is None or value not in df.columns:
            ax.scatter(xv, yv, zv, s=6, alpha=0.55)
        else:
            col = df[value]
            if col.dtype.kind in "biufc" and col.nunique() > 12:
                sc = ax.scatter(xv, yv, zv, c=col.to_numpy(dtype=float), cmap="viridis", s=6, alpha=0.6)
                fig.colorbar(sc, ax=ax, pad=0.1, label=value)
            else:
                cats = col.astype(str)
                uniq = sorted(cats.unique())
                cmap = plt.get_cmap("tab20", len(uniq))
                for i, u in enumerate(uniq):
                    m = cats == u
                    ax.scatter(xv[m], yv[m], zv[m], s=6, alpha=0.6, color=cmap(i), label=u)
                ax.legend(markerscale=3, fontsize=8, loc="best")

        ax.set_xlabel(f"{x}{' (log2)' if x_log2 else ''}")
        ax.set_ylabel(f"{y}{' (log2)' if y_log2 else ''}")
        ax.set_zlabel(f"{z}{' (log2)' if z_log2 else ''}")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if zlim is not None:
            ax.set_zlim(*zlim)
        ax.set_title("3D feature scatter")
        plt.tight_layout()
        return ax

    def __getattr__(self, name: str):
        # compatibility aliases requested in notebooks/discussions
        if name == "2d_plot":
            return self.plot_2d
        if name == "3d_plot":
            return self.plot_3d
        raise AttributeError(name)
