from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

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

    def _resolve_signal(self, sample_id: str, current: str) -> tuple[np.ndarray, np.ndarray]:
        trace = self.analysis.traces[sample_id]
        if current == "denoise":
            if not self.analysis.denoised:
                self.analysis.denoise()
            y = self.analysis.denoised[sample_id]
        elif current == "raw":
            y = trace.current
        else:
            raise ValueError("current must be 'denoise' or 'raw'")
        return trace.time * 1000.0, y

    @staticmethod
    def _slice_events(events: list, start_event: int, end_event: int):
        if start_event < 1 or end_event < start_event:
            raise ValueError("start_event must be >=1 and end_event must be >= start_event")
        start_i = start_event - 1
        end_i = min(end_event, len(events))
        return events[start_i:end_i]

    def _event_current_core(
        self,
        events_map: dict[str, list],
        sample_id: str | None,
        current: str,
        start_event: int,
        end_event: int,
        start_ms: float,
        end_ms: float,
        ylim: tuple[float, float] | None,
        width: float,
        height: float,
        title_prefix: str,
    ):
        if not self.analysis.traces:
            self.analysis.load()
        if sample_id is None:
            sample_id = next(iter(self.analysis.traces))
        if sample_id not in events_map:
            raise ValueError(f"sample_id {sample_id} has no detected events")

        t_ms, y = self._resolve_signal(sample_id, current)
        selected_events = self._slice_events(events_map[sample_id], start_event=start_event, end_event=end_event)

        if selected_events:
            event_start_ms = float(t_ms[selected_events[0].start_idx])
            event_end_ms = float(t_ms[selected_events[-1].end_idx - 1])
            win_start_ms, win_end_ms = event_start_ms, event_end_ms
        else:
            win_start_ms, win_end_ms = start_ms, end_ms
        mask = (t_ms >= win_start_ms) & (t_ms <= win_end_ms)

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            raise ImportError("event_current visualization requires matplotlib") from exc

        fig, ax = plt.subplots(figsize=(width, height))
        ax.plot(t_ms[mask], y[mask], lw=0.8)
        for e in selected_events:
            ax.axvline(float(t_ms[e.start_idx]), color="red", linestyle="--", linewidth=1.0)
            ax.axvline(float(t_ms[e.end_idx - 1]), color="red", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Current")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(f"{title_prefix} | {sample_id} | {current} | event {start_event}-{end_event}")
        plt.tight_layout()
        return ax

    def event_current_simple(
        self,
        sample_id: str | None = None,
        current: str = "denoise",
        start_event: int = 1,
        end_event: int = 5,
        start_ms: float = 0.0,
        end_ms: float = 1.0,
        ylim: tuple[float, float] | None = None,
        width: float = 10.0,
        height: float = 3.0,
    ):
        if not self.analysis.detect_events_simple_object:
            self.analysis.detect_events_simple(sample_id=sample_id, current=current)
        return self._event_current_core(
            self.analysis.detect_events_simple_object,
            sample_id=sample_id,
            current=current,
            start_event=start_event,
            end_event=end_event,
            start_ms=start_ms,
            end_ms=end_ms,
            ylim=ylim,
            width=width,
            height=height,
            title_prefix="Simple events",
        )

    def event_current(
        self,
        sample_id: str | None = None,
        current: str = "denoise",
        start_event: int = 1,
        end_event: int = 5,
        start_ms: float = 0.0,
        end_ms: float = 1.0,
        ylim: tuple[float, float] | None = None,
        width: float = 10.0,
        height: float = 3.0,
        ):
        if not self.analysis.events:
            self.analysis.detect_events()
        return self._event_current_core(
            self.analysis.events,
            sample_id=sample_id,
            current=current,
            start_event=start_event,
            end_event=end_event,
            start_ms=start_ms,
            end_ms=end_ms,
            ylim=ylim,
            width=width,
            height=height,
            title_prefix="Detected events",
        )

    def event_current_label(
        self,
        sample_id: str | None = None,
        current: str = "denoise",
        start_event: int = 1,
        end_event: int = 5,
        lable_col: str = "pred_label",
        label_size: float = 9.0,
        lable_color: dict[str, str] | None = None,
        label_offset: float = 3.0,
        start_ms: float = 0.0,
        end_ms: float = 1.0,
        ylim: tuple[float, float] | None = None,
        width: float = 10.0,
        height: float = 3.0,
    ):
        """Plot detected events with per-event text labels from feature_df."""
        if not self.analysis.events:
            self.analysis.detect_events()
        if self.analysis.feature_df is None:
            self.analysis.extract_features()
        assert self.analysis.feature_df is not None

        if sample_id is None:
            if not self.analysis.traces:
                self.analysis.load()
            sample_id = next(iter(self.analysis.traces))

        ax = self._event_current_core(
            self.analysis.events,
            sample_id=sample_id,
            current=current,
            start_event=start_event,
            end_event=end_event,
            start_ms=start_ms,
            end_ms=end_ms,
            ylim=ylim,
            width=width,
            height=height,
            title_prefix=f"Detected events ({lable_col})",
        )

        trace = self.analysis.traces[sample_id]
        t_ms = trace.time * 1000.0
        if current == "denoise":
            y = self.analysis.denoised[sample_id]
        else:
            y = trace.current

        selected_events = self._slice_events(self.analysis.events[sample_id], start_event=start_event, end_event=end_event)
        label_df = self.analysis.feature_df
        if "trace_id" in label_df.columns:
            sub = label_df[label_df["trace_id"] == sample_id]
        else:
            sub = label_df.iloc[0:0].copy()
        if len(sub) == 0 and "sample_id" in label_df.columns:
            sub = label_df[label_df["sample_id"] == sample_id]

        color_map = lable_color or {}
        start_i = start_event - 1
        for offset, e in enumerate(selected_events):
            event_idx = start_i + offset
            txt = ""
            if "event_id" in sub.columns and lable_col in sub.columns:
                hit = sub[sub["event_id"] == event_idx]
                if len(hit):
                    txt = str(hit.iloc[0][lable_col])
            if txt == "":
                continue
            x_mid = float((t_ms[e.start_idx] + t_ms[e.end_idx - 1]) / 2.0)
            seg_mean = float(np.mean(y[e.start_idx:e.end_idx])) if e.end_idx > e.start_idx else float(y[e.start_idx])
            y_text = seg_mean + float(label_offset)
            txt_color = color_map.get(txt, "black")
            ax.text(x_mid, y_text, txt, color=txt_color, fontsize=label_size, ha="center", va="bottom")
        return ax

    def model_cm(
        self,
        model_name: str,
        split: str = "test",
        width: float = 6.0,
        height: float = 5.0,
        cmap: str = "Reds",
        decimals: int = 1,
    ):
        """Visualize aggregated row-wise confusion matrix percentage for train or test split."""
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if model_name not in self.analysis.model_cv_results:
            raise ValueError("model_name not found, run build_best_model first")

        result = self.analysis.model_cv_results[model_name]
        folds = result["folds"]
        labels = result.get("labels")
        key = f"{split}_cm"
        cm = np.sum([f[key] for f in folds], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0
        cm_pct = np.nan_to_num(cm_pct)
        if labels is None:
            labels = [str(i) for i in range(cm.shape[0])]
        else:
            labels = [str(x) for x in labels]

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            raise ImportError("analysis.pl.model_cm requires matplotlib") from exc

        fig, ax = plt.subplots(figsize=(width, height))
        im = ax.imshow(cm_pct, cmap=cmap, vmin=0.0, vmax=100.0)
        ax.set_title(f"{model_name} | {split} confusion matrix (%)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        for i in range(cm_pct.shape[0]):
            for j in range(cm_pct.shape[1]):
                ax.text(j, i, f"{cm_pct[i, j]:.{decimals}f}", ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, label="% within true class")
        plt.tight_layout()
        return ax

    def model_metric_bar(
        self,
        metric: str = "accuracy",
        split: str = "test",
        width: float = 8.0,
        height: float = 4.0,
        cmap: str = "RdBu_r",
        decimals: int = 3,
    ):
        """Bar plot for model weighted metrics with descending sorting and gradient color."""
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
        order = np.argsort(np.asarray(vals))[::-1]
        names = [names[i] for i in order]
        vals = [vals[i] for i in order]
        v_arr = np.asarray(vals, dtype=float)
        finite_mask = np.isfinite(v_arr)
        if not finite_mask.any():
            norm_vals = np.full_like(v_arr, 0.5, dtype=float)
        else:
            v_min, v_max = float(np.min(v_arr[finite_mask])), float(np.max(v_arr[finite_mask]))
            if abs(v_max - v_min) < 1e-12:
                norm_vals = np.full_like(v_arr, 0.5, dtype=float)
            else:
                norm_vals = (v_arr - v_min) / (v_max - v_min)
                norm_vals[~finite_mask] = 0.0

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            raise ImportError("analysis.pl.model_metric_bar requires matplotlib") from exc

        fig, ax = plt.subplots(figsize=(width, height))
        cm = plt.get_cmap(cmap)
        colors = [cm(v) for v in norm_vals]
        bars = ax.bar(names, vals, color=colors)
        ax.set_ylabel(key)
        ax.set_title(f"Model comparison | {key}")
        ax.tick_params(axis="x", rotation=35)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.{decimals}f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
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

    def stacked_bar(
        self,
        group_col: str = "sample_id",
        value_col: str = "pred_label",
        data: str = "filtered",
        label_color: dict[str, str] | None = None,
        cmap: str = "tab20",
        width: float = 8.0,
        height: float = 4.5,
    ):
        df = self._pick_df(data=data).copy()
        if group_col not in df.columns:
            raise ValueError(f"group_col not found: {group_col}")
        if value_col not in df.columns:
            raise ValueError(f"value_col not found: {value_col}")
        ctab = pd.crosstab(df[group_col].astype(str), df[value_col].astype(str), normalize="index")
        categories = list(ctab.columns)

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            raise ImportError("analysis.pl.stacked_bar requires matplotlib") from exc

        palette = plt.get_cmap(cmap, max(1, len(categories)))
        colors = []
        for i, c in enumerate(categories):
            if label_color and c in label_color:
                colors.append(label_color[c])
            else:
                colors.append(palette(i))

        fig, ax = plt.subplots(figsize=(width, height))
        bottom = np.zeros(len(ctab), dtype=float)
        x = np.arange(len(ctab))
        for i, c in enumerate(categories):
            vals = ctab[c].to_numpy(dtype=float)
            ax.bar(x, vals, bottom=bottom, label=c, color=colors[i], width=0.8)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(ctab.index.astype(str), rotation=45, ha="right")
        ax.set_ylabel("Proportion")
        ax.set_xlabel(group_col)
        ax.set_title(f"Stacked proportion: {value_col} by {group_col}")
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        return ax

    @staticmethod
    def _format_p_value(p_value: float) -> str:
        if p_value < 2.2e-16:
            return "P < 2.2e-16"
        if p_value > 0.01:
            return f"P = {round(p_value, 2):.2f}"
        return f"P = {p_value:.2e}"

    def box_significance(
        self,
        group_col: str = "label",
        value_col: str = "blockade_ratio",
        data: str = "filtered",
        method: str = "ttest",
        label_color: dict[str, str] | None = None,
        cmap: str = "tab20",
        reference_group: str | None = None,
        line_offset: float = 0.02,
        line_height: float = 1.7,
        ylim: tuple[float, float] | None = None,
        log2: bool = False,
        width: float = 6.0,
        height: float = 5.0,
    ):
        df = self._pick_df(data=data).copy()
        if group_col not in df.columns:
            raise ValueError(f"group_col not found: {group_col}")
        if value_col not in df.columns:
            raise ValueError(f"value_col not found: {value_col}")
        vals = df[value_col].to_numpy(dtype=float)
        if log2:
            vals = np.log2(np.abs(vals) + 1e-12)
        plot_df = pd.DataFrame({group_col: df[group_col].astype(str), value_col: vals})
        medians = plot_df.groupby(group_col)[value_col].median().sort_values(ascending=False)
        order = medians.index.tolist()
        if len(order) < 2:
            raise ValueError("box_significance requires at least 2 groups")
        ref = reference_group if reference_group is not None else order[0]
        if ref not in order:
            raise ValueError("reference_group not found in grouped data")

        try:
            import matplotlib.pyplot as plt
            from scipy.stats import ranksums, ttest_ind
        except Exception as exc:  # pragma: no cover
            raise ImportError("analysis.pl.box_significance requires matplotlib and scipy") from exc

        cmap_obj = plt.get_cmap(cmap, max(1, len(order)))
        colors = []
        for i, g in enumerate(order):
            if label_color and g in label_color:
                colors.append(label_color[g])
            else:
                colors.append(cmap_obj(i))

        grouped_data = [plot_df.loc[plot_df[group_col] == g, value_col].to_numpy(dtype=float) for g in order]
        fig, ax = plt.subplots(figsize=(width, height))
        bp = ax.boxplot(grouped_data, labels=order, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.8)

        for i, g in enumerate(order, start=1):
            median = float(medians[g])
            ax.text(i, median, f"{median:.2f}", ha="center", va="center", fontsize=9, color="black", fontweight="bold")

        ref_vals = plot_df.loc[plot_df[group_col] == ref, value_col].to_numpy(dtype=float)
        y_base = float(line_height)
        ref_idx = order.index(ref) + 1
        for i, g in enumerate(order):
            if g == ref:
                continue
            cur_vals = plot_df.loc[plot_df[group_col] == g, value_col].to_numpy(dtype=float)
            if method == "ttest":
                _, p_value = ttest_ind(cur_vals, ref_vals, equal_var=False, nan_policy="omit")
            elif method == "ranksum":
                _, p_value = ranksums(cur_vals, ref_vals)
            else:
                raise ValueError("method must be 'ttest' or 'ranksum'")
            label = self._format_p_value(float(p_value))
            x1, x2 = ref_idx, i + 1
            y = y_base
            y_base += float(line_offset)
            ax.plot([x1, x1, x2, x2], [y, y + 0.01, y + 0.01, y], color="black", lw=1.2)
            ax.text((x1 + x2) / 2.0, y + 0.012, label, ha="center", va="bottom", fontsize=8, color="black", fontweight="bold")

        ax.set_title(f"{value_col} by {group_col}")
        ax.set_xlabel(group_col)
        ax.set_ylabel(f"{value_col}{' (log2)' if log2 else ''}")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            y_min = float(np.nanmin(plot_df[value_col].to_numpy(dtype=float)))
            y_max = float(np.nanmax(plot_df[value_col].to_numpy(dtype=float)))
            margin = max(1e-6, 0.1 * (y_max - y_min + 1e-12))
            ax.set_ylim(y_min - margin, y_max + margin)
        plt.tight_layout()
        return ax

    def __getattr__(self, name: str):
        # compatibility aliases requested in notebooks/discussions
        if name == "2d_plot":
            return self.plot_2d
        if name == "3d_plot":
            return self.plot_3d
        raise AttributeError(name)
