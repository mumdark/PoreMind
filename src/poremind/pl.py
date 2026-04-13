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
