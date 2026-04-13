# `PlotAccessor.plot_2d`
**Module:** `pl.py`

Creates 2D scatter of selected feature axes with optional group/value coloring.

## Parameters
- `x` (`str`): X-axis column.
- `y` (`str`): Y-axis column.
- `xlim` (`tuple[float,float] | None`): X-axis limits.
- `ylim` (`tuple[float,float] | None`): Y-axis limits.
- `x_log2` (`bool`): Apply log2 transform to x.
- `y_log2` (`bool`): Apply log2 transform to y.
- `data` (`str`): filtered or full dataframe source.
- `value` (`str | None`): Column used for color mapping.
- `width` (`float`): Figure width.
- `height` (`float`): Figure height.

## Returns
- `matplotlib Axes`
