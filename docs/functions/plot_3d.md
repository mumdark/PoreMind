# `PlotAccessor.plot_3d`
**Module:** `pl.py`

Creates 3D scatter of selected feature axes with optional value mapping.

## Parameters
- `x` (`str`): X-axis column.
- `y` (`str`): Y-axis column.
- `z` (`str`): Z-axis column.
- `xlim` (`tuple[float,float] | None`): X-axis limits.
- `ylim` (`tuple[float,float] | None`): Y-axis limits.
- `zlim` (`tuple[float,float] | None`): Z-axis limits.
- `x_log2` (`bool`): Apply log2 transform to x.
- `y_log2` (`bool`): Apply log2 transform to y.
- `z_log2` (`bool`): Apply log2 transform to z.
- `data` (`str`): filtered or full dataframe source.
- `value` (`str | None`): Column used for color mapping.
- `width` (`float`): Figure width.
- `height` (`float`): Figure height.

## Returns
- `matplotlib Axes`
