# `PlotAccessor.box_significance`
**Module:** `pl.py`

Plot grouped boxplots for a numeric column and annotate significance against a reference group.

## Parameters
- `group_col` (`str`): Group column. Default `label`.
- `value_col` (`str`): Numeric column. Default `blockade_ratio`.
- `data` (`str`): Data source (`filtered` or `full/feature` fallback via accessor).
- `method` (`str`): Significance test method: `ttest` or `ranksum`.
- `label_color` (`dict[str, str] | None`): Optional color mapping per group.
- `cmap` (`str`): Colormap used for group colors when `label_color` not provided.
- `reference_group` (`str | None`): Group used as reference; default is highest-median group.
- `line_offset` (`float`): Vertical increment factor between significance lines (scaled by y-range).
- `line_height` (`float`): Initial offset factor above the top value (scaled by y-range).
- `ylim` (`tuple[float, float] | None`): Optional y-axis range.
- `log2` (`bool`): Whether to apply `log2(abs(x)+1e-12)` transform before plotting/testing.
- `width` (`float`): Figure width.
- `height` (`float`): Figure height.

## Returns
- `matplotlib Axes`
