# `PlotAccessor.stacked_bar`
**Module:** `pl.py`

Plot stacked proportion bar chart for a categorical variable by sample/group.

## Parameters
- `group_col` (`str`): Grouping column. Default `sample_id`.
- `value_col` (`str`): Categorical column to stack. Default `pred_label`.
- `data` (`str`): Data source (`filtered` or `full/feature` fallback via accessor).
- `label_color` (`dict[str, str] | None`): Optional color mapping per category.
- `cmap` (`str`): Colormap name used when `label_color` not provided.
- `width` (`float`): Figure width.
- `height` (`float`): Figure height.

## Returns
- `matplotlib Axes`
