# `PlotAccessor.current`
**Module:** `pl.py`

Plots raw/denoised current in a specified time range (ms).

## Parameters
- `sample_id` (`str | None`): Trace id; None selects first trace.
- `current` (`str`): raw or denoise.
- `start_ms` (`float`): Start time in ms.
- `end_ms` (`float`): End time in ms.
- `width` (`float`): Figure width.
- `height` (`float`): Figure height.

## Returns
- `matplotlib Axes`
