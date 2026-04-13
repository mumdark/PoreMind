# `MultiSampleAnalysis.denoise`
**Module:** `workflow.py`

Applies selected preprocessing method to each loaded trace and caches denoised signals.

## Parameters
- `method` (`str`): Preprocessing method name. Default: butterworth_filtfilt.
- `**kwargs` (`Any`): Method-specific parameters.

## Returns
- `Self (MultiSampleAnalysis).`
