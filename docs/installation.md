# Installation

## 1) Create and activate environment

```bash
conda create -n poremind python=3.10 -y
conda activate poremind
```

## 2) Install package (editable mode)

```bash
pip install -e .
```

## 3) Install ABF reader dependency

```bash
pip install pyabf
```

## 4) Optional: launch local UI

```bash
poremind-ui
# or
python -m ui.app
```

## 5) Smoke test

```python
from poremind import __version__, create_analysis_object
print(__version__)
```
