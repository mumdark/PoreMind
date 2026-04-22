# Single-Molecule Nanopore Current Analysis Framework (v0.1)

## 1. Scope

This document defines the method framework and implementation structure for single-molecule nanopore current analysis in two execution modes:

1. Offline analysis
2. Near-real-time processing

The framework includes the following functional chain:

1. Current data ingestion (ABF/CSV/HDF5/custom acquisition format)
2. Denoising and drift correction
3. Baseline estimation and event detection (blockade event identification)
4. Event-level and trajectory-level feature extraction
5. Classification and anomaly detection
6. Visualization, review, and report export

---

## 2. Layered Architecture

The system uses a three-layer architecture with a fully pluggable algorithm layer.

### 2.1 Data Layer
- `io/reader`: Unified input adapters (Reader Interface)
- `io/schema`: Internal canonical data model (timestamp, current, sampling rate, metadata)
- `storage/cache`: Waveform cache, event index cache, and feature table cache

### 2.2 Algorithm Layer
- `preprocess`: Spike removal, low-pass/band-pass filtering, wavelet denoising, drift correction
- `baseline`: Sliding robust baseline, piecewise baseline, EM/state-space baseline
- `event_detect`: Threshold method, CUSUM, change-point detection, HMM segmentation
- `feature`: Duration, blockade amplitude, area, rise/recovery time, spectral descriptors
- `model`: Classifiers, clustering, and out-of-distribution detection

### 2.3 Application Layer
- `pipeline/orchestrator`: Configurable DAG pipeline orchestration
- `ui`: Waveform viewer, event replay, manual annotation correction
- `report`: PDF/HTML report generation (parameters, figures, confusion matrix, ROC)
- `service`: CLI and REST API interface

---

## 3. Core Data Structures

### 3.1 Trace (continuous current trajectory)
- `trace_id`
- `time[]` / `current[]`
- `sampling_rate`
- `voltage`, `buffer_info`, `pore_id`, `temperature`
- `preprocess_meta` (filter parameters and version)

### 3.2 Event (candidate blockade event)
- `event_id`, `trace_id`
- `start_idx`, `end_idx`
- `baseline_local`
- `delta_I` (blockade depth)
- `dwell_time`
- `snr`
- `quality_flag`

### 3.3 FeatureVector
- Statistical features: `mean`, `std`, `skew`, `kurtosis`
- Morphological features: `rise_time`, `fall_time`, `plateau_ratio`
- Frequency-domain features: dominant frequency, band energy, spectral entropy
- Task-specific features: multilevel step count, sub-event count

---

## 4. Analysis Workflow

### 4.1 Preprocessing
1. Suppress line-frequency and high-frequency noise via sampling-rate-aware low-pass filtering
2. Remove isolated spikes with robust filters
3. Correct baseline drift using large-window trend estimation

### 4.2 Baseline Estimation and Event Detection
1. Compute a robust local baseline on the trajectory
2. Derive residual current `r(t) = I(t) - baseline(t)`
3. Trigger candidate events using residual statistics
4. Confirm boundaries with change-point-aware segmentation
5. Apply event-level validity constraints (minimum duration/depth and adjacency merge rules)

### 4.3 Feature Extraction and Decision
1. Extract single-event features and context-window stability features
2. Construct feature matrix and model inputs
3. Perform supervised classification or unsupervised anomaly screening
4. Output class scores and uncertainty-related indicators

### 4.4 Quality Control
- Trajectory-level QC: noise RMS, drift slope, valid event rate
- Model-level QC: recall, PR-AUC, calibration error
- Runtime-level QC: throughput and end-to-end latency

---

## 5. Reference Project Structure

```text
nanopore_tool/
  pyproject.toml
  src/nanopore_tool/
    io/
    preprocess/
    baseline/
    detect/
    feature/
    model/
    pipeline/
    qc/
    viz/
    cli/
  configs/
    default.yaml
  tests/
    test_io.py
    test_detect.py
    test_feature.py
```

---

## 6. Implementation Mapping

- Data ingestion and schema normalization are implemented in `io/`.
- Signal preprocessing, baseline estimation, and event detection are implemented in `preprocess/`, `baseline/`, and `detect/`.
- Event descriptors and feature vectors are implemented in `feature/`.
- Classification and anomaly detection are implemented in `model/`.
- Pipeline scheduling, quality control, visualization, and interfaces are implemented in `pipeline/`, `qc/`, `viz/`, and `cli/`.

