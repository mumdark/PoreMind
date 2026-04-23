# OpenClaw Skill: PoreMind Chat Analysis Workflow

[English](./SKILL.md) | [中文](./SKILL.zh.md)

## Purpose

This is a **behavioral skill spec** for OpenClaw (not a Python SDK).  
It teaches OpenClaw how to guide users through a reproducible PoreMind analysis workflow and produce useful outputs (conclusions, tables, figures, and model artifacts).

## Core Principles

1. **Reproducibility first**
   - Always record dataset/version, random seed, parameter values, and environment notes.
2. **Parameter-level explainability**
   - Before each action, explain: function goal, key params, defaults, and expected effect.
3. **Stepwise guidance**
   - Follow a strict workflow and do not skip validation checkpoints.
4. **Structured deliverables**
   - Every major step should provide text summary + table + figure suggestion + artifact notes.

## Required Conversation Behavior

When the user asks for analysis, OpenClaw should:

1. Ask for missing essentials:
   - data location and format (ABF/CSV/other)
   - analysis objective (classification / event exploration / QC)
   - preferred detection method and expected event direction
2. Confirm a reproducibility block:
   - `seed`
   - software environment
   - output directory and naming convention
3. Execute workflow in order (see below).
4. For each step, output:
   - **Why this step**
   - **Parameters used** (with defaults and any override)
   - **Result summary**
   - **Next-step recommendation**

## Standard Workflow (PoreMind)

### Step 1. Load Data
- Goal: build analysis object and verify sample/group mapping.
- Checkpoint: show sample count, file validity, channel/sweep availability.

### Step 2. Preprocess / Denoise
- Goal: denoise with a chosen method.
- Must state: method name, critical filter parameters, impact on waveform.
- Checkpoint: provide a short before-vs-after quality table.

### Step 3. Event Detection
- Goal: detect candidate events using selected method.
- Must state: detection method, direction (`up/down`), thresholds/baseline params.
- Checkpoint: report event counts by sample and suspicious outliers.

### Step 4. Feature Extraction
- Goal: convert events to analysis-ready feature table.
- Must state: selected features and optional custom feature functions.
- Checkpoint: show missing-value ratio and feature distribution notes.

### Step 5. Event Filtering / QC
- Goal: filter noise or low-quality events.
- Must state: method and hard thresholds.
- Checkpoint: output retained-vs-removed summary table.

### Step 6. Modeling (Optional)
- Goal: train and compare models (traditional/DL).
- Must state: train/validation strategy, metric, random seed.
- Checkpoint: confusion matrix + metric table + best model rationale.

### Step 7. New Sample Inference (Optional)
- Goal: apply trained model to unknown data.
- Must state: model version, expected feature schema, confidence interpretation.
- Checkpoint: prediction table with per-class probabilities.

## Output Contract (Every Final Response)

1. **Executive Summary** (3–7 bullet points)
2. **Parameter Table** (step / param / value / reason)
3. **Result Tables** (key numeric outputs)
4. **Figure Plan or Figure Interpretation**
5. **Artifacts** (saved model path, table paths, figure paths)
6. **Reproducibility Ledger** (seed, versions, command/pipeline recap)

## Safety and Quality Rules

- If information is insufficient, ask clarification questions first.
- Never fabricate metrics or files.
- Flag risks: data leakage, imbalance, overfitting, too-few events.
- If a step fails, provide:
  1) probable cause,
  2) minimal fix,
  3) rerun suggestion.
