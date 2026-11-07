# Graph Generation & Reproducibility Guide

This directory contains the automated, self-contained Python scripts used to validate experimental data, recalculate metrics, and generate publication-quality figures (PNG at 300 DPI and vector PDF) for the E-RAG research paper and methodology audit.

---

## 1. Directory Structure & Script Overview

- **`validate_and_export_data.py`:**
  - Loads ground-truth items from `data/manifests/test.jsonl` ($N=400$).
  - Validates question IDs and record sequence across all five systems (`closed_book_test.jsonl`, `standard_test.jsonl`, `ircot_test.jsonl`, `erag_observe_test.jsonl`, `erag_active_test.jsonl`).
  - Recalculates Exact Match, Token F1, Gold Recall, All-Gold, and Coverage from raw answer strings.
  - Exports validated CSV datasets into `research/methodology_audit/validated_plot_data/`.
- **`plot_five_systems.py`:**
  - Reads `validated_plot_data/test400_five_systems_per_question.csv` and `test400_aggregate_metrics.csv`.
  - Generates Figures J1 through J6 into `research/methodology_audit/five_system_plots/`.
- **`plot_ircot_vs_active.py`:**
  - Reads `validated_plot_data/ircot_vs_active_paired.csv`.
  - Generates Figures K3.1 through K3.7 into `research/methodology_audit/ircot_vs_erag_active_plots/`.

---

## 2. Environment & Dependency Specifications

All figures require standard scientific Python libraries:
- `python >= 3.10`
- `numpy >= 1.24.0`
- `pandas >= 2.0.0`
- `matplotlib >= 3.8.0`
- `scipy >= 1.10.0`
- `scikit-learn >= 1.3.0`

Styling uses `seaborn-v0_8-whitegrid` with serif typography, standard IEEE/Springer bounding boxes, and consistent system palettes:
- Closed-Book: `#7f7f7f` (Grey)
- Standard RAG: `#d62728` (Red)
- IRCoT: `#ff7f0e` (Orange)
- E-RAG Observe: `#1f77b4` (Blue)
- E-RAG Active: `#2ca02c` (Green)

---

## 3. How to Regenerate All Figures

Execute the following commands from the repository root:

```bash
# Step 1: Validate raw test results and export full-precision data
.venv/Scripts/python.exe research/methodology_audit/graph_generation/validate_and_export_data.py

# Step 2: Generate all Five-System Figures (J1 - J6)
.venv/Scripts/python.exe research/methodology_audit/graph_generation/plot_five_systems.py

# Step 3: Generate all IRCoT vs Active Paired Figures (K3.1 - K3.7)
.venv/Scripts/python.exe research/methodology_audit/graph_generation/plot_ircot_vs_active.py
```

All figures will be regenerated directly from source JSONL files, matching the metadata in `graph_data_manifest.csv`.

