# HO-RLSL Subspace Tracking & FCCA Implementation Guide

This document outlines the technical implementation of the High-Order Recursive Least Squares Subspace (HO-RLSL) tracking and Fiedler Consensus Clustering Algorithm (FCCA) within the **TrackingTensor3D** project, based on **Ozdemir et al. (2017)**.

---

## 1. HO-RLSL Subspace Tracking (`algorithms/ho_rlsl.py`)

The HO-RLSL engine tracks the non-stationary dynamics of the EEG connectivity tensor.

### 1.1 Key Functions
- `HORLSLRunner._initial_bases()`: Builds the initial Tucker-mode bases from the first 10 time points across subjects.
- `HORLSLRunner.run()`: Applies the literal HO-RLSL loop: orthogonal projection, sparse recovery, batch update every `alpha`, then delete/add direction checks for change-point detection.
- `gtcs_s_recovery()`: Shared sparse recovery helper used to estimate the low-rank plus sparse split.

---

## 2. FCCA (`algorithms/fcca.py`)

FCCA identifies stable functional communities at the group level by aggregating individual subspace reconfigurations.

### 2.1 Methodology
1. **Consensus Matrix Computation**: `fcca_on_lowrank_interval()` builds the co-occurrence matrix $W$ across all subjects and time points in a selected interval.
2. **Recursive Repartitioning**: `recursive_repartitioning()` applies repeated Fiedler-based bipartitioning within each graph before consensus accumulation.
3. **Final Spectral Partitioning**: The Fiedler vector of the consensus matrix yields the final binary partition used for interval-level reporting.

---

## 3. Workflow & Execution

To replicate the analysis, follow these steps in order:

### A. Tensor Decomposition
```bash
python3 -m algorithms.ho_rlsl
```
- Input: `data/processed_v2/03_connectivity_tensors/tensor_incorrect_4d.npy`
- Output: Low-rank tensors and detected change-points (`horls_*.npy`).

### B. Table V-style Comparison
```bash
python3 -m analysis.reproduce_behavioral_metrics
```
- Reports ERP CORE intervals against the Ozdemir Table V targets.

### C. Dynamic Network Comparison
```bash
python3 -m analysis.evaluate_fcca
```
- Evaluates modularity across the primary Pre-ERN / ERN / Post-ERN windows and reports any supplementary late detections.

---

## 4. Parameter Standards (Paper-Comparison Path)

- **HO-RLSL train steps:** 10.
- **Update Window ($\alpha$):** 8 samples (62.5ms).
- **Sensitivity ($\sigma_{min}$):** 0.11.
- **HO-RLSL sparse penalty ($\lambda$):** 0.2 for the Table V-style analysis path.
- **Band:** Theta (4-8 Hz).
