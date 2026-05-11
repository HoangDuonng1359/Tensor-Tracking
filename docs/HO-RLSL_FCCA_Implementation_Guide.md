# HO-RLSL Subspace Tracking & FCCA Implementation Guide

This document outlines the technical implementation of the High-Order Recursive Least Squares Subspace (HO-RLSL) tracking and Fiedler Consensus Clustering Algorithm (FCCA) within the **TrackingTensor3D** project, based on **Ozdemir et al. (2017)**.

---

## 1. HO-RLSL Subspace Tracking (`algorithms/ho_rlsl.py`)

The HO-RLSL engine tracks the non-stationary dynamics of the EEG connectivity tensor.

### 1.1 Key Functions
- `initialize_tucker_subspace()`: Performs HOSVD on the baseline tensor (-1000ms to 0ms) to create the initial spatial ($U$) and subject ($V$) bases.
- `update_recursive_subspace()`: The core online update step. It projects new data onto the orthogonal complement of $U$ and identifies novel directions using an adaptive threshold ($\sigma_{min}$).
- `extract_sparse_components()`: Uses ISTA ($L_1$ minimization) to isolate sparse noise and artifacts from the low-rank neural signal.
- `decompose_tensor_stream()`: The main loop that iterates through time, managing subspace velocity calculation and change-point detection.

---

## 2. FCCA (Fiedler Consensus Clustering) (`fcca/`)

FCCA identifies stable functional communities at the group level by aggregating individual subspace reconfigurations.

### 2.1 Methodology
1. **Consensus Matrix Computation**: `compute_consensus_clusters()` in `fcca/consensus.py` builds a co-occurrence matrix $W$ across all subjects and time windows.
2. **Spectral Partitioning**: Nodes are assigned to clusters based on the sign of the Fiedler vector (derived from the Laplacian of $W$).
3. **Dynamic Assessment**: `fcca/dynamics.py` compares the consensus topology between Baseline and ERN periods to quantify network integration/segregation.

---

## 3. Workflow & Execution

To replicate the analysis, follow these steps in order:

### A. Tensor Decomposition
```bash
python3 -m algorithms.ho_rlsl
```
- Input: `connectivity/tensor_incorrect_4d.npy`
- Output: Low-rank tensors and detected change-points (`horls_*.npy`).

### B. Group Consensus Clustering
```bash
python3 -m fcca.consensus
```
- Identifies global functional communities during the ERN window (0-150ms).

### C. Dynamic Network Comparison
```bash
python3 -m fcca.dynamics
```
- Compares Pre-Response vs. Post-Response network states.

---

## 4. Parameter Standards (Ozdemir 2017)

- **Rank ($r$):** 5 (Captures ~85% variance in connectivity).
- **Update Window ($\alpha$):** 8 samples (62.5ms).
- **Sensitivity ($\sigma_{min}$):** 0.063 (Calibrated for ERN detection).
- **Band:** Theta (4-8 Hz).

