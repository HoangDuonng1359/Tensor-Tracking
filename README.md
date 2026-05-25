# TrackingTensor3D: Dynamic Brain Network Analysis

## Overview
This repository implements **Higher-Order Recursive Least Squares Subspace Learning (HO-RLSL)** and **Fiedler Consensus Clustering Approach (FCCA)** for the analysis of dynamic functional connectivity networks (dFCNs). The implementation is organized as a **high-fidelity Ozdemir (2017) reproduction on ERP CORE**, with explicit documentation of the places where ERP CORE differs from the original study.

The pipeline is specifically applied to BIDS-compatible EEG data to track and analyze **Error-Related Negativity (ERN)** network reconfigurations. It extracts phase-locking value (PLV) tensors and identifies critical change points where the low-rank community structure of the functional connectivity network alters significantly.

---

## 🏗️ Project Structure & Architecture

The codebase follows a modular, research-oriented architecture, clearly separating data extraction, mathematical modeling, and evaluation:

- **`core/`**: Centralized configurations (`config.py`) and signal timing constants (`timing.py`).
- **`preprocessing/`**: The robust ETL pipeline. 
  - Handles BIDS EEG loading and Current Source Density (CSD) transformation.
  - Generates 1:1 balanced epochs using temporal matching.
  - Computes high-resolution Phase-Locking Value (PLV) tensors using RID-Rihaczek TFD.
- **`algorithms/`**: The core mathematical engines.
  - `ho_rlsl.py`: The recursive tensor tracking algorithm with batch update ($\alpha$-window) support.
  - `hosvd.py`: Baseline Tucker decomposition and Windowed tracking.
  - `time_frequency.py`: Time-Frequency Distribution utilities.
  - `fcca.py`: Fiedler Consensus Clustering algorithms.
- **`analysis/`**: Scientific evaluation and reporting.
  - `paper_alignment.py`: Shared paper-comparison defaults, timing helpers, and interval semantics.
  - `reproduce_behavioral_metrics.py`: Table V-style comparison on ERP CORE.
  - `evaluate_fcca.py`: FCCA modularity analysis over primary ERN windows plus supplementary late detections.
- **`data/`**: (Ignored in VCS) Local storage for raw BIDS data and processed 4D `.npy` and `.mat` tensors.

---

## 🚀 Installation & Setup

1. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Preparation**:
   Ensure the raw BIDS-compatible EEG dataset is placed in `data/raw/ERN Raw Data BIDS-Compatible/`.

---

## ⚙️ Execution Pipeline

The full analysis pipeline should be executed sequentially from the project root:

### Phase 1: Preprocessing & Tensor Construction
Converts raw EEG into dynamic functional connectivity tensors (4D Tensors: Time × Frequency × Channel × Subject).
```bash
# 1. Apply CSD (Current Source Density) and create initial epochs
python3 -m preprocessing.pipeline

# 2. Filter Theta band and perform 1:1 Correct/Incorrect trial balancing (Temporal matching)
python3 -m preprocessing.sampling

# 3. Compute Phase-Locking Value (PLV) tensors using RID-Rihaczek TFD
python3 -m preprocessing.connectivity
```
*Outputs are stored as 4D `.npy` and `.mat` tensors in `data/processed_v2/03_connectivity_tensors/`.*

### Phase 2: Scientific Evaluation & Tracking
Runs the paper-comparison analyses on the saved tensors.
```bash
python3 -m analysis.reproduce_behavioral_metrics
python3 -m analysis.evaluate_fcca
```

---

## 📊 Scientific Validation (HO-RLSL vs HOSVD)

This repository aims for a **Table V-style high-fidelity comparison** between HO-RLSL and HOSVD, validated against the ERP CORE dataset.

### 1. Change Point (CP) Extraction & Network Topology
HO-RLSL identifies robust change points defining the response-related window:

| Condition | Number of CPs | Response Interval (ms) | Modularity | W/B Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **ERN (Incorrect)** | 6 | **15.6 – 132.8** | 0.5593 | 3.9804 |
| **CRN (Correct)** | 6 | **15.6 – 132.8** | 0.5427 | 6.4429 |

*Note on CRN Functional Integration*: In the CRN window, the W/B Ratio spikes to **6.44**, while Modularity slightly decreases compared to the PRE-CRN state. This indicates the network re-organizes from several fragmented communities into a highly integrated giant component to process the correct response.

### 2. Baseline Comparison (HO-RLSL vs HOSVD)

| Method | Response Interval | Overlap (0-150ms) | Modularity | W/B Ratio | Dist ERN-CRN | Permutation p-value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HOSVD** | 0.0–117.2 ms | 78.1% | 0.0000 | N/A (1 cluster) | **7.7460** | < 0.0001 |
| **HO-RLSL** | **15.6–132.8 ms**| **78.1%** | 0.5593 | **3.9804** | 1.3340 | < 0.0001 |

### 3. Subject Bootstrapping Stability (0-200ms, 100 runs, ±20ms)

| Algorithm | Change Point (ms) | Stability (%) | Cluster Stability (ARI) |
| :--- | :--- | :--- | :--- |
| **HO-RLSL** | -54.7, 7.8, 70.3, 132.8, 195.3 | ~31 - 37% | **0.6187 ± 0.1283** |
| **HOSVD** | None | 0% | 0.0038 ± 0.0647 |

**Conclusion:** 
Permutation tests confirm that both methods separate ERN from CRN significantly ($p < 0.0001$). However, HO-RLSL is markedly superior for tracking dynamic changes: it maintains strong community structure (high W/B ratio) and provides more stable response-related change points under bootstrapping, whereas HOSVD suffers from network degeneracy (Modularity = 0).

## Known Differences From Ozdemir 2017

- Dataset: ERP CORE contributes 40 participants instead of the original 91.
- Sensor layout: this repo uses a 30-channel subset rather than the paper's 63-channel montage.
- Trial balancing: correct trials are selected by temporal matching instead of random subsampling.
- Late dynamics: HO-RLSL may detect supplementary post-ERN reconfigurations after the primary Table V-style post-ERN window.

---

## ⚙️ Key Configurations
All tracking and preprocessing parameters are tightly controlled in `core/config.py`:
- `SFREQ`: 128Hz
- `TMIN` / `TMAX`: [-1.0, 1.0]s (0ms represents the behavioral response).
- `alpha`: 8 (Window size for tracking and batch updates).
- `sigma_min`: 0.11 (Singular value threshold for noise reduction).
