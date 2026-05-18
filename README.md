# TrackingTensor3D: Dynamic Brain Network Analysis

## Overview
This repository implements **Higher-Order Recursive Least Squares Subspace Learning (HO-RLSL)** and **Fiedler Consensus Clustering Approach (FCCA)** for the analysis of dynamic functional connectivity networks (dFCNs). The implementation is organized as a **high-fidelity Ozdemir (2017) reproduction on ERP CORE**, with explicit documentation of the places where ERP CORE differs from the original study.

The pipeline is specifically applied to BIDS-compatible EEG data to track and analyze **Error-Related Negativity (ERN)** network reconfigurations. It extracts phase-locking value (PLV) tensors and identifies critical change points where the low-rank community structure of the functional connectivity network alters significantly.

---

## 🏗️ Project Structure & Clean Architecture

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
Converts raw EEG into dynamic functional connectivity tensors.
```bash
# 1. Apply CSD and create initial epochs
python3 -m preprocessing.pipeline

# 2. Filter Theta band and perform 1:1 Correct/Incorrect trial balancing
python3 -m preprocessing.sampling

# 3. Compute Phase-Locking Value (PLV) tensors (RID-Rihaczek)
python3 -m preprocessing.connectivity
```
*Outputs are stored in `data/processed_v2/03_connectivity_tensors/`.*

### Phase 2: Scientific Evaluation & Tracking
Runs the paper-comparison analyses on the saved tensors.
```bash
python3 -m analysis.reproduce_behavioral_metrics
python3 -m analysis.evaluate_fcca
```

---

## 📊 Scientific Validation

This repository aims for a **Table V-style high-fidelity comparison**, not an exact replication claim. The active analysis path uses one shared paper-comparison configuration for HO-RLSL and labels any remaining differences as ERP CORE dataset effects unless a code-level mismatch is demonstrated.

**Detected ERN Intervals (N=40 Subjects)**

| Interval | HO-RLSL (ERP CORE) | HoSVD (ERP CORE) | Original Paper (HO-RLSL) | Original Paper (HoSVD) |
| :--- | :--- | :--- | :--- |
| **Pre-ERN** | about -484 to -70 ms | about -484 to -47 ms | -484 to -109 ms | -484 to -47 ms |
| **ERN** | about -70 to +117 ms | about -47 to +141 ms | -109 to +141 ms | -47 to +141 ms |
| **Post-ERN**| about +117 to +766 ms | about +141 to +766 ms | +141 to +766 ms | +141 to +766 ms |

Notes:
- HO-RLSL uses the canonical paper-comparison configuration shared by the analysis scripts.
- HoSVD table values come from an explicit windowed paper-comparison approximation, not from the canonical `HOSVDRunner`.
- Residual timing shifts are expected because ERP CORE provides 40 subjects and 30 channels instead of the original 91 subjects and 63 channels.

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
