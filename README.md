# TrackingTensor3D

This repository implements High-Order Recursive Least Squares Subspace (HO-RLSL) tracking and Fiedler Consensus Clustering Algorithm (FCCA) for EEG connectivity analysis, based on **Ozdemir et al. (2017)**.

## Project Structure

The project is organized into a modular, research-oriented architecture:

- **`core/`**: Configuration system using Python dataclasses.
- **`preprocessing/`**: ETL, CSD transformation, and connectivity (PLV) computation.
- **`algorithms/`**: Mathematical engines for HO-RLSL and Time-Frequency Distributions.
- **`fcca/`**: Group-level consensus clustering and dynamic assessment.
- **`eda/`**: Exploratory Data Analysis, diagnostics, and visualization scripts.

## Data Output Structure

Processed data is stored in `data/processed_v2/` with a numbered sequence for clarity:
- **`01_initial_epochs/`**: Standardized epochs after CSD and filtering.
- **`02_balanced_theta_epochs/`**: Refined epochs (Theta band, 1:1 balanced).
- **`03_connectivity_tensors/`**: Core 4D PLV tensors in `.npy` format.
- **`04_matlab_results/`**: Consolidated results in `.mat` format for MATLAB analysis.

## Installation

1. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Execution Pipeline

The pipeline should be executed in the following order from the project root:

1. **Step 1: Initial Preprocessing**
   Processes raw EEG data, applies CSD, and creates epochs.
   ```bash
   python3 -m preprocessing.pipeline
   ```

2. **Step 2: Group Connectivity Computation**
   Computes PLV tensors using RID-Rihaczek TFD with temporal matching (all 40 subjects included).
   ```bash
   python3 -m preprocessing.connectivity
   ```

3. **Step 3: Dataset Refinement**
   Filters data in the Theta band and performs 1:1 trial balancing.
   ```bash
   python3 -m preprocessing.cleaning
   ```

4. **Step 4: Tensor Decomposition (HO-RLSL)**
   Runs the subspace tracking algorithm to detect change-points and recover low-rank dynamics.
   ```bash
   python3 -m algorithms.ho_rlsl
   ```

5. **Step 5: Consensus Clustering (FCCA)**
   Identifies stable neural communities across the group.
   ```bash
   python3 -m fcca.consensus
   ```

## Visualization & Diagnostics

Visualization scripts are located in the `eda/` folder. Examples:
- View preprocessing QC: `python3 -m eda.viz_preprocessing_qc`
- View connectivity dynamics: `python3 -m eda.viz_fcca_connectivity`
- View group comparison: `python3 -m eda.group_comparison`

## Configuration

All paths and hyperparameters are centralized in `core/config.py`. Key parameters include:
- `SFREQ`: Sampling frequency (128Hz).
- `THETA_BAND`: Frequency range for analysis (4-8Hz).
- `MIN_INCORRECT_TRIALS`: Set to 1 to include all subjects with errors.
