# Data Directory Structure

This directory contains the raw and processed data for the TrackingTensor3D project. Due to size constraints and privacy regulations, the actual data files are not included in this repository.

## Directory Layout

- **`raw/ERN Raw Data BIDS-Compatible/`**: Place the raw ERP CORE (Ozdemir style) EEG datasets here. Each subject should have its own folder (e.g., `sub-001/eeg/*.set`).
- **`processed_v2/01_initial_epochs/`**: Output for initial preprocessing (Resampling, Filtering, CSD).
- **`processed_v2/02_balanced_theta_epochs/`**: Output for theta-band filtered and 1:1 balanced epochs.
- **`processed_v2/03_connectivity_tensors/`**: Stores the generated 4D connectivity tensors (`.npy`).
- **`processed_v2/04_matlab_results/`**: Stores the consolidated results for MATLAB analysis (`.mat`).

## How to use
1. Download the ERP CORE dataset.
2. Ensure the structure matches the `config.py` paths.
3. Run the pipeline as described in the root `README.md`.
