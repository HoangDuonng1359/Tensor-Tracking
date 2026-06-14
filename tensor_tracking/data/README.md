# EEG Dataset Configuration (ERN Raw Data)

This folder contains the raw EEG dataset used in the brain network tracking experiment.

## Data Setup Instructions

Depending on your use case, you can run the project either from scratch (preprocessing raw files) or directly from the pre-computed connectivity tensors:

### Mode A: Run from Pre-computed Tensors (Recommended & Fast)
If you only want to run the HO-RLSL low-rank extraction, change point detection, and FCCA consensus community visualization:
1. **No raw EEG data is required.**
2. The project comes pre-packaged with the computed connectivity tensors inside the outputs folder:
   * `tensor_tracking/outputs/processed_1024/03_connectivity_tensors/tensor_incorrect_4d.npy`
   * `tensor_tracking/outputs/processed_1024/03_connectivity_tensors/tensor_correct_4d.npy`
3. Simply open and run the Jupyter Notebook `tensor_tracking/notebooks/Experiment_Changepoints.ipynb`. It will automatically fall back to mock ERP coordinates for scalp visualization and load these pre-computed tensors without any errors.

---

### Mode B: Preprocess Raw EEG Data from Scratch
If you wish to run the entire EEG signal processing and connectivity pipeline (using `src/preprocessing/pipeline_1024.py` or `src/main.py` without `--skip-preprocessing`):
1. **Raw EEG Dataset:** Place your raw BIDS-compatible EEG dataset in this directory under the name `ERN_Raw_Data_BIDS-Compatible`.
2. **Directory Structure:**
   ```
   tensor_tracking/data/ERN_Raw_Data_BIDS-Compatible/
   ├── sub-001/
   │   └── eeg/
   │       ├── sub-001_task-ERN_channels.tsv
   │       ├── sub-001_task-ERN_electrodes.tsv     <-- Crucial for FCCA Scalp Coordinates
   │       └── sub-001_task-ERN_eeg.set (and .fdt) <-- Raw EEG Recording
   ├── sub-002/
   ...
   ```
3. **Electrode Layout TSV files (`*_electrodes.tsv`):** These files contain the 3D coordinates (x, y, z) for mapping channels (e.g., `Fp1`, `Fz`, `Cz`, `Pz`) to 2D screen positions. The FCCA visualization scripts parse these files to render brain networks on a 2D scalp outline. If they are not found, the plotting scripts will fallback to displaying nodes in a circular layout (`N1` to `N30`).
