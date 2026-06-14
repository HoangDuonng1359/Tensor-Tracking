# Brain Network Tracking: HO-RLSL v1 vs HO-RLSL v2

This project compares **HO-RLSL v1** and **HO-RLSL v2** for tracking low-rank subspace dynamics and change points in functional brain connectivity tensors on the ERN (Error-Related Negativity) EEG dataset.

---

## 📂 Repository Structure

The project has a portable layout:

```
tensor_tracking_new-optimized/
├── requirements.txt            # Python package dependencies
├── .gitignore                  # Git tracking rules
├── README.md                   # Project documentation
└── tensor_tracking/
    ├── data/                   # Raw EEG BIDS dataset
    │   └── README.md
    ├── src/                    # Source packages
    │   ├── preprocessing/      # Epoch cleaning and config
    │   ├── low_rank_extraction/# HO-RLSL v1 & v2 trackers
    │   ├── change_point_detection/ # CPD routines
    │   └── fcca/               # Consensus community clustering
    ├── notebooks/              # Jupyter walkthroughs
    │   └── Experiment_Changepoints.ipynb
    └── outputs/                # Pre-computed tensors and figure outputs
```

---

## 🛠️ Environment Setup & Installation

To run this repository on any computer without errors, follow these steps:

### 1. Create a Virtual Environment (Optional but recommended)
Open your terminal inside the cloned project directory and run:
```bash
# Python 3.10+ is recommended
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies
Install all required libraries from the unified dependencies list:
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Walkthrough Notebook

The core experiments are conducted in the Jupyter Notebook:
`tensor_tracking/notebooks/Experiment_Changepoints.ipynb`

### Dynamic Path Resolution
This project uses **100% relative and dynamic path resolution**:
* All absolute machine paths have been removed.
* The notebook resolves `ROOT` dynamically based on the current workspace location.
* Running all cells from top to bottom will successfully execute the entire comparison pipeline and display figures inline without path-not-found errors.

---

## 📊 Outputs & Visualizations

Running the notebook generates the following primary output figures inside `tensor_tracking/outputs/`:

1. **`ho_rlsl_v1_vs_v2_comparison.png`**: A 2x2 grid plotting the Grand Average ERP alongside detected change points for v1 and v2 under both ERN (Incorrect) and CRN (Correct) trials.
2. **`fcca_consensus_matrices_comparison.png`**: Consensus matrices ($W$) across Pre-Event, Event, and Post-Event intervals.
3. **`fcca_brain_networks_comparison.png`**: Scalp outline network topologies mapping out community structures using Fiedler consensus clustering.
