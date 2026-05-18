import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from algorithms.hosvd import HOSVDRunner, default_config_for_condition as hosvd_cfg
from algorithms.ho_rlsl import HORLSLRunner, default_config_for_condition as horlsl_cfg
from algorithms.fcca import fcca_on_lowrank_interval
from algorithms.common import convert_subject_tensor_to_stream
from core.timing import eeg_timing_from_array

# ERP CORE 30-channel layout with Region Labels
CHANNELS = ["FP1", "F3", "F7", "FC3", "C3", "C5", "P3", "P7", "P9", "PO7", "PO3", "O1", "Oz", "Pz", "CPz", "FP2", "Fz", "F4", "F8", "FC4", "FCz", "Cz", "C4", "C6", "P4", "P8", "P10", "PO8", "PO4", "O2"]
REGIONS = {
    "Frontal": [0, 1, 2, 15, 16, 17, 18],
    "Central": [3, 4, 5, 19, 20, 21, 22, 23],
    "Parietal": [6, 7, 8, 13, 14, 24, 25, 26],
    "Occipital": [9, 10, 11, 12, 27, 28, 29]
}

def run_fcca_for_phases(result, stream_lowrank, timing):
    # Standardized phases using timing in ms
    pre_slice = timing.ms_window(-687.5, -375.0)
    post_slice = timing.ms_window(406.25, 718.75)
    
    pre_window = [pre_slice.start, pre_slice.stop - 1]
    post_window = [post_slice.start, post_slice.stop - 1]
    
    # ERN window detection near 0ms
    ern_slice = timing.ms_window(-31.25, 31.25)
    ern_window = [ern_slice.start, ern_slice.stop - 1] 
    
    for interval in result.intervals:
        if interval[0] <= timing.zero_index <= interval[1]:
            ern_window = interval
            break
            
    _, W_pre = fcca_on_lowrank_interval(stream_lowrank, pre_window[0], pre_window[1], threshold=0.15)
    _, W_ern = fcca_on_lowrank_interval(stream_lowrank, ern_window[0], ern_window[1], threshold=0.15)
    _, W_post = fcca_on_lowrank_interval(stream_lowrank, post_window[0], post_window[1], threshold=0.15)
    
    return (W_pre, W_ern, W_post), (pre_window, ern_window, post_window)

def plot_final_comparison():
    print("🎬 Running Final Scientific Comparison: HOSVD vs HO-RLSL...")
    os.makedirs("outputs", exist_ok=True)
    
    data = np.load("data/processed_v2/03_connectivity_tensors/tensor_incorrect_4d.npy").astype(np.float32)
    stream = convert_subject_tensor_to_stream(data)
    timing = eeg_timing_from_array(stream, time_axis=0)
    
    algos = {
        "HOSVD (Static Baseline)": HOSVDRunner(hosvd_cfg()),
        "HO-RLSL (Ozdemir 2017)": HORLSLRunner(horlsl_cfg())
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    
    for row_idx, (name, runner) in enumerate(algos.items()):
        print(f"   Processing {name}...")
        result = runner.run(stream)
        ws, windows = run_fcca_for_phases(result, result.lowrank_stream, timing)
        
        # 1. Change Point Plot
        ax_cp = axes[row_idx, 0]
        ax_cp.plot(result.residual_energy, color='blue', alpha=0.3)
        for cp in result.change_points:
            ax_cp.axvline(x=cp, color='red', linestyle='--', linewidth=0.8)
        ax_cp.axvline(x=timing.zero_index, color='black', linewidth=2, label='Response')
        ax_cp.set_title(f"{name}\nChange Point Dynamics")
        ax_cp.set_ylabel("Normalized Energy")
        
        # 2-4. Phase Heatmaps
        phases = ["Pre-ERN", "ERN Phase", "Post-ERN"]
        for col_idx in range(3):
            ax_h = axes[row_idx, col_idx + 1]
            sns.heatmap(ws[col_idx], ax=ax_h, cmap='magma', cbar=False, square=True)
            ax_h.set_title(f"{phases[col_idx]}\n{windows[col_idx]}")
            
            # Add Region Borders for interpretability
            for region, indices in REGIONS.items():
                start, end = indices[0], indices[-1]
                ax_h.axvline(x=start, color='white', alpha=0.2, linewidth=0.5)
                ax_h.axhline(y=start, color='white', alpha=0.2, linewidth=0.5)

    plt.suptitle("Functional Connectivity Reconfiguration: HOSVD vs HO-RLSL", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("outputs/final_scientific_comparison.png", dpi=300)
    print("✅ Final comparison saved to outputs/final_scientific_comparison.png")

if __name__ == "__main__":
    plot_final_comparison()
