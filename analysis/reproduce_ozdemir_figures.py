import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from algorithms.hosvd import HOSVDRunner, default_config_for_condition as hosvd_cfg
from algorithms.ho_rlsl import HORLSLRunner, default_config_for_condition as horlsl_cfg
from fcca import fcca_on_lowrank_interval
from algorithms.common import convert_subject_tensor_to_stream
from core.timing import eeg_timing_from_array

# Electrode Labels from ERP CORE 30-channel layout
CHANNELS = ["FP1", "F3", "F7", "FC3", "C3", "C5", "P3", "P7", "P9", "PO7", "PO3", "O1", "Oz", "Pz", "CPz", "FP2", "Fz", "F4", "F8", "FC4", "FCz", "Cz", "C4", "C6", "P4", "P8", "P10", "PO8", "PO4", "O2"]

def run_fcca_for_phases(result, stream_lowrank, timing):
    # Pre-ERN: frames 40-80 (roughly -687.5ms to -375ms)
    # ERN: interval containing zero_index
    # Post-ERN: frames 180-220 (roughly 406.25ms to 718.75ms)
    
    pre_slice = timing.ms_window(-687.5, -375.0)
    post_slice = timing.ms_window(406.25, 718.75)
    
    pre_window = [pre_slice.start, pre_slice.stop - 1]
    post_window = [post_slice.start, post_slice.stop - 1]
    
    ern_slice = timing.ms_window(-31.25, 31.25)
    ern_window = [ern_slice.start, ern_slice.stop - 1] # Default
    
    for interval in result.intervals:
        if interval[0] <= timing.zero_index <= interval[1]:
            ern_window = interval
            break
            
    _, W_pre = fcca_on_lowrank_interval(stream_lowrank, pre_window[0], pre_window[1], threshold=0.2)
    _, W_ern = fcca_on_lowrank_interval(stream_lowrank, ern_window[0], ern_window[1], threshold=0.2)
    _, W_post = fcca_on_lowrank_interval(stream_lowrank, post_window[0], post_window[1], threshold=0.2)
    
    return (W_pre, W_ern, W_post), (pre_window, ern_window, post_window)

def plot_reproduction():
    print("🎬 Reproducing Ozdemir Figures for HOSVD and HO-RLSL...")
    os.makedirs("outputs", exist_ok=True)
    
    data = np.load("data/processed_v2/03_connectivity_tensors/tensor_incorrect_4d.npy").astype(np.float32)
    stream = convert_subject_tensor_to_stream(data)
    timing = eeg_timing_from_array(stream, time_axis=0)
    
    algos = {
        "HOSVD": HOSVDRunner(hosvd_cfg()),
        "HO-RLSL": HORLSLRunner(horlsl_cfg())
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    
    for row_idx, (name, runner) in enumerate(algos.items()):
        print(f"   Processing {name}...")
        result = runner.run(stream)
        ws, windows = run_fcca_for_phases(result, result.lowrank_stream, timing)
        
        # 1. Change Point Plot
        ax_cp = axes[row_idx, 0]
        ax_cp.plot(result.residual_energy, color='gray', alpha=0.5)
        for cp in result.change_points:
            ax_cp.axvline(x=cp, color='red', linestyle='--', linewidth=1)
        ax_cp.axvline(x=timing.zero_index, color='black', linewidth=2)
        ax_cp.set_title(f"{name} Tracking & CPs")
        ax_cp.set_ylabel("Energy / Residual")
        
        # 2. Pre-ERN Heatmap
        sns.heatmap(ws[0], ax=axes[row_idx, 1], cmap='viridis', cbar=False, square=True)
        axes[row_idx, 1].set_title(f"Pre-ERN {windows[0]}")
        
        # 3. During-ERN Heatmap
        sns.heatmap(ws[1], ax=axes[row_idx, 2], cmap='viridis', cbar=False, square=True)
        axes[row_idx, 2].set_title(f"ERN Phase {windows[1]}")
        
        # 4. Post-ERN Heatmap
        sns.heatmap(ws[2], ax=axes[row_idx, 3], cmap='viridis', cbar=False, square=True)
        axes[row_idx, 3].set_title(f"Post-ERN {windows[2]}")
        
        # Labels for heatmaps (only for first row/column to save space)
        if row_idx == 0:
            axes[row_idx, 1].set_xlabel("Nodes (30)")
        
    plt.tight_layout()
    plt.savefig("outputs/ozdemir_reproduction_comparison.png", dpi=300)
    print("✅ Full reproduction comparison saved to outputs/ozdemir_reproduction_comparison.png")

if __name__ == "__main__":
    plot_reproduction()
