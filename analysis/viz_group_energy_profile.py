import numpy as np
import matplotlib.pyplot as plt
from core.config import config
from core.timing import eeg_timing
from algorithms.ho_rlsl import HORLSLRunner, default_config_for_condition
from algorithms.hosvd import HOSVDRunner
from dataclasses import replace

def plot_group_energy_profile():
    print("Generating Group Energy Profile (40 Subjects Combined)...")
    
    # 1. Load Real Data
    data = np.load("data/processed_v2/03_connectivity_tensors/tensor_incorrect_4d.npy")
    stream = np.transpose(data, (3, 1, 2, 0)).astype(np.float32)
    timing = eeg_timing(stream.shape[0])
    time_ms = timing.time_ms

    # 2. Setup Algorithms with Optimal Group Parameters
    cfg = default_config_for_condition()
    cfg.sigma_min = 15.0 # Calibrated for 40 subjects scale
    
    # Run HO-RLSL (with Sparsity)
    print(" Running HO-RLSL...")
    runner_rlsl = HORLSLRunner(replace(cfg, lambda_sparse=0.2))
    res_rlsl = runner_rlsl.run(stream)
    
    # Run HOSVD (Baseline)
    print(" Running HOSVD...")
    runner_hosvd = HOSVDRunner(replace(cfg, lambda_sparse=0.0))
    res_hosvd = runner_hosvd.run(stream)

    # 3. Visualization
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Normalize for visual comparison
    e_rlsl = res_rlsl.residual_energy / np.max(res_rlsl.residual_energy)
    e_hosvd = res_hosvd.residual_energy / np.max(res_hosvd.residual_energy)
    
    ax.plot(time_ms, e_rlsl, color='#FF4B4B', label='HO-RLSL (Sparsity Enabled)', linewidth=2)
    ax.plot(time_ms, e_hosvd, color='#4B9BFF', label='HOSVD (Baseline)', linewidth=1.5, alpha=0.6)
    
    # Mark the ERN Window
    ax.axvspan(50, 150, color='white', alpha=0.1, label='Expected ERN Window')
    ax.axvline(0, color='yellow', linestyle='--', label='Response (0ms)')
    
    ax.set_title("Group Residual Energy Profile (N=40 Subjects)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (ms) relative to response")
    ax.set_ylabel("Normalized Residual Energy")
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    output_path = config.paths.OUTPUTS_DIR / "group_energy_comparison.png"
    plt.savefig(output_path, dpi=200)
    print(f"Comparison plot saved to: {output_path}")

if __name__ == "__main__":
    plot_group_energy_profile()
