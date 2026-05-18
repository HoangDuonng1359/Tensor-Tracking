import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from algorithms.toucan import TOUCANRunner, default_config_for_toucan
from algorithms.fcca import fcca_on_lowrank_interval
from algorithms.common import convert_subject_tensor_to_stream
from core.timing import eeg_timing_from_array
import os

def visualize_fcca_results():
    print("🎨 Generating Detailed FCCA Visualizations (TOUCAN Perfected)...")
    
    # 1. Load data & Run TOUCAN
    data = np.load("data/processed_v2/03_connectivity_tensors/tensor_incorrect_4d.npy").astype(np.float32)
    stream = convert_subject_tensor_to_stream(data)
    timing = eeg_timing_from_array(stream, time_axis=0)
    
    config = default_config_for_toucan()
    config.max_rank = 5
    runner = TOUCANRunner(config)
    result = runner.run(stream)
    
    # 2. Identify Intervals
    # We look for the interval that contains the ERN (usually starts just before 0ms)
    ern_interval = None
    zero_idx = timing.zero_index
    for interval in result.intervals:
        if interval[0] <= zero_idx <= interval[1]:
            ern_interval = interval
            break
    
    if ern_interval is None:
        # Fallback to a fixed window if no interval detected near 0ms
        ern_slice = timing.ms_window(-31.25, 156.25)
        ern_interval = [ern_slice.start, ern_slice.stop - 1] 
    
    baseline_interval = [0, 15] # Initial training period
    
    print(f"📍 Baseline Interval: {baseline_interval}")
    print(f"📍 ERN Interval: {ern_interval} (Detected by TOUCAN)")
    
    # 3. Run FCCA
    clusters_base, W_base = fcca_on_lowrank_interval(result.lowrank_stream, baseline_interval[0], baseline_interval[1])
    clusters_ern, W_ern = fcca_on_lowrank_interval(result.lowrank_stream, ern_interval[0], ern_interval[1])
    
    # 4. PLOTTING
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # A. Tracking Energy and Change Points
    ax = axes[0, 0]
    ax.plot(result.residual_energy, color='blue', alpha=0.6, label='TOUCAN Residual Energy')
    for cp in result.change_points:
        ax.axvline(x=cp, color='red', linestyle='--', alpha=0.4)
    ax.axvline(x=zero_idx, color='black', linewidth=2, label='Response (0ms)')
    ax.set_title("TOUCAN Tracking Dynamics")
    ax.set_xlabel("Time (Samples)")
    ax.set_ylabel("Energy")
    ax.legend()

    # B. Baseline Consensus Matrix
    sns.heatmap(W_base, ax=axes[0, 1], cmap='viridis', square=True)
    axes[0, 1].set_title(f"Baseline Consensus Matrix (t={baseline_interval})")

    # C. ERN Consensus Matrix
    sns.heatmap(W_ern, ax=axes[1, 0], cmap='viridis', square=True)
    axes[1, 0].set_title(f"ERN Consensus Matrix (t={ern_interval})")
    
    # D. Consensus Difference (ERN - Baseline)
    diff = W_ern - W_base
    sns.heatmap(diff, ax=axes[1, 1], cmap='RdBu_r', center=0, square=True)
    axes[1, 1].set_title("Network Reconfiguration (ERN minus Baseline)")

    plt.tight_layout()
    output_path = "outputs/toucan_fcca_detailed_report.png"
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"✅ Visualization saved to: {output_path}")
    
    # Text Analysis
    print("\n📝 Detailed Analysis:")
    print(f"- Detected Change Points: {result.change_points}")
    print(f"- ERN Interval Duration: {ern_interval[1] - ern_interval[0]} samples")
    print(f"- Avg Consensus Probability (ERN): {np.mean(W_ern):.4f}")
    
    # Identifying most stable connections in ERN
    top_indices = np.unravel_index(np.argsort(W_ern, axis=None)[-10:], W_ern.shape)
    print("- Strongest Consensus Edges (ERN):")
    for i in range(len(top_indices[0])):
        u, v = top_indices[0][i], top_indices[1][i]
        if u < v:
            print(f"  Node {u} <-> Node {v}: Prob = {W_ern[u, v]:.4f}")

if __name__ == "__main__":
    visualize_fcca_results()
