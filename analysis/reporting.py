import numpy as np
import matplotlib.pyplot as plt
import mne
from pathlib import Path
from core.config import config
from core.timing import eeg_timing_from_bundle
from fcca import fcca_on_lowrank_interval

def plot_premium_dynamics():
    """Consolidated plot of Energy, Mass, and Rank dynamics."""
    print("Generating Premium Dynamics Plot...")
    bundle_inc = config.paths.TRACKING_DIR / "horls_incorrect_bundle.npz"
    bundle_cor = config.paths.TRACKING_DIR / "horls_correct_bundle.npz"
    
    if not bundle_inc.exists():
        print("Error: incorrect bundle not found.")
        return

    with np.load(bundle_inc) as b:
        energy_inc = b['residual_energy']
        mass_inc = b['sparse_mass']
        ranks_inc = b['mode_ranks'][:, 0]
        cp_inc = b['change_points']
        
    energy_cor = np.load(bundle_cor)['residual_energy'] if bundle_cor.exists() else None
    
    timing = eeg_timing_from_bundle(bundle_inc)
    time_ms = timing.time_ms

    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    colors = {'inc': '#FF4B4B', 'cor': '#4B9BFF', 'cp': '#FFD700'}
    
    axes[0].plot(time_ms, energy_inc, color=colors['inc'], label='Incorrect (ERN)')
    if energy_cor is not None:
        axes[0].plot(time_ms, energy_cor, color=colors['cor'], label='Correct (CRN)', alpha=0.6)
    axes[0].set_title("A. Residual Energy (Error Signal Intensity)")
    axes[0].legend()
    
    for cp in cp_inc:
        axes[0].axvline(time_ms[cp], color=colors['cp'], linestyle='--', alpha=0.5)

    axes[1].plot(time_ms, mass_inc, color=colors['inc'])
    axes[1].set_title("B. Sparse Mass (L1-norm of Transients)")
    
    axes[2].plot(time_ms, ranks_inc, color=colors['inc'])
    axes[2].set_title("C. Subspace Rank (Neural Complexity)")
    axes[2].set_xlabel("Time (ms)")
    
    plt.tight_layout()
    plt.savefig(config.paths.OUTPUTS_DIR / "paper_dynamics_report.png")
    print("Paper dynamics report saved.")

def plot_cluster_transitions():
    """Visualization of FCCA cluster evolution across detected change-points."""
    print("Generating Cluster Transition Maps...")
    bundle_path = config.paths.TRACKING_DIR / "horls_incorrect_bundle.npz"
    with np.load(bundle_path) as b:
        tensor = b['lowrank_stream']
        intervals = b['intervals']
        
    timing = eeg_timing_from_bundle(bundle_path)
    time_ms = timing.time_ms
    
    ch_names = config.eeg.CHANNELS
    info = mne.create_info(ch_names, config.eeg.SFREQ, ch_types='eeg')
    info.set_montage(config.eeg.MONTAGE_NAME)
    
    n_intervals = len(intervals)
    fig, axes = plt.subplots(1, n_intervals, figsize=(4 * n_intervals, 4))
    if n_intervals == 1: axes = [axes]
    
    for i, (start, end) in enumerate(intervals):
        t_win = (time_ms[int(start)], time_ms[int(end)])
        clusters, _ = fcca_on_lowrank_interval(tensor, int(start), int(end))
        
        mne.viz.plot_topomap(clusters, info, axes=axes[i], show=False, cmap='RdBu_r', contours=0)
        axes[i].set_title(f"Stage {i+1}\n{t_win[0]:.0f}-{t_win[1]:.0f}ms")
        
    plt.savefig(config.paths.OUTPUTS_DIR / "cluster_transitions_report.png")
    print("Cluster transitions report saved.")

if __name__ == "__main__":
    plot_premium_dynamics()
    plot_cluster_transitions()
