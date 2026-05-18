from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from scipy.linalg import svd

from core.config import config

def main() -> None:
    print("======================================================================")
    print("  RUNNING GROUP-LEVEL TENSOR CONNECTIVITY DIAGNOSTIC PIPELINE")
    print("======================================================================")
    
    # Establish output directory for connectivity diagnostics
    output_dir = config.paths.OUTPUTS_DIR / "connectivity_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Connectivity Diagnostics will be saved in: {output_dir}\n")

    # Styling configuration for publication-grade plots
    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "DejaVu Sans", "Arial"],
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16,
    })

    # 1. Load the generated 4D tensors [Subjects x Channels x Channels x Times]
    if not config.tensor_correct_file.exists() or not config.tensor_incorrect_file.exists():
        raise FileNotFoundError(
            "Balanced 4D tensors not found. Please make sure step 3 preprocessing (connectivity.py) ran successfully."
        )
        
    print(f"Loading Correct 4D Tensor: {config.tensor_correct_file.name}")
    tensor_cor = np.load(config.tensor_correct_file) # Shape: (n_subjects, n_channels, n_channels, n_times)
    
    print(f"Loading Incorrect 4D Tensor: {config.tensor_incorrect_file.name}")
    tensor_inc = np.load(config.tensor_incorrect_file) # Shape: (n_subjects, n_channels, n_channels, n_times)

    n_subjects, n_channels, _, n_times = tensor_cor.shape
    times_ms = np.linspace(config.eeg.TMIN * 1000.0, config.eeg.TMAX * 1000.0, n_times)

    # Establish time window of interest (ERN peak: 0 to 100ms)
    ern_window = (times_ms >= 0.0) & (times_ms <= 100.0)
    ern_indices = np.where(ern_window)[0]

    # Load info template from standard epochs for topomap plotting
    sample_epochs = mne.read_epochs(
        sorted(list(config.paths.REFINED_DIR.glob("*_theta_balanced-epo.fif")))[0],
        preload=False, verbose=False
    )
    info = sample_epochs.info

    # ======================================================================
    # DIAGNOSTIC 1: Grand Average Connectivity Matrices (Correct, Incorrect, Diff)
    # ======================================================================
    print("Generating Diagnostic 1: Grand Average PLV Connectivity Matrices...")
    # Average across subjects, times (inside 0-100ms window)
    mean_cor_mat = tensor_cor[:, :, :, ern_indices].mean(axis=(0, 3))
    mean_inc_mat = tensor_inc[:, :, :, ern_indices].mean(axis=(0, 3))
    diff_mat = mean_inc_mat - mean_cor_mat

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    
    # Correct Matrix
    sns.heatmap(mean_cor_mat, ax=axes[0], cmap="viridis", vmin=0.1, vmax=0.7, cbar=True, square=True,
                xticklabels=False, yticklabels=False)
    axes[0].set_title("A. Correct Response Network (CRN)\n[PLV in 0-100 ms]", fontweight="bold")
    
    # Incorrect Matrix
    sns.heatmap(mean_inc_mat, ax=axes[1], cmap="viridis", vmin=0.1, vmax=0.7, cbar=True, square=True,
                xticklabels=False, yticklabels=False)
    axes[1].set_title("B. Incorrect Response Network (ERN)\n[PLV in 0-100 ms]", fontweight="bold")
    
    # Difference Matrix
    sns.heatmap(diff_mat, ax=axes[2], cmap="RdBu_r", vmin=-0.15, vmax=0.15, cbar=True, square=True,
                xticklabels=False, yticklabels=False)
    axes[2].set_title("C. Network Difference\n[Incorrect - Correct]", fontweight="bold")

    for ax in axes:
        ax.set_xlabel("Channels (1 to 30)")
        ax.set_ylabel("Channels (1 to 30)")

    fig.suptitle("Diagnostic 1: Theta-Band Grand Average Functional Connectivity", fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(output_dir / "connectivity_01_grand_average_matrices.png", dpi=300)
    plt.close(fig)
    print(" -> Saved: connectivity_01_grand_average_matrices.png")

    # ======================================================================
    # DIAGNOSTIC 2: Global Connectivity Dynamics over Time
    # ======================================================================
    print("Generating Diagnostic 2: Global PLV Dynamics & Shading...")
    # Calculate global connectivity at each timepoint (mean PLV across all unique pairs)
    # Shape of global_plv: (n_subjects, n_times)
    triu_indices = np.triu_indices(n_channels, k=1)
    
    global_plv_cor = np.zeros((n_subjects, n_times))
    global_plv_inc = np.zeros((n_subjects, n_times))
    
    for s in range(n_subjects):
        for t in range(n_times):
            global_plv_cor[s, t] = tensor_cor[s, :, :, t][triu_indices].mean()
            global_plv_inc[s, t] = tensor_inc[s, :, :, t][triu_indices].mean()
            
    mean_g_cor = global_plv_cor.mean(axis=0)
    sem_g_cor = global_plv_cor.std(axis=0) / np.sqrt(n_subjects)
    
    mean_g_inc = global_plv_inc.mean(axis=0)
    sem_g_inc = global_plv_inc.std(axis=0) / np.sqrt(n_subjects)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    
    # Plot Correct
    ax.plot(times_ms, mean_g_cor, color="#1abc9c", linewidth=2.5, label="Correct Response (CRN)")
    ax.fill_between(times_ms, mean_g_cor - sem_g_cor, mean_g_cor + sem_g_cor, color="#1abc9c", alpha=0.15)
    
    # Plot Incorrect
    ax.plot(times_ms, mean_g_inc, color="#e74c3c", linewidth=2.5, label="Incorrect Response (ERN)")
    ax.fill_between(times_ms, mean_g_inc - sem_g_inc, mean_g_inc + sem_g_inc, color="#e74c3c", alpha=0.15)
    
    # Highlight window of interest
    ax.axvspan(0.0, 100.0, color="#f1c40f", alpha=0.12, label="ERN Peak Window (0-100 ms)")
    ax.axvline(0.0, color="black", linestyle="--", alpha=0.8, label="Button Press Response")
    
    ax.set_title("Diagnostic 2: Theta-Band Global Phase Synchronization Dynamics", fontweight="bold", pad=15)
    ax.set_xlabel("Time relative to response (ms)")
    ax.set_ylabel("Global Phase Locking Value (Mean PLV)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper left")
    sns.despine()
    fig.tight_layout()
    fig.savefig(output_dir / "connectivity_02_global_dynamics.png", dpi=300)
    plt.close(fig)
    print(" -> Saved: connectivity_02_global_dynamics.png")

    # ======================================================================
    # DIAGNOSTIC 3: Connectivity Network Hubs (Node Strength Topography)
    # ======================================================================
    print("Generating Diagnostic 3: Connectivity Hubs (Node Strength Topomap)...")
    # Node strength = sum of PLV values connected to a node
    # Compute for each subject, then average
    strength_cor = np.zeros((n_subjects, n_channels))
    strength_inc = np.zeros((n_subjects, n_channels))
    
    for s in range(n_subjects):
        strength_cor[s] = tensor_cor[s, :, :, ern_indices].mean(axis=2).sum(axis=0)
        strength_inc[s] = tensor_inc[s, :, :, ern_indices].mean(axis=2).sum(axis=0)
        
    mean_str_cor = strength_cor.mean(axis=0)
    mean_str_inc = strength_inc.mean(axis=0)
    diff_strength = mean_str_inc - mean_str_cor

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    
    # Correct Topomap
    im_c, _ = mne.viz.plot_topomap(mean_str_cor, info, axes=axes[0], cmap="viridis", show=False, res=300)
    axes[0].set_title("A. CRN Network Hubs\n[Correct Node Strength]", fontweight="bold")
    fig.colorbar(im_c, ax=axes[0], orientation="horizontal", shrink=0.7, pad=0.08)
    
    # Incorrect Topomap
    im_i, _ = mne.viz.plot_topomap(mean_str_inc, info, axes=axes[1], cmap="viridis", show=False, res=300)
    axes[1].set_title("B. ERN Network Hubs\n[Incorrect Node Strength]", fontweight="bold")
    fig.colorbar(im_i, ax=axes[1], orientation="horizontal", shrink=0.7, pad=0.08)
    
    # Difference Topomap
    im_d, _ = mne.viz.plot_topomap(diff_strength, info, axes=axes[2], cmap="RdBu_r", show=False, res=300)
    axes[2].set_title("C. Coherence Hub Shift\n[Incorrect - Correct]", fontweight="bold")
    fig.colorbar(im_d, ax=axes[2], orientation="horizontal", shrink=0.7, pad=0.08)
    
    fig.suptitle("Diagnostic 3: Network Synchronization Hubs (0-100 ms)", fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(output_dir / "connectivity_03_network_hubs.png", dpi=300)
    plt.close(fig)
    print(" -> Saved: connectivity_03_network_hubs.png")

    # ======================================================================
    # DIAGNOSTIC 4: Singular Value/Energy Decay Profile (Subspace Quality Audit)
    # ======================================================================
    print("Generating Diagnostic 4: Subspace Quality (SVD/Eigenvalue Decay)...")
    # Reshape the Incorrect matrix during ERN peak to perform SVD
    # An unfolded representation of [Channels x Channels] for checking Low-Rank structure
    U, S, Vt = svd(mean_inc_mat)
    
    # Compute cumulative energy explained
    cumulative_energy = np.cumsum(S**2) / np.sum(S**2) * 100.0
    
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    
    # Plot singular values
    color = "#0984e3"
    ax1.plot(range(1, len(S) + 1), S, 'o-', color=color, linewidth=2.5, markersize=8, label="Singular Value")
    ax1.set_xlabel("Subspace Component Index (Rank)")
    ax1.set_ylabel("Singular Value (Spectrum)", color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Second axis for cumulative energy
    ax2 = ax1.twinx()
    color = "#27ae60"
    ax2.plot(range(1, len(S) + 1), cumulative_energy, 's--', color=color, linewidth=2.0, markersize=6, label="Cumulative Energy")
    ax2.set_ylabel("Cumulative Explained Energy (%)", color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Draw 80% threshold line
    ax2.axhline(80.0, color="#d35400", linestyle=":", label="80% Energy Threshold")
    
    # Find components needed for 80% energy
    comp_80 = np.where(cumulative_energy >= 80.0)[0][0] + 1
    ax1.axvline(comp_80, color="#d35400", linestyle="-.", alpha=0.7)
    
    ax1.set_title("Diagnostic 4: Subspace Low-Rank Quality & Energy Profile", fontweight="bold", pad=15)
    sns.despine(right=False)
    fig.tight_layout()
    fig.savefig(output_dir / "connectivity_04_energy_profile.png", dpi=300)
    plt.close(fig)
    print(" -> Saved: connectivity_04_energy_profile.png")
    
    print("\n======================================================================")
    print("SUCCESS: ALL 4 CONNECTIVITY TENSOR DIAGNOSTIC PLOTS GENERATED!")
    print("======================================================================")

if __name__ == "__main__":
    main()
