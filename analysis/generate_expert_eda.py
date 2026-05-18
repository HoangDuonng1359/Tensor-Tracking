from __future__ import annotations

import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne

from core.config import config

def main() -> None:
    print("======================================================================")
    print("      GENERATING PREMIUM SCIENTIFIC EEG EDA VISUALIZATIONS")
    print("======================================================================")
    
    # Establish output directory
    output_dir = config.paths.OUTPUTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to: {output_dir}\n")

    # Set up publication-quality styling
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

    # ======================================================================
    # PANEL 1: Power Spectral Density (PSD) Spectrum (Signal Quality Control)
    # ======================================================================
    print("Step 1: Generating PSD Spectrum (Signal Quality Control)...")
    balanced_dir = config.paths.REFINED_DIR
    balanced_files = sorted(list(balanced_dir.glob("sub-*_theta_balanced-epo.fif")))
    
    if not balanced_files:
        print(" [WARNING] No balanced epochs found. Trying master epochs...")
        initial_dir = config.paths.EPOCHS_DIR
        balanced_files = sorted(list(initial_dir.glob("sub-*_master-epo.fif")))
        
    if not balanced_files:
        print(" [ERROR] No epochs files found in processed directory!")
        return

    # Load first subject as representative
    rep_file = balanced_files[0]
    print(f" -> Loading representative file for PSD: {rep_file.name}")
    epochs = mne.read_epochs(rep_file, preload=True)
    
    # Compute PSD (1 to 30 Hz)
    spectrum = epochs.compute_psd(fmin=1.0, fmax=30.0)
    psds, freqs = spectrum.get_data(return_freqs=True)  # (n_epochs, n_channels, n_freqs)
    mean_psds = np.mean(psds, axis=0)  # average over trials -> (n_channels, n_freqs)
    
    # Convert to dB for standard representation
    mean_psds_db = 10 * np.log10(mean_psds)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(epochs.ch_names)))
    
    for i, ch_name in enumerate(epochs.ch_names):
        ax.plot(freqs, mean_psds_db[i], color=colors[i], alpha=0.6, linewidth=1)
        
    # Highlight Theta Band (4-8 Hz)
    ax.axvspan(4.0, 8.0, color="#ffeeba", alpha=0.3, label="Theta Band (4-8 Hz)")
    
    # Grand average line
    grand_avg_psd = np.mean(mean_psds_db, axis=0)
    ax.plot(freqs, grand_avg_psd, color="#d9534f", linewidth=2.5, label="Grand Average")

    ax.set_title("EEG Power Spectral Density (PSD) - Signal Quality Audit", fontweight="bold", pad=15)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density (dB/Hz)")
    ax.set_xlim(1.0, 30.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    sns.despine()
    fig.tight_layout()
    fig.savefig(output_dir / "01_eeg_psd_spectrum.png", dpi=300)
    plt.close(fig)
    print(" -> Saved: 01_eeg_psd_spectrum.png")

    # ======================================================================
    # PANEL 2: Grand Average ERP at FCz (Neurophysiological Ground Truth)
    # ======================================================================
    print("\nStep 2: Generating Grand Average ERP at FCz (Neurophysiological Validation)...")
    initial_dir = config.paths.EPOCHS_DIR
    master_files = sorted(list(initial_dir.glob("sub-*_master-epo.fif")))
    
    # Select first 10 subjects for robust grand average
    n_subs_to_avg = min(10, len(master_files))
    subs_files = master_files[:n_subs_to_avg]
    print(f" -> Averaging ERPs across {n_subs_to_avg} subjects at channel FCz...")
    
    correct_erps = []
    incorrect_erps = []
    times = None
    fcz_idx = None

    for f in subs_files:
        sub_epochs = mne.read_epochs(f, preload=True, verbose=False)
        if times is None:
            times = sub_epochs.times * 1000.0  # convert to ms
            fcz_idx = sub_epochs.ch_names.index("FCz") if "FCz" in sub_epochs.ch_names else sub_epochs.ch_names.index("Cz")
            
        correct_data = sub_epochs["Correct"].get_data(picks=sub_epochs.ch_names[fcz_idx])
        incorrect_data = sub_epochs["Incorrect"].get_data(picks=sub_epochs.ch_names[fcz_idx])
        
        correct_erps.append(correct_data.mean(axis=0)[0])  # shape: (n_times,)
        incorrect_erps.append(incorrect_data.mean(axis=0)[0])  # shape: (n_times,)

    grand_correct = np.mean(correct_erps, axis=0) * 1e6  # convert to microvolts
    grand_incorrect = np.mean(incorrect_erps, axis=0) * 1e6  # convert to microvolts

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, grand_correct, color="#008080", linewidth=2.5, label="Correct Response (CRN)")
    ax.plot(times, grand_incorrect, color="#e056fd", linewidth=2.5, label="Incorrect Response (ERN)")
    
    # Shaded ERN window (0 to 100ms)
    ax.axvspan(0.0, 100.0, color="#f6e58d", alpha=0.3, label="ERN Analysis Window (0-100 ms)")
    
    # Customize axes
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(5, ax.get_ylim()[0]*0.9, "Response Trigger", fontsize=10, fontweight="bold")
    
    # EEG standard negative-up convention (optional but let's keep typical positive-up for general user readability, but invert y-axis label to note)
    ax.invert_yaxis()  # Standard clinical ERP convention: negative voltage is plotted UPWARDS
    
    ax.set_title("Grand Average ERP at Channel FCz (Clinical Negative-Up)", fontweight="bold", pad=15)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (μV, Negative UP)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    sns.despine()
    fig.tight_layout()
    fig.savefig(output_dir / "02_eeg_grand_average_erp.png", dpi=300)
    plt.close(fig)
    print(" -> Saved: 02_eeg_grand_average_erp.png")

    # ======================================================================
    # PANEL 3: 2D Scalp Topography (Spatial Mapping of ERN)
    # ======================================================================
    print("\nStep 3: Generating 2D Scalp Topography of the ERN Effect...")
    # Calculate difference topomap data (Incorrect - Correct) in the 0-100 ms window
    diff_erps_all_ch = []
    
    for f in subs_files:
        sub_epochs = mne.read_epochs(f, preload=True, verbose=False)
        corr_ch_erp = sub_epochs["Correct"].get_data().mean(axis=0)  # shape: (n_channels, n_times)
        incorr_ch_erp = sub_epochs["Incorrect"].get_data().mean(axis=0)  # shape: (n_channels, n_times)
        diff_erps_all_ch.append(incorr_ch_erp - corr_ch_erp)

    grand_diff_erp = np.mean(diff_erps_all_ch, axis=0) * 1e6  # (n_channels, n_times) in microvolts
    
    # Average in the 0-100 ms window
    time_mask = (sub_epochs.times >= 0.0) & (sub_epochs.times <= 0.100)
    topomap_data = grand_diff_erp[:, time_mask].mean(axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot topomap
    im, _ = mne.viz.plot_topomap(
        topomap_data,
        sub_epochs.info,
        axes=ax,
        cmap="RdBu_r",
        show=False,
        contours=6,
        sensors=True,
        res=300
    )
    
    # Add colorbar
    colorbar = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.7)
    colorbar.set_label("Voltage Difference (μV: Incorrect - Correct)", fontsize=11)
    
    ax.set_title("Scalp Topography of ERN Effect (0-100 ms Window)\nFrontocentral ACC Hotspot", fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(output_dir / "03_eeg_scalp_topography.png", dpi=300)
    plt.close(fig)
    print(" -> Saved: 03_eeg_scalp_topography.png")

    # ======================================================================
    # PANEL 4: Functional Connectivity Difference Matrix (Theta Phase Synchrony)
    # ======================================================================
    print("\nStep 4: Generating Theta Band Functional Connectivity Difference Matrix...")
    tensor_correct_path = config.tensor_correct_file
    tensor_incorrect_path = config.tensor_incorrect_file

    if not tensor_correct_path.exists() or not tensor_incorrect_path.exists():
        print(f" [ERROR] Connectivity tensors not found at expected paths!")
        return

    # Load 4D tensors: (n_subjects, n_channels, n_channels, n_times)
    print(f" -> Loading Correct tensor: {tensor_correct_path.name}")
    tensor_correct = np.load(tensor_correct_path)
    print(f" -> Loading Incorrect tensor: {tensor_incorrect_path.name}")
    tensor_incorrect = np.load(tensor_incorrect_path)

    # ERN window is indices 128 to 141 (0 to 100 ms at 128 Hz, midpoint 128)
    # Let's average across all 40 subjects (axis 0) and the ERN time window (axis 3)
    mean_incorr_conn = np.mean(tensor_incorrect[:, :, :, 128:141], axis=(0, 3))
    mean_corr_conn = np.mean(tensor_correct[:, :, :, 128:141], axis=(0, 3))
    diff_conn = mean_incorr_conn - mean_corr_conn

    # Plot as Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        diff_conn,
        xticklabels=config.eeg.CHANNELS,
        yticklabels=config.eeg.CHANNELS,
        cmap="RdBu_r",
        center=0.0,
        cbar_kws={"label": "Phase Locking Value (PLV) Difference (Incorrect - Correct)"},
        ax=ax,
        square=True
    )
    
    ax.set_title("Theta-Band (4-8 Hz) PLV Connectivity Difference Matrix\n(Incorrect minus Correct during ERN peak)", fontweight="bold", pad=15)
    ax.set_xlabel("Channels")
    ax.set_ylabel("Channels")
    fig.tight_layout()
    fig.savefig(output_dir / "04_eeg_connectivity_matrix.png", dpi=300)
    plt.close(fig)
    print(" -> Saved: 04_eeg_connectivity_matrix.png")

    print("\n======================================================================")
    print("SUCCESS: ALL 4 EXPERT EEG EDA FIGURES GENERATED SUCCESSFULLY!")
    print("======================================================================")

if __name__ == "__main__":
    main()
