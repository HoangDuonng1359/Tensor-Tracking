from __future__ import annotations

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from pathlib import Path

from core.config import config
from algorithms.time_frequency import RIDRihaczek
from preprocessing.sampling import match_temporal_trials

def main() -> None:
    print("======================================================================")
    print("     GENERATING 5 ADVANCED EEG PREPROCESSING DIAGNOSTIC CHARTS")
    print("======================================================================")
    
    output_dir = config.paths.OUTPUTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Diagnostics will be saved in: {output_dir}\n")

    # Publication-ready layout settings
    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "DejaVu Sans", "Arial"],
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    # Raw file paths for sub-001
    raw_path = config.paths.DATA_RAW / "sub-001" / "eeg" / "sub-001_task-ERN_eeg.set"
    if not raw_path.exists():
        print(f" [ERROR] Raw file not found: {raw_path}")
        return

    print("Loading raw EEG data for sub-001 (1024 Hz)...")
    raw_raw = mne.io.read_raw_eeglab(raw_path, preload=True, verbose=False)
    
    # Rename & Montage mapping
    rename_map = {'FP1': 'Fp1', 'FP2': 'Fp2'}
    raw_raw.rename_channels(lambda x: rename_map.get(x, x))
    raw_raw.set_montage(config.eeg.MONTAGE_NAME, on_missing='ignore')
    
    # Keep 30 channels
    available_eeg = [ch for ch in config.eeg.CHANNELS if ch in raw_raw.ch_names]
    raw_raw.pick_channels(available_eeg)

    # ======================================================================
    # CHART 1: Resampling & Anti-Aliasing PSD Comparison
    # ======================================================================
    # Raw PSD (1024 Hz) up to 100 Hz
    spectrum_raw = raw_raw.compute_psd(fmin=1.0, fmax=100.0, verbose=False)
    psd_raw, freqs_raw = spectrum_raw.get_data(return_freqs=True)
    psd_raw_db = 10 * np.log10(psd_raw)  # Keep shape: (n_channels, n_freqs)

    # Resampled PSD (128 Hz)
    raw_resampled = raw_raw.copy().resample(config.eeg.SFREQ, verbose=False)
    spectrum_res = raw_resampled.compute_psd(fmin=1.0, fmax=64.0, verbose=False)
    psd_res, freqs_res = spectrum_res.get_data(return_freqs=True)
    psd_res_db = 10 * np.log10(psd_res)  # Keep shape: (n_channels, n_freqs)

    # Find Cz or FCz idx
    ch_idx = raw_raw.ch_names.index("FCz") if "FCz" in raw_raw.ch_names else 0
    ch_name = raw_raw.ch_names[ch_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freqs_raw, psd_raw_db[ch_idx], color="#95a5a6", alpha=0.8, linewidth=1.5, label=f"Original (1024 Hz) - {ch_name}")
    ax.plot(freqs_res, psd_res_db[ch_idx], color="#e74c3c", linewidth=2.0, label=f"Resampled (128 Hz, Anti-Aliased) - {ch_name}")
    ax.axvline(64.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="Nyquist Frequency (64 Hz)")
    ax.set_title("Resampling & Anti-Aliasing Filter Verification (PSD Comparison)", fontweight="bold", pad=12)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density (dB/Hz)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    sns.despine()
    fig.tight_layout()
    fig.savefig(output_dir / "diag_01_resampling_psd.png", dpi=300)
    plt.close(fig)
    print(" -> Saved: diag_01_resampling_psd.png")

    # ======================================================================
    # CHART 2: 2D Sensor Placement Map
    # ======================================================================
    print("\nStep 2: Creating Chart 2 (2D Sensor Placement Map)...")
    fig, ax = plt.subplots(figsize=(6, 6))
    mne.viz.plot_sensors(raw_raw.info, show_names=True, axes=ax, show=False, kind="topomap")
    ax.set_title("2D Scalp Sensor Layout (30 Selected EEG Channels)", fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(output_dir / "diag_02_sensor_layout.png", dpi=300)
    plt.close(fig)
    print(" -> Saved: diag_02_sensor_layout.png")

    # ======================================================================
    # CHART 3: Spatial Sharpening Contrast (Voltage vs. CSD Topomaps)
    # ======================================================================
    print("\nStep 3: Creating Chart 3 (Spatial Sharpening: Voltage vs. CSD)...")
    # Create epochs on raw voltage data
    events, event_id = mne.events_from_annotations(raw_resampled, verbose=False)
    correct_codes = [l for l in event_id.keys() if len(l) == 3 and l[0] == l[2]]
    incorrect_codes = [l for l in event_id.keys() if len(l) == 3 and l[0] != l[2]]
    resp_id = {label: event_id[label] for label in correct_codes + incorrect_codes}
    
    epochs_volt = mne.Epochs(
        raw_resampled, events, event_id=resp_id, 
        tmin=config.eeg.TMIN, tmax=config.eeg.TMAX, 
        baseline=(-0.6, -0.4), preload=True, verbose=False
    )
    
    # Calculate CSD transform
    raw_csd = mne.preprocessing.compute_current_source_density(raw_resampled, verbose=False)
    epochs_csd = mne.Epochs(
        raw_csd, events, event_id=resp_id, 
        tmin=config.eeg.TMIN, tmax=config.eeg.TMAX, 
        baseline=(-0.6, -0.4), preload=True, verbose=False
    )

    # Rename event IDs to include Correct/ and Incorrect/ hierarchy as in pipeline
    mapping = {l: f"Correct/{l}" for l in correct_codes}
    mapping.update({l: f"Incorrect/{l}" for l in incorrect_codes})
    epochs_volt.event_id = {mapping[k]: v for k, v in epochs_volt.event_id.items()}
    epochs_csd.event_id = {mapping[k]: v for k, v in epochs_csd.event_id.items()}
    
    # Extract Grand Average ERN peak (0-100 ms difference: Incorrect - Correct)
    time_mask = (epochs_volt.times >= 0.0) & (epochs_volt.times <= 0.100)
    
    # Voltage Difference
    volt_inc = epochs_volt['Incorrect'].get_data().mean(axis=0) * 1e6 # in microvolts
    volt_cor = epochs_volt['Correct'].get_data().mean(axis=0) * 1e6
    volt_diff = (volt_inc - volt_cor)[:, time_mask].mean(axis=1)
    
    # CSD Difference
    csd_inc = epochs_csd['Incorrect'].get_data().mean(axis=0) * 1e6 # in CSD units (microvolts / m^2)
    csd_cor = epochs_csd['Correct'].get_data().mean(axis=0) * 1e6
    csd_diff = (csd_inc - csd_cor)[:, time_mask].mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    
    # Plot Voltage Topomap
    im1, _ = mne.viz.plot_topomap(volt_diff, epochs_volt.info, axes=axes[0], cmap="RdBu_r", show=False, contours=6, res=300)
    axes[0].set_title("Voltage (Incorrect - Correct)\n[Spurious Diffuse Blurring]", fontweight="bold")
    fig.colorbar(im1, ax=axes[0], orientation="vertical", shrink=0.7, label="Voltage Difference (μV)")
    
    # Plot CSD Topomap
    im2, _ = mne.viz.plot_topomap(csd_diff, epochs_csd.info, axes=axes[1], cmap="RdBu_r", show=False, contours=6, res=300)
    axes[1].set_title("CSD Laplacian (Incorrect - Correct)\n[Sharp Localized Frontocentral ACC Source]", fontweight="bold")
    fig.colorbar(im2, ax=axes[1], orientation="vertical", shrink=0.7, label="CSD Difference (μV/m²)")

    fig.suptitle("Volume Conduction Contrast: Scalp Voltage vs. CSD Laplacian (0-100 ms)", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(output_dir / "diag_03_csd_sharpening.png", dpi=300)
    plt.close(fig)
    print(" -> Saved: diag_03_csd_sharpening.png")

    # ======================================================================
    # CHART 4: Session Trial Timeline Distribution (Temporal Matching)
    # ======================================================================
    print("\nStep 4: Creating Chart 4 (Session Trial Timeline & Temporal Matching)...")
    # Identify indices
    evs = epochs_volt.events
    ev_ids = epochs_volt.event_id
    
    correct_idx = np.where(np.isin(evs[:, 2], [ev_ids[k] for k in ev_ids if k.startswith('Correct')]))[0]
    incorrect_idx = np.where(np.isin(evs[:, 2], [ev_ids[k] for k in ev_ids if k.startswith('Incorrect')]))[0]
    
    # Run match temporal trials
    balanced_epochs = match_temporal_trials(epochs_volt)
    # Get indices of balanced epochs in the original set
    balanced_events = balanced_epochs.events
    
    # Find which correct trials were matched
    selected_correct_idx = []
    for event_row in balanced_events:
        if event_row[2] in [ev_ids[k] for k in ev_ids if k.startswith('Correct')]:
            # Find matching original event index
            orig_match = np.where((evs[:, 0] == event_row[0]) & (evs[:, 2] == event_row[2]))[0]
            if len(orig_match) > 0:
                selected_correct_idx.append(orig_match[0])

    fig, ax = plt.subplots(figsize=(10, 3.5))
    
    # Scatter plot representing the sequence of trials
    ax.scatter(correct_idx, np.ones_like(correct_idx) * 2, color="#2ecc71", alpha=0.3, s=40, label="Unselected Correct Trials")
    ax.scatter(selected_correct_idx, np.ones_like(selected_correct_idx) * 2, color="#008080", alpha=1.0, s=60, marker="o", facecolors='none', edgecolors='#008080', linewidths=1.5, label="Matched Selected Correct Trials")
    ax.scatter(incorrect_idx, np.ones_like(incorrect_idx) * 1, color="#e056fd", alpha=1.0, s=65, marker="x", label="Incorrect (Error) Trials")
    
    ax.set_yticks([1, 2])
    ax.set_yticklabels(["Incorrect", "Correct"])
    ax.set_ylim(0.4, 2.6)
    ax.set_xlabel("Trial Order Index in Experiment Session")
    ax.set_title("Temporal Matching Subsampling: Controlling for Fatigue / Drift Bias", fontweight="bold", pad=12)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="lower right")
    sns.despine(left=True)
    fig.tight_layout()
    fig.savefig(output_dir / "diag_04_temporal_matching.png", dpi=300)
    plt.close(fig)
    print(" -> Saved: diag_04_temporal_matching.png")

    # ======================================================================
    # CHART 5: Inter-Trial Phase Coherence (ITPC) Spectrogram
    # ======================================================================
    print("\nStep 5: Creating Chart 5 (Theta ITPC Spectrogram)...")
    # Load RID-Rihaczek Engine
    n_points = epochs_csd.get_data().shape[2]
    tfd_engine = RIDRihaczek(n_points)
    
    # Get Incorrect data at FCz
    fcz_idx = epochs_csd.ch_names.index("FCz") if "FCz" in epochs_csd.ch_names else 0
    inc_data = epochs_csd['Incorrect'].get_data()[:, fcz_idx, :] # (trials, times)
    
    # Move to GPU or keep CPU as engine defaults
    inc_tensor = torch.tensor(inc_data, dtype=torch.float32, device=tfd_engine.device)
    
    # Compute TFD phases
    tfr = tfd_engine.compute_tfd(inc_tensor) # (trials, times, freq)
    phases = tfr / (torch.abs(tfr) + 1e-12) # normalize amplitude to extract phase
    
    # Calculate ITPC = |mean_trials( phase )|
    itpc = torch.abs(torch.mean(phases, dim=0)).cpu().numpy() # (times, freqs)
    
    # Crop to Theta Band frequencies (4 to 8 Hz)
    freq_axis = np.fft.fftfreq(n_points, d=1/config.eeg.SFREQ)
    theta_mask = (freq_axis >= 2.0) & (freq_axis <= 15.0) # Extend slightly (2-15Hz) for visual context
    theta_freqs = freq_axis[theta_mask]
    itpc_theta = itpc[:, theta_mask]

    # Plot ITPC
    times_ms = epochs_csd.times * 1000.0
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    # Heatmap/contour representation
    c = ax.contourf(times_ms, theta_freqs, itpc_theta.T, levels=40, cmap="inferno")
    fig.colorbar(c, ax=ax, label="Inter-Trial Phase Coherence (ITPC)")
    
    # Add guidelines
    ax.axvline(0.0, color="white", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axhspan(4.0, 8.0, color="cyan", alpha=0.15, label="Theta Band Focus (4-8 Hz)")
    
    ax.set_title("Inter-Trial Phase Coherence (ITPC) at Channel FCz\nEvent-Locked Theta Band Synchrony Burst", fontweight="bold", pad=12)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(2.0, 14.0)
    ax.legend(frameon=True, facecolor="black", edgecolor="none", labelcolor="white", loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "diag_05_theta_itpc_spectrogram.png", dpi=300)
    plt.close(fig)
    print(" -> Saved: diag_05_theta_itpc_spectrogram.png")

    print("\n======================================================================")
    print("SUCCESS: ALL 5 DIAGNOSTIC CHARTS GENERATED AND SAVED TO OUTPUTS/EDA!")
    print("======================================================================")

if __name__ == "__main__":
    main()
