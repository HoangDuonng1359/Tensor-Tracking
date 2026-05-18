from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from mne.preprocessing import compute_current_source_density

from core.config import config

def main() -> None:
    print("======================================================================")
    print("  RUNNING INDIVIDUAL SUBJECT PREPROCESSING DIAGNOSTIC PIPELINE")
    print("======================================================================")
    
    # Establish output directory for diagnostics
    output_dir = config.paths.OUTPUTS_DIR / "preprocessing_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Diagnostics will be saved in: {output_dir}\n")

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

    # 1. Load raw data of representative subject sub-001
    subject_id = "sub-001"
    raw_path = config.paths.DATA_RAW / subject_id / "eeg" / f"{subject_id}_task-ERN_eeg.set"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found for diagnostic: {raw_path}")
        
    print(f"Loading raw dataset: {raw_path.name}")
    raw_1024 = mne.io.read_raw_eeglab(raw_path, preload=True, verbose=False)
    
    # Standardize electrode names
    rename_map = {'FP1': 'Fp1', 'FP2': 'Fp2'}
    raw_1024.rename_channels(lambda x: rename_map.get(x, x))
    
    # Save a copy of raw before resampling for PSD comparison
    raw_1024_eeg = raw_1024.copy().pick_channels([ch for ch in config.eeg.CHANNELS if ch in raw_1024.ch_names])

    # ======================================================================
    # DIAGNOSTIC 1: PSD Downsampling & Anti-Aliasing Audit (1024Hz vs 128Hz)
    # ======================================================================
    print("Generating Diagnostic 1: Resampling PSD Spectrum Comparison...")
    raw_128 = raw_1024_eeg.copy().resample(config.eeg.SFREQ, verbose=False)
    
    # Compute PSD for raw 1024Hz and 128Hz at channel FCz
    psd_1024_spec = raw_1024_eeg.compute_psd(fmin=1.0, fmax=50.0, picks="FCz", verbose=False)
    psd_128_spec = raw_128.compute_psd(fmin=1.0, fmax=50.0, picks="FCz", verbose=False)
    
    psds_1024, freqs_1024 = psd_1024_spec.get_data(return_freqs=True)
    psds_128, freqs_128 = psd_128_spec.get_data(return_freqs=True)
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(freqs_1024, 10 * np.log10(psds_1024[0]), color="#0984e3", linewidth=2.5, label="Original Signal (1024 Hz)")
    ax.plot(freqs_128, 10 * np.log10(psds_128[0]), color="#d63031", linestyle="--", linewidth=2.0, label="Resampled Signal (128 Hz)")
    ax.axvline(64.0, color="black", linestyle=":", label="Nyquist Limit (64 Hz)")
    
    ax.set_title("Diagnostic 1: Resampling & Anti-Aliasing Verification at FCz", fontweight="bold", pad=15)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density (dB/Hz)")
    ax.set_xlim(1.0, 80.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    sns.despine()
    fig.tight_layout()
    fig.savefig(output_dir / "diag_01_resampling_psd.png", dpi=300)
    plt.close(fig)
    print(" -> Saved: diag_01_resampling_psd.png")

    # ======================================================================
    # DIAGNOSTIC 2: Sensor Montage Coordinate Verification
    # ======================================================================
    print("Generating Diagnostic 2: EEG Electrode Layout Geometry...")
    raw_128.set_montage(config.eeg.MONTAGE_NAME, on_missing='ignore', verbose=False)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    raw_128.plot_sensors(show_names=True, axes=ax, title="Diagnostic 2: Scalp Geometry Layout (30 Selected Kênh EEG)")
    fig.tight_layout()
    fig.savefig(output_dir / "diag_02_sensor_layout.png", dpi=300)
    plt.close(fig)
    print(" -> Saved: diag_02_sensor_layout.png")

    # ======================================================================
    # DIAGNOSTIC 3: Raw Voltage vs CSD Laplacian Timeseries Comparison
    # ======================================================================
    print("Generating Diagnostic 3: Timeseries Waveform Sharpening (CSD Laplacian)...")
    raw_csd = compute_current_source_density(raw_128, verbose=False)
    
    # Get 5 seconds of raw data from 100s to 105s at FCz
    start_sec, stop_sec = 100.0, 105.0
    start_samp = int(start_sec * config.eeg.SFREQ)
    stop_samp = int(stop_sec * config.eeg.SFREQ)
    time_axis = np.linspace(start_sec, stop_sec, stop_samp - start_samp)
    
    fcz_idx = raw_128.ch_names.index("FCz")
    volt_data = raw_128.get_data(picks=[fcz_idx], start=start_samp, stop=stop_samp)[0] * 1e6  # microvolts
    csd_data = raw_csd.get_data(picks=[fcz_idx], start=start_samp, stop=stop_samp)[0] * 1e6  # microvolts/m^2 (scale matches standard CSD visualization)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    
    ax1.plot(time_axis, volt_data, color="#2d3436", linewidth=1.8)
    ax1.set_title("Raw EEG Voltage Waveform at FCz (Smeared/Drifted)", fontweight="bold")
    ax1.set_ylabel("Amplitude (μV)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2.plot(time_axis, csd_data, color="#d63031", linewidth=1.8)
    ax2.set_title("Current Source Density (CSD) Laplacian at FCz (Sharpened/Detrended)", fontweight="bold")
    ax2.set_ylabel("CSD Amplitude (μV/m²)")
    ax2.set_xlabel("Time (seconds)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    sns.despine()
    fig.tight_layout()
    fig.savefig(output_dir / "diag_03_csd_timeseries.png", dpi=300)
    plt.close(fig)
    print(" -> Saved: diag_03_csd_timeseries.png")

    # ======================================================================
    # DIAGNOSTIC 4: Raw Voltage vs CSD Laplacian 2D Topography Comparison
    # ======================================================================
    print("Generating Diagnostic 4: Volume Conduction Suppression (Scalp Topomap Comparison)...")
    # Slice a single incorrect response trial
    events, event_id = mne.events_from_annotations(raw_128, verbose=False)
    incorrect_codes = [l for l in event_id.keys() if len(l) == 3 and l[0] != l[2]]
    inc_events = events[np.isin(events[:, 2], [event_id[k] for k in incorrect_codes])]
    
    if len(inc_events) > 0:
        target_sample = inc_events[0, 0]
        # Get 100ms average voltage around 50ms post-response
        onset_sec = target_sample / config.eeg.SFREQ
        t_start = int((onset_sec + 0.04) * config.eeg.SFREQ)
        t_stop = int((onset_sec + 0.07) * config.eeg.SFREQ)
        
        volt_topomap = raw_128.get_data(start=t_start, stop=t_stop).mean(axis=1) * 1e6
        csd_topomap = raw_csd.get_data(start=t_start, stop=t_stop).mean(axis=1) * 1e6
        
        fig, (ax_v, ax_c) = plt.subplots(1, 2, figsize=(11, 5.5))
        
        # Plot Voltage
        im_v, _ = mne.viz.plot_topomap(volt_topomap, raw_128.info, axes=ax_v, cmap="RdBu_r", show=False, sensors=True, res=300)
        ax_v.set_title("Raw Voltage Scalp Map (μV)\n[Smeared ACC source]", fontweight="bold", pad=10)
        fig.colorbar(im_v, ax=ax_v, orientation="horizontal", shrink=0.7, pad=0.08)
        
        # Plot CSD
        im_c, _ = mne.viz.plot_topomap(csd_topomap, raw_csd.info, axes=ax_c, cmap="RdBu_r", show=False, sensors=True, res=300)
        ax_c.set_title("CSD Laplacian Scalp Map (μV/m²)\n[Localized ACC Current Sink]", fontweight="bold", pad=10)
        fig.colorbar(im_c, ax=ax_c, orientation="horizontal", shrink=0.7, pad=0.08)
        
        fig.suptitle("Diagnostic 4: Spatial Volume Conduction Suppression", fontweight="bold", y=0.98)
        fig.tight_layout()
        fig.savefig(output_dir / "diag_04_csd_topomap.png", dpi=300)
        plt.close(fig)
        print(" -> Saved: diag_04_csd_topomap.png")
    else:
        print(" [WARNING] No incorrect trials found for topomap diagnostic!")

    # ======================================================================
    # DIAGNOSTIC 5: Master Epochs ERP Comparison (Before temporal matching)
    # ======================================================================
    print("Generating Diagnostic 5: Master Epochs ERP at FCz...")
    correct_codes = [l for l in event_id.keys() if len(l) == 3 and l[0] == l[2]]
    resp_id = {label: event_id[label] for label in correct_codes + incorrect_codes}
    
    epochs = mne.Epochs(
        raw_csd, events, event_id=resp_id,
        tmin=config.eeg.TMIN, tmax=config.eeg.TMAX,
        baseline=None, preload=True, verbose=False
    )
    
    mapping = {l: f"Correct/{l}" for l in correct_codes}
    mapping.update({l: f"Incorrect/{l}" for l in incorrect_codes})
    epochs.event_id = {mapping[k]: v for k, v in epochs.event_id.items()}
    
    times = epochs.times * 1000.0  # ms
    correct_erp = epochs["Correct"].get_data(picks=["FCz"]).mean(axis=0)[0] * 1e6
    incorrect_erp = epochs["Incorrect"].get_data(picks=["FCz"]).mean(axis=0)[0] * 1e6
    
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(times, correct_erp, color="#1abc9c", linewidth=2.5, label="Correct Trials (CRN)")
    ax.plot(times, incorrect_erp, color="#e74c3c", linewidth=2.5, label="Incorrect Trials (ERN)")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
    
    ax.invert_yaxis()  # Negative up convention
    
    ax.set_title("Diagnostic 5: Individual Subject ERP at FCz (sub-001, Master Epochs)", fontweight="bold", pad=15)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("CSD Amplitude (μV/m², Negative UP)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    sns.despine()
    fig.tight_layout()
    fig.savefig(output_dir / "diag_05_master_erp.png", dpi=300)
    plt.close(fig)
    print(" -> Saved: diag_05_master_erp.png")
    
    print("\n======================================================================")
    print("SUCCESS: ALL 5 INDIVIDUAL PREPROCESSING DIAGNOSTIC PLOTS GENERATED!")
    print("======================================================================")

if __name__ == "__main__":
    main()
