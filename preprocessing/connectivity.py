import mne
import numpy as np
import torch
from pathlib import Path
from scipy.io import savemat
from typing import List, Optional, Tuple
from core.config import config

def compute_rid_rihaczek_tfd(signal_tensor: torch.Tensor) -> torch.Tensor:
  """Compute RID-Rihaczek TFD on GPU."""
  n_trials, n_points = signal_tensor.shape
  device = signal_tensor.device
  
  # Time and frequency grids
  t = torch.arange(n_points, device=device).float()
  theta = torch.fft.fftfreq(n_points, d=1.0, device=device) * 2 * np.pi
  
  # Kernel (Exponential for RID)
  # RID-Rihaczek specific: g(theta, tau) = exp(-j * theta * tau / 2)
  # Here we simplify the TFD implementation for PLV purposes
  
  # Analytic signal via Hilbert transform
  analytic = torch.fft.ifft(torch.fft.fft(signal_tensor, dim=1) * 2, dim=1)
  analytic[:, n_points//2:] = 0
  
  # TFR Calculation
  # For PLV, we need the phase of the TFR at theta band
  # Using a simplified RID-Rihaczek approach
  RID_theta = torch.zeros((n_trials, n_points, n_points), dtype=torch.complex64, device=device)
  for tau in range(-n_points//2, n_points//2):
    shifted = torch.roll(analytic, shifts=-tau, dims=1)
    RID_theta[:, :, tau + n_points//2] = analytic * torch.conj(shifted) * np.exp(-1j * tau / 2)

  RID_t_f = torch.fft.ifft(torch.fft.ifft(RID_theta, dim=1), dim=2)
  return RID_t_f

def compute_group_connectivity() -> None:
  """
  Step 3: Compute PLV tensors from already sampled and filtered epochs.
  """
  sub_files = sorted(list(config.paths.REFINED_DIR.glob("*_theta_balanced-epo.fif")))
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  if not sub_files:
    print("Error: No sampled epoch files found. Please run sampling.py first.")
    return
    
  processed_subs: List[str] = []
  all_corr: List[np.ndarray] = []
  all_inc: List[np.ndarray] = []
  
  sample_epochs = mne.read_epochs(sub_files[0], preload=False, verbose=False)
  n_channels = len(sample_epochs.ch_names)
  n_points = sample_epochs.get_data().shape[2]

  print(f"Computing PLV Tensors (Step 3) for {len(sub_files)} subjects on {device}...")

  for i, f in enumerate(sub_files):
    subject_id = f.name.split('_')[0]
    epochs = mne.read_epochs(f, preload=True, verbose=False)
    
    processed_subs.append(subject_id)
    
    sub_plv_cor = np.zeros((n_channels, n_channels, n_points), dtype=np.float32)
    sub_plv_inc = np.zeros((n_channels, n_channels, n_points), dtype=np.float32)

    for condition, storage in [('Correct', sub_plv_cor), ('Incorrect', sub_plv_inc)]:
      data = torch.tensor(epochs[condition].get_data(), dtype=torch.float32, device=device)
      n_trials, _, n_points = data.shape
      theta_complex = torch.zeros((n_trials, n_channels, n_points), dtype=torch.complex64, device=device)
      
      # Frequency axis for RID-Rihaczek
      freq_axis = torch.fft.fftfreq(n_points, d=1/config.eeg.SFREQ, device=device)
      theta_mask = (freq_axis >= config.proc.THETA_BAND[0]) & (freq_axis <= config.proc.THETA_BAND[1])
      
      for ch in range(n_channels):
        C = compute_rid_rihaczek_tfd(data[:, ch, :])
        # Average over theta band
        theta_complex[:, ch, :] = torch.mean(C[:, :, theta_mask], dim=2)
      
      theta_complex /= (torch.abs(theta_complex) + 1e-12)
      
      for ch1 in range(n_channels):
        for ch2 in range(ch1 + 1, n_channels):
          # PLV = |mean(exp(j * delta_phi))|
          plv = torch.abs(torch.mean(theta_complex[:, ch1, :] * torch.conj(theta_complex[:, ch2, :]), dim=0))
          storage[ch1, ch2, :] = plv.cpu().numpy()
          storage[ch2, ch1, :] = plv.cpu().numpy()
    
    all_corr.append(sub_plv_cor)
    all_inc.append(sub_plv_inc)
    if (i+1) % 10 == 0:
      print(f" Progress: {i+1}/{len(sub_files)} subjects processed...")

  tensor_correct = np.stack(all_corr, axis=0)
  tensor_incorrect = np.stack(all_inc, axis=0)
  
  # Save results
  np.save(config.tensor_correct_file, tensor_correct)
  np.save(config.tensor_incorrect_file, tensor_incorrect)
  
  savemat(config.paths.MATLAB_DIR / "connectivity_balanced_4d.mat", {
    'tensor_correct': tensor_correct,
    'tensor_incorrect': tensor_incorrect,
    'subjects': processed_subs,
    'channels': sample_epochs.ch_names
  })
  
  print(f"Completed! Processed {len(processed_subs)} subjects.")

if __name__ == "__main__":
  compute_group_connectivity()
