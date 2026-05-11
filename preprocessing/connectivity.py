import mne
import numpy as np
import torch
import torch.fft
from pathlib import Path
from scipy.io import savemat
from typing import List, Optional, Tuple
from core.config import config

def compute_rid_rihaczek_tfd(signal_tensor: torch.Tensor) -> torch.Tensor:
  """Compute RID-Rihaczek TFD on GPU."""
  n_trials, n_points = signal_tensor.shape
  device = signal_tensor.device
  
  tau = torch.arange(-n_points // 2, n_points // 2, device=device, dtype=torch.float32)
  theta = torch.fft.fftfreq(n_points, d=1.0, device=device) * 2 * np.pi
  
  THETA, TAU = torch.meshgrid(theta, tau, indexing='ij')
  kernel = torch.exp(-(THETA * TAU)**2 / 1.0) * torch.exp(1j * THETA * TAU / 2)
  
  AF = torch.zeros((n_trials, n_points, n_points), dtype=torch.complex64, device=device)
  for i, t in enumerate(tau):
    sh = int(t.item())
    if sh >= 0:
      R = signal_tensor[:, sh:] * torch.conj(signal_tensor[:, :n_points-sh])
      AF[:, :, i] = torch.nn.functional.pad(R, (0, int(abs(sh))))
    else:
      R = signal_tensor[:, :n_points+sh] * torch.conj(signal_tensor[:, int(abs(sh)):])
      AF[:, :, i] = torch.nn.functional.pad(R, (int(abs(sh)), 0))

  AF_theta = torch.fft.fft(AF, dim=1)
  RID_theta = AF_theta * kernel.unsqueeze(0)
  RID_t_f = torch.fft.ifft(torch.fft.ifft(RID_theta, dim=1), dim=2)
  return RID_t_f

def match_temporal_trials(epochs: mne.Epochs) -> Optional[Tuple[mne.Epochs, mne.Epochs]]:
  """
  Balance Correct and Incorrect trials using temporal matching to avoid biological drift bias.
  """
  n_inc = len(epochs['Incorrect'])
  n_cor = len(epochs['Correct'])
  
  if n_inc == 0:
    return None
    
  times_inc = epochs['Incorrect'].events[:, 0]
  times_cor = epochs['Correct'].events[:, 0]
  
  selected_cor_indices = []
  available_cor_indices = list(range(n_cor))
  
  for t_i in times_inc:
    diffs = np.abs(times_cor[available_cor_indices] - t_i)
    best_match_idx_in_available = np.argmin(diffs)
    real_idx = available_cor_indices.pop(best_match_idx_in_available)
    selected_cor_indices.append(real_idx)
    
  epochs_inc = epochs['Incorrect']
  epochs_cor_matched = epochs['Correct'][selected_cor_indices]
  
  return epochs_inc, epochs_cor_matched

def compute_group_connectivity() -> None:
  sub_files = sorted(list(config.paths.EPOCHS_DIR.glob("*_master-epo.fif")))
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # List to store processed data
  all_corr: List[np.ndarray] = []
  all_inc: List[np.ndarray] = []
  processed_subs: List[str] = []

  print(f"Running Connectivity with TEMPORAL MATCHING on {device}...")
  
  sample_epochs = mne.read_epochs(sub_files[0], preload=False, verbose=False)
  n_channels = len(sample_epochs.ch_names)

  for i, f in enumerate(sub_files):
    subject_id = f.name.split('_')[0]
    epochs = mne.read_epochs(f, preload=True, verbose=False)
    
    balanced = match_temporal_trials(epochs)
    if balanced is None:
      print(f" Skipping {subject_id}: 0 incorrect trials found")
      continue
      
    processed_subs.append(subject_id)
    epochs_inc, epochs_cor = balanced
    
    sub_plv_cor = np.zeros((n_channels, n_channels, 256), dtype=np.float32)
    sub_plv_inc = np.zeros((n_channels, n_channels, 256), dtype=np.float32)

    for ep_obj, output_arr in [(epochs_cor, sub_plv_cor), (epochs_inc, sub_plv_inc)]:
      raw_data = ep_obj.get_data()[:, :, :256] * 1e6
      data = torch.tensor(raw_data, dtype=torch.float32, device=device)
      n_trials = data.shape[0]
      
      theta_complex = torch.zeros((n_trials, n_channels, 256), dtype=torch.complex64, device=device)
      freq_axis = np.fft.fftfreq(256, 1/config.eeg.SFREQ)
      theta_mask = (freq_axis >= config.proc.THETA_BAND[0]) & (freq_axis <= config.proc.THETA_BAND[1])
      
      for ch in range(n_channels):
        C = compute_rid_rihaczek_tfd(data[:, ch, :])
        theta_complex[:, ch, :] = torch.mean(C[:, :, theta_mask], dim=2)
      
      theta_complex /= (torch.abs(theta_complex) + 1e-12)
      
      for t in range(256):
        Z_t = theta_complex[:, :, t]
        plv = torch.abs(torch.matmul(Z_t.H, Z_t)) / n_trials
        output_arr[:, :, t] = plv.cpu().numpy()
    
    all_corr.append(sub_plv_cor)
    all_inc.append(sub_plv_inc)
    print(f" Processed {subject_id}: Balanced with {len(epochs_inc)} trials")

  tensor_correct = np.stack(all_corr, axis=0)
  tensor_incorrect = np.stack(all_inc, axis=0)

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
