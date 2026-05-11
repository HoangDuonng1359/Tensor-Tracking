import numpy as np
import mne
from scipy.linalg import eigh
from typing import Tuple, List, Optional
from core.config import config

def compute_fiedler_vector(adj_matrix: np.ndarray) -> np.ndarray:
  """Compute the Fiedler vector from an adjacency matrix."""
  n = adj_matrix.shape[0]
  d = np.diag(np.sum(adj_matrix, axis=1))
  laplacian = d - adj_matrix
  
  try:
    eigenvalues, eigenvectors = eigh(laplacian)
    return eigenvectors[:, 1]
  except Exception as e:
    print(f"Eigenvalue solver error: {e}")
    return np.zeros(n)

def compute_consensus_clusters(connectivity_tensor: np.ndarray, 
                 time_window: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
  """
  Dynamic FCCA implementation to compare clustering across different time windows.
  """
  n_subs, n_nodes, _, n_times = connectivity_tensor.shape
  start_t, end_t = time_window
  
  W = np.zeros((n_nodes, n_nodes))
  count = 0
  
  time_ms = np.linspace(-1000, 1000, n_times)
  time_indices = np.where((time_ms >= start_t) & (time_ms <= end_t))[0]
  
  for s in range(n_subs):
    sub_data = connectivity_tensor[s, :, :, :]
    sub_win = sub_data[:, :, time_indices]
    adj_sub_win = np.mean(sub_win, axis=2)
    
    f_vec = compute_fiedler_vector(adj_sub_win)
    clusters = (f_vec > 0).astype(int)
    
    T_r = (clusters[:, None] == clusters[None, :]).astype(float)
    W += T_r
    count += 1
      
  W /= count
  consensus_f_vec = compute_fiedler_vector(W)
  final_clusters = (consensus_f_vec > 0).astype(int)
  return final_clusters, W

def main() -> None:
  print("Starting Dynamic FCCA Analysis (Baseline vs ERN)...")
  
  lowrank_path = config.paths.TENSOR_DIR / "horls_lowrank_incorrect.npy"
  if not lowrank_path.exists():
    print("Error: Low-rank tensor not found.")
    return
    
  tensor_inc = np.load(lowrank_path)
  
  baseline_win = (-400, -200)
  ern_win = (50, 150)
  
  cls_base, W_base = compute_consensus_clusters(tensor_inc, baseline_win)
  cls_ern, W_ern = compute_consensus_clusters(tensor_inc, ern_win)
  
  # Calculate Modularity (estimated by variance of consensus matrix)
  mod_base = np.var(W_base)
  mod_ern = np.var(W_ern)
  
  print("\nRESULTS ASSESSMENT:")
  print(f" Modularity (Baseline): {mod_base:.4f}")
  print(f" Modularity (ERN): {mod_ern:.4f}")
  
  if mod_ern < mod_base:
    print(" CONCLUSION: Network becomes more INTEGRATED during errors (ERN).")
  else:
    print(" CONCLUSION: Network maintains or increases segregation.")
  
  # Save dynamic results
  np.save(config.paths.TENSOR_DIR / "fcca_dynamic_results.npy", {
    'baseline_clusters': cls_base,
    'ern_clusters': cls_ern,
    'W_baseline': W_base,
    'W_ern': W_ern,
    'modularity': {'baseline': mod_base, 'ern': mod_ern}
  }, allow_pickle=True)
  print("Dynamic FCCA results saved.")

if __name__ == "__main__":
  main()
