import numpy as np
import mne
from scipy.linalg import eigh
from scipy.io import savemat
from typing import Tuple, List
from core.config import config

def compute_fiedler_vector(adj_matrix: np.ndarray) -> np.ndarray:
  """
  Compute the Fiedler vector (eigenvector of the second smallest eigenvalue)
  of the Laplacian matrix derived from adj_matrix.
  """
  n = adj_matrix.shape[0]
  # Degree matrix
  d = np.diag(np.sum(adj_matrix, axis=1))
  # Laplacian
  laplacian = d - adj_matrix
  
  # Solve eigenvalue problem for symmetric matrix
  eigenvalues, eigenvectors = eigh(laplacian)
  
  # Second smallest eigenvalue's eigenvector
  fiedler_vec = eigenvectors[:, 1]
  return fiedler_vec

def compute_consensus_clusters(connectivity_tensor: np.ndarray, 
                 time_window: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
  """
  Fiedler Consensus Clustering Algorithm (FCCA) implementation based on Ozdemir (2017).
  """
  n_subs, n_nodes, _, n_times = connectivity_tensor.shape
  start_t, end_t = time_window
  
  W = np.zeros((n_nodes, n_nodes))
  count = 0
  
  print(f"Running FCCA on time window {start_t}ms to {end_t}ms...")
  
  time_ms = np.linspace(-1000, 1000, n_times)
  time_indices = np.where((time_ms >= start_t) & (time_ms <= end_t))[0]
  
  for s in range(n_subs):
    for t in time_indices:
      adj = connectivity_tensor[s, :, :, t]
      
      f_vec = compute_fiedler_vector(adj)
      clusters = (f_vec > 0).astype(int)
      
      T_r = (clusters[:, None] == clusters[None, :]).astype(float)
      W += T_r
      count += 1
      
  W /= count
  
  consensus_f_vec = compute_fiedler_vector(W)
  final_clusters = (consensus_f_vec > 0).astype(int)
  
  return final_clusters, W

def main() -> None:
  print("Starting FCCA Analysis...")
  
  lowrank_file = config.paths.TENSOR_DIR / "horls_lowrank_incorrect.npy"
  if not lowrank_file.exists():
    print("Error: Low-rank tensor file not found. Please run HO-RLSL decomposition first.")
    return
    
  tensor_inc = np.load(lowrank_file)
  ern_window = (0, 150) 
  
  clusters, W_consensus = compute_consensus_clusters(tensor_inc, ern_window)
  
  # Load channel names for reporting
  sample_files = list(config.paths.EPOCHS_DIR.glob("*.fif"))
  if not sample_files:
    print("Error: No epoch files found to extract channel names.")
    return
    
  sample_epo = mne.read_epochs(sample_files[0], preload=False, verbose=False)
  ch_names = sample_epo.ch_names
  
  print("\nFCCA CONSENSUS CLUSTERS:")
  cluster_a = [ch_names[i] for i, c in enumerate(clusters) if c == 0]
  cluster_b = [ch_names[i] for i, c in enumerate(clusters) if c == 1]
  
  print(f"Cluster A: {', '.join(cluster_a)}")
  print(f"Cluster B: {', '.join(cluster_b)}")
  
  # Save results
  np.save(config.paths.TENSOR_DIR / "fcca_clusters.npy", clusters)
  np.save(config.paths.TENSOR_DIR / "fcca_consensus_matrix.npy", W_consensus)
  
  # Export to MATLAB
  savemat(config.paths.MATLAB_DIR / "fcca_results.mat", {
    'clusters': clusters,
    'W_consensus': W_consensus,
    'channel_names': ch_names
  })
  print("FCCA results saved successfully.")

if __name__ == "__main__":
  main()
