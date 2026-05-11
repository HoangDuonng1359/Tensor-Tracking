import torch
import numpy as np
from scipy.io import savemat
from typing import List, Tuple, Optional
from core.config import config

class HORLSDecomposer:
  """
  High-Order Recursive Least Squares Subspace (HO-RLSL) Decomposer.
  Implements the tensor tracking algorithm from Ozdemir et al. (2017).
  """
  def __init__(self, n_nodes: int, n_subs: int, r: int = 5, alpha_win: int = 8, 
         sigma_min: float = 0.063, device: str = 'cpu'):
    self.N = n_nodes
    self.S = n_subs
    self.r = r # Tucker rank
    self.alpha = alpha_win # Window size for updates
    self.sigma_min_paper = sigma_min # Sensitivity threshold
    self.device = device
    
    self.U: Optional[torch.Tensor] = None 
    self.V: Optional[torch.Tensor] = None

  def initialize_tucker_subspace(self, tensor_4d: torch.Tensor, n_init: int = 128) -> None:
    """
    Initialization phase using baseline (pre-stimulus) data.
    """
    m_train = tensor_4d[:, :, :, :n_init].to(self.device)
    
    # Channel mode basis
    m_chan = m_train.permute(1, 0, 2, 3).reshape(self.N, -1)
    u_chan, s_chan, _ = torch.svd(m_chan)
    self.U = u_chan[:, :self.r]
    
    # Adaptive thresholding based on baseline singular values
    self.sigma_min = 0.1 * s_chan[0].item()
    
    # Subject mode basis
    m_sub = m_train.reshape(self.S, -1)
    u_sub, _, _ = torch.svd(m_sub)
    self.V = u_sub[:, :self.r]
    
    print(f"Subspace initialized on {n_init} samples. Adaptive sigma_min: {self.sigma_min:.4f}")
    
  def update_recursive_subspace(self, data_window: torch.Tensor) -> int:
    """
    Recursive subspace update step. Adds/removes directions based on energy.
    """
    eye_n = torch.eye(self.N, device=self.device)
    proj_perp = eye_n - torch.matmul(self.U, self.U.T)
    
    flat_data = data_window.permute(1, 0, 2, 3).reshape(self.N, -1)
    w = flat_data.shape[1] 
    projected_data = torch.matmul(proj_perp, flat_data)
    
    u_new, s_new, _ = torch.svd(projected_data)
    threshold = np.sqrt(self.sigma_min_paper * w)
    added_indices = torch.where(s_new > threshold)[0]
    
    if len(added_indices) > 0:
      new_dirs = u_new[:, added_indices]
      combined = torch.cat([self.U, new_dirs], dim=1)
      
      u_rot, _, _ = torch.svd(torch.matmul(combined.T, flat_data))
      self.U = torch.matmul(combined, u_rot[:, :self.r])
      self.U, _ = torch.linalg.qr(self.U)
      
    return len(added_indices)

  def extract_sparse_components(self, X_t: torch.Tensor, max_iter: int = 10, lambda_sparse: float = 0.05) -> torch.Tensor:
    """
    Extract sparse noise/artifacts using l1-regularization (ISTA).
    """
    eye_n = torch.eye(self.N, device=self.device)
    phi_u = eye_n - torch.matmul(self.U, self.U.T)
    
    eye_s = torch.eye(self.S, device=self.device)
    phi_v = eye_s - torch.matmul(self.V, self.V.T)

    S_t = torch.zeros_like(X_t)
    
    for _ in range(max_iter):
      residual = X_t - S_t
      proj_res = torch.tensordot(residual, phi_u, dims=([1], [0]))
      proj_res = torch.tensordot(proj_res, phi_u, dims=([1], [0]))
      proj_res = torch.tensordot(proj_res, phi_v, dims=([0], [0]))
      proj_res = proj_res.permute(2, 0, 1)

      S_t = S_t + proj_res
      S_t = torch.sign(S_t) * torch.clamp(torch.abs(S_t) - lambda_sparse, min=0)
      
    return S_t

  def decompose_tensor_stream(self, tensor_4d: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the full HO-RLSL decomposition pipeline across all time points.
    """
    n_subs, n_chan, _, n_times = tensor_4d.shape
    self.initialize_tucker_subspace(tensor_4d, n_init=128)
    
    energy = torch.zeros(n_times, device=self.device)
    weights_evolution = torch.zeros((n_chan, n_times), device=self.device)
    subspace_velocity = torch.zeros(n_times, device=self.device)
    lowrank_tensor = torch.zeros_like(tensor_4d, device=self.device)
    lowrank_buffer = torch.zeros_like(tensor_4d, device=self.device)
    change_points: List[int] = []
    
    U_prev = self.U.clone()
    
    for t in range(n_times):
      X_t = tensor_4d[:, :, :, t].to(self.device)
      
      S_t = self.extract_sparse_components(X_t)
      L_clean_t = X_t - S_t
      lowrank_buffer[:, :, :, t] = L_clean_t
      
      temp = torch.tensordot(L_clean_t, self.U, dims=([1], [0]))
      temp = torch.tensordot(temp, self.U, dims=([1], [0]))
      core = torch.tensordot(temp, self.V, dims=([0], [0]))
      
      energy[t] = torch.norm(core)**2
      weights_evolution[:, t] = self.U[:, 0]
      
      proj_current = torch.matmul(self.U, self.U.T)
      proj_prev = torch.matmul(U_prev, U_prev.T)
      subspace_velocity[t] = torch.norm(proj_current - proj_prev)
      U_prev = self.U.clone()
      
      l_temp = torch.tensordot(core, self.V, dims=([2], [1]))
      l_temp = torch.tensordot(l_temp, self.U, dims=([0], [1]))
      lowrank_tensor[:, :, :, t] = torch.tensordot(l_temp, self.U, dims=([0], [1]))
      
      if t >= 128 and t % self.alpha == 0: 
        window = lowrank_buffer[:, :, :, t-self.alpha : t]
        if self.update_recursive_subspace(window) >= 1:
          change_points.append(t)

    return (
      weights_evolution.cpu().numpy(), 
      energy.cpu().numpy(), 
      subspace_velocity.cpu().numpy(), 
      lowrank_tensor.cpu().numpy(), 
      np.array(change_points, dtype=int)
    )

def main() -> None:
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Starting HO-RLSL decomposition on {device}...")
  
  for cond in ['incorrect', 'correct']:
    tensor_file = config.tensor_incorrect_file if cond == 'incorrect' else config.tensor_correct_file
    print(f" Processing {cond} condition...")
    
    data = np.load(tensor_file)
    tensor = torch.tensor(data, dtype=torch.float32, device=device)
    n_subs, n_nodes = tensor.shape[0], tensor.shape[1]
    
    decomposer = HORLSDecomposer(n_nodes=n_nodes, n_subs=n_subs, device=device)
    w_t, energy, velocity, L_t, cp_indices = decomposer.decompose_tensor_stream(tensor)
    
    np.save(config.paths.TENSOR_DIR / f"horls_weights_{cond}.npy", w_t)
    np.save(config.paths.TENSOR_DIR / f"horls_energy_{cond}.npy", energy)
    np.save(config.paths.TENSOR_DIR / f"horls_lowrank_{cond}.npy", L_t)
    np.save(config.paths.TENSOR_DIR / f"horls_cp_{cond}.npy", cp_indices)
    
    # Export to MATLAB
    savemat(config.paths.MATLAB_DIR / f"horls_results_{cond}.mat", {
      'weights': w_t,
      'energy': energy,
      'velocity': velocity,
      'change_points': cp_indices
    })
    
  print("HO-RLSL decomposition completed successfully.")

if __name__ == "__main__":
  main()
