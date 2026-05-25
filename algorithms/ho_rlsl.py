from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

from algorithms.common import DecompositionConfig
from algorithms.common import DecompositionResult
from algorithms.common import add_direction
from algorithms.common import concatenate_mode_unfoldings
from algorithms.common import compute_intervals
from algorithms.common import config_to_dict
from algorithms.common import convert_subject_tensor_to_stream
from algorithms.common import delete_direction
from algorithms.common import ensure_symmetric
from algorithms.common import gtcs_s_recovery
from algorithms.common import project_orthogonal
from algorithms.common import reconstruct_from_bases
from algorithms.common import result_to_legacy_layout
from algorithms.common import save_result_bundle
from algorithms.common import soft_threshold
from algorithms.common import truncated_basis
from algorithms.common import unfold_tensor
from core.config import config


class HORLSLRunner:
  def __init__(self, decomposition_config: DecompositionConfig):
    self.config = decomposition_config
    self.reinit_buffer = []

  def _initial_bases(self, stream: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, float, float]:
    train_window = stream[: self.config.train_steps]
    mode_count = train_window[0].ndim
    bases: list[np.ndarray] = []
    correlations: list[np.ndarray] = []
    thresholds: list[float] = []
    
    # Track residual energy during training for statistical thresholding
    training_residuals = []

    for mode in range(mode_count):
      unfolded = concatenate_mode_unfoldings(train_window, mode)
      # Initial correlation matrix R = L * L^T
      R = unfolded @ unfolded.T
      u, s_vals, _ = np.linalg.svd(unfolded, full_matrices=False)
      print(f"Mode {mode} training top 10 singular values: {s_vals[:10]}")
      
      basis, threshold = truncated_basis(
        unfolded,
        sigma_min=self.config.sigma_min,
        sigma_scale=self.config.sigma_scale,
        max_rank=self.config.max_rank,
      )
      bases.append(basis)
      correlations.append(R)
      thresholds.append(threshold)
    
    # Calculate baseline residual energy stats
    for t in range(self.config.train_steps):
        observed = stream[t]
        projected = observed.copy()
        for mode, b in enumerate(bases):
            P = np.eye(b.shape[0]) - b @ b.T
            projected = np.moveaxis(np.tensordot(P, np.moveaxis(projected, mode, 0), axes=([1], [0])), 0, mode)
        training_residuals.append(np.mean(projected**2))
    
    mu_res = np.mean(training_residuals)
    std_res = np.std(training_residuals)

    return bases, correlations, np.asarray(thresholds, dtype=np.float32), mu_res, std_res

  def run(self, stream: np.ndarray) -> DecompositionResult:
    n_times = stream.shape[0]
    
    def delete_direction(data: np.ndarray, P: np.ndarray, sigma_min: float) -> np.ndarray:
        if P.shape[1] == 0: return P
        w = data.shape[1]
        lambdas = np.diag((P.T @ data) @ (P.T @ data).T) / w
        keep_idx = np.where(lambdas >= sigma_min)[0]
        if len(keep_idx) == 0: return P[:, :1]
        return P[:, keep_idx]

    def add_direction(data: np.ndarray, P: np.ndarray, sigma_min: float, max_rank: int) -> np.ndarray:
        w = data.shape[1]
        proj_matrix = np.eye(P.shape[0]) - P @ P.T
        D_proj = proj_matrix @ data
        cov = (D_proj @ D_proj.T) / w
        evals, evecs = np.linalg.eigh(cov)
        idx = np.argsort(evals)[::-1]
        evals, evecs = evals[idx], evecs[:, idx]
        
        if data.shape[1] > 0:
            print(f"  Residual evals (top 5): {evals[:5]}")
            
        add_idx = np.where(evals > sigma_min)[0]
        if len(add_idx) == 0:
            return P
            
        new_directions = evecs[:, add_idx]
        combined = np.hstack([P, new_directions])
        # Paper-aligned strict rank cap (Stage 4)
        limit = min(max_rank, P.shape[0] - 1) if max_rank is not None else P.shape[0] - 1
        if combined.shape[1] > limit:
            combined = combined[:, :limit]
        return combined

    bases, _, thresholds, _, _ = self._initial_bases(stream)
    
    lowrank_stream = np.zeros_like(stream, dtype=np.float32)
    sparse_stream = np.zeros_like(stream, dtype=np.float32)
    residual_energy = np.zeros(n_times, dtype=np.float32)
    mode_ranks = np.zeros((n_times, 3), dtype=np.int32)
    change_points: list[int] = []
    
    self.reinit_buffer = []
    t_j = self.config.train_steps  # Anchor for dynamic window alignment (Stage 2)

    for t in range(n_times):
      observed = stream[t]
      
      # Sparse Component Recovery (L+S model)
      projected_t = observed.copy()
      projectors = []
      for mode, b in enumerate(bases):
          P = np.eye(b.shape[0]) - b @ b.T
          projectors.append(P)
          projected_t = np.moveaxis(np.tensordot(P, np.moveaxis(projected_t, mode, 0), axes=([1], [0])), 0, mode)
      
      S_t = gtcs_s_recovery(projected_t, projectors, self.config.lambda_sparse, recovery_mode=self.config.recovery_mode)
      if self.config.symmetric_modes: S_t = ensure_symmetric(S_t)
      L_t = observed - S_t

      # Subspace Tracking updates only occur for t >= train_steps (as per Algorithm 1: t > t_train)
      if t >= self.config.train_steps:
          # Accumulate denoised Low-Rank L_t instead of observed raw M_t (Stage 1)
          self.reinit_buffer.append(L_t)
          
          # Dynamic modulo check aligned to last change point t_j (Stage 2)
          if (t - t_j + 1) % self.config.alpha == 0:
              subspace_changed = False
              window_data = np.array(self.reinit_buffer)
              self.reinit_buffer = []
              
              for mode in range(len(bases)):
                  unfolded_window = concatenate_mode_unfoldings(window_data, mode)
                  rank_before = bases[mode].shape[1]
                  
                  sig_min = self.config.sigma_min if self.config.sigma_min is not None else float(thresholds[mode])
                  # Step 1: Delete Direction
                  bases[mode] = delete_direction(unfolded_window, bases[mode], sig_min)
                  # Step 2: Add Direction
                  bases[mode] = add_direction(unfolded_window, bases[mode], sig_min, self.config.max_rank)
                  
                  rank_after = bases[mode].shape[1]
                  if rank_after != rank_before:
                      subspace_changed = True
              
              if subspace_changed:
                  change_points.append(t)
                  t_j = t + 1  # Reset anchor to the first sample of the next window (Stage 2)

      lowrank_stream[t] = L_t.astype(np.float32)
      sparse_stream[t] = S_t.astype(np.float32)
      residual_energy[t] = float(np.mean(projected_t**2))
      mode_ranks[t] = np.asarray([b.shape[1] for b in bases[:3]], dtype=np.int32)

    intervals = compute_intervals(change_points, n_times)
    return DecompositionResult(
      algorithm="ho_rlsl_literal",
      config=config_to_dict(self.config),
      lowrank_stream=lowrank_stream,
      sparse_stream=sparse_stream,
      change_points=np.sort(np.unique(np.asarray(change_points, dtype=np.int32))),
      intervals=intervals,
      mode_ranks=mode_ranks,
      residual_energy=residual_energy,
      sparse_mass=np.zeros(n_times),
      sigma_thresholds=thresholds,
    )


def load_condition_tensor(condition: str) -> np.ndarray:
  tensor_file = config.tensor_incorrect_file if condition == "incorrect" else config.tensor_correct_file
  return np.load(tensor_file).astype(np.float32)


def default_config_for_condition() -> DecompositionConfig:
  return DecompositionConfig(
    train_steps=10,    
    alpha=8,           
    sigma_min=0.11,    
    max_rank=4,        
    lambda_sparse=0.1, 
    symmetric_modes=True,
    name="eeg_research_standard",
  )


def save_condition_outputs(condition: str, result: DecompositionResult) -> None:
  output_dir = config.paths.TRACKING_DIR
  artifact_prefix = output_dir / f"horls_{condition}"
  bundle_path = output_dir / f"horls_{condition}_bundle.npz"
  save_result_bundle(bundle_path, result)
  lowrank_legacy, sparse_legacy = result_to_legacy_layout(result)

  np.save(output_dir / f"{artifact_prefix.name}_lowrank.npy", lowrank_legacy)
  np.save(output_dir / f"{artifact_prefix.name}_sparse.npy", sparse_legacy)
  np.save(output_dir / f"{artifact_prefix.name}_cp.npy", result.change_points)
  np.save(output_dir / f"{artifact_prefix.name}_intervals.npy", result.intervals)
  np.save(output_dir / f"{artifact_prefix.name}_energy.npy", result.residual_energy)
  np.save(output_dir / f"{artifact_prefix.name}_sparse_mass.npy", result.sparse_mass)
  np.save(output_dir / f"{artifact_prefix.name}_mode_ranks.npy", result.mode_ranks)
  np.save(output_dir / f"{artifact_prefix.name}_mode_rank_trace.npy", result.mode_ranks[:, 0].astype(np.float32))

  savemat(config.paths.MATLAB_DIR / f"horls_results_{condition}.mat", {
    "lowrank": lowrank_legacy,
    "sparse": sparse_legacy,
    "change_points": result.change_points,
    "intervals": result.intervals,
    "residual_energy": result.residual_energy,
    "sparse_mass": result.sparse_mass,
    "mode_ranks": result.mode_ranks,
    "sigma_thresholds": result.sigma_thresholds,
  })


def run_condition(condition: str, decomposition_config: DecompositionConfig | None = None) -> DecompositionResult:
  condition_config = decomposition_config or default_config_for_condition()
  subject_tensor = load_condition_tensor(condition)
  stream = convert_subject_tensor_to_stream(subject_tensor)
  return HORLSLRunner(condition_config).run(stream)


def main() -> None:
  parser = argparse.ArgumentParser(description="Run paper-aligned HO-RLSL on EEG tensors.")
  parser.add_argument("--condition", choices=["correct", "incorrect", "both"], default="both")
  parser.add_argument("--train-steps", type=int, default=10)
  parser.add_argument("--alpha", type=int, default=8)
  parser.add_argument("--sigma-min", type=float, default=0.11)
  parser.add_argument("--lambda-sparse", type=float, default=0.05)
  args = parser.parse_args()

  run_config = DecompositionConfig(
    train_steps=args.train_steps,
    alpha=args.alpha,
    sigma_min=args.sigma_min,
    lambda_sparse=args.lambda_sparse,
    symmetric_modes=True,
    name="eeg_cli",
  )

  conditions = ["correct", "incorrect"] if args.condition == "both" else [args.condition]
  for condition in conditions:
    result = run_condition(condition, decomposition_config=replace(run_config))
    save_condition_outputs(condition, result)
    print(
      f"HO-RLSL {condition}: {len(result.change_points)} change points, "
      f"{len(result.intervals)} intervals, thresholds={result.sigma_thresholds.tolist()}"
    )


if __name__ == "__main__":
  main()
