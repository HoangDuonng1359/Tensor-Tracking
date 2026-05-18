from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import savemat

from core.timing import eeg_timing


@dataclass
class DecompositionConfig:
  train_steps: int
  alpha: int
  sigma_min: float | None = None
  sigma_scale: float = 0.1
  lambda_sparse: float = 0.05
  max_rank: int | None = None
  symmetric_modes: bool = True
  name: str = "default"
  recovery_mode: str = "pgd"


@dataclass
class DecompositionResult:
  algorithm: str
  config: dict[str, Any]
  lowrank_stream: np.ndarray
  sparse_stream: np.ndarray
  change_points: np.ndarray
  intervals: np.ndarray
  mode_ranks: np.ndarray
  residual_energy: np.ndarray
  sparse_mass: np.ndarray
  sigma_thresholds: np.ndarray


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
  return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def gtcs_s_recovery(
  Y: np.ndarray,
  projectors: list[np.ndarray],
  lambda_sparse: float,
  iterations: int = 5,
  recovery_mode: str = "pgd",
) -> np.ndarray:
  """
  Generalized Tensor Compressive Sensing recovery.
  Supports both parallel PGD (Proximal Gradient Descent) and literal serial GTCS-S.
  """
  if recovery_mode == "gtcs_s_serial":
      return gtcs_s_serial_recovery(Y, projectors, lambda_sparse, iterations)
      
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # Convert to torch for performance
  Y_t = torch.as_tensor(Y, device=device, dtype=torch.float32)
  P_t = [torch.as_tensor(p, device=device, dtype=torch.float32) for p in projectors]
  S_t = Y_t.clone()
  
  for _ in range(iterations):
    # Forward projection
    Y_est = S_t
    for mode, P in enumerate(P_t):
      # mode_product using torch.tensordot
      Y_est = torch.moveaxis(torch.tensordot(P, torch.moveaxis(Y_est, mode, 0), dims=([1], [0])), 0, mode)

    # Gradient update
    error = Y_t - Y_est
    update = error
    for mode, P in enumerate(P_t):
      update = torch.moveaxis(torch.tensordot(P, torch.moveaxis(update, mode, 0), dims=([1], [0])), 0, mode)

    S_t = S_t + update
    # Proximal step (soft thresholding)
    S_t = torch.sign(S_t) * torch.clamp(torch.abs(S_t) - lambda_sparse, min=0.0)

  return S_t.cpu().numpy()


def gtcs_s_serial_recovery(
  Y: np.ndarray,
  projectors: list[np.ndarray],
  lambda_sparse: float,
  iterations: int = 5,
) -> np.ndarray:
  """
  Literal serial recovery procedure for compressed tensors (GTCS-S).
  Repeatedly unfolds the tensor along each mode and applies coordinate-descent style
  l1 proximal steps (soft thresholding) sequentially.
  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  Y_t = torch.as_tensor(Y, device=device, dtype=torch.float32)
  P_t = [torch.as_tensor(p, device=device, dtype=torch.float32) for p in projectors]
  S_t = Y_t.clone()

  for _ in range(iterations):
    # Mode-by-mode sequential update
    for mode in range(3):
      # Current projection of S along this mode
      P = P_t[mode]
      S_proj = torch.moveaxis(torch.tensordot(P, torch.moveaxis(S_t, mode, 0), dims=([1], [0])), 0, mode)
      
      # Residual error along this mode
      error = Y_t - S_proj
      update = torch.moveaxis(torch.tensordot(P, torch.moveaxis(error, mode, 0), dims=([1], [0])), 0, mode)
      
      # Update S for this mode
      S_t = S_t + update
      
      # Proximal soft-thresholding step
      S_t = torch.sign(S_t) * torch.clamp(torch.abs(S_t) - lambda_sparse, min=0.0)

  return S_t.cpu().numpy()


def calculate_nmse(original: np.ndarray, estimated: np.ndarray) -> float:
  """Normalized Mean Square Error."""
  denom = np.linalg.norm(original) ** 2
  if denom < 1e-10: return 0.0
  return float(np.linalg.norm(original - estimated) ** 2 / denom)


def subspace_distance(U_true: np.ndarray, U_est: np.ndarray) -> float:
  """
  Subspace Projection Distance. 
  Measures the distance between the true subspace and the estimated subspace.
  Range: [0, 1], where 0 means identical subspaces.
  """
  k = U_true.shape[1]
  if k == 0: return 1.0
  # Projection distance: 1 - (1/k) * ||U_est.T @ U_true||_F^2
  val = np.linalg.norm(U_est.T @ U_true) ** 2 / k
  return float(1.0 - val)


def ensure_symmetric(tensor: np.ndarray) -> np.ndarray:
  return 0.5 * (tensor + np.swapaxes(tensor, 0, 1))


def mode_product(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
  moved = np.moveaxis(tensor, mode, 0)
  projected = np.tensordot(matrix, moved, axes=(1, 0))
  return np.moveaxis(projected, 0, mode)


def unfold_tensor(tensor: np.ndarray, mode: int) -> np.ndarray:
  moved = np.moveaxis(tensor, mode, 0)
  return moved.reshape(moved.shape[0], -1)


def concatenate_mode_unfoldings(window: np.ndarray, mode: int) -> np.ndarray:
  # Window shape: (time, n1, n2, n3)
  matrices = [unfold_tensor(frame, mode) for frame in window]
  return np.concatenate(matrices, axis=1)


def truncated_basis(
  matrix: np.ndarray,
  sigma_min: float | None,
  sigma_scale: float,
  max_rank: int | None = None,
) -> tuple[np.ndarray, float]:
  u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
  if singular_values.size == 0:
    return np.eye(matrix.shape[0], dtype=np.float32), 0.0

  threshold = sigma_min if sigma_min is not None else sigma_scale * float(singular_values[0])
  keep = singular_values >= threshold
  if not np.any(keep):
    keep[0] = True

  basis = u[:, keep]
  if max_rank is not None and basis.shape[1] > max_rank:
    basis = basis[:, :max_rank]

  return basis.astype(np.float32), float(threshold)


def reconstruct_from_bases(tensor: np.ndarray, bases: list[np.ndarray]) -> np.ndarray:
  core = tensor
  for mode, basis in enumerate(bases):
    core = mode_product(core, basis.T, mode)
  recon = core
  for mode, basis in enumerate(bases):
    recon = mode_product(recon, basis, mode)
  return recon


def project_orthogonal(tensor: np.ndarray, bases: list[np.ndarray]) -> np.ndarray:
  projected = tensor
  for mode, basis in enumerate(bases):
    projector = np.eye(basis.shape[0], dtype=np.float32) - basis @ basis.T
    projected = mode_product(projected, projector, mode)
  return projected


def delete_direction(data: np.ndarray, basis: np.ndarray, sigma_min: float) -> np.ndarray:
  if basis.size == 0:
    return basis

  basis_projection = basis.T @ data
  energies = np.sum(basis_projection * basis_projection, axis=1) / max(data.shape[1], 1)
  keep = energies >= sigma_min
  if not np.any(keep):
    keep[np.argmax(energies)] = True
  return basis[:, keep]


def add_direction(
  data: np.ndarray,
  basis: np.ndarray,
  sigma_min: float,
  max_rank: int | None = None,
) -> np.ndarray:
  projector = np.eye(basis.shape[0], dtype=np.float32) - basis @ basis.T
  projected = projector @ data
  covariance = projected @ projected.T / max(data.shape[1], 1)
  eigenvalues, eigenvectors = np.linalg.eigh(covariance)
  keep = eigenvalues > sigma_min
  additions = eigenvectors[:, keep]
  if additions.size == 0:
    return basis

  updated = np.concatenate([basis, additions], axis=1)
  q, _ = np.linalg.qr(updated)
  if max_rank is not None and q.shape[1] > max_rank:
    q = q[:, :max_rank]
  return q.astype(np.float32)


def compute_intervals(change_points: list[int], n_times: int) -> np.ndarray:
  if n_times == 0:
    return np.zeros((0, 2), dtype=int)

  intervals: list[list[int]] = []
  start = 0
  for cp in sorted(set(int(point) for point in change_points if 0 <= point < n_times)):
    intervals.append([start, cp])
    start = cp + 1
  intervals.append([start, n_times - 1])
  return np.asarray(intervals, dtype=int)


def convert_subject_tensor_to_stream(tensor_4d: np.ndarray) -> np.ndarray:
  # (subjects, channels, channels, time) -> (time, channels, channels, subjects)
  return np.transpose(tensor_4d, (3, 1, 2, 0)).astype(np.float32)


def convert_stream_to_subject_tensor(stream: np.ndarray) -> np.ndarray:
  # (time, channels, channels, subjects) -> (subjects, channels, channels, time)
  return np.transpose(stream, (3, 1, 2, 0)).astype(np.float32)


def result_to_legacy_layout(result: DecompositionResult) -> tuple[np.ndarray, np.ndarray]:
  lowrank_legacy = convert_stream_to_subject_tensor(result.lowrank_stream)
  sparse_legacy = convert_stream_to_subject_tensor(result.sparse_stream)
  return lowrank_legacy, sparse_legacy


def save_result_bundle(path: Path, result: DecompositionResult) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    path,
    lowrank_stream=result.lowrank_stream,
    sparse_stream=result.sparse_stream,
    change_points=result.change_points,
    intervals=result.intervals,
    mode_ranks=result.mode_ranks,
    residual_energy=result.residual_energy,
    sparse_mass=result.sparse_mass,
    sigma_thresholds=result.sigma_thresholds,
  )
  metadata_path = path.with_suffix(".json")
  timing = eeg_timing(result.lowrank_stream.shape[0])
  metadata = {
    "algorithm": result.algorithm,
    "config": result.config,
    "timing": timing.to_metadata(),
    "artifacts": {
      "change_points": f"{path.stem.replace('_bundle', '')}_cp.npy",
      "intervals": f"{path.stem.replace('_bundle', '')}_intervals.npy",
      "residual_energy": f"{path.stem.replace('_bundle', '')}_energy.npy",
      "sparse_mass": f"{path.stem.replace('_bundle', '')}_sparse_mass.npy",
      "mode_ranks": f"{path.stem.replace('_bundle', '')}_mode_ranks.npy",
      "mode_rank_trace": f"{path.stem.replace('_bundle', '')}_mode_rank_trace.npy",
      "lowrank_tensor": f"{path.stem.replace('_bundle', '')}_lowrank.npy",
      "sparse_tensor": f"{path.stem.replace('_bundle', '')}_sparse.npy",
    },
  }
  metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_legacy_outputs(prefix: Path, result: DecompositionResult) -> None:
  lowrank_legacy, sparse_legacy = result_to_legacy_layout(result)
  np.save(prefix.parent / f"{prefix.name}_lowrank.npy", lowrank_legacy)
  np.save(prefix.parent / f"{prefix.name}_sparse.npy", sparse_legacy)
  np.save(prefix.parent / f"{prefix.name}_cp.npy", result.change_points)
  np.save(prefix.parent / f"{prefix.name}_intervals.npy", result.intervals)
  np.save(prefix.parent / f"{prefix.name}_energy.npy", result.residual_energy)
  np.save(prefix.parent / f"{prefix.name}_sparse_mass.npy", result.sparse_mass)
  np.save(prefix.parent / f"{prefix.name}_mode_ranks.npy", result.mode_ranks)
  np.save(prefix.parent / f"{prefix.name}_mode_rank_trace.npy", result.mode_ranks[:, 0].astype(np.float32))


def save_summary_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def config_to_dict(config: DecompositionConfig) -> dict[str, Any]:
  return asdict(config)
