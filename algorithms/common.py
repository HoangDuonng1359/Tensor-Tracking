from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


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
  metadata = {
    "algorithm": result.algorithm,
    "config": result.config,
  }
  metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_legacy_outputs(prefix: Path, result: DecompositionResult) -> None:
  lowrank_legacy, _ = result_to_legacy_layout(result)
  np.save(prefix.parent / f"{prefix.name}_lowrank.npy", lowrank_legacy)
  np.save(prefix.parent / f"{prefix.name}_cp.npy", result.change_points)
  np.save(prefix.parent / f"{prefix.name}_energy.npy", result.residual_energy)
  first_mode_weights = result.mode_ranks[:, 0].astype(np.float32)
  np.save(prefix.parent / f"{prefix.name}_weights.npy", first_mode_weights)


def save_summary_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def config_to_dict(config: DecompositionConfig) -> dict[str, Any]:
  return asdict(config)
