from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
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
from algorithms.common import reconstruct_from_bases
from algorithms.common import result_to_legacy_layout
from algorithms.common import save_result_bundle
from algorithms.common import truncated_basis
from core.config import config


class HOSVDRunner:
  def __init__(self, decomposition_config: DecompositionConfig):
    self.config = decomposition_config

  def _initial_bases(self, stream: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    train_window = stream[: self.config.train_steps]
    bases: list[np.ndarray] = []
    thresholds: list[float] = []

    mode0_basis, mode0_threshold = truncated_basis(
      concatenate_mode_unfoldings(train_window, 0),
      sigma_min=self.config.sigma_min,
      sigma_scale=self.config.sigma_scale,
      max_rank=self.config.max_rank,
    )
    bases.append(mode0_basis)
    thresholds.append(mode0_threshold)

    if self.config.symmetric_modes:
      bases.append(mode0_basis.copy())
      thresholds.append(mode0_threshold)
    else:
      mode1_basis, mode1_threshold = truncated_basis(
        concatenate_mode_unfoldings(train_window, 1),
        sigma_min=self.config.sigma_min,
        sigma_scale=self.config.sigma_scale,
        max_rank=self.config.max_rank,
      )
      bases.append(mode1_basis)
      thresholds.append(mode1_threshold)

    basis2, threshold2 = truncated_basis(
      concatenate_mode_unfoldings(train_window, 2),
      sigma_min=self.config.sigma_min,
      sigma_scale=self.config.sigma_scale,
      max_rank=self.config.max_rank,
    )
    bases.append(basis2)
    thresholds.append(threshold2)
    return bases, np.asarray(thresholds, dtype=np.float32)

  def _update_bases(
    self,
    bases: list[np.ndarray],
    thresholds: np.ndarray,
    window: np.ndarray,
  ) -> tuple[list[np.ndarray], bool]:
    updated = [basis.copy() for basis in bases]
    changed = False

    mode0_data = concatenate_mode_unfoldings(window, 0)
    candidate = delete_direction(mode0_data, updated[0], float(thresholds[0]))
    candidate = add_direction(mode0_data, candidate, float(thresholds[0]), self.config.max_rank)
    if candidate.shape != updated[0].shape or not np.allclose(candidate, updated[0], atol=1e-5):
      changed = True
    updated[0] = candidate

    if self.config.symmetric_modes:
      updated[1] = updated[0].copy()
    else:
      mode1_data = concatenate_mode_unfoldings(window, 1)
      candidate = delete_direction(mode1_data, updated[1], float(thresholds[1]))
      candidate = add_direction(mode1_data, candidate, float(thresholds[1]), self.config.max_rank)
      if candidate.shape != updated[1].shape or not np.allclose(candidate, updated[1], atol=1e-5):
        changed = True
      updated[1] = candidate

    mode2_data = concatenate_mode_unfoldings(window, 2)
    candidate = delete_direction(mode2_data, updated[2], float(thresholds[2]))
    candidate = add_direction(mode2_data, candidate, float(thresholds[2]), self.config.max_rank)
    if candidate.shape != updated[2].shape or not np.allclose(candidate, updated[2], atol=1e-5):
      changed = True
    updated[2] = candidate

    return updated, changed

  def run(self, stream: np.ndarray) -> DecompositionResult:
    n_times = stream.shape[0]
    bases, thresholds = self._initial_bases(stream)

    lowrank_stream = np.zeros_like(stream, dtype=np.float32)
    sparse_stream = np.zeros_like(stream, dtype=np.float32)
    residual_energy = np.zeros(n_times, dtype=np.float32)
    sparse_mass = np.zeros(n_times, dtype=np.float32)
    mode_ranks = np.zeros((n_times, 3), dtype=np.int32)
    change_points: list[int] = []
    interval_anchor = self.config.train_steps

    for t in range(n_times):
      observed = stream[t]
      lowrank_estimate = reconstruct_from_bases(observed, bases)
      if self.config.symmetric_modes:
        lowrank_estimate = ensure_symmetric(lowrank_estimate)

      sparse_estimate = observed - lowrank_estimate
      lowrank_stream[t] = lowrank_estimate.astype(np.float32)
      sparse_stream[t] = sparse_estimate.astype(np.float32)
      residual_energy[t] = float(np.linalg.norm(sparse_estimate) ** 2)
      sparse_mass[t] = float(np.sum(np.abs(sparse_estimate)))
      mode_ranks[t] = np.asarray([basis.shape[1] for basis in bases[:3]], dtype=np.int32)

      update_ready = t + 1 >= self.config.train_steps and (t - interval_anchor + 1) % self.config.alpha == 0
      if update_ready:
        window = lowrank_stream[t - self.config.alpha + 1 : t + 1]
        updated_bases, changed = self._update_bases(bases, thresholds, window)
        bases = updated_bases
        if changed:
          change_points.append(t)
          interval_anchor = t + 1

    intervals = compute_intervals(change_points, n_times)
    return DecompositionResult(
      algorithm="hosvd",
      config=config_to_dict(self.config),
      lowrank_stream=lowrank_stream,
      sparse_stream=sparse_stream,
      change_points=np.asarray(change_points, dtype=np.int32),
      intervals=intervals,
      mode_ranks=mode_ranks,
      residual_energy=residual_energy,
      sparse_mass=sparse_mass,
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
    lambda_sparse=0.0,
    symmetric_modes=True,
    name="eeg_default",
  )


def save_condition_outputs(condition: str, result: DecompositionResult) -> None:
  output_dir = config.paths.TENSOR_DIR
  bundle_path = output_dir / f"hosvd_{condition}_bundle.npz"
  save_result_bundle(bundle_path, result)

  lowrank_legacy, sparse_legacy = result_to_legacy_layout(result)
  np.save(output_dir / f"hosvd_lowrank_{condition}.npy", lowrank_legacy)
  np.save(output_dir / f"hosvd_cp_{condition}.npy", result.change_points)
  np.save(output_dir / f"hosvd_energy_{condition}.npy", result.residual_energy)
  np.save(output_dir / f"hosvd_sparse_{condition}.npy", sparse_legacy)
  np.save(output_dir / f"hosvd_weights_{condition}.npy", result.mode_ranks[:, 0].astype(np.float32))

  savemat(config.paths.MATLAB_DIR / f"hosvd_results_{condition}.mat", {
    "lowrank": lowrank_legacy,
    "sparse": sparse_legacy,
    "change_points": result.change_points,
    "intervals": result.intervals,
    "residual_energy": result.residual_energy,
    "mode_ranks": result.mode_ranks,
    "sigma_thresholds": result.sigma_thresholds,
  })


def run_condition(condition: str, decomposition_config: DecompositionConfig | None = None) -> DecompositionResult:
  condition_config = decomposition_config or default_config_for_condition()
  subject_tensor = load_condition_tensor(condition)
  stream = convert_subject_tensor_to_stream(subject_tensor)
  return HOSVDRunner(condition_config).run(stream)


def main() -> None:
  parser = argparse.ArgumentParser(description="Run HoSVD baseline on EEG tensors.")
  parser.add_argument("--condition", choices=["correct", "incorrect", "both"], default="both")
  parser.add_argument("--train-steps", type=int, default=10)
  parser.add_argument("--alpha", type=int, default=8)
  parser.add_argument("--sigma-min", type=float, default=0.11)
  args = parser.parse_args()

  run_config = DecompositionConfig(
    train_steps=args.train_steps,
    alpha=args.alpha,
    sigma_min=args.sigma_min,
    lambda_sparse=0.0,
    symmetric_modes=True,
    name="eeg_cli",
  )

  conditions = ["correct", "incorrect"] if args.condition == "both" else [args.condition]
  for condition in conditions:
    result = run_condition(condition, decomposition_config=replace(run_config))
    save_condition_outputs(condition, result)
    print(
      f"HoSVD {condition}: {len(result.change_points)} change points, "
      f"{len(result.intervals)} intervals, thresholds={result.sigma_thresholds.tolist()}"
    )


if __name__ == "__main__":
  main()
