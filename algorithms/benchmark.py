from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from algorithms.common import DecompositionConfig, DecompositionResult, convert_subject_tensor_to_stream
from algorithms.ho_rlsl import HORLSLRunner
from algorithms.hosvd import HOSVDRunner
from core.config import config
from core.timing import eeg_timing, EEGTiming


def _subject_sample_from_template(template: np.ndarray, rng: np.random.Generator) -> np.ndarray:
  noise = rng.normal(loc=0.0, scale=0.03, size=template.shape)
  sample = np.clip(template + noise, 0.0, 1.0)
  sample = 0.5 * (sample + sample.T)
  np.fill_diagonal(sample, 0.0)
  return sample.astype(np.float32)


def _module_membership(network_type: str, interval: int, n_nodes: int) -> np.ndarray:
  if network_type == "modular":
    if interval in (0, 2):
      membership = np.zeros((n_nodes, 2), dtype=np.float32)
      membership[: n_nodes // 2, 0] = 1.0
      membership[n_nodes // 2 :, 1] = 1.0
      return membership
    membership = np.eye(4, dtype=np.float32)[np.repeat(np.arange(4), n_nodes // 4)]
    return membership

  if network_type == "hierarchical":
    if interval in (0, 2):
      membership = np.zeros((n_nodes, 2), dtype=np.float32)
      membership[: n_nodes // 2, 0] = 1.0
      membership[n_nodes // 2 :, 1] = 1.0
      return membership
    membership = np.eye(4, dtype=np.float32)[np.repeat(np.arange(4), n_nodes // 4)]
    membership[: n_nodes // 2, 0] += 0.5
    membership[n_nodes // 2 :, 1] += 0.5
    return membership

  if network_type == "overlapping":
    membership = np.zeros((n_nodes, 2), dtype=np.float32)
    membership[: n_nodes // 2, 0] = 1.0
    membership[n_nodes // 2 :, 1] = 1.0
    if interval == 1:
      overlap = slice(n_nodes // 4, n_nodes // 2)
      membership[overlap, 1] = 0.75
      overlap = slice(n_nodes // 2, 3 * n_nodes // 4)
      membership[overlap, 0] = 0.75
    return membership

  raise ValueError(f"Unsupported network_type={network_type}")


def _template_from_membership(network_type: str, membership: np.ndarray) -> np.ndarray:
  base = membership @ membership.T
  if network_type == "hierarchical" and membership.shape[1] == 4:
    super_groups = np.zeros((membership.shape[0], 2), dtype=np.float32)
    super_groups[: membership.shape[0] // 2, 0] = 1.0
    super_groups[membership.shape[0] // 2 :, 1] = 1.0
    base += 0.35 * (super_groups @ super_groups.T)

  base = 0.1 + 0.5 * np.clip(base, 0.0, 1.0)
  base = 0.5 * (base + base.T)
  np.fill_diagonal(base, 0.0)
  return np.clip(base, 0.0, 1.0).astype(np.float32)


def generate_synthetic_tensor(
  network_type: str,
  n_nodes: int = 64,
  n_subjects: int = 60,
  n_times: int = 80,
  noise_sparsity: float = 0.1,
  seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  rng = np.random.default_rng(seed)
  templates = [
    _template_from_membership(network_type, _module_membership(network_type, interval, n_nodes))
    for interval in range(3)
  ]

  first_boundary = max(1, int(round(0.25 * n_times)))
  second_boundary = max(first_boundary + 1, int(round(0.75 * n_times)))
  second_boundary = min(second_boundary, n_times - 1)

  truth = np.zeros((n_times, n_nodes, n_nodes, n_subjects), dtype=np.float32)
  observed = np.zeros_like(truth)
  interval_truth = np.asarray(
    [
      [0, first_boundary - 1],
      [first_boundary, second_boundary - 1],
      [second_boundary, n_times - 1],
    ],
    dtype=int,
  )

  for t in range(n_times):
    template = templates[0]
    if first_boundary <= t < second_boundary:
      template = templates[1]
    elif t >= second_boundary:
      template = templates[2]

    for subject in range(n_subjects):
      clean = _subject_sample_from_template(template, rng)
      truth[t, :, :, subject] = clean

      sparse_noise = np.zeros((n_nodes, n_nodes), dtype=np.float32)
      mask = rng.random((n_nodes, n_nodes)) < noise_sparsity
      amplitudes = rng.beta(4, 2, size=(n_nodes, n_nodes)).astype(np.float32)
      sparse_noise[mask] = amplitudes[mask] * rng.choice([-1.0, 1.0], size=np.count_nonzero(mask))
      sparse_noise = 0.5 * (sparse_noise + sparse_noise.T)
      np.fill_diagonal(sparse_noise, 0.0)

      observed_sample = np.clip(clean + sparse_noise, 0.0, 1.0)
      observed_sample = 0.5 * (observed_sample + observed_sample.T)
      np.fill_diagonal(observed_sample, 0.0)
      observed[t, :, :, subject] = observed_sample.astype(np.float32)

  return observed, truth, interval_truth


def interval_mse(estimate: np.ndarray, truth: np.ndarray, intervals: np.ndarray) -> list[float]:
  mses: list[float] = []
  for start, end in intervals:
    segment_estimate = estimate[start : end + 1]
    segment_truth = truth[start : end + 1]
    mses.append(float(np.mean((segment_estimate - segment_truth) ** 2)))
  return mses


def run_synthetic_benchmark(output_dir: Path, simulations: int = 20) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  algorithms = {
    "ho_rlsl": HORLSLRunner,
    "hosvd": HOSVDRunner,
  }
  noise_levels = [0.1, 0.2, 0.3, 0.4]
  network_types = ["modular", "hierarchical", "overlapping"]
  rows: list[dict[str, object]] = []

  for network_type in network_types:
    for noise_level in noise_levels:
      metric_accumulator: dict[str, list[list[float]]] = {name: [] for name in algorithms}

      for simulation in range(simulations):
        observed, truth, truth_intervals = generate_synthetic_tensor(
          network_type=network_type,
          noise_sparsity=noise_level,
          seed=simulation,
        )
        run_config = DecompositionConfig(
          train_steps=5,
          alpha=5,
          sigma_min=None,
          sigma_scale=0.1,
          lambda_sparse=0.05,
          symmetric_modes=True,
          name=f"synthetic_{network_type}_{noise_level}",
        )

        for algorithm_name, runner_cls in algorithms.items():
          runner = runner_cls(replace(run_config))
          result = runner.run(observed)
          metric_accumulator[algorithm_name].append(interval_mse(result.lowrank_stream, truth, truth_intervals))

      for algorithm_name, simulation_metrics in metric_accumulator.items():
        mean_metrics = np.mean(np.asarray(simulation_metrics, dtype=np.float32), axis=0)
        row = {
          "network_type": network_type,
          "noise_level": noise_level,
          "algorithm": algorithm_name,
          "interval_1_mse": float(mean_metrics[0]),
          "interval_2_mse": float(mean_metrics[1]),
          "interval_3_mse": float(mean_metrics[2]),
        }
        rows.append(row)

  csv_path = output_dir / "synthetic_benchmark_summary.csv"
  with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def _rows_for_eeg_result(
  condition: str,
  algorithm: str,
  result: DecompositionResult,
  timing: EEGTiming,
) -> list[dict[str, Any]]:
  rows = []
  for interval_index, (start, end) in enumerate(result.intervals.tolist(), start=1):
    rows.append({
      "condition": condition,
      "algorithm": algorithm,
      "interval_index": interval_index,
      "start_idx": int(start),
      "end_idx": int(end),
      "start_s": float(timing.index_to_s(start)),
      "end_s": float(timing.index_to_s(end)),
      "start_ms": float(timing.index_to_ms(start)),
      "end_ms": float(timing.index_to_ms(end)),
      "change_point_count": len(result.change_points),
      "timing_reference": timing.zero_reference,
    })
  return rows


def _write_eeg_interval_csv(output_dir: Path, rows: list[dict[str, object]]) -> None:
  csv_path = output_dir / "eeg_interval_summary.csv"
  with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def _plot_eeg_condition(output_dir: Path, condition: str, results: dict[str, object], time_axis: np.ndarray) -> None:
  figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
  for algorithm_name, result in results.items():
    axes[0].plot(time_axis, result.residual_energy, label=f"{algorithm_name} residual")
    axes[1].plot(time_axis, result.sparse_mass, label=f"{algorithm_name} sparse mass")
    for cp in result.change_points.tolist():
      cp_time = float(time_axis[cp])
      axes[0].axvline(cp_time, color="gray", linestyle="--", alpha=0.25)
      axes[1].axvline(cp_time, color="gray", linestyle="--", alpha=0.25)

  axes[0].set_title(f"{condition} residual energy")
  axes[0].legend()
  axes[1].set_title(f"{condition} sparse mass")
  axes[1].legend()
  axes[1].set_xlabel("Time (s)")
  figure.tight_layout()
  figure.savefig(output_dir / f"eeg_{condition}_benchmark.png", dpi=150)
  plt.close(figure)


def run_eeg_benchmark(output_dir: Path) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  algorithms = {
    "ho_rlsl": HORLSLRunner,
    "hosvd": HOSVDRunner,
  }
  rows: list[dict[str, object]] = []

  for condition, tensor_path in {
    "correct": config.tensor_correct_file,
    "incorrect": config.tensor_incorrect_file,
  }.items():
    subject_tensor = np.load(tensor_path).astype(np.float32)
    stream = convert_subject_tensor_to_stream(subject_tensor)
    timing = eeg_timing(stream.shape[0])
    time_axis = timing.time_s

    run_config = DecompositionConfig(
      train_steps=10,
      alpha=8,
      sigma_min=0.11,
      lambda_sparse=0.05,
      symmetric_modes=True,
      name=f"eeg_{condition}",
    )

    condition_results = {}
    for algorithm_name, runner_cls in algorithms.items():
      algorithm_config = replace(run_config, lambda_sparse=0.0 if algorithm_name == "hosvd" else 0.05)
      result = runner_cls(algorithm_config).run(stream)
      condition_results[algorithm_name] = result
      rows.extend(_rows_for_eeg_result(condition, algorithm_name, result, timing))

    _plot_eeg_condition(output_dir, condition, condition_results, time_axis)

  _write_eeg_interval_csv(output_dir, rows)


def main() -> None:
  parser = argparse.ArgumentParser(description="Run HoSVD vs HO-RLSL benchmark suite.")
  parser.add_argument("--dataset", choices=["synthetic", "eeg", "all"], default="all")
  parser.add_argument("--output-dir", default=str(config.paths.OUTPUTS_DIR / "benchmark"))
  parser.add_argument("--synthetic-simulations", type=int, default=20)
  args = parser.parse_args()

  output_dir = Path(args.output_dir)
  if args.dataset in {"synthetic", "all"}:
    run_synthetic_benchmark(output_dir / "synthetic", simulations=args.synthetic_simulations)
    print(f"Wrote synthetic benchmark outputs to {output_dir / 'synthetic'}")
  if args.dataset in {"eeg", "all"}:
    run_eeg_benchmark(output_dir / "eeg")
    print(f"Wrote EEG benchmark outputs to {output_dir / 'eeg'}")


if __name__ == "__main__":
  main()
