from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from Ho_RLSL import (
    HoRLSL,
    HoRLSLConfig,
    mode_dot,
    paper_time_last_to_subject_first,
    parse_optional_int_tuple,
    subject_first_to_paper_time_last,
    unfold,
)


@dataclass
class SyntheticCase:
    observed: np.ndarray
    low_rank: np.ndarray
    change_points: tuple[int, ...]
    model: str
    noise_sparsity: float


def contiguous_modules(n_nodes: int, n_modules: int) -> list[set[int]]:
    indices = np.array_split(np.arange(n_nodes), n_modules)
    return [set(int(i) for i in group) for group in indices]


def module_adjacency(
    n_nodes: int,
    modules: list[set[int]],
    intra_weight: float = 0.7,
    inter_weight: float = 0.1,
) -> np.ndarray:
    adjacency = np.full((n_nodes, n_nodes), inter_weight, dtype=float)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if any(i in module and j in module for module in modules):
                adjacency[i, j] = intra_weight
                adjacency[j, i] = intra_weight
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def hierarchical_adjacency(n_nodes: int, interval: int) -> np.ndarray:
    big_modules = contiguous_modules(n_nodes, 2)
    small_modules = contiguous_modules(n_nodes, 4)
    adjacency = np.full((n_nodes, n_nodes), 0.08, dtype=float)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            same_big = any(i in module and j in module for module in big_modules)
            same_small = any(i in module and j in module for module in small_modules)
            if same_small:
                weight = 0.75 if interval == 1 else 0.68
            elif same_big:
                weight = 0.45 if interval == 1 else 0.62
            else:
                weight = 0.08
            adjacency[i, j] = weight
            adjacency[j, i] = weight

    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def overlapping_modules(n_nodes: int) -> list[set[int]]:
    modules = contiguous_modules(n_nodes, 2)
    overlap_start = n_nodes // 4
    overlap_end = n_nodes // 2
    overlap_nodes = set(range(overlap_start, overlap_end))
    modules[0] = modules[0] | overlap_nodes
    modules[1] = modules[1] | overlap_nodes
    return modules


def base_network(n_nodes: int, model: str, interval: int) -> np.ndarray:
    if model == "modular":
        modules = contiguous_modules(n_nodes, 2 if interval != 1 else 4)
        return module_adjacency(n_nodes, modules)
    if model == "hierarchical":
        return hierarchical_adjacency(n_nodes, interval)
    if model == "overlap":
        modules = contiguous_modules(n_nodes, 2) if interval != 1 else overlapping_modules(n_nodes)
        return module_adjacency(n_nodes, modules)
    raise ValueError(f"Unknown synthetic model: {model}")


def add_sparse_noise(
    low_rank: np.ndarray,
    sparsity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    observed = low_rank.copy()
    n_subjects, n_nodes, _, n_times = observed.shape
    upper = np.triu_indices(n_nodes, k=1)
    n_edges = len(upper[0])
    n_noisy_edges = max(1, int(round(sparsity * n_edges))) if sparsity > 0 else 0

    for subject in range(n_subjects):
        for time_idx in range(n_times):
            if n_noisy_edges == 0:
                continue
            selected = rng.choice(n_edges, size=n_noisy_edges, replace=False)
            values = rng.beta(4.0, 2.0, size=n_noisy_edges)
            rows = upper[0][selected]
            cols = upper[1][selected]
            observed[subject, rows, cols, time_idx] += values
            observed[subject, cols, rows, time_idx] += values

    return observed


def generate_synthetic_case(
    model: str,
    n_subjects: int,
    n_nodes: int,
    n_times: int,
    noise_sparsity: float,
    rng: np.random.Generator,
) -> SyntheticCase:
    change_points = (n_times // 3, 2 * n_times // 3)
    subject_scale = rng.normal(1.0, 0.04, size=n_subjects)
    low_rank = np.zeros((n_subjects, n_nodes, n_nodes, n_times), dtype=float)

    for time_idx in range(n_times):
        interval = int(time_idx >= change_points[0]) + int(time_idx >= change_points[1])
        adjacency = base_network(n_nodes, model, interval)
        for subject in range(n_subjects):
            low_rank[subject, :, :, time_idx] = subject_scale[subject] * adjacency

    observed = add_sparse_noise(low_rank, noise_sparsity, rng)
    return SyntheticCase(
        observed=observed,
        low_rank=low_rank,
        change_points=change_points,
        model=model,
        noise_sparsity=noise_sparsity,
    )


def hosvd_per_time_baseline(tensor: np.ndarray, ranks: tuple[int, int, int]) -> np.ndarray:
    sequence = subject_first_to_paper_time_last(tensor)
    reconstructed = np.zeros_like(sequence, dtype=float)

    for time_idx in range(sequence.shape[-1]):
        sample = sequence[..., time_idx]
        factors = []
        for mode, rank in enumerate(ranks):
            data = unfold(sample, mode)
            u, _, _ = np.linalg.svd(data, full_matrices=False)
            factors.append(u[:, : min(rank, u.shape[1])])

        core = sample
        for mode, factor in enumerate(factors):
            core = mode_dot(core, factor.T, mode)

        estimate = core
        for mode, factor in enumerate(factors):
            estimate = mode_dot(estimate, factor, mode)

        reconstructed[..., time_idx] = estimate

    return paper_time_last_to_subject_first(reconstructed)


def mse_by_interval(
    truth: np.ndarray,
    estimate: np.ndarray,
    change_points: tuple[int, ...],
) -> tuple[float, ...]:
    boundaries = (0, *change_points, truth.shape[-1])
    errors = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        errors.append(float(np.mean((truth[..., start:end] - estimate[..., start:end]) ** 2)))
    return tuple(errors)


def nearest_change_point_errors(detected: list[int], truth: tuple[int, ...]) -> tuple[int | None, ...]:
    if not detected:
        return tuple(None for _ in truth)
    return tuple(min(abs(int(point) - int(candidate)) for candidate in detected) for point in truth)


def evaluate_case(
    case: SyntheticCase,
    config: HoRLSLConfig,
    ranks: tuple[int, int, int],
) -> dict[str, object]:
    start = time.perf_counter()
    result = HoRLSL(config).fit_transform_subject_first(case.observed)
    ho_rlsl_seconds = time.perf_counter() - start

    start = time.perf_counter()
    hosvd_estimate = hosvd_per_time_baseline(case.observed, ranks)
    hosvd_seconds = time.perf_counter() - start

    ho_rlsl_mse = mse_by_interval(case.low_rank, result.low_rank, case.change_points)
    hosvd_mse = mse_by_interval(case.low_rank, hosvd_estimate, case.change_points)

    return {
        "model": case.model,
        "noise": case.noise_sparsity,
        "truth_change_points": case.change_points,
        "detected_change_points": tuple(result.change_points),
        "change_point_errors": nearest_change_point_errors(result.change_points, case.change_points),
        "ho_rlsl_mse": ho_rlsl_mse,
        "hosvd_mse": hosvd_mse,
        "ho_rlsl_seconds": ho_rlsl_seconds,
        "hosvd_seconds": hosvd_seconds,
        "final_ranks": tuple(base.shape[1] for base in result.final_bases),
        "reconstruction_error": float(np.max(np.abs(result.low_rank + result.sparse - case.observed))),
    }


def format_tuple(values: tuple[object, ...], precision: int = 5) -> str:
    formatted = []
    for value in values:
        if value is None:
            formatted.append("None")
        elif isinstance(value, float):
            formatted.append(f"{value:.{precision}f}")
        else:
            formatted.append(str(value))
    return "(" + ", ".join(formatted) + ")"


def print_results(results: list[dict[str, object]]) -> None:
    header = (
        "model        noise  cp_true       cp_detected   cp_error      "
        "Ho-RLSL MSE intervals        HoSVD MSE intervals          seconds"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['model']:<12} "
            f"{row['noise']:<5.2f} "
            f"{format_tuple(row['truth_change_points'], 0):<13} "
            f"{format_tuple(row['detected_change_points'], 0):<13} "
            f"{format_tuple(row['change_point_errors'], 0):<13} "
            f"{format_tuple(row['ho_rlsl_mse']):<27} "
            f"{format_tuple(row['hosvd_mse']):<27} "
            f"{row['ho_rlsl_seconds']:.2f}/{row['hosvd_seconds']:.2f}"
        )


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_model_list(value: str) -> list[str]:
    models = [item.strip() for item in value.split(",") if item.strip()]
    valid = {"modular", "hierarchical", "overlap"}
    unknown = sorted(set(models) - valid)
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic Ho-RLSL evaluation inspired by Ozdemir 2017.")
    parser.add_argument("--models", default="modular,hierarchical,overlap")
    parser.add_argument("--noise-levels", default="0.1,0.3")
    parser.add_argument("--subjects", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--times", type=int, default=36)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-length", type=int, default=6)
    parser.add_argument("--alpha", type=int, default=6)
    parser.add_argument("--sigma-min", type=float, default=0.5)
    parser.add_argument("--max-ranks", default="8,8,4")
    parser.add_argument("--sparse-solver", choices=["fista", "proxy"], default="fista")
    parser.add_argument("--fista-max-iter", type=int, default=15)
    parser.add_argument("--check", action="store_true", help="Fail if smoke-test invariants are violated.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    models = parse_model_list(args.models)
    noise_levels = parse_float_list(args.noise_levels)
    max_ranks = parse_optional_int_tuple(args.max_ranks)
    if max_ranks is None or any(rank is None for rank in max_ranks):
        raise ValueError("--max-ranks must provide three integer ranks for this synthetic test")
    ranks = tuple(int(rank) for rank in max_ranks)

    config = HoRLSLConfig(
        train_length=args.train_length,
        alpha=args.alpha,
        sigma_min=args.sigma_min,
        max_ranks=ranks,
        sparse_solver=args.sparse_solver,
        fista_max_iter=args.fista_max_iter,
        tie_symmetric_modes=(0, 1),
    )

    results = []
    for model in models:
        for noise in noise_levels:
            case = generate_synthetic_case(
                model=model,
                n_subjects=args.subjects,
                n_nodes=args.nodes,
                n_times=args.times,
                noise_sparsity=noise,
                rng=rng,
            )
            results.append(evaluate_case(case, config, ranks))

    print_results(results)

    if args.check:
        for row in results:
            if not np.isfinite(row["reconstruction_error"]):
                raise AssertionError(f"Non-finite reconstruction error for {row['model']}")
            if row["reconstruction_error"] > 1e-8:
                raise AssertionError(f"Reconstruction error too high for {row['model']}: {row['reconstruction_error']}")
            if not all(np.isfinite(value) for value in row["ho_rlsl_mse"]):
                raise AssertionError(f"Non-finite Ho-RLSL MSE for {row['model']}")


if __name__ == "__main__":
    main()
