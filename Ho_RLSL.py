from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


SUBJECT_FIRST_LAYOUT = "subjects,nodes,nodes,time"
PAPER_TIME_LAST_LAYOUT = "nodes,nodes,subjects,time"


def validate_subject_first_tensor(tensor: np.ndarray) -> np.ndarray:
    tensor = np.asarray(tensor)
    if tensor.ndim != 4:
        raise ValueError(f"Expected tensor shape (subjects, nodes, nodes, time), got {tensor.shape}")
    if tensor.shape[1] != tensor.shape[2]:
        raise ValueError(
            "Expected square node-node connectivity matrices in axes 1 and 2, "
            f"got shape {tensor.shape}"
        )
    return tensor


def validate_paper_time_last_tensor(sequence: np.ndarray) -> np.ndarray:
    sequence = np.asarray(sequence)
    if sequence.ndim != 4:
        raise ValueError(f"Expected tensor shape (nodes, nodes, subjects, time), got {sequence.shape}")
    if sequence.shape[0] != sequence.shape[1]:
        raise ValueError(
            "Expected square node-node connectivity matrices in axes 0 and 1, "
            f"got shape {sequence.shape}"
        )
    return sequence


def subject_first_to_paper_time_last(tensor: np.ndarray) -> np.ndarray:
    """Convert `(subjects, nodes, nodes, time)` to `(nodes, nodes, subjects, time)`."""
    tensor = validate_subject_first_tensor(tensor)
    return np.transpose(tensor, (1, 2, 0, 3))


def paper_time_last_to_subject_first(sequence: np.ndarray) -> np.ndarray:
    """Convert `(nodes, nodes, subjects, time)` to `(subjects, nodes, nodes, time)`."""
    sequence = validate_paper_time_last_tensor(sequence)
    return np.transpose(sequence, (2, 0, 1, 3))


def unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    """Return mode-n unfolding with mode fibers as columns."""
    tensor = np.asarray(tensor)
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def mode_dot(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    """Compute tensor x_mode matrix."""
    result = np.tensordot(matrix, tensor, axes=(1, mode))
    return np.moveaxis(result, 0, mode)


def multi_mode_dot(
    tensor: np.ndarray,
    matrices: Iterable[np.ndarray],
    modes: Iterable[int] | None = None,
) -> np.ndarray:
    result = np.asarray(tensor)
    matrices = tuple(matrices)
    if modes is None:
        modes = range(len(matrices))
    for matrix, mode in zip(matrices, modes):
        result = mode_dot(result, matrix, int(mode))
    return result


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def orthonormalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.reshape(matrix.shape[0], 0)
    q, r = np.linalg.qr(matrix)
    keep = np.abs(np.diag(r)) > np.finfo(float).eps
    return q[:, keep]


def _empty_basis(n_rows: int) -> np.ndarray:
    return np.zeros((n_rows, 0), dtype=float)


def subspace_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Projection Frobenius distance normalized to roughly [0, sqrt(2)]."""
    if left.shape[0] != right.shape[0]:
        raise ValueError("Subspaces must have the same ambient dimension")
    if left.shape[1] == 0 and right.shape[1] == 0:
        return 0.0
    left_projection = left @ left.T if left.shape[1] else np.zeros((left.shape[0], left.shape[0]))
    right_projection = right @ right.T if right.shape[1] else np.zeros((right.shape[0], right.shape[0]))
    scale = np.sqrt(max(left.shape[1], right.shape[1], 1))
    return float(np.linalg.norm(left_projection - right_projection, ord="fro") / scale)


def parse_optional_int_tuple(value: str | None) -> tuple[int | None, ...] | None:
    if value is None or value.strip() == "":
        return None
    entries: list[int | None] = []
    for item in value.split(","):
        item = item.strip().lower()
        if item in {"none", "null", "-"}:
            entries.append(None)
        else:
            entries.append(int(item))
    return tuple(entries)


def parse_int_pair(value: str | None) -> tuple[int, int] | None:
    if value is None or value.strip().lower() in {"none", "null", "-"}:
        return None
    parts = [int(item.strip()) for item in value.split(",")]
    if len(parts) != 2:
        raise ValueError("Expected an integer pair like '0,1'")
    return parts[0], parts[1]


@dataclass
class HoRLSLConfig:
    """Configuration for Higher-Order Recursive Low-Rank + Sparse Learning.

    This implementation follows the Ho-RLSL structure from Ozdemir et al.:
    initialize Tucker-mode subspaces, remove sparse outliers, recursively
    update subspace directions, and mark updates as change points.

    The paper recovers the sparse tensor from the orthogonally projected
    measurement with GTCS-S. This module uses a tensor FISTA l1 solver by
    default, with a proxy thresholding option for quick experiments.
    """

    train_length: int = 10
    alpha: int = 8
    sigma_min: float = 0.1
    init_method: str = "hosvd"
    threshold_mode: str = "relative"
    normalize_unfoldings: bool = True
    sparse_solver: str = "fista"
    lambda_sparse: float | None = None
    epsilon: float | None = None
    epsilon_mode: str = "absolute"
    fista_max_iter: int = 50
    fista_tol: float = 1e-4
    fista_step_size: float = 1.0
    sparse_threshold: float = 3.0
    sparse_threshold_mode: str = "mad"
    enforce_symmetric_connectivity: bool = True
    zero_diagonal: bool = True
    max_ranks: tuple[int | None, ...] | None = None
    tie_symmetric_modes: tuple[int, int] | None = (0, 1)
    use_paper_change_rule: bool = True
    subspace_change_threshold: float = 0.15
    store_rank_history: bool = True


@dataclass
class HoRLSLResult:
    low_rank: np.ndarray
    sparse: np.ndarray
    change_points: list[int]
    final_bases: list[np.ndarray]
    rank_history: list[tuple[int, tuple[int, ...]]]
    update_times: list[int]
    change_history: list[dict[str, int]]
    input_layout: str
    output_layout: str
    internal_layout: str
    original_shape: tuple[int, ...]
    internal_shape: tuple[int, ...]
    initial_ranks: tuple[int, ...]
    initial_thresholds: tuple[float, ...]
    config: dict[str, object]


class HoRLSL:
    """Higher-order recursive low-rank + sparse tracker for time-last tensors.

    The paper-like internal layout is `(nodes, nodes, subjects, time)`. For
    this repo's saved connectivity tensors, use `fit_transform_subject_first`,
    because those arrays are `(subjects, nodes, nodes, time)`.
    """

    def __init__(self, config: HoRLSLConfig | None = None):
        self.config = config or HoRLSLConfig()
        self.bases_: list[np.ndarray] | None = None
        self.thresholds_: list[float] | None = None
        self.initial_core_: np.ndarray | None = None
        self.initial_full_factors_: list[np.ndarray] | None = None
        self.initial_time_basis_: np.ndarray | None = None
        self.initial_singular_values_: list[np.ndarray] | None = None
        self.initial_ranks_: tuple[int, ...] | None = None

    def fit_transform(self, sequence: np.ndarray) -> HoRLSLResult:
        sequence = validate_paper_time_last_tensor(sequence).astype(float, copy=False)

        n_modes = sequence.ndim - 1
        n_times = sequence.shape[-1]
        if n_times <= self.config.train_length:
            raise ValueError("train_length must be smaller than the number of time samples")

        self._validate_config(n_modes)
        train = sequence[..., : self.config.train_length]
        self.bases_, self.thresholds_ = self._initial_subspaces(train)
        self._apply_tied_modes()
        self.initial_ranks_ = tuple(base.shape[1] for base in self.bases_)

        low_rank = np.zeros_like(sequence, dtype=float)
        sparse = np.zeros_like(sequence, dtype=float)
        rank_history: list[tuple[int, tuple[int, ...]]] = []
        change_points: list[int] = []
        update_times: list[int] = []
        change_history: list[dict[str, int]] = []
        update_buffer: list[np.ndarray] = []

        for t in range(n_times):
            m_t = sequence[..., t]
            l_hat, s_hat = self._separate(m_t)
            low_rank[..., t] = l_hat
            sparse[..., t] = s_hat

            if t < self.config.train_length:
                continue

            update_buffer.append(l_hat)
            if len(update_buffer) < self.config.alpha:
                continue

            changed, update_events = self._update_subspaces(update_buffer, update_time=t)
            self._apply_tied_modes()
            update_times.append(t)
            update_buffer = []
            change_history.extend(update_events)

            ranks = tuple(base.shape[1] for base in self.bases_)
            if self.config.store_rank_history:
                rank_history.append((t, ranks))
            if changed:
                change_points.append(t)

        return HoRLSLResult(
            low_rank=low_rank,
            sparse=sparse,
            change_points=change_points,
            final_bases=[basis.copy() for basis in self.bases_],
            rank_history=rank_history,
            update_times=update_times,
            change_history=change_history,
            input_layout=PAPER_TIME_LAST_LAYOUT,
            output_layout=PAPER_TIME_LAST_LAYOUT,
            internal_layout=PAPER_TIME_LAST_LAYOUT,
            original_shape=tuple(sequence.shape),
            internal_shape=tuple(sequence.shape),
            initial_ranks=tuple(int(rank) for rank in (self.initial_ranks_ or ())),
            initial_thresholds=tuple(float(value) for value in self.thresholds_),
            config=asdict(self.config),
        )

    def fit_transform_subject_first(self, tensor: np.ndarray) -> HoRLSLResult:
        """Run Ho-RLSL on `(subjects, nodes, nodes, time)` data.

        The internal sequence is reordered to `(nodes, nodes, subjects, time)`
        to match the paper's `N x N x subjects` tensor at each time point. The
        returned low-rank and sparse tensors are converted back to the original
        subject-first shape.
        """
        tensor = validate_subject_first_tensor(tensor)
        sequence = subject_first_to_paper_time_last(tensor)
        result = self.fit_transform(sequence)
        result.low_rank = np.transpose(result.low_rank, (2, 0, 1, 3))
        result.sparse = np.transpose(result.sparse, (2, 0, 1, 3))
        result.input_layout = SUBJECT_FIRST_LAYOUT
        result.output_layout = SUBJECT_FIRST_LAYOUT
        result.original_shape = tuple(tensor.shape)
        result.internal_shape = tuple(sequence.shape)
        return result

    def _validate_config(self, n_modes: int) -> None:
        if self.config.train_length < 1:
            raise ValueError("train_length must be positive")
        if self.config.alpha < 1:
            raise ValueError("alpha must be positive")
        if self.config.init_method not in {"hosvd"}:
            raise ValueError("init_method must be 'hosvd'")
        if self.config.threshold_mode not in {"relative", "absolute"}:
            raise ValueError("threshold_mode must be 'relative' or 'absolute'")
        if self.config.sparse_solver not in {"fista", "proxy"}:
            raise ValueError("sparse_solver must be 'fista' or 'proxy'")
        if self.config.lambda_sparse is not None and self.config.lambda_sparse < 0:
            raise ValueError("lambda_sparse must be non-negative")
        if self.config.epsilon is not None and self.config.epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if self.config.epsilon_mode not in {"absolute", "relative"}:
            raise ValueError("epsilon_mode must be 'absolute' or 'relative'")
        if self.config.fista_max_iter < 1:
            raise ValueError("fista_max_iter must be positive")
        if self.config.fista_tol < 0:
            raise ValueError("fista_tol must be non-negative")
        if self.config.fista_step_size <= 0:
            raise ValueError("fista_step_size must be positive")
        if self.config.sparse_threshold_mode not in {"mad", "relative", "absolute"}:
            raise ValueError("sparse_threshold_mode must be 'mad', 'relative', or 'absolute'")
        if self.config.max_ranks is not None and len(self.config.max_ranks) != n_modes:
            raise ValueError(f"max_ranks must contain {n_modes} entries")
        if self.config.tie_symmetric_modes is not None:
            source, target = self.config.tie_symmetric_modes
            if not (0 <= source < n_modes and 0 <= target < n_modes):
                raise ValueError(f"tie_symmetric_modes entries must be in [0, {n_modes - 1}]")

    def _initial_subspaces(self, train: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        if self.config.init_method == "hosvd":
            return self._initial_subspaces_hosvd(train)
        raise ValueError(f"Unsupported init_method: {self.config.init_method}")

    def _initial_subspaces_hosvd(self, train: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        """Initialize mode subspaces from the training tensor via HOSVD.

        The paper writes the training block as
        `M_train = C x_1 P0_1 x_2 P0_2 x_3 P0_3 x_4 P0_4`.
        HOSVD obtains each `P0_i` from the left singular vectors of the
        corresponding mode unfolding. Only the first three factors are used as
        online low-rank subspaces; the fourth factor is the training-time mode.
        """
        train = np.asarray(train, dtype=float)
        full_factors = []
        singular_values_by_mode = []

        for mode in range(train.ndim):
            data = unfold(train, mode)
            if self.config.normalize_unfoldings:
                data = data / np.sqrt(max(data.shape[1], 1))
            u, singular_values, _ = np.linalg.svd(data, full_matrices=False)
            full_factors.append(u)
            singular_values_by_mode.append(singular_values)

        self.initial_full_factors_ = full_factors
        self.initial_time_basis_ = full_factors[-1]
        self.initial_singular_values_ = singular_values_by_mode
        self.initial_core_ = multi_mode_dot(
            train,
            [factor.T for factor in full_factors],
            modes=range(train.ndim),
        )

        bases = []
        thresholds = []
        for mode in range(train.ndim - 1):
            u = full_factors[mode]
            singular_values = singular_values_by_mode[mode]
            threshold = self._threshold_from_values(singular_values)
            max_rank = self._max_rank(mode)
            keep = singular_values >= threshold
            if max_rank is not None:
                keep_indices = np.flatnonzero(keep)[:max_rank]
            else:
                keep_indices = np.flatnonzero(keep)
            if keep_indices.size == 0 and singular_values.size > 0:
                keep_indices = np.array([0])
            bases.append(orthonormalize(u[:, keep_indices]))
            thresholds.append(float(threshold))
        return bases, thresholds

    def _threshold_from_values(self, values: np.ndarray) -> float:
        if values.size == 0:
            return float(self.config.sigma_min)
        if self.config.threshold_mode == "relative":
            return float(self.config.sigma_min * values[0])
        return float(self.config.sigma_min)

    def _sparse_threshold(self, residual: np.ndarray) -> float:
        mode = self.config.sparse_threshold_mode
        value = float(self.config.sparse_threshold)
        abs_residual = np.abs(residual)

        if mode == "absolute":
            return value
        if mode == "relative":
            return value * float(abs_residual.max(initial=0.0))

        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        robust_sigma = 1.4826 * mad
        if robust_sigma <= np.finfo(float).eps:
            robust_sigma = float(abs_residual.std())
        return value * robust_sigma

    def _separate(self, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        projectors = self._orthogonal_projectors()
        projected_measurement = self._project_orthogonal(observed, projectors)
        sparse_hat = self._recover_sparse(projected_measurement, projectors)
        low_hat = observed - sparse_hat
        low_hat = self._sanitize_low_rank(low_hat)
        sparse_hat = observed - low_hat
        return low_hat, sparse_hat

    def _orthogonal_projectors(self) -> list[np.ndarray]:
        """Return phi_i = I - P_i P_i^T for each tracked mode."""
        if self.bases_ is None:
            raise RuntimeError("Subspaces have not been initialized")

        projectors = []
        for basis in self.bases_:
            identity = np.eye(basis.shape[0], dtype=float)
            if basis.shape[1] == 0:
                projectors.append(identity)
            else:
                projectors.append(identity - basis @ basis.T)
        return projectors

    def _project_orthogonal(self, tensor: np.ndarray, projectors: list[np.ndarray]) -> np.ndarray:
        """Compute Y_t = M_t x_1 phi_1 x_2 phi_2 x_3 phi_3."""
        return multi_mode_dot(tensor, projectors, modes=range(len(projectors)))

    def _backproject_sparse_proxy(
        self,
        projected_measurement: np.ndarray,
        projectors: list[np.ndarray],
    ) -> np.ndarray:
        """Apply A* to a projected tensor, where A(S)=S x_i phi_i."""
        adjoint_projectors = [projector.T for projector in projectors]
        return multi_mode_dot(
            projected_measurement,
            adjoint_projectors,
            modes=range(len(adjoint_projectors)),
        )

    def _recover_sparse(self, projected_measurement: np.ndarray, projectors: list[np.ndarray]) -> np.ndarray:
        if self.config.sparse_solver == "proxy":
            sparse_proxy = self._backproject_sparse_proxy(projected_measurement, projectors)
            return soft_threshold(sparse_proxy, self._sparse_threshold(sparse_proxy))
        return self._recover_sparse_fista(projected_measurement, projectors)

    def _lambda_sparse(self, projected_measurement: np.ndarray, projectors: list[np.ndarray]) -> float:
        if self.config.lambda_sparse is not None:
            return float(self.config.lambda_sparse)
        sparse_proxy = self._backproject_sparse_proxy(projected_measurement, projectors)
        return self._sparse_threshold(sparse_proxy)

    def _epsilon_value(self, projected_measurement: np.ndarray) -> float | None:
        if self.config.epsilon is None:
            return None
        epsilon = float(self.config.epsilon)
        if self.config.epsilon_mode == "relative":
            return epsilon * float(np.linalg.norm(projected_measurement))
        return epsilon

    def _recover_sparse_fista(
        self,
        projected_measurement: np.ndarray,
        projectors: list[np.ndarray],
    ) -> np.ndarray:
        """L1 sparse recovery proxy for GTCS-S using FISTA.

        Solves `0.5 * ||A(S) - Y||_F^2 + lambda * ||S||_1`, where
        `A(S) = S x_1 phi_1 x_2 phi_2 x_3 phi_3`.
        """
        sparse = np.zeros_like(projected_measurement, dtype=float)
        momentum = sparse.copy()
        t_value = 1.0
        step = float(self.config.fista_step_size)
        lambda_step = step * self._lambda_sparse(projected_measurement, projectors)
        epsilon = self._epsilon_value(projected_measurement)

        for _ in range(self.config.fista_max_iter):
            residual = self._project_orthogonal(momentum, projectors) - projected_measurement
            gradient = self._backproject_sparse_proxy(residual, projectors)
            next_sparse = soft_threshold(momentum - step * gradient, lambda_step)

            denominator = max(float(np.linalg.norm(sparse)), 1.0)
            relative_change = float(np.linalg.norm(next_sparse - sparse) / denominator)

            next_t_value = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_value * t_value))
            momentum = next_sparse + ((t_value - 1.0) / next_t_value) * (next_sparse - sparse)
            sparse = next_sparse
            t_value = next_t_value

            if epsilon is not None:
                constrained_residual = self._project_orthogonal(sparse, projectors) - projected_measurement
                if float(np.linalg.norm(constrained_residual)) <= epsilon:
                    break

            if relative_change <= self.config.fista_tol:
                break

        return sparse

    def _sanitize_low_rank(self, low_rank: np.ndarray) -> np.ndarray:
        low_rank = np.asarray(low_rank, dtype=float).copy()
        low_rank[~np.isfinite(low_rank)] = 0.0

        if self.config.enforce_symmetric_connectivity and low_rank.ndim >= 2 and low_rank.shape[0] == low_rank.shape[1]:
            low_rank = 0.5 * (low_rank + np.swapaxes(low_rank, 0, 1))

        if self.config.zero_diagonal and low_rank.ndim >= 2 and low_rank.shape[0] == low_rank.shape[1]:
            diagonal = np.arange(low_rank.shape[0])
            low_rank[diagonal, diagonal, ...] = 0.0

        return low_rank

    def _project_low_rank(self, tensor: np.ndarray) -> np.ndarray:
        if self.bases_ is None:
            raise RuntimeError("Subspaces have not been initialized")
        core = tensor
        for mode, basis in enumerate(self.bases_):
            core = mode_dot(core, basis.T, mode)
        result = core
        for mode, basis in enumerate(self.bases_):
            result = mode_dot(result, basis, mode)
        return result

    def _update_subspaces(
        self,
        tensors: list[np.ndarray],
        update_time: int,
    ) -> tuple[bool, list[dict[str, int]]]:
        if self.bases_ is None or self.thresholds_ is None:
            raise RuntimeError("Subspaces have not been initialized")

        previous_bases = [basis.copy() for basis in self.bases_]
        changed_by_direction_update = False
        update_events: list[dict[str, int]] = []
        for mode in range(len(self.bases_)):
            if self._is_tied_target(mode):
                continue

            data = np.concatenate([unfold(tensor, mode) for tensor in tensors], axis=1)
            basis = self.bases_[mode]
            threshold = self.thresholds_[mode]
            old_rank = basis.shape[1]

            basis, deleted_count = self._delete_directions(data, basis, threshold)
            basis, added_count = self._add_directions(data, basis, threshold, mode)

            self.bases_[mode] = basis
            changed = deleted_count > 0 or added_count > 0
            changed_by_direction_update = changed_by_direction_update or changed
            update_events.append(
                {
                    "time": int(update_time),
                    "mode": int(mode),
                    "old_rank": int(old_rank),
                    "new_rank": int(basis.shape[1]),
                    "deleted": int(deleted_count),
                    "added": int(added_count),
                    "changed": int(changed),
                }
            )

        self._apply_tied_modes()
        tied = self.config.tie_symmetric_modes
        if tied is not None:
            source, target = tied
            if source < len(self.bases_) and target < len(self.bases_):
                target_changed = subspace_distance(previous_bases[target], self.bases_[target]) > 0.0
                source_changed = any(event["mode"] == source and event["changed"] for event in update_events)
                if target_changed and source_changed:
                    update_events.append(
                        {
                            "time": int(update_time),
                            "mode": int(target),
                            "old_rank": int(previous_bases[target].shape[1]),
                            "new_rank": int(self.bases_[target].shape[1]),
                            "deleted": 0,
                            "added": 0,
                            "changed": 1,
                            "tied_to": int(source),
                        }
                    )
        if self.config.use_paper_change_rule:
            return changed_by_direction_update, update_events

        distance = max(
            subspace_distance(old_basis, new_basis)
            for old_basis, new_basis in zip(previous_bases, self.bases_)
        )
        changed_by_distance = distance >= self.config.subspace_change_threshold
        if update_events:
            update_events[-1]["subspace_distance_scaled"] = int(round(distance * 1_000_000))
        return changed_by_distance, update_events

    def _delete_directions(
        self,
        data: np.ndarray,
        basis: np.ndarray,
        threshold: float,
    ) -> tuple[np.ndarray, int]:
        if basis.shape[1] == 0:
            return basis, 0

        coefficients = basis.T @ data
        eigenvalues = np.mean(coefficients * coefficients, axis=1)
        keep = eigenvalues >= threshold * threshold
        if not np.any(keep):
            keep[np.argmax(eigenvalues)] = True

        deleted_count = int(np.count_nonzero(~keep))
        return orthonormalize(basis[:, keep]), deleted_count

    def _add_directions(
        self,
        data: np.ndarray,
        basis: np.ndarray,
        threshold: float,
        mode: int,
    ) -> tuple[np.ndarray, int]:
        if basis.shape[1] > 0:
            projected = data - basis @ (basis.T @ data)
        else:
            projected = data

        projected = projected / np.sqrt(max(projected.shape[1], 1))
        u, singular_values, _ = np.linalg.svd(projected, full_matrices=False)
        eigenvalues = singular_values * singular_values
        add_indices = np.flatnonzero(eigenvalues >= threshold * threshold)
        if add_indices.size == 0:
            return basis, 0

        max_rank = self._max_rank(mode)
        if max_rank is not None:
            remaining = max(max_rank - basis.shape[1], 0)
            add_indices = add_indices[:remaining]
        if add_indices.size == 0:
            return basis, 0

        updated = np.column_stack([basis, u[:, add_indices]])
        return orthonormalize(updated), int(add_indices.size)

    def _max_rank(self, mode: int) -> int | None:
        if self.config.max_ranks is None:
            return None
        return self.config.max_ranks[mode]

    def _is_tied_target(self, mode: int) -> bool:
        tied = self.config.tie_symmetric_modes
        return tied is not None and mode == tied[1]

    def _apply_tied_modes(self) -> None:
        tied = self.config.tie_symmetric_modes
        if tied is None or self.bases_ is None:
            return
        source, target = tied
        if source >= len(self.bases_) or target >= len(self.bases_):
            return
        if self.bases_[source].shape[0] != self.bases_[target].shape[0]:
            return
        self.bases_[target] = self.bases_[source].copy()


def fit_transform_subject_first(
    tensor: np.ndarray,
    config: HoRLSLConfig | None = None,
) -> HoRLSLResult:
    return HoRLSL(config).fit_transform_subject_first(tensor)


def fit_transform_connectivity(
    tensor: np.ndarray,
    config: HoRLSLConfig | None = None,
) -> HoRLSLResult:
    """Alias for repo-native `(subjects, nodes, nodes, time)` connectivity tensors."""
    return fit_transform_subject_first(tensor, config)


def fit_transform_time_last(
    sequence: np.ndarray,
    config: HoRLSLConfig | None = None,
) -> HoRLSLResult:
    return HoRLSL(config).fit_transform(sequence)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run Ho-RLSL on a time-last connectivity tensor.")
    parser.add_argument("input", type=Path, help="Input .npy tensor")
    parser.add_argument("--output", type=Path, default=Path("ho_rlsl_results.npz"))
    parser.add_argument(
        "--layout",
        choices=["subject-first", "paper"],
        default="subject-first",
        help=(
            "subject-first means (subjects, nodes, nodes, time); "
            "paper means (nodes, nodes, subjects, time)"
        ),
    )
    parser.add_argument("--train-length", type=int, default=10)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--sigma-min", type=float, default=0.1)
    parser.add_argument("--init-method", choices=["hosvd"], default="hosvd")
    parser.add_argument("--threshold-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument(
        "--raw-hosvd-singular-values",
        action="store_true",
        help="Use raw HOSVD singular values instead of covariance-scaled values for thresholding.",
    )
    parser.add_argument("--sparse-threshold", type=float, default=3.0)
    parser.add_argument("--sparse-threshold-mode", choices=["mad", "relative", "absolute"], default="mad")
    parser.add_argument("--sparse-solver", choices=["fista", "proxy"], default="fista")
    parser.add_argument("--lambda-sparse", type=float, default=None)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Optional residual constraint for sparse recovery: ||A(S)-Y||_F <= epsilon.",
    )
    parser.add_argument("--epsilon-mode", choices=["absolute", "relative"], default="absolute")
    parser.add_argument("--fista-max-iter", type=int, default=50)
    parser.add_argument("--fista-tol", type=float, default=1e-4)
    parser.add_argument("--fista-step-size", type=float, default=1.0)
    parser.add_argument(
        "--max-ranks",
        default=None,
        help="Optional comma-separated ranks for modes 0,1,2, e.g. '10,10,10' or '10,10,None'.",
    )
    parser.add_argument(
        "--tie-symmetric-modes",
        default="0,1",
        help="Mode pair to tie for symmetric connectivity, e.g. '0,1', or 'None' to disable.",
    )
    parser.add_argument(
        "--no-symmetric-connectivity",
        action="store_true",
        help="Do not symmetrize low-rank node-node connectivity estimates.",
    )
    parser.add_argument(
        "--keep-diagonal",
        action="store_true",
        help="Do not force low-rank node-node connectivity diagonals to zero.",
    )
    parser.add_argument(
        "--subspace-distance-change-rule",
        action="store_true",
        help="Use the fallback subspace-distance change rule instead of the paper add/delete rule.",
    )
    args = parser.parse_args()

    x = np.load(args.input)
    max_ranks = parse_optional_int_tuple(args.max_ranks)
    tie_symmetric_modes = parse_int_pair(args.tie_symmetric_modes)
    cfg = HoRLSLConfig(
        train_length=args.train_length,
        alpha=args.alpha,
        sigma_min=args.sigma_min,
        init_method=args.init_method,
        threshold_mode=args.threshold_mode,
        normalize_unfoldings=not args.raw_hosvd_singular_values,
        sparse_solver=args.sparse_solver,
        lambda_sparse=args.lambda_sparse,
        epsilon=args.epsilon,
        epsilon_mode=args.epsilon_mode,
        fista_max_iter=args.fista_max_iter,
        fista_tol=args.fista_tol,
        fista_step_size=args.fista_step_size,
        sparse_threshold=args.sparse_threshold,
        sparse_threshold_mode=args.sparse_threshold_mode,
        enforce_symmetric_connectivity=not args.no_symmetric_connectivity,
        zero_diagonal=not args.keep_diagonal,
        max_ranks=max_ranks,
        tie_symmetric_modes=tie_symmetric_modes,
        use_paper_change_rule=not args.subspace_distance_change_rule,
    )
    if args.layout == "subject-first":
        out = fit_transform_subject_first(x, cfg)
    else:
        out = fit_transform_time_last(x, cfg)
    np.savez_compressed(
        args.output,
        low_rank=out.low_rank.astype(np.float32),
        sparse=out.sparse.astype(np.float32),
        change_points=np.asarray(out.change_points, dtype=int),
        update_times=np.asarray(out.update_times, dtype=int),
        rank_history=np.asarray(out.rank_history, dtype=object),
        change_history=np.asarray(out.change_history, dtype=object),
        input_layout=np.asarray(out.input_layout),
        output_layout=np.asarray(out.output_layout),
        internal_layout=np.asarray(out.internal_layout),
        original_shape=np.asarray(out.original_shape, dtype=int),
        internal_shape=np.asarray(out.internal_shape, dtype=int),
        initial_ranks=np.asarray(out.initial_ranks, dtype=int),
        initial_thresholds=np.asarray(out.initial_thresholds, dtype=float),
        config=np.asarray(out.config, dtype=object),
    )
    print(f"Saved {args.output}")
    print(f"Detected change points: {out.change_points}")
