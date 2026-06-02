from pathlib import Path
import argparse
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

try:
    import tensorly as tl
    from tensorly.decomposition import tucker

    HAS_TENSORLY = True
except ImportError:
    tl = None
    tucker = None
    HAS_TENSORLY = False

try:
    from Ho_RLSL import HoRLSL, HoRLSLConfig, parse_optional_int_tuple

    HAS_HO_RLSL = True
except ImportError:
    HoRLSL = None
    HoRLSLConfig = None
    parse_optional_int_tuple = None
    HAS_HO_RLSL = False

try:
    import mne

    HAS_MNE = True
except ImportError:
    mne = None
    HAS_MNE = False

try:
    import ruptures as rpt

    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False

try:
    import networkx as nx

    if not hasattr(nx, "from_numpy_matrix") and hasattr(nx, "from_numpy_array"):
        nx.from_numpy_matrix = nx.from_numpy_array

    from dyconnmap.graphs import nodal_global_efficiency
    from dyconnmap.graphs import threshold_omst_global_cost_efficiency

    HAS_DYCONNMAP = True
except ImportError:
    HAS_DYCONNMAP = False


if HAS_TENSORLY:
    tl.set_backend("numpy")

INPUT_FILE = Path("tensor_4d/tensor_incorrect_4d.npy")
CONDITION_FILES = {
    "incorrect": Path("tensor_4d/tensor_incorrect_4d.npy"),
    "correct": Path("tensor_4d/tensor_correct_4d.npy"),
}
OUTPUT_DIR = Path("fcca_results")
FIGURE_DIR = OUTPUT_DIR / "figures"
TUCKER_FCCA_RESULTS_FILE = OUTPUT_DIR / "fcca_results.npz"
HO_RLSL_RESULTS_FILE = OUTPUT_DIR / "ho_rlsl_results.npz"
HO_RLSL_FCCA_RESULTS_FILE = OUTPUT_DIR / "ho_rlsl_fcca_results.npz"
HO_RLSL_PAPER_LIKE_REPORT_FILE = OUTPUT_DIR / "paper_like_ho_rlsl_report.json"
EPOCH_DIR = Path("01_initial_epochs")
RAW_BIDS_DIR = Path("ERN Raw Data BIDS-Compatible")
RAW_BIDS_DIR_ALIASES = (
    RAW_BIDS_DIR,
    Path("ERN_Raw_Data_BIDS-Compatible"),
    Path("preprocessing/data/raw/ERN Raw Data BIDS-Compatible"),
)

TUCKER_RANK = (10, 10, 10, 40)
DEFAULT_HO_RLSL_MAX_RANKS = (10, 10, 10)

# Detect two change points and split the time axis into three ERN phases.
INTERVAL_NAMES = ("pre_ern", "ern", "post_ern")
N_CHANGE_POINTS = 2
MIN_INTERVAL_FRAMES = 10
ADAPTIVE_RECURSIVE_MODULARITY_TOLERANCE = 0.06
CORE_INTERVAL_NAMES = {"ern", "crn"}
COMPACT_INTERVAL_NAMES = {"pre_ern", "post_ern", "pre_crn", "post_crn"}
PREFERRED_ERP_CHANNELS = ("FCz", "Cz", "FC1", "FC2")
ERP_BASELINE_MS = (-200.0, 0.0)
ERP_TMIN = -1.0
ERP_TMAX = 1.0
RAW_CONDITION_EVENT_CODES = {
    "incorrect": ("112", "122", "211", "221"),
    "correct": ("111", "121", "212", "222"),
}


def tucker_low_rank_decomposition(X, rank=TUCKER_RANK):
    """
    X shape: (subjects, nodes, nodes, time_windows)

    Returns:
        L: low-rank reconstruction
        S: residual
        core, factors: Tucker decomposition objects
    """
    if not HAS_TENSORLY:
        raise RuntimeError("tensorly is not installed; Tucker low-rank decomposition is unavailable.")
    core, factors = tucker(X, rank=rank, init="svd")
    L = tl.tucker_to_tensor((core, factors))
    S = X - L
    return L, S, core, factors


def ho_rlsl_low_rank_decomposition(
    X,
    train_length=10,
    alpha=8,
    sigma_min=0.11,
    max_ranks=DEFAULT_HO_RLSL_MAX_RANKS,
    sparse_solver="gtcs_s_omp",
    fista_max_iter=20,
    lambda_sparse=None,
    epsilon=None,
    epsilon_mode="absolute",
    change_point_position="midpoint",
    change_point_modes=None,
    sparsity=8,
    residual_tol=1e-3,
    min_cp_distance_ms=50.0,
    score_smoothing_ms=25.0,
    threshold_k=3.0,
    target_window_ms=(25.0, 75.0),
):
    if not HAS_HO_RLSL:
        raise RuntimeError("Ho_RLSL.py could not be imported.")

    config = HoRLSLConfig(
        train_length=train_length,
        alpha=alpha,
        sigma_min=sigma_min,
        max_ranks=max_ranks,
        sparse_solver=sparse_solver,
        fista_max_iter=fista_max_iter,
        sparsity=sparsity,
        residual_tol=residual_tol,
        lambda_sparse=lambda_sparse,
        epsilon=epsilon,
        epsilon_mode=epsilon_mode,
        tie_symmetric_modes=(0, 1),
        change_point_position=change_point_position,
        change_point_modes=change_point_modes,
        min_cp_distance_ms=min_cp_distance_ms,
        score_smoothing_ms=score_smoothing_ms,
        threshold_k=threshold_k,
        threshold_method="mad",
        mode_weights=(1.0, 1.0, 0.4),
        sampling_rate=1024.0,
        target_window_ms=target_window_ms,
        target_anchor_ms=50.0,
    )
    result = HoRLSL(config).fit_transform_subject_first(X)
    return result.low_rank, result.sparse, result


def save_ho_rlsl_result(result, output_path=HO_RLSL_RESULTS_FILE, save_tensors=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "change_points": np.asarray(result.change_points, dtype=int),
        "raw_change_points": np.asarray(result.raw_change_points, dtype=int),
        "filtered_change_points": np.asarray(result.filtered_change_points, dtype=int),
        "change_point_ms": np.asarray(result.change_point_ms, dtype=float),
        "change_point_modes": np.asarray(result.change_point_modes, dtype=int),
        "change_scores": np.asarray(result.change_scores, dtype=float),
        "change_score_times": np.asarray(result.change_score_times, dtype=int),
        "change_point_score_components": np.asarray(result.change_point_score_components, dtype=object),
        "sparse_solver": np.asarray(result.config.get("sparse_solver", "")),
        "solver_params": np.asarray(result.solver_params, dtype=object),
        "update_times": np.asarray(result.update_times, dtype=int),
        "rank_history": np.asarray(result.rank_history, dtype=object),
        "change_history": np.asarray(result.change_history, dtype=object),
        "input_layout": np.asarray(result.input_layout),
        "output_layout": np.asarray(result.output_layout),
        "internal_layout": np.asarray(result.internal_layout),
        "original_shape": np.asarray(result.original_shape, dtype=int),
        "internal_shape": np.asarray(result.internal_shape, dtype=int),
        "initial_ranks": np.asarray(result.initial_ranks, dtype=int),
        "initial_thresholds": np.asarray(result.initial_thresholds, dtype=float),
        "config": np.asarray(result.config, dtype=object),
    }
    if save_tensors:
        save_dict["low_rank"] = result.low_rank.astype(np.float32)
        save_dict["sparse"] = result.sparse.astype(np.float32)

    np.savez_compressed(output_path, **save_dict)


def sanitize_adjacency(A):
    A = np.asarray(A, dtype=float).copy()
    A[~np.isfinite(A)] = 0
    A = (A + A.T) / 2
    A[A < 0] = 0
    np.fill_diagonal(A, 0)
    return A


def fiedler_split(A):
    A = sanitize_adjacency(A)
    n_nodes = A.shape[0]

    if n_nodes < 2 or np.allclose(A, 0):
        return np.zeros(n_nodes, dtype=int)

    D = np.diag(A.sum(axis=1))
    L_graph = D - A
    eigenvalues, eigenvectors = eigh(L_graph)

    fiedler_vector = eigenvectors[:, 1]
    labels = (fiedler_vector > 0).astype(int)

    if len(np.unique(labels)) < 2:
        labels = (fiedler_vector > np.median(fiedler_vector)).astype(int)

    if len(np.unique(labels)) < 2:
        order = np.argsort(fiedler_vector)
        labels = np.zeros(n_nodes, dtype=int)
        labels[order[n_nodes // 2 :]] = 1

    return labels


def labels_to_communities(labels):
    labels = np.asarray(labels, dtype=int)
    return [
        set(np.flatnonzero(labels == label).tolist())
        for label in sorted(np.unique(labels))
    ]


def partition_to_labels(partition, n_nodes):
    labels = np.zeros(n_nodes, dtype=int)
    ordered = sorted((sorted(group) for group in partition if group), key=lambda group: group[0])
    for label, group in enumerate(ordered):
        labels[np.asarray(group, dtype=int)] = label
    return labels


def weighted_modularity(A, labels):
    A = sanitize_adjacency(A)
    labels = np.asarray(labels, dtype=int)
    total_weight = A.sum() / 2.0
    if total_weight <= 0:
        return np.nan

    degrees = A.sum(axis=1)
    score = 0.0
    for label in np.unique(labels):
        nodes = np.flatnonzero(labels == label)
        internal_weight = A[np.ix_(nodes, nodes)].sum() / 2.0
        degree_sum = degrees[nodes].sum()
        score += internal_weight / total_weight - (degree_sum / (2.0 * total_weight)) ** 2
    return float(score)


def recursive_fiedler_communities(A, min_size=3, min_gain=1e-4):
    A = sanitize_adjacency(A)
    n_nodes = A.shape[0]
    partition = [list(range(n_nodes))]

    changed = True
    while changed:
        changed = False
        best_gain = min_gain
        best_index = None
        best_split = None

        for idx, nodes in enumerate(partition):
            if len(nodes) < 2 * min_size:
                continue

            subA = A[np.ix_(nodes, nodes)]
            sublabels = fiedler_split(subA)
            if len(np.unique(sublabels)) < 2:
                continue

            left = [nodes[i] for i in np.flatnonzero(sublabels == 0)]
            right = [nodes[i] for i in np.flatnonzero(sublabels == 1)]
            if len(left) < min_size or len(right) < min_size:
                continue

            gain = weighted_modularity(subA, sublabels)
            if np.isfinite(gain) and gain > best_gain:
                best_gain = gain
                best_index = idx
                best_split = [left, right]

        if best_index is not None and best_split is not None:
            partition = partition[:best_index] + best_split + partition[best_index + 1 :]
            changed = True

    labels = partition_to_labels(partition, n_nodes)
    return labels, weighted_modularity(A, labels), "Adaptive recursive Fiedler"


def communities_to_labels(communities, n_nodes):
    labels = np.zeros(n_nodes, dtype=int)
    for community_idx, community in enumerate(communities):
        for node in community:
            labels[node] = community_idx
    return labels


def clustering_profile_for_interval(interval_name=None):
    if interval_name in CORE_INTERVAL_NAMES:
        return "detailed"
    if interval_name in COMPACT_INTERVAL_NAMES:
        return "compact"
    return "balanced"


def modularity_communities(A, seed=42, clustering_profile="balanced"):
    """
    Split a weighted graph by maximizing modularity.

    Returns:
        labels: integer community label per node
        score: NetworkX modularity score for the selected partition
        method: algorithm used
    """
    A = sanitize_adjacency(A)
    n_nodes = A.shape[0]

    if n_nodes < 2 or np.allclose(A, 0):
        return np.zeros(n_nodes, dtype=int), np.nan, "empty graph"

    if clustering_profile == "detailed":
        recursive_tolerance = 0.08
        recursive_min_size = 3
    elif clustering_profile == "compact":
        recursive_tolerance = 0.0
        recursive_min_size = 6
    else:
        recursive_tolerance = ADAPTIVE_RECURSIVE_MODULARITY_TOLERANCE
        recursive_min_size = 3

    recursive_labels, recursive_score, recursive_method = recursive_fiedler_communities(
        A,
        min_size=recursive_min_size,
    )

    if "nx" not in globals() or nx is None:
        return recursive_labels, recursive_score, recursive_method

    G = nx.from_numpy_array(A)
    if G.number_of_edges() == 0:
        return np.zeros(n_nodes, dtype=int), np.nan, "empty graph"

    candidates = [(recursive_labels, recursive_score, recursive_method)]

    try:
        communities = nx.community.louvain_communities(G, weight="weight", seed=seed)
        method = "Louvain modularity"
    except Exception:
        communities = nx.community.greedy_modularity_communities(G, weight="weight")
        method = "Greedy modularity"

    communities = [set(community) for community in communities if len(community) > 0]
    labels = communities_to_labels(communities, n_nodes)

    try:
        score = nx.community.modularity(G, communities, weight="weight")
    except Exception:
        score = np.nan
    candidates.append((labels, float(score), method))

    finite_candidates = [
        candidate for candidate in candidates if np.isfinite(candidate[1])
    ]
    if not finite_candidates:
        return recursive_labels, recursive_score, recursive_method

    best_labels, best_score, best_method = max(finite_candidates, key=lambda item: item[1])
    recursive_k = len(np.unique(recursive_labels))
    best_k = len(np.unique(best_labels))
    if recursive_k > best_k and recursive_score >= best_score - recursive_tolerance:
        return recursive_labels, float(recursive_score), recursive_method

    return best_labels, float(best_score), best_method


def build_consensus_matrix(labels_all, n_nodes):
    W = np.zeros((n_nodes, n_nodes), dtype=float)

    for labels in labels_all:
        same = labels[:, None] == labels[None, :]
        W += same.astype(float)

    W /= max(len(labels_all), 1)
    return W


def graph_feature_matrix(X):
    """
    Represent each time window as one graph-feature vector.

    The detector uses the subject-averaged upper-triangle edge weights, so
    change points reflect shifts in connectivity structure over time.
    """
    if X.ndim != 4:
        raise ValueError(f"Expected a 4D tensor, got shape {X.shape}")

    _, n_nodes, n_nodes_2, _ = X.shape
    if n_nodes != n_nodes_2:
        raise ValueError(f"Expected square connectivity matrices, got shape {X.shape}")

    upper = np.triu_indices(n_nodes, k=1)
    features = X[:, upper[0], upper[1], :].mean(axis=0).T
    features = np.asarray(features, dtype=float)
    features[~np.isfinite(features)] = 0

    std = features.std(axis=0)
    valid = std > 0
    features[:, valid] = (features[:, valid] - features[:, valid].mean(axis=0)) / std[valid]
    features[:, ~valid] = 0
    return features


def segment_sse(prefix_sum, prefix_sq_sum, start, end):
    n_samples = end - start
    if n_samples <= 0:
        return np.inf

    segment_sum = prefix_sum[end] - prefix_sum[start]
    segment_sq_sum = prefix_sq_sum[end] - prefix_sq_sum[start]
    return float(segment_sq_sum.sum() - np.square(segment_sum).sum() / n_samples)


def detect_change_points_exact(features, n_bkps=N_CHANGE_POINTS, min_size=MIN_INTERVAL_FRAMES):
    """
    Exact dynamic-programming change-point detector for squared-error segments.
    Returns boundary frame indices where a new interval starts.
    """
    n_times = features.shape[0]
    n_segments = n_bkps + 1
    min_size = max(1, min(min_size, n_times // n_segments))

    if n_times < n_segments:
        return []

    prefix_sum = np.vstack([np.zeros(features.shape[1]), np.cumsum(features, axis=0)])
    prefix_sq_sum = np.vstack([np.zeros(features.shape[1]), np.cumsum(features * features, axis=0)])

    dp = np.full((n_segments + 1, n_times + 1), np.inf)
    previous = np.full((n_segments + 1, n_times + 1), -1, dtype=int)
    dp[0, 0] = 0

    for segment_idx in range(1, n_segments + 1):
        min_end = segment_idx * min_size
        max_end = n_times - (n_segments - segment_idx) * min_size
        for end in range(min_end, max_end + 1):
            best_cost = np.inf
            best_start = -1
            min_start = (segment_idx - 1) * min_size
            max_start = end - min_size
            for start in range(min_start, max_start + 1):
                cost = dp[segment_idx - 1, start] + segment_sse(
                    prefix_sum,
                    prefix_sq_sum,
                    start,
                    end,
                )
                if cost < best_cost:
                    best_cost = cost
                    best_start = start
            dp[segment_idx, end] = best_cost
            previous[segment_idx, end] = best_start

    if not np.isfinite(dp[n_segments, n_times]):
        return []

    boundaries = []
    end = n_times
    for segment_idx in range(n_segments, 0, -1):
        start = previous[segment_idx, end]
        if start <= 0:
            break
        boundaries.append(start)
        end = start

    return sorted(boundaries)


def detect_change_points(X, n_bkps=N_CHANGE_POINTS, min_size=MIN_INTERVAL_FRAMES):
    features = graph_feature_matrix(X)

    if HAS_RUPTURES:
        try:
            bkps = rpt.Binseg(model="l2", min_size=min_size).fit(features).predict(n_bkps=n_bkps)
            return sorted(int(bkp) for bkp in bkps[:-1]), "ruptures Binseg"
        except Exception as exc:
            print(f"[WARN] ruptures change-point detection failed: {exc}")

    return detect_change_points_exact(features, n_bkps=n_bkps, min_size=min_size), "exact SSE"


def make_intervals_from_change_points(n_times, change_points, names=INTERVAL_NAMES):
    boundaries = [0]
    boundaries.extend(int(point) for point in sorted(change_points) if 0 < point < n_times)
    boundaries.append(n_times)

    expected_boundaries = len(names) + 1
    if len(boundaries) != expected_boundaries:
        frame_splits = np.array_split(np.arange(n_times), len(names))
        return {
            name: frames
            for name, frames in zip(names, frame_splits)
            if len(frames) > 0
        }

    return {
        name: np.arange(start, end)
        for name, start, end in zip(names, boundaries[:-1], boundaries[1:])
        if end > start
    }


def make_change_point_intervals(X, names=INTERVAL_NAMES):
    n_times = X.shape[-1]
    change_points, method = detect_change_points(X, n_bkps=len(names) - 1)
    intervals = make_intervals_from_change_points(n_times, change_points, names)
    return intervals, change_points, method


def frames_to_ms(n_times, start_ms=-1000.0, end_ms=1000.0):
    return np.linspace(start_ms, end_ms, n_times)


def connectivity_change_scores(tensor, candidate_points, window=64):
    """Score connectivity changes in a low-rank tensor around candidate frames.

    The paper defines ERN intervals from change points of the connectivity
    mode, not from subject-mode updates. This score compares the mean
    low-rank adjacency vector before and after each candidate update point.
    """
    tensor = np.asarray(tensor)
    if tensor.ndim != 4:
        raise ValueError(f"Expected tensor shape (subjects, nodes, nodes, time), got {tensor.shape}")

    n_subjects, n_nodes, n_nodes_2, n_times = tensor.shape
    if n_nodes != n_nodes_2:
        raise ValueError(f"Expected square node-node connectivity matrices, got {tensor.shape}")

    upper = np.triu_indices(n_nodes, k=1)
    points = np.asarray(candidate_points, dtype=int).ravel()
    valid_points = []
    scores = []
    half_window = max(1, int(window))

    for point in points:
        point = int(point)
        left_start = max(0, point - half_window)
        left_end = point
        right_start = point
        right_end = min(n_times, point + half_window)
        if left_end <= left_start or right_end <= right_start:
            continue

        left = tensor[:, upper[0], upper[1], left_start:left_end].mean(axis=(0, 2))
        right = tensor[:, upper[0], upper[1], right_start:right_end].mean(axis=(0, 2))
        scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1.0)
        valid_points.append(point)
        scores.append(float(np.linalg.norm(right - left) / scale))

    return np.asarray(valid_points, dtype=int), np.asarray(scores, dtype=float)


def select_connectivity_change_points(
    tensor,
    candidate_points,
    target_window_ms=(25.0, 75.0),
    anchor_ms=50.0,
    score_window=64,
    start_ms=-1000.0,
    end_ms=1000.0,
):
    n_times = tensor.shape[-1]
    times_ms = frames_to_ms(n_times, start_ms=start_ms, end_ms=end_ms)
    points, scores = connectivity_change_scores(tensor, candidate_points, window=score_window)
    if points.size == 0:
        return points, {
            "connectivity_candidate_points": [],
            "connectivity_candidate_ms": [],
            "connectivity_scores": [],
            "connectivity_selection": "unavailable",
        }

    in_target = (times_ms[points] >= target_window_ms[0]) & (times_ms[points] <= target_window_ms[1])
    if np.any(in_target):
        target_indices = np.flatnonzero(in_target)
        anchor_idx = int(target_indices[np.argmax(scores[target_indices])])
        selection = "highest connectivity score in target window"
    else:
        anchor_idx = int(np.argmin(np.abs(times_ms[points] - anchor_ms)))
        selection = "closest connectivity candidate to target center fallback"

    anchor_point = int(points[anchor_idx])
    before_anchor = np.flatnonzero(points < anchor_point)
    after_anchor = np.flatnonzero(points > anchor_point)
    selected_indices = []
    if before_anchor.size >= 2:
        selected_indices.append(int(before_anchor[-2]))
    if before_anchor.size >= 1:
        selected_indices.append(int(before_anchor[-1]))
    selected_indices.append(anchor_idx)
    if after_anchor.size >= 1:
        selected_indices.append(int(after_anchor[0]))
    if after_anchor.size >= 2:
        selected_indices.append(int(after_anchor[1]))

    selected_indices = sorted(set(selected_indices), key=lambda idx: int(points[idx]))
    selected_points = points[selected_indices]
    metadata = {
        "connectivity_candidate_points": [int(point) for point in points],
        "connectivity_candidate_ms": [float(times_ms[point]) for point in points],
        "connectivity_scores": [float(score) for score in scores],
        "connectivity_selected_points": [int(point) for point in selected_points],
        "connectivity_selected_ms": [float(times_ms[point]) for point in selected_points],
        "connectivity_anchor_point": anchor_point,
        "connectivity_anchor_ms": float(times_ms[anchor_point]),
        "connectivity_selection": selection,
    }
    return selected_points, metadata


def ho_rlsl_mode_change_counts(change_history):
    counts = {}
    for event in change_history or []:
        event = dict(event)
        if int(event.get("changed", 0)):
            mode = int(event.get("mode", -1))
            counts[mode] = counts.get(mode, 0) + 1
    return counts


def make_paper_like_ho_rlsl_intervals(
    n_times,
    change_points,
    connectivity_tensor=None,
    connectivity_candidate_points=None,
    names=INTERVAL_NAMES,
    target_window_ms=(25.0, 75.0),
    anchor_ms=50.0,
    start_ms=-1000.0,
    end_ms=1000.0,
):
    """Build pre/ERN/post intervals from Ho-RLSL update change points.

    Paper-like flow: Ho-RLSL update change points define intervals; a separate
    segmentation detector is not used. For ERN, choose the change point closest
    to the expected ERN latency, then use its nearest neighboring Ho-RLSL
    change points as interval boundaries.
    """
    times_ms = frames_to_ms(n_times, start_ms=start_ms, end_ms=end_ms)
    connectivity_metadata = {}
    if connectivity_tensor is not None and connectivity_candidate_points is not None:
        points, connectivity_metadata = select_connectivity_change_points(
            connectivity_tensor,
            connectivity_candidate_points,
            target_window_ms=target_window_ms,
            anchor_ms=anchor_ms,
            score_window=64,
            start_ms=start_ms,
            end_ms=end_ms,
        )
    else:
        points = np.asarray(change_points, dtype=int).ravel()
    points = np.unique(points[(points > 0) & (points < n_times)])

    if points.size < 3:
        intervals = make_intervals_from_change_points(n_times, [], names)
        metadata = {
            "method": "Ho-RLSL update rule intervals unavailable; fallback equal split",
            "anchor_point": None,
            "anchor_ms": None,
            "boundary_points": [],
            "boundary_ms": [],
            "all_change_points": points.astype(int).tolist(),
            "all_change_points_ms": [float(times_ms[p]) for p in points],
            "target_window_ms": [float(target_window_ms[0]), float(target_window_ms[1])],
            **connectivity_metadata,
        }
        return intervals, [], metadata

    point_times = times_ms[points]
    in_target = (point_times >= target_window_ms[0]) & (point_times <= target_window_ms[1])
    candidate_indices = np.flatnonzero(in_target)
    if candidate_indices.size:
        anchor_local = int(candidate_indices[np.argmin(np.abs(point_times[candidate_indices] - anchor_ms))])
        anchor_source = "within target window"
    else:
        anchor_local = int(np.argmin(np.abs(point_times - anchor_ms)))
        anchor_source = "closest to target center fallback"

    anchor_point = int(points[anchor_local])
    before = points[points < anchor_point]
    after = points[points > anchor_point]

    if before.size and after.size:
        ern_start = int(before[-1])
        ern_end = int(after[0])
    elif before.size:
        ern_start = int(before[-1])
        ern_end = min(n_times - 1, anchor_point + max(1, anchor_point - ern_start))
    elif after.size:
        ern_end = int(after[0])
        ern_start = max(1, anchor_point - max(1, ern_end - anchor_point))
    else:
        ern_start = max(1, anchor_point - 1)
        ern_end = min(n_times - 1, anchor_point + 1)

    pre_duration = max(ern_end - ern_start, 1)
    post_duration = max(ern_end - ern_start, 1)
    pre_start = max(1, ern_start - pre_duration)
    post_end = min(n_times - 1, ern_end + post_duration)

    # Use neighboring Ho-RLSL points when available. If there are no extra
    # neighbors, mirror the ERN duration so the displayed intervals have four
    # explicit paper-style boundaries instead of using epoch endpoints.
    extra_before = points[points < ern_start]
    extra_after = points[points > ern_end]
    if extra_before.size:
        pre_start = int(extra_before[-1])
    if extra_after.size:
        post_end = int(extra_after[0])

    interval_boundaries = [pre_start, ern_start, ern_end, post_end]
    interval_boundaries = sorted({int(point) for point in interval_boundaries if 0 <= point < n_times})
    if len(interval_boundaries) != 4:
        intervals = make_intervals_from_change_points(n_times, [], names)
        boundaries = []
    else:
        pre_start, ern_start, ern_end, post_end = interval_boundaries
        intervals = {
            names[0]: np.arange(pre_start, ern_start),
            names[1]: np.arange(ern_start, ern_end),
            names[2]: np.arange(ern_end, post_end),
        }
        boundaries = interval_boundaries
    source_label = (
        "connectivity-mode low-rank score"
        if connectivity_metadata
        else "Ho-RLSL update rule"
    )
    method = (
        f"Ho-RLSL {source_label} paper-like intervals "
        f"({anchor_source}; anchor {times_ms[anchor_point]:.1f} ms)"
    )
    metadata = {
        "method": method,
        "anchor_point": anchor_point,
        "anchor_ms": float(times_ms[anchor_point]),
        "anchor_source": anchor_source,
        "boundary_points": [int(point) for point in boundaries],
        "boundary_ms": [float(times_ms[point]) for point in boundaries],
        "interval_boundary_points": [int(point) for point in boundaries],
        "interval_boundary_ms": [float(times_ms[point]) for point in boundaries],
        "all_change_points": points.astype(int).tolist(),
        "all_change_points_ms": [float(times_ms[point]) for point in points],
        "target_window_ms": [float(target_window_ms[0]), float(target_window_ms[1])],
        **connectivity_metadata,
    }
    return intervals, boundaries, metadata


def condition_event_names(epochs, condition):
    names = list(getattr(epochs, "event_id", {}).keys())
    if not names:
        return []

    selected = []
    for name in names:
        normalized = name.lower()
        has_error = any(token in normalized for token in ("incorrect", "error", "wrong", "ern"))
        has_correct = any(token in normalized for token in ("correct", "crn"))

        if condition == "incorrect" and has_error:
            selected.append(name)
        elif condition == "correct" and has_correct and not any(
            token in normalized for token in ("incorrect", "error", "wrong")
        ):
            selected.append(name)

    return selected


def choose_erp_channel(epochs, preferred_channels=PREFERRED_ERP_CHANNELS):
    channel_lookup = {name.lower(): idx for idx, name in enumerate(epochs.ch_names)}
    for preferred in preferred_channels:
        idx = channel_lookup.get(preferred.lower())
        if idx is not None:
            return idx, epochs.ch_names[idx]

    picks = mne.pick_types(epochs.info, eeg=True, meg=False, exclude="bads")
    if len(picks) > 0:
        idx = int(picks[0])
        return idx, epochs.ch_names[idx]

    return 0, epochs.ch_names[0]


def choose_raw_channel(raw, preferred_channels=PREFERRED_ERP_CHANNELS):
    channel_lookup = {name.lower(): idx for idx, name in enumerate(raw.ch_names)}
    for preferred in preferred_channels:
        idx = channel_lookup.get(preferred.lower())
        if idx is not None:
            return raw.ch_names[idx]

    picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude="bads")
    if len(picks) > 0:
        return raw.ch_names[int(picks[0])]

    return raw.ch_names[0]


def baseline_correct_trace(trace, times_ms, baseline_ms=ERP_BASELINE_MS):
    baseline_mask = (times_ms >= baseline_ms[0]) & (times_ms <= baseline_ms[1])
    if not baseline_mask.any():
        baseline_mask = times_ms < 0
    if baseline_mask.any():
        return trace - np.nanmean(trace[baseline_mask])
    return trace - np.nanmean(trace)


def convert_eeg_trace_to_microvolts(trace):
    # MNE stores EEG data in volts. Plot ERP waveforms in microvolts.
    return np.asarray(trace, dtype=float) * 1e6


def epochs_to_condition_trace(epochs, condition):
    event_names = condition_event_names(epochs, condition)
    if event_names:
        epochs = epochs[event_names]

    if len(epochs) == 0:
        return None

    channel_idx, channel_name = choose_erp_channel(epochs)
    channel_type = epochs.get_channel_types(picks=[channel_idx])[0]
    data = epochs.get_data()[:, channel_idx, :]
    trace = data.mean(axis=0)
    times_ms = np.asarray(epochs.times * 1000.0, dtype=float)

    if channel_type == "eeg":
        trace = convert_eeg_trace_to_microvolts(trace)

    trace = baseline_correct_trace(np.asarray(trace, dtype=float), times_ms)

    return {
        "trace": trace,
        "times_ms": times_ms,
        "n_epochs": len(epochs),
        "channel": channel_name,
        "channel_type": channel_type,
        "events": tuple(event_names) if event_names else ("all",),
    }


def raw_to_condition_trace(raw_path, condition):
    raw = mne.io.read_raw_eeglab(raw_path, preload=False, verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    wanted_codes = RAW_CONDITION_EVENT_CODES[condition]
    selected_event_id = {
        code: event_id[code]
        for code in wanted_codes
        if code in event_id
    }

    if not selected_event_id:
        return None

    channel_name = choose_raw_channel(raw)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=selected_event_id,
        tmin=ERP_TMIN,
        tmax=ERP_TMAX,
        baseline=None,
        picks=[channel_name],
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )

    if len(epochs) == 0:
        return None

    trace = epochs.get_data()[:, 0, :].mean(axis=0)
    times_ms = np.asarray(epochs.times * 1000.0, dtype=float)
    trace = convert_eeg_trace_to_microvolts(trace)
    trace = baseline_correct_trace(trace, times_ms)

    return {
        "trace": trace,
        "times_ms": times_ms,
        "n_epochs": len(epochs),
        "channel": channel_name,
        "channel_type": "eeg",
        "events": tuple(selected_event_id.keys()),
    }


def load_raw_bids_erp_traces(raw_bids_dir=RAW_BIDS_DIR, conditions=CONDITION_FILES.keys()):
    if not HAS_MNE:
        return {}

    raw_dirs = [Path(raw_bids_dir)]
    if Path(raw_bids_dir) == RAW_BIDS_DIR:
        raw_dirs.extend(path for path in RAW_BIDS_DIR_ALIASES if path not in raw_dirs)

    raw_paths = []
    raw_source = None
    for candidate in raw_dirs:
        raw_paths = sorted(candidate.glob("sub-*/eeg/*_task-ERN_eeg.set"))
        if raw_paths:
            raw_source = candidate
            break
    if not raw_paths:
        return {}

    collected = {
        condition: {
            "traces": [],
            "times_ms": None,
            "n_epochs": 0,
            "subjects": 0,
            "channels": [],
            "channel_types": [],
            "events": set(),
        }
        for condition in conditions
    }

    for raw_path in raw_paths:
        for condition in conditions:
            try:
                item = raw_to_condition_trace(raw_path, condition)
            except Exception as exc:
                print(f"[WARN] Could not build {condition} ERP from {raw_path}: {exc}")
                continue

            if item is None:
                continue

            target = collected[condition]
            if target["times_ms"] is None:
                target["times_ms"] = item["times_ms"]
                trace = item["trace"]
            else:
                trace = np.interp(target["times_ms"], item["times_ms"], item["trace"])

            target["traces"].append(trace)
            target["n_epochs"] += item["n_epochs"]
            target["subjects"] += 1
            target["channels"].append(item["channel"])
            target["channel_types"].append(item["channel_type"])
            target["events"].update(item["events"])

    erp = {}
    for condition, item in collected.items():
        if item["traces"]:
            erp[condition] = {
                "trace": np.mean(np.vstack(item["traces"]), axis=0),
                "times_ms": item["times_ms"],
                "n_epochs": item["n_epochs"],
                "subjects": item["subjects"],
                "channel": max(set(item["channels"]), key=item["channels"].count),
                "channel_type": max(set(item["channel_types"]), key=item["channel_types"].count),
                "events": tuple(sorted(item["events"])),
                "source": f"raw BIDS EEGLAB: {raw_source}",
            }

    return erp


def load_erp_traces(epoch_dir=EPOCH_DIR, conditions=CONDITION_FILES.keys()):
    if not HAS_MNE:
        print("[WARN] mne is not installed; plotting connectivity trace instead of EEG ERP.")
        return {}

    raw_erp = load_raw_bids_erp_traces(conditions=conditions)
    if raw_erp:
        missing_conditions = [condition for condition in conditions if condition not in raw_erp]
        if not missing_conditions:
            print(f"Using raw BIDS EEG ERPs from {RAW_BIDS_DIR}")
            return raw_erp

        print(
            "[WARN] Raw BIDS ERPs were incomplete for "
            f"{missing_conditions}; falling back to epoch FIF files where needed."
        )

    fif_files = sorted(Path(epoch_dir).glob("*-epo.fif"))
    if not fif_files:
        print(f"[WARN] No *-epo.fif files found in {epoch_dir}; plotting connectivity trace.")
        return raw_erp

    collected = {
        condition: {
            "traces": [],
            "times_ms": None,
            "n_epochs": 0,
            "subjects": 0,
            "channels": [],
            "events": set(),
        }
        for condition in conditions
    }

    for fif_file in fif_files:
        try:
            epochs = mne.read_epochs(fif_file, preload=True, verbose=False)
        except Exception as exc:
            print(f"[WARN] Could not read {fif_file}: {exc}")
            continue

        for condition in conditions:
            item = epochs_to_condition_trace(epochs, condition)
            if item is None:
                continue

            target = collected[condition]
            if target["times_ms"] is None:
                target["times_ms"] = item["times_ms"]
                trace = item["trace"]
            else:
                trace = np.interp(target["times_ms"], item["times_ms"], item["trace"])

            target["traces"].append(trace)
            target["n_epochs"] += item["n_epochs"]
            target["subjects"] += 1
            target["channels"].append(item["channel"])
            target.setdefault("channel_types", []).append(item["channel_type"])
            target["events"].update(item["events"])

    erp = {}
    for condition, item in collected.items():
        if item["traces"]:
            erp[condition] = {
                "trace": np.mean(np.vstack(item["traces"]), axis=0),
                "times_ms": item["times_ms"],
                "n_epochs": item["n_epochs"],
                "subjects": item["subjects"],
                "channel": max(set(item["channels"]), key=item["channels"].count),
                "channel_type": max(set(item["channel_types"]), key=item["channel_types"].count),
                "events": tuple(sorted(item["events"])),
                "source": "epoch FIF",
            }

    erp.update(raw_erp)
    return erp


def tensor_mean_connectivity_trace(X):
    """
    Convert a 4D connectivity tensor to one scalar time-course by averaging
    over subjects and unique node pairs.

    Expected X shape: (subjects, nodes, nodes, time_windows).
    """
    if X.ndim != 4:
        raise ValueError(f"Expected a 4D tensor, got shape {X.shape}")

    _, n_nodes, n_nodes_2, _ = X.shape
    if n_nodes != n_nodes_2:
        raise ValueError(f"Expected square connectivity matrices, got shape {X.shape}")

    upper = np.triu_indices(n_nodes, k=1)
    trace = X[:, upper[0], upper[1], :].mean(axis=(0, 1))
    trace = np.asarray(trace, dtype=float)

    std = trace.std()
    if std > 0:
        trace = (trace - trace.mean()) / std
    else:
        trace = trace - trace.mean()

    return trace


def smooth_trace(trace, window=7):
    if window <= 1 or len(trace) < window:
        return trace

    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    padded = np.pad(trace, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def draw_interval_markers(ax, intervals, times_ms, condition_label):
    interval_list = list(intervals.values())
    boundary_frames = []
    for frames in interval_list:
        if len(frames) > 0:
            boundary_frames.append(int(frames[0]))
    if interval_list and len(interval_list[-1]) > 0:
        boundary_frames.append(int(interval_list[-1][-1]))
    for frame in sorted(set(boundary_frames)):
        if 0 <= frame < len(times_ms):
            ax.axvline(times_ms[frame], color="#7587ff", linewidth=1.5, alpha=0.85)

    condition_upper = condition_label.upper()
    if condition_label == "incorrect":
        labels = ("PRE-ERN", "ERN", "POST-ERN")
    elif condition_label == "correct":
        labels = ("PRE-CRN", "CRN", "POST-CRN")
    else:
        labels = (f"PRE-{condition_upper}", condition_upper, f"POST-{condition_upper}")

    label_frames = [
        intervals[name]
        for name in INTERVAL_NAMES
        if name in intervals
    ]

    y_min, y_max = ax.get_ylim()
    y_text = y_max - 0.12 * (y_max - y_min)
    for text, frames in zip(labels, label_frames):
        center_frame = int(round((frames[0] + frames[-1]) / 2))
        ax.text(
            times_ms[center_frame],
            y_text,
            text,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
        )


def visualize_condition_timecourses(condition_files=CONDITION_FILES):
    """
    Paper-style two-panel figure for incorrect/correct ERPs.
    The red trace comes from raw BIDS EEG when available;
    blue boundaries come from connectivity-tensor change points.
    """
    available = {
        name: path
        for name, path in condition_files.items()
        if path.exists()
    }
    if not available:
        print("[WARN] No condition tensor files found for time-course visualization.")
        return

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    erp_traces = load_erp_traces(conditions=available.keys())
    fig, axes = plt.subplots(len(available), 1, figsize=(10.5, 5.8), sharex=True)
    axes = np.atleast_1d(axes)

    for panel_idx, (condition_name, tensor_path) in enumerate(available.items()):
        X = np.load(tensor_path)
        n_times = X.shape[-1]
        tensor_times_ms = frames_to_ms(n_times)
        intervals, change_points, method = make_change_point_intervals(X)
        erp = erp_traces.get(condition_name)

        if erp is not None:
            trace = smooth_trace(erp["trace"], window=7)
            times_ms = erp["times_ms"]
            y_label = "Amplitude (μV)" if erp["channel_type"] == "eeg" else "Baseline-corrected CSD"
            event_name = "ERN" if condition_name == "incorrect" else "CRN"
            title = (
                f"({chr(97 + panel_idx)}) Average {event_name} waveform: "
                f"{erp['subjects']} subjects, {erp['n_epochs']} epochs, "
                f"{erp['channel']}, {erp['channel_type']}"
            )
        else:
            trace = smooth_trace(tensor_mean_connectivity_trace(X), window=7)
            times_ms = tensor_times_ms
            y_label = "z(mean PLV)"
            title = (
                f"({chr(97 + panel_idx)}) {condition_name.capitalize()} tensor: "
                f"{X.shape[0]} subjects, {X.shape[1]} nodes, {n_times} windows"
            )

        ax = axes[panel_idx]
        ax.plot(times_ms, trace, color="red", linewidth=1.7)
        ax.axhline(0, color="0.65", linewidth=0.8)
        ax.set_xlim(-1000, 1000)
        ax.set_ylabel(y_label)
        ax.set_title(title, loc="left")
        draw_interval_markers(ax, intervals, tensor_times_ms, condition_name)
        ax.text(
            0.99,
            0.08,
            f"{method}: frames {change_points}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.25",
        )
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.tick_params(direction="in", top=True, right=True)

    axes[-1].set_xlabel("time (ms)")
    fig.suptitle("Average ERN/CRN Waveforms and Connectivity Change Points", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    save_path = FIGURE_DIR / "incorrect_correct_tensor_timecourses.png"
    fig.savefig(save_path, dpi=250)
    plt.close(fig)
    print(f"Saved condition time-course figure: {save_path}")


def run_fcca_interval(X_clean, frames, interval_name=None):
    """
    Minimal FCCA:
      1. Modularity split each subject-time graph.
      2. Build channel co-clustering consensus matrix.
      3. Modularity split the consensus matrix and keep its quality score.
    """
    n_subjects, n_nodes, _, _ = X_clean.shape
    graph_labels = []
    graph_modularity_scores = []
    graphs = []
    community_method = None

    for s in range(n_subjects):
        for t in frames:
            A = sanitize_adjacency(X_clean[s, :, :, t])
            labels, score, method = modularity_communities(A)
            graph_labels.append(labels)
            graph_modularity_scores.append(score)
            community_method = method
            graphs.append(A)

    graph_labels = np.asarray(graph_labels)
    graph_modularity_scores = np.asarray(graph_modularity_scores, dtype=float)
    graphs = np.asarray(graphs)

    consensus_matrix = build_consensus_matrix(graph_labels, n_nodes)
    clustering_profile = clustering_profile_for_interval(interval_name)
    consensus_labels, consensus_modularity, consensus_method = modularity_communities(
        consensus_matrix,
        clustering_profile=clustering_profile,
    )
    mean_adjacency = graphs.mean(axis=0)

    return {
        "frames": np.asarray(frames),
        "graph_labels": graph_labels,
        "graph_modularity_scores": graph_modularity_scores,
        "consensus_matrix": consensus_matrix,
        "consensus_labels": consensus_labels,
        "consensus_modularity": np.asarray(consensus_modularity),
        "community_method": np.asarray(
            f"{consensus_method or community_method or 'unknown'} ({clustering_profile})"
        ),
        "mean_adjacency": mean_adjacency,
    }


def dyconnmap_threshold(A):
    """
    Use dyconnmap's OMST thresholding when installed. Fall back to the
    strongest 20% of edges so visualization still works without dyconnmap.
    """
    A = sanitize_adjacency(A)

    if HAS_DYCONNMAP:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="divide by zero encountered in divide",
                    category=RuntimeWarning,
                )
                _, thresholded, *_ = threshold_omst_global_cost_efficiency(A)
            thresholded = sanitize_adjacency(thresholded)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="divide by zero encountered in divide",
                    category=RuntimeWarning,
                )
                efficiency = nodal_global_efficiency(thresholded)
            return thresholded, np.asarray(efficiency).ravel(), "dyconnmap OMST"
        except Exception as exc:
            print(f"[WARN] dyconnmap threshold failed: {exc}")

    upper = A[np.triu_indices_from(A, k=1)]
    positive = upper[upper > 0]
    thresholded = np.zeros_like(A)

    if len(positive) > 0:
        cutoff = np.quantile(positive, 0.80)
        thresholded[A >= cutoff] = A[A >= cutoff]
        thresholded = sanitize_adjacency(thresholded)

    return thresholded, np.full(A.shape[0], np.nan), "top 20% fallback"


def plot_matrix(ax, matrix, title, labels=None):
    if labels is not None:
        order = np.argsort(labels)
        matrix = matrix[np.ix_(order, order)]
    else:
        order = np.arange(matrix.shape[0])

    im = ax.imshow(matrix, cmap="viridis", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Node")
    ax.set_ylabel("Node")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return order


COMMUNITY_PALETTE = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#e377c2",
    "#8c564b",
    "#bcbd22",
    "#7f7f7f",
)


def community_node_colors(labels):
    labels = np.asarray(labels, dtype=int)
    communities = sorted(int(label) for label in np.unique(labels))
    color_map = {
        community: COMMUNITY_PALETTE[idx % len(COMMUNITY_PALETTE)]
        for idx, community in enumerate(communities)
    }
    return [color_map[int(label)] for label in labels]


def community_edge_style(i, j, labels):
    labels = np.asarray(labels, dtype=int)
    communities = sorted(int(label) for label in np.unique(labels))
    color_map = {
        community: COMMUNITY_PALETTE[idx % len(COMMUNITY_PALETTE)]
        for idx, community in enumerate(communities)
    }
    if labels[i] == labels[j]:
        return color_map[int(labels[i])], 0.58, 1.25
    return "0.70", 0.22, 0.65


def plot_circular_graph(ax, A, labels, title):
    A_thr, efficiency, threshold_label = dyconnmap_threshold(A)
    n_nodes = A_thr.shape[0]
    theta = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    xy = np.column_stack([np.cos(theta), np.sin(theta)])

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if A_thr[i, j] > 0:
                color, alpha, linewidth = community_edge_style(i, j, labels)
                ax.plot(
                    [xy[i, 0], xy[j, 0]],
                    [xy[i, 1], xy[j, 1]],
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth,
                )

    node_colors = community_node_colors(labels)
    ax.scatter(xy[:, 0], xy[:, 1], c=node_colors, s=80, edgecolors="white", zorder=3)

    for idx, (x_pos, y_pos) in enumerate(xy):
        ax.text(x_pos * 1.14, y_pos * 1.14, str(idx + 1), ha="center", va="center", fontsize=7)

    ax.set_title(f"{title}\n{threshold_label}")
    ax.set_aspect("equal")
    ax.axis("off")

    if HAS_DYCONNMAP and np.isfinite(efficiency).any():
        ax.text(
            0.0,
            -1.25,
            f"mean nodal efficiency: {np.nanmean(efficiency):.3f}",
            ha="center",
            va="center",
            fontsize=8,
        )


def visualize_fcca_results(results):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for interval_name, result in results.items():
        consensus = result["consensus_matrix"]
        labels = result["consensus_labels"]
        mean_adjacency = result["mean_adjacency"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        plot_matrix(axes[0], consensus, f"{interval_name}: consensus", labels)
        plot_matrix(axes[1], mean_adjacency, f"{interval_name}: mean low-rank graph", labels)
        plot_circular_graph(axes[2], mean_adjacency, labels, f"{interval_name}: graph")

        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"{interval_name}_fcca.png", dpi=200)
        plt.close(fig)


def save_fcca_results(
    results,
    change_points=None,
    change_point_method=None,
    output_path=TUCKER_FCCA_RESULTS_FILE,
    extra_fields=None,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_dict = {}
    for interval_name, result in results.items():
        for key, value in result.items():
            save_dict[f"{interval_name}_{key}"] = value

    if change_points is not None:
        save_dict["change_points"] = np.asarray(change_points, dtype=int)
    if change_point_method is not None:
        save_dict["change_point_method"] = np.asarray(change_point_method)
    if extra_fields:
        for key, value in extra_fields.items():
            save_dict[key] = value

    np.savez_compressed(output_path, **save_dict)


def save_paper_like_report(report, output_path=HO_RLSL_PAPER_LIKE_REPORT_FILE):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Run tensor low-rank denoising and interval FCCA analysis.")
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--low-rank-method", choices=["tucker", "ho-rlsl"], default="tucker")
    parser.add_argument("--skip-timecourse", action="store_true")
    parser.add_argument("--save-ho-rlsl-tensors", action="store_true")
    parser.add_argument("--ho-train-length", type=int, default=10)
    parser.add_argument("--ho-alpha", type=int, default=8)
    parser.add_argument("--ho-sigma-min", type=float, default=0.11)
    parser.add_argument("--ho-max-ranks", default="10,10,10")
    parser.add_argument("--ho-sparse-solver", choices=["fista", "fista_l1", "proxy", "gtcs_s_omp"], default="gtcs_s_omp")
    parser.add_argument("--ho-fista-max-iter", type=int, default=20)
    parser.add_argument("--ho-sparsity", type=int, default=8)
    parser.add_argument("--ho-residual-tol", type=float, default=1e-3)
    parser.add_argument("--ho-min-cp-distance-ms", type=float, default=50.0)
    parser.add_argument("--ho-score-smoothing-ms", type=float, default=25.0)
    parser.add_argument("--ho-threshold-k", type=float, default=3.0)
    parser.add_argument("--ho-lambda-sparse", type=float, default=None)
    parser.add_argument("--ho-epsilon", type=float, default=None)
    parser.add_argument("--ho-epsilon-mode", choices=["absolute", "relative"], default="absolute")
    parser.add_argument("--ho-change-point-position", choices=["start", "midpoint", "end"], default="midpoint")
    parser.add_argument(
        "--ho-change-point-modes",
        default=None,
        help=(
            "Optional comma-separated Ho-RLSL internal modes allowed to create change points. "
            "Modes 0/1 are connectivity; mode 2 is subjects."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.low_rank_method == "tucker" and not HAS_TENSORLY:
        raise RuntimeError("tensorly is not installed. Use --low-rank-method ho-rlsl or install tensorly.")
    if not args.skip_timecourse:
        visualize_condition_timecourses()

    X = np.load(args.input)

    ho_result = None
    output_path = TUCKER_FCCA_RESULTS_FILE
    if args.low_rank_method == "tucker":
        L, S, core, factors = tucker_low_rank_decomposition(X, rank=TUCKER_RANK)
        X_clean = L
        low_rank_label = "Tucker"
    else:
        max_ranks = parse_optional_int_tuple(args.ho_max_ranks)
        if max_ranks is None or any(rank is None for rank in max_ranks):
            raise ValueError("--ho-max-ranks must provide three integer ranks, e.g. 10,10,10")
        change_point_modes = parse_optional_int_tuple(args.ho_change_point_modes)
        if change_point_modes is not None and any(mode is None for mode in change_point_modes):
            raise ValueError("--ho-change-point-modes must contain integers only, e.g. 0 or 0,1")
        L, S, ho_result = ho_rlsl_low_rank_decomposition(
            X,
            train_length=args.ho_train_length,
            alpha=args.ho_alpha,
            sigma_min=args.ho_sigma_min,
            max_ranks=tuple(int(rank) for rank in max_ranks),
            sparse_solver=args.ho_sparse_solver,
            fista_max_iter=args.ho_fista_max_iter,
            sparsity=args.ho_sparsity,
            residual_tol=args.ho_residual_tol,
            lambda_sparse=args.ho_lambda_sparse,
            epsilon=args.ho_epsilon,
            epsilon_mode=args.ho_epsilon_mode,
            change_point_position=args.ho_change_point_position,
            change_point_modes=None
            if change_point_modes is None
            else tuple(int(mode) for mode in change_point_modes),
            min_cp_distance_ms=args.ho_min_cp_distance_ms,
            score_smoothing_ms=args.ho_score_smoothing_ms,
            threshold_k=args.ho_threshold_k,
        )
        save_ho_rlsl_result(ho_result, save_tensors=args.save_ho_rlsl_tensors)
        X_clean = L
        output_path = HO_RLSL_FCCA_RESULTS_FILE
        low_rank_label = "Ho-RLSL"

    print("Original:", X.shape)
    print(f"Low-rank ({low_rank_label}):", L.shape)
    print("Sparse residual:", S.shape)
    if ho_result is not None:
        print("Ho-RLSL filtered change points:", ho_result.change_points)
        print("Ho-RLSL raw update change points:", ho_result.raw_change_points)
        print(f"Saved Ho-RLSL metadata: {HO_RLSL_RESULTS_FILE}")
    print("dyconnmap visualization helpers:", "available" if HAS_DYCONNMAP else "not installed")
    print("change-point detector:", "ruptures available" if HAS_RUPTURES else "NumPy exact SSE fallback")

    n_subjects, n_nodes, _, n_times = X_clean.shape
    times_ms = frames_to_ms(n_times)
    paper_like_metadata = None
    if ho_result is not None:
        intervals, change_points, paper_like_metadata = make_paper_like_ho_rlsl_intervals(
            n_times,
            ho_result.raw_change_points if ho_result.raw_change_points else ho_result.change_points,
            connectivity_tensor=X_clean,
            connectivity_candidate_points=ho_result.change_score_times
            if len(ho_result.change_score_times)
            else ho_result.update_times,
        )
        change_point_method = paper_like_metadata["method"]
    else:
        intervals, change_points, change_point_method = make_change_point_intervals(X_clean)
    print(
        f"Detected change points with {change_point_method}: "
        f"frames={change_points} | ms={[round(float(times_ms[p]), 1) for p in change_points]}"
    )

    fcca_results = {}
    for interval_name, frames in intervals.items():
        result = run_fcca_interval(X_clean, frames, interval_name=interval_name)
        fcca_results[interval_name] = result
        n_communities = len(np.unique(result["consensus_labels"]))
        print(
            f"{interval_name}: frames {frames[0]}-{frames[-1]} | "
            f"ms={times_ms[frames[0]]:.1f}-{times_ms[frames[-1]]:.1f} | "
            f"graphs={n_subjects * len(frames)} | communities={n_communities} | "
            f"modularity={float(result['consensus_modularity']):.4f} | "
            f"labels={result['consensus_labels']}"
        )

    extra_fields = None
    if ho_result is not None and paper_like_metadata is not None:
        extra_fields = {
            "ho_rlsl_update_change_points": np.asarray(ho_result.change_points, dtype=int),
            "ho_rlsl_raw_change_points": np.asarray(ho_result.raw_change_points, dtype=int),
            "ho_rlsl_filtered_change_points": np.asarray(ho_result.filtered_change_points, dtype=int),
            "ho_rlsl_change_scores": np.asarray(ho_result.change_scores, dtype=float),
            "ho_rlsl_change_score_times": np.asarray(ho_result.change_score_times, dtype=int),
            "ho_rlsl_change_point_modes": np.asarray(ho_result.change_point_modes, dtype=int),
            "ho_rlsl_update_change_points_ms": np.asarray(
                [times_ms[p] for p in ho_result.change_points if 0 <= p < n_times],
                dtype=float,
            ),
            "paper_like_anchor_point": np.asarray(
                -1 if paper_like_metadata["anchor_point"] is None else paper_like_metadata["anchor_point"],
                dtype=int,
            ),
            "paper_like_anchor_ms": np.asarray(
                np.nan if paper_like_metadata["anchor_ms"] is None else paper_like_metadata["anchor_ms"],
                dtype=float,
            ),
        }
    save_fcca_results(
        fcca_results,
        change_points,
        change_point_method,
        output_path=output_path,
        extra_fields=extra_fields,
    )
    visualize_fcca_results(fcca_results)

    if ho_result is not None and paper_like_metadata is not None:
        report = {
            "input": str(args.input),
            "tensor_shape": [int(value) for value in X_clean.shape],
            "low_rank_method": low_rank_label,
            "ho_rlsl_config": ho_result.config,
            "ho_rlsl_change_points": [int(point) for point in ho_result.change_points],
            "ho_rlsl_raw_change_points": [int(point) for point in ho_result.raw_change_points],
            "ho_rlsl_change_points_ms": [
                float(times_ms[point]) for point in ho_result.change_points if 0 <= point < n_times
            ],
            "ho_rlsl_change_point_modes": [int(mode) for mode in ho_result.change_point_modes],
            "ho_rlsl_solver_params": ho_result.solver_params,
            "ho_rlsl_change_score_components": [
                dict(item) for item in ho_result.change_point_score_components.tolist()
            ],
            "ho_rlsl_mode_change_counts": {
                str(mode): int(count)
                for mode, count in ho_rlsl_mode_change_counts(ho_result.change_history).items()
            },
            "paper_like_intervals": {
                name: {
                    "start_frame": int(frames[0]),
                    "end_frame": int(frames[-1]),
                    "start_ms": float(times_ms[frames[0]]),
                    "end_ms": float(times_ms[frames[-1]]),
                    "n_frames": int(len(frames)),
                    "communities": int(len(np.unique(fcca_results[name]["consensus_labels"]))),
                    "modularity": float(fcca_results[name]["consensus_modularity"]),
                }
                for name, frames in intervals.items()
                if name in fcca_results
            },
            "paper_like_interval_selection": paper_like_metadata,
        }
        save_paper_like_report(report)
        print(f"Saved paper-like Ho-RLSL report: {HO_RLSL_PAPER_LIKE_REPORT_FILE}")

    print(f"Saved FCCA arrays: {output_path}")
    print(f"Saved figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
