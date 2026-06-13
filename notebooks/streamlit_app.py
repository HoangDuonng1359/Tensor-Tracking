import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "src/low_rank_extraction"))

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh
import streamlit as st

from preprocessing.config_1024 import config as prep_config

try:
    import tensorly as tl
    from tensorly.decomposition import tucker

    HAS_TENSORLY = True
    tl.set_backend("numpy")
except ImportError:
    tl = None
    tucker = None
    HAS_TENSORLY = False

try:
    from HoSVD import hosvd_low_rank

    HAS_HOSVD = True
except ImportError:
    hosvd_low_rank = None
    HAS_HOSVD = False

try:
    import mne

    HAS_MNE = True
except ImportError:
    mne = None
    HAS_MNE = False

try:
    import networkx as nx

    if not hasattr(nx, "from_numpy_matrix") and hasattr(nx, "from_numpy_array"):
        nx.from_numpy_matrix = nx.from_numpy_array
except ImportError:
    nx = None

try:
    from dyconnmap.graphs import threshold_omst_global_cost_efficiency

    HAS_DYCONNMAP = True
except ImportError:
    HAS_DYCONNMAP = False

try:
    import ruptures as rpt

    HAS_RUPTURES = True
except ImportError:
    rpt = None
    HAS_RUPTURES = False


ROOT = Path(__file__).resolve().parent
TENSOR_FILES = {
    "incorrect": prep_config.tensor_incorrect_file,
    "correct": prep_config.tensor_correct_file,
}
OUTPUT_DIR = Path("../outputs/fcca_results")
RESULTS_FILE = ROOT / OUTPUT_DIR / "fcca_results.npz"
HO_RLSL_RESULTS_FILE = ROOT / "../outputs/ho_rlsl_results.npz"
HO_RLSL_FCCA_RESULTS_FILE = ROOT / OUTPUT_DIR / "ho_rlsl_fcca_results.npz"
HOSVD_RESULTS_FILE = ROOT / "../outputs/fcca_results" / "hosvd_change_points.npz"
PRECOMPUTED_FCCA_ALL_FILE = ROOT / "../outputs/fcca_results/precomputed_fcca_all.npz"

# Per-algorithm pre-computed change-point files (ms values stored as dicts).
CP_ALGO_FILES = {
    "HOSVD":       ROOT / "../outputs/fcca_results/HOSVD/HOSVD_ChangePoints.npy",
    "HO-RLSL":     ROOT / "../outputs/fcca_results/HO_RLSL/HORLSL_ChangePoints.npy",
    "PELT":        ROOT / "../outputs/fcca_results/PELT/PELT_ChangePoints.npy",
    "DMD":         ROOT / "../outputs/fcca_results/DMD/DMD_ChangePoints.npy",
    "CP-Tracking": ROOT / "../outputs/fcca_results/CP_Tracking/CP_ChangePoints.npy",
}
# Key suffix used inside the .npy dict for each condition.
_CP_KEY_MAP = {"incorrect": "1024Hz_ERN", "correct": "1024Hz_CRN"}
# Pre-computed diagnostic plot images from each CP algorithm's optimal run.
PLOTS_DIR = ROOT / "../outputs/fcca_results/Plots"
CP_PLOT_FILES = {
    "HOSVD": {
        "incorrect": PLOTS_DIR / "HOSVD_1024Hz_ERN_Optimal_Plot.png",
        "correct":   PLOTS_DIR / "HOSVD_1024Hz_CRN_Optimal_Plot.png",
    },
    "HO-RLSL": {
        "incorrect": PLOTS_DIR / "HORLSL_1024Hz_ERN_Optimal_Plot.png",
        "correct":   PLOTS_DIR / "HORLSL_1024Hz_CRN_Optimal_Plot.png",
    },
    "PELT": {
        "incorrect": PLOTS_DIR / "PELT_1024Hz_ERN_Optimal_Plot.png",
        "correct":   PLOTS_DIR / "PELT_1024Hz_CRN_Optimal_Plot.png",
    },
    "DMD": {
        "incorrect": PLOTS_DIR / "DMD_1024Hz_ERN_Plot.png",
        "correct":   PLOTS_DIR / "DMD_1024Hz_CRN_Plot.png",
    },
    "CP-Tracking": {
        "incorrect": PLOTS_DIR / "CP_Tracking_1024Hz_ERN_Optimal.png",
        "correct":   PLOTS_DIR / "CP_Tracking_1024Hz_CRN_Optimal.png",
    },
}
FIGURE_DIR = ROOT / OUTPUT_DIR / "figures"
EPOCH_DIR = prep_config.paths.EPOCHS_DIR
RAW_BIDS_DIR = prep_config.paths.DATA_RAW
RAW_BIDS_DIR_ALIASES = (
    RAW_BIDS_DIR,
    ROOT / "../data/ERN Raw Data BIDS-Compatible",
    ROOT / "../data/ERN_Raw_Data_BIDS-Compatible",
)
INTERVAL_NAMES = ("pre_ern", "ern", "post_ern")
N_CHANGE_POINTS = prep_config.algo.N_CHANGE_POINTS
MIN_INTERVAL_FRAMES = prep_config.algo.MIN_INTERVAL_FRAMES
ADAPTIVE_RECURSIVE_MODULARITY_TOLERANCE = 0.06
CORE_INTERVAL_NAMES = {"ern", "crn"}
COMPACT_INTERVAL_NAMES = {"pre_ern", "post_ern", "pre_crn", "post_crn"}
PREFERRED_ERP_CHANNELS = ("FCz", "Cz", "FC1", "FC2")
ERP_BASELINE_MS = (-200.0, 0.0)
TUCKER_RANK = (10, 10, 10, 40)
RAW_CONDITION_EVENT_CODES = {
    "incorrect": ("112", "122", "211", "221"),
    "correct": ("111", "121", "212", "222"),
}
ERP_TMIN = -1.0
ERP_TMAX = 1.0


st.set_page_config(
    page_title="Brain Connectivity Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


@st.cache_data(show_spinner=False)
def load_tensor(path_str):
    return np.load(path_str)


@st.cache_data(show_spinner="Running Tucker low-rank decomposition...")
def load_analysis_tensor(path_str, low_rank_method):
    X = np.load(path_str)
    if low_rank_method == "Raw tensor":
        return X

    if low_rank_method == "HoSVD low-rank":
        if not HAS_HOSVD:
            raise RuntimeError("HoSVD.py could not be imported.")
        return hosvd_low_rank(X, ranks=(10, 10, 10))

    if low_rank_method == "Ho-RLSL saved low-rank":
        if Path(path_str).resolve() != TENSOR_FILES["incorrect"].resolve():
            raise ValueError("Saved Ho-RLSL low_rank is available only for the incorrect/ERN tensor.")
        with np.load(HO_RLSL_RESULTS_FILE, allow_pickle=True) as npz:
            if "low_rank" not in npz.files:
                raise ValueError(
                    "Ho-RLSL result file exists but does not contain low_rank. "
                    "Run ho_rlsl_cp.py to save the low rank tensor."
                )
            L = npz["low_rank"]
        if L.shape != X.shape:
            raise ValueError(f"Saved Ho-RLSL low_rank shape {L.shape} does not match tensor shape {X.shape}.")
        return L

    L, _, _, _ = tucker_low_rank_decomposition(X, rank=TUCKER_RANK)
    return L


@st.cache_data(show_spinner=False)
def has_saved_ho_rlsl_low_rank(path_str):
    path = Path(path_str)
    if path.resolve() != TENSOR_FILES["incorrect"].resolve():
        return False
    if not HO_RLSL_RESULTS_FILE.exists() or not path.exists():
        return False
    try:
        tensor_shape = np.load(path, mmap_mode="r").shape
        with np.load(HO_RLSL_RESULTS_FILE, allow_pickle=True) as npz:
            return "low_rank" in npz.files and npz["low_rank"].shape == tensor_shape
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def load_saved_ho_rlsl_change_points():
    if not HO_RLSL_RESULTS_FILE.exists():
        return None
    try:
        with np.load(HO_RLSL_RESULTS_FILE, allow_pickle=True) as npz:
            if "change_points" not in npz.files:
                return None
            change_points = np.asarray(npz["change_points"], dtype=int).ravel()
            raw_change_points = (
                np.asarray(npz["raw_change_points"], dtype=int).ravel()
                if "raw_change_points" in npz.files
                else change_points
            )
            filtered_change_points = (
                np.asarray(npz["filtered_change_points"], dtype=int).ravel()
                if "filtered_change_points" in npz.files
                else change_points
            )
            update_times = (
                np.asarray(npz["update_times"], dtype=int).ravel()
                if "update_times" in npz.files
                else np.asarray([], dtype=int)
            )
            change_scores = (
                np.asarray(npz["change_scores"], dtype=float)
                if "change_scores" in npz.files
                else np.zeros((0, 0), dtype=float)
            )
            change_score_times = (
                np.asarray(npz["change_score_times"], dtype=int).ravel()
                if "change_score_times" in npz.files
                else np.asarray([], dtype=int)
            )
            change_point_modes = (
                np.asarray(npz["change_point_modes"], dtype=int).ravel()
                if "change_point_modes" in npz.files
                else np.asarray([], dtype=int)
            )
            sparse_solver = (
                str(np.asarray(npz["sparse_solver"]).item())
                if "sparse_solver" in npz.files
                else "unknown"
            )
    except Exception:
        return None

    return {
        "change_points": change_points,
        "raw_change_points": raw_change_points,
        "filtered_change_points": filtered_change_points,
        "update_times": update_times,
        "change_scores": change_scores,
        "change_score_times": change_score_times,
        "change_point_modes": change_point_modes,
        "sparse_solver": sparse_solver,
        "method": "Ho-RLSL update rule",
        "source": str(HO_RLSL_RESULTS_FILE),
    }


@st.cache_data(show_spinner=False)
def load_saved_hosvd_change_points():
    if not HOSVD_RESULTS_FILE.exists():
        return None
    try:
        with np.load(HOSVD_RESULTS_FILE, allow_pickle=True) as npz:
            if "change_points" not in npz.files:
                return None
            change_points = np.asarray(npz["change_points"], dtype=int).ravel()
            method = (
                str(np.asarray(npz["method"]).item())
                if "method" in npz.files
                else "HoSVD"
            )
    except Exception:
        return None

    return {
        "change_points": change_points,
        "method": method,
        "source": str(HOSVD_RESULTS_FILE),
    }


@st.cache_data(show_spinner=False)
def load_fcca_results(path_str):
    path = Path(path_str)
    if not path.exists():
        return {}

    with np.load(path, allow_pickle=True) as npz:
        return {key: npz[key] for key in npz.files}


def find_electrodes_file():
    for root in RAW_BIDS_DIR_ALIASES:
        if not root.exists():
            continue
        matches = sorted(root.glob("sub-*/eeg/*_task-ERN_electrodes.tsv"))
        if matches:
            return matches[0]
    return None


@st.cache_data(show_spinner=False)
def load_electrode_layout(n_nodes):
    electrodes_file = find_electrodes_file()
    if electrodes_file is None:
        return None

    electrodes = pd.read_csv(electrodes_file, sep="\t")
    required = {"name", "x", "y"}
    if not required.issubset(electrodes.columns):
        return None

    electrodes = electrodes.iloc[:n_nodes].copy()
    if len(electrodes) != n_nodes:
        return None

    for column in ("x", "y"):
        electrodes[column] = pd.to_numeric(electrodes[column], errors="coerce")
    electrodes = electrodes.dropna(subset=["x", "y"])
    if len(electrodes) != n_nodes:
        return None

    # BIDS coordinates here use x for anterior-posterior and y for left-right.
    # Plot with left on the left side of the screen and frontal channels on top.
    plot_x = -electrodes["y"].to_numpy(dtype=float)
    plot_y = electrodes["x"].to_numpy(dtype=float)
    plot_x = plot_x / max(np.max(np.abs(plot_x)), 1.0)
    plot_y = plot_y / max(np.max(np.abs(plot_y)), 1.0)

    return {
        "names": electrodes["name"].astype(str).tolist(),
        "xy": np.column_stack([plot_x, plot_y]),
        "source": str(electrodes_file),
    }


@st.cache_data(show_spinner="Loading ERP traces with local pipeline...")
def load_analysis_erp_traces(conditions):
    if not HAS_MNE:
        return {}, "mne is not installed, so EEG ERP traces cannot be read."

    erp, err = load_erp_traces(str(EPOCH_DIR), conditions)
    if not erp:
        return {}, "No ERP traces could be extracted; falling back to tensor connectivity."

    return erp, None


def build_consensus_matrix(labels_all, n_nodes):
    W = np.zeros((n_nodes, n_nodes), dtype=float)

    for labels in labels_all:
        same = labels[:, None] == labels[None, :]
        W += same.astype(float)

    W /= max(len(labels_all), 1)
    return W


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


@st.cache_data(show_spinner="Running FCCA interval analysis...")
def run_fcca_interval_cached(path_str, frames_tuple, low_rank_method, interval_name):
    X = load_analysis_tensor(path_str, low_rank_method)
    return run_fcca_interval(X, np.asarray(frames_tuple, dtype=int), interval_name=interval_name)


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


def baseline_correct_trace(trace, times_ms, baseline_ms=ERP_BASELINE_MS):
    baseline_mask = (times_ms >= baseline_ms[0]) & (times_ms <= baseline_ms[1])
    if not baseline_mask.any():
        baseline_mask = times_ms < 0
    if baseline_mask.any():
        return trace - np.nanmean(trace[baseline_mask])
    return trace - np.nanmean(trace)


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

    # MNE stores EEG in volts. CSD data are not voltage, so do not convert them to uV.
    if channel_type == "eeg" and np.nanmax(np.abs(trace)) < 1e-3:
        trace = trace * 1e6

    trace = baseline_correct_trace(np.asarray(trace, dtype=float), times_ms)

    return {
        "trace": trace,
        "times_ms": times_ms,
        "n_epochs": len(epochs),
        "channel": channel_name,
        "channel_type": channel_type,
        "events": tuple(event_names) if event_names else ("all",),
    }


@st.cache_data(show_spinner=False)
def load_erp_traces(epoch_dir_str, conditions):
    if not HAS_MNE:
        return {}, "mne is not installed, so FIF EEG epochs cannot be read."

    epoch_dir = Path(epoch_dir_str)
    fif_files = sorted(epoch_dir.glob("*-epo.fif"))
    if not fif_files:
        return {}, f"No *-epo.fif files were found in {epoch_dir}."

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
        except Exception:
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
            }

    if not erp:
        return {}, "Epoch files were found, but no condition waveforms could be extracted."

    return erp, None


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

    L = np.diag(A.sum(axis=1)) - A
    _, eigenvectors = eigh(L)
    fvec = eigenvectors[:, 1]
    labels = (fvec > 0).astype(int)

    if len(np.unique(labels)) < 2:
        labels = (fvec > np.median(fvec)).astype(int)

    if len(np.unique(labels)) < 2:
        order = np.argsort(fvec)
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

    if nx is None:
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


def graph_feature_matrix(X):
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
            min_start = (segment_idx - 1) * min_size
            max_start = end - min_size
            for start in range(min_start, max_start + 1):
                cost = dp[segment_idx - 1, start] + segment_sse(
                    prefix_sum,
                    prefix_sq_sum,
                    start,
                    end,
                )
                if cost < dp[segment_idx, end]:
                    dp[segment_idx, end] = cost
                    previous[segment_idx, end] = start

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


def make_intervals_from_change_points(n_times, change_points, names=INTERVAL_NAMES):
    boundaries = [0]
    boundaries.extend(int(point) for point in sorted(change_points) if 0 < point < n_times)
    boundaries.append(n_times)

    if len(boundaries) != len(names) + 1:
        splits = np.array_split(np.arange(n_times), len(names))
        return {
            name: frames
            for name, frames in zip(names, splits)
            if len(frames) > 0
        }

    return {
        name: np.arange(start, end)
        for name, start, end in zip(names, boundaries[:-1], boundaries[1:])
        if end > start
    }


def detect_change_points(X, n_bkps=N_CHANGE_POINTS, min_size=MIN_INTERVAL_FRAMES):
    features = graph_feature_matrix(X)

    if HAS_RUPTURES:
        try:
            bkps = rpt.Binseg(model="l2", min_size=min_size).fit(features).predict(n_bkps=n_bkps)
            return sorted(int(bkp) for bkp in bkps[:-1]), "ruptures Binseg"
        except Exception as exc:
            print(f"[WARN] ruptures change-point detection failed: {exc}")

    return detect_change_points_exact(features, n_bkps=n_bkps, min_size=min_size), "exact SSE"


def make_change_point_intervals(X, names=INTERVAL_NAMES):
    n_times = X.shape[-1]
    change_points, method = detect_change_points(X, n_bkps=len(names) - 1)
    intervals = make_intervals_from_change_points(n_times, change_points, names)
    return intervals, change_points, method


def make_paper_like_ho_rlsl_intervals(
    n_times,
    change_points,
    names=INTERVAL_NAMES,
    target_window_ms=(25.0, 75.0),
    anchor_ms=50.0,
    start_ms=-1000.0,
    end_ms=1000.0,
    **kwargs
):
    """Build pre/ERN/post intervals from Ho-RLSL update change points.

    Paper-like flow: Ho-RLSL update change points define intervals; a separate
    segmentation detector is not used. For ERN, choose the change point closest
    to the expected ERN latency, then use its nearest neighboring Ho-RLSL
    change points as interval boundaries.
    """
    times_ms = frames_to_ms(n_times, start_ms=start_ms, end_ms=end_ms)
    points = np.asarray(change_points, dtype=int).ravel()
    points = np.unique(points[(points > 0) & (points < n_times)])

    if points.size < 2:
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
        boundaries = [int(before[-1]), int(after[0])]
    elif before.size:
        boundaries = [int(before[-1]), min(n_times - 1, anchor_point + max(1, anchor_point - int(before[-1])))]
    elif after.size:
        boundaries = [max(1, anchor_point - max(1, int(after[0]) - anchor_point)), int(after[0])]
    else:
        boundaries = []

    boundaries = sorted({point for point in boundaries if 0 < point < n_times})
    intervals = make_intervals_from_change_points(n_times, boundaries, names)
    method = (
        "Ho-RLSL update rule paper-like intervals "
        f"({anchor_source}; anchor {times_ms[anchor_point]:.1f} ms)"
    )
    metadata = {
        "method": method,
        "anchor_point": anchor_point,
        "anchor_ms": float(times_ms[anchor_point]),
        "anchor_source": anchor_source,
        "boundary_points": [int(point) for point in boundaries],
        "boundary_ms": [float(times_ms[point]) for point in boundaries],
        "all_change_points": points.astype(int).tolist(),
        "all_change_points_ms": [float(times_ms[point]) for point in points],
        "target_window_ms": [float(target_window_ms[0]), float(target_window_ms[1])],
    }
    return intervals, boundaries, metadata


@st.cache_data(show_spinner=False)
def make_change_point_intervals_cached(X):
    intervals, change_points, method = make_change_point_intervals(X)
    return intervals, change_points, method


def display_change_points_for_timecourse(change_point_tensor, low_rank_method, condition):
    intervals, detected_points, detected_method = make_change_point_intervals_cached(change_point_tensor)
    if low_rank_method == "HoSVD low-rank" and condition == "incorrect":
        saved = load_saved_hosvd_change_points()
        if saved is not None:
            n_times = change_point_tensor.shape[-1]
            points = saved["change_points"]
            points = points[(points >= 0) & (points < n_times)]
            intervals = make_intervals_from_change_points(n_times, points)
            return intervals, points.astype(int).tolist(), saved["method"]
    if low_rank_method == "Ho-RLSL saved low-rank" and condition == "incorrect":
        saved = load_saved_ho_rlsl_change_points()
        if saved is not None:
            n_times = change_point_tensor.shape[-1]
            points = saved["filtered_change_points"]
            points = points[(points >= 0) & (points < n_times)]
            intervals, _, metadata = make_paper_like_ho_rlsl_intervals(n_times, points)
            raw_points = saved["raw_change_points"]
            raw_points = raw_points[(raw_points >= 0) & (raw_points < n_times)]
            method = f"{metadata['method']} ({saved['sparse_solver']})"
            return (
                intervals,
                {
                    "filtered": points.astype(int).tolist(),
                    "raw": raw_points.astype(int).tolist(),
                    "scores": saved["change_scores"],
                    "score_times": saved["change_score_times"],
                    "modes": saved["change_point_modes"],
                },
                method,
            )
    return intervals, detected_points, detected_method


def normalize_change_point_display(change_points):
    if isinstance(change_points, dict):
        return change_points
    points = [int(point) for point in change_points]
    return {
        "filtered": points,
        "raw": points,
        "scores": np.zeros((0, 0), dtype=float),
        "score_times": np.asarray([], dtype=int),
        "modes": np.asarray([], dtype=int),
    }


def frames_to_ms(n_times, start_ms=-1000.0, end_ms=1000.0):
    return np.linspace(start_ms, end_ms, n_times)


def channel_labels(n_nodes):
    layout = load_electrode_layout(n_nodes)
    if layout is not None:
        return layout["names"]
    return [f"Ch{idx:02d}" for idx in range(1, n_nodes + 1)]


def mean_connectivity_trace(X):
    _, n_nodes, _, _ = X.shape
    upper = np.triu_indices(n_nodes, k=1)
    trace = X[:, upper[0], upper[1], :].mean(axis=(0, 1))
    std = trace.std()
    if std > 0:
        return (trace - trace.mean()) / std
    return trace - trace.mean()


def smooth_trace(trace, window):
    if window <= 1 or len(trace) < window:
        return trace

    if window % 2 == 0:
        window += 1

    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(trace, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def interval_mean_adjacency(X, frames):
    A = X[:, :, :, frames].mean(axis=(0, 3))
    return sanitize_adjacency(A)


def threshold_adjacency(A, mode, edge_percent):
    A = sanitize_adjacency(A)

    if mode == "dyconnmap OMST" and HAS_DYCONNMAP:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="divide by zero encountered in divide",
                    category=RuntimeWarning,
                )
                _, thresholded, *_ = threshold_omst_global_cost_efficiency(A)
            return sanitize_adjacency(thresholded), "dyconnmap OMST"
        except Exception as exc:
            st.warning(f"dyconnmap threshold failed; using top edges instead. {exc}")

    upper_idx = np.triu_indices_from(A, k=1)
    weights = A[upper_idx]
    positive = weights[weights > 0]
    thresholded = np.zeros_like(A)

    if len(positive) == 0:
        return thresholded, "empty graph"

    keep_fraction = max(edge_percent, 1) / 100.0
    cutoff = np.quantile(positive, 1 - keep_fraction)
    thresholded[A >= cutoff] = A[A >= cutoff]
    thresholded = sanitize_adjacency(thresholded)
    return thresholded, f"top {edge_percent:.0f}% edges"


def get_interval_labels(condition, intervals):
    if condition == "incorrect":
        return ("PRE-ERN", "ERN", "POST-ERN")
    if condition == "correct":
        return ("PRE-CRN", "CRN", "POST-CRN")
    return ("PRE", "EVENT", "POST")


def draw_interval_labels(ax, intervals, times_ms, condition):
    interval_values = list(intervals.values())
    for frames in interval_values[1:]:
        ax.axvline(times_ms[frames[0]], color="#6075ff", linewidth=1.4, alpha=0.85)

    labels = get_interval_labels(condition, intervals)
    label_frames = [
        intervals[name]
        for name in INTERVAL_NAMES
        if name in intervals
    ]

    y_min, y_max = ax.get_ylim()
    y_pos = y_max - 0.12 * (y_max - y_min)
    for text, frames in zip(labels, label_frames):
        center = int(round((frames[0] + frames[-1]) / 2))
        ax.text(
            times_ms[center],
            y_pos,
            text,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )


def plot_timecourses(tensors, smooth_window, erp_traces=None, low_rank_method="Tucker low-rank"):
    fig, axes = plt.subplots(len(tensors), 1, figsize=(11, 5.8), sharex=True)
    axes = np.atleast_1d(axes)

    for idx, (condition, X) in enumerate(tensors.items()):
        n_times = X.shape[-1]
        tensor_times_ms = frames_to_ms(n_times)
        condition_low_rank_method = low_rank_method
        if low_rank_method == "Ho-RLSL saved low-rank" and condition != "incorrect":
            condition_low_rank_method = "Raw tensor"
        change_point_tensor = load_analysis_tensor(str(TENSOR_FILES[condition]), condition_low_rank_method)
        intervals, change_points, change_point_method = display_change_points_for_timecourse(
            change_point_tensor,
            condition_low_rank_method,
            condition,
        )
        cp_display = normalize_change_point_display(change_points)
        erp = erp_traces.get(condition) if erp_traces else None

        if erp is not None:
            times_ms = erp["times_ms"]
            trace = smooth_trace(erp["trace"], smooth_window)
            ylabel = "Amplitude (uV)" if erp["channel_type"] == "eeg" else "Baseline-corrected CSD"
            event_name = "ERN" if condition == "incorrect" else "CRN"
            title = (
                f"({chr(97 + idx)}) Average {event_name} waveform "
                f"({erp['subjects']} subjects, {erp['n_epochs']} epochs, {erp['channel']}, {erp['channel_type']})"
            )
        else:
            times_ms = tensor_times_ms
            trace = smooth_trace(mean_connectivity_trace(X), smooth_window)
            ylabel = "z(mean PLV)"
            title = (
                f"({chr(97 + idx)}) {condition.capitalize()} tensor: "
                f"{X.shape[0]} subjects, {X.shape[1]} nodes, {n_times} windows"
            )

        ax = axes[idx]
        ax.plot(times_ms, trace, color="red", linewidth=1.8)
        ax.axhline(0, color="0.65", linewidth=0.8)
        ax.set_xlim(-1000, 1000)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        draw_interval_labels(ax, intervals, tensor_times_ms, condition)
        for point in cp_display["raw"]:
            ax.axvline(
                tensor_times_ms[point],
                color="0.45",
                linewidth=0.8,
                alpha=0.28,
                linestyle=":",
            )
        for point in cp_display["filtered"]:
            ax.axvline(
                tensor_times_ms[point],
                color="#d62728",
                linewidth=1.0,
                alpha=0.65,
                linestyle="--",
            )
            ax.text(
                tensor_times_ms[point],
                ax.get_ylim()[1],
                f"{tensor_times_ms[point]:.0f}ms",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        # ax.text(
        #     0.99,
        #     0.08,
        #     f"{change_point_method}: frames {cp_display['filtered']}",
        #     transform=ax.transAxes,
        #     ha="right",
        #     va="bottom",
        #     fontsize=8,
        #     color="0.25",
        # )
        ax.tick_params(direction="in", top=True, right=True)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    return fig


def plot_heatmap(A, labels=None, sort_by_community=True, title="Connectivity Matrix"):
    A = sanitize_adjacency(A)
    order = np.arange(A.shape[0])
    if labels is not None and sort_by_community:
        order = np.argsort(labels)
        A = A[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    im = ax.imshow(A, cmap="viridis", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Node")
    ax.set_ylabel("Node")
    ax.set_xticks(np.arange(A.shape[0]))
    ax.set_yticks(np.arange(A.shape[0]))
    ax.set_xticklabels([str(idx + 1) for idx in order], rotation=90, fontsize=7)
    ax.set_yticklabels([str(idx + 1) for idx in order], fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def draw_head_outline(ax, xy):
    center = xy.mean(axis=0)
    radius = max(np.max(np.linalg.norm(xy - center, axis=1)) * 1.08, 1.0)
    theta = np.linspace(0, 2 * np.pi, 240)
    ax.plot(
        center[0] + radius * np.cos(theta),
        center[1] + radius * np.sin(theta),
        color="0.35",
        linewidth=1.0,
        zorder=0,
    )
    ax.plot(
        [center[0] - 0.10 * radius, center[0], center[0] + 0.10 * radius],
        [center[1] + radius, center[1] + 1.12 * radius, center[1] + radius],
        color="0.35",
        linewidth=1.0,
        zorder=0,
    )


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


def community_color_map(labels):
    if labels is None:
        return {}
    labels = np.asarray(labels, dtype=int)
    communities = sorted(int(label) for label in np.unique(labels))
    return {
        community: COMMUNITY_PALETTE[idx % len(COMMUNITY_PALETTE)]
        for idx, community in enumerate(communities)
    }


def community_node_colors(labels, n_nodes):
    if labels is None:
        return ["#ff8c1a"] * n_nodes
    labels = np.asarray(labels, dtype=int)
    colors = community_color_map(labels)
    return [colors[int(label)] for label in labels]


def community_edge_style(i, j, labels):
    if labels is None:
        return "0.35", 0.35, 1.0

    labels = np.asarray(labels, dtype=int)
    color_map = community_color_map(labels)
    if labels[i] == labels[j]:
        return color_map[int(labels[i])], 0.58, 1.35
    return "0.70", 0.22, 0.70


def community_legend_handles(labels):
    colors = community_color_map(labels)
    return [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=8,
            linewidth=0,
            label=f"community {community}",
        )
        for community, color in colors.items()
    ]


def add_network_legend(ax, labels):
    handles = community_legend_handles(labels)
    if not handles:
        return
    ax.legend(
        handles=handles,
        title="communities",
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0),
        frameon=False,
        fontsize=8,
        title_fontsize=8,
    )


def plot_electrode_network(A, labels, threshold_mode, edge_percent, show_node_labels):
    A_thr, threshold_label = threshold_adjacency(A, threshold_mode, edge_percent)
    n_nodes = A_thr.shape[0]
    layout = load_electrode_layout(n_nodes)
    if layout is None:
        return plot_circular_network(A, labels, threshold_mode, edge_percent, show_node_labels)

    xy = layout["xy"]
    names = layout["names"]
    node_colors = community_node_colors(labels, n_nodes)

    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    draw_head_outline(ax, xy)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            weight = A_thr[i, j]
            if weight <= 0:
                continue
            color, alpha, linewidth = community_edge_style(i, j, labels)
            ax.plot(
                [xy[i, 0], xy[j, 0]],
                [xy[i, 1], xy[j, 1]],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                zorder=1,
            )

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=node_colors,
        s=76,
        edgecolors="white",
        linewidths=1.0,
        zorder=3,
    )

    if show_node_labels:
        for idx, (x_pos, y_pos) in enumerate(xy):
            ax.text(
                x_pos,
                y_pos,
                names[idx],
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="black",
                zorder=4,
            )

    ax.set_title(f"Scalp Connectivity Graph\n{threshold_label}")
    add_network_legend(ax, labels)
    ax.set_aspect("equal")
    ax.axis("off")
    pad_x = max((xy[:, 0].max() - xy[:, 0].min()) * 0.12, 0.12)
    pad_y = max((xy[:, 1].max() - xy[:, 1].min()) * 0.15, 0.15)
    ax.set_xlim(xy[:, 0].min() - pad_x, xy[:, 0].max() + pad_x)
    ax.set_ylim(xy[:, 1].min() - pad_y, xy[:, 1].max() + pad_y)
    fig.tight_layout()
    return fig, A_thr


def plot_circular_network(A, labels, threshold_mode, edge_percent, show_node_labels):
    A_thr, threshold_label = threshold_adjacency(A, threshold_mode, edge_percent)
    n_nodes = A_thr.shape[0]
    theta = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    xy = np.column_stack([np.cos(theta), np.sin(theta)])

    fig, ax = plt.subplots(figsize=(6.6, 6.2))

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            weight = A_thr[i, j]
            if weight <= 0:
                continue
            color, alpha, linewidth = community_edge_style(i, j, labels)
            ax.plot(
                [xy[i, 0], xy[j, 0]],
                [xy[i, 1], xy[j, 1]],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                zorder=1,
            )

    node_colors = community_node_colors(labels, n_nodes)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=node_colors,
        s=110,
        edgecolors="white",
        linewidths=1.0,
        zorder=3,
    )

    names = channel_labels(n_nodes)
    if show_node_labels:
        for idx, (x_pos, y_pos) in enumerate(xy):
            ax.text(
                x_pos * 1.14,
                y_pos * 1.14,
                names[idx],
                ha="center",
                va="center",
                fontsize=7,
            )

    ax.set_title(f"Circular Connectivity Graph\n{threshold_label}")
    add_network_legend(ax, labels)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig, A_thr


def node_metric_table(A, labels, A_thr):
    names = channel_labels(A.shape[0])
    return pd.DataFrame(
        {
            "channel": names,
            "community": labels.astype(int),
            "strength": A.sum(axis=1),
            "thresholded_degree": (A_thr > 0).sum(axis=1),
            "thresholded_strength": A_thr.sum(axis=1),
        }
    )


def load_available_tensors():
    tensors = {}
    missing = []
    for condition, path in TENSOR_FILES.items():
        if path.exists():
            tensors[condition] = load_tensor(str(path))
        else:
            missing.append(path)
    return tensors, missing


def get_fcca_interval_matrix(results, interval_name, source):
    if source == "FCCA consensus matrix":
        matrix_key = f"{interval_name}_consensus_matrix"
    elif source == "FCCA mean adjacency":
        matrix_key = f"{interval_name}_mean_adjacency"
    else:
        return None, None

    label_key = f"{interval_name}_consensus_labels"
    matrix = results.get(matrix_key)
    labels = results.get(label_key)
    return matrix, labels


def saved_fcca_matches_current_intervals(results, low_rank_method):
    if low_rank_method != "Ho-RLSL saved low-rank":
        return True
    method = results.get("change_point_method")
    if method is None:
        return False
    method_text = str(np.asarray(method).item())
    return method_text.startswith("Ho-RLSL update rule paper-like intervals")


def main():
    st.title("Brain Connectivity Explorer")
    st.caption("Interactive views for 4D connectivity tensors and interval-wise FCCA outputs.")

    tensors, missing = load_available_tensors()
    results = load_fcca_results(str(RESULTS_FILE))
    ho_fcca_results = load_fcca_results(str(HO_RLSL_FCCA_RESULTS_FILE))
    erp_traces, erp_error = load_analysis_erp_traces(tuple(tensors.keys()))

    if not tensors:
        st.error("No tensor files were found in tensor_4d/.")
        st.stop()

    with st.sidebar:
        st.header("Controls")
        condition = st.selectbox("Condition", list(tensors.keys()), index=0)
        low_rank_options = ["Tucker low-rank", "Raw tensor"] if HAS_TENSORLY else ["Raw tensor"]
        if HAS_HOSVD:
            insert_at = 1 if "Raw tensor" in low_rank_options else len(low_rank_options)
            low_rank_options.insert(insert_at, "HoSVD low-rank")
        if has_saved_ho_rlsl_low_rank(str(TENSOR_FILES[condition])):
            low_rank_options.insert(1, "Ho-RLSL saved low-rank")
        default_low_rank = "Ho-RLSL saved low-rank" if "Ho-RLSL saved low-rank" in low_rank_options else "Raw tensor"
        low_rank_method = st.selectbox(
            "Analysis tensor",
            low_rank_options,
            index=low_rank_options.index(default_low_rank),
            help=(
                "HoSVD is computed live and cached. Ho-RLSL appears only when "
                "fcca_results/ho_rlsl_results.npz contains a matching low_rank array."
            ),
        )
        cp_method = st.selectbox(
            "Change point method",
            ("Live detection", "HOSVD", "HO-RLSL", "PELT", "DMD", "CP-Tracking"),
            index=2,
            help="Select the change point detection algorithm to partition the timecourse."
        )
        analysis_tensor = load_analysis_tensor(str(TENSOR_FILES[condition]), low_rank_method)
        n_times = analysis_tensor.shape[-1]
        
        # Resolve change points based on cp_method
        if cp_method == "Live detection":
            intervals, change_points, change_point_method = make_change_point_intervals_cached(analysis_tensor)
        elif cp_method == "HOSVD":
            saved = load_saved_hosvd_change_points()
            if saved is not None:
                points = saved["change_points"]
                points = points[(points >= 0) & (points < n_times)]
                intervals = make_intervals_from_change_points(n_times, points)
                change_points = points.astype(int).tolist()
                change_point_method = saved["method"]
            else:
                intervals, change_points, change_point_method = make_change_point_intervals_cached(analysis_tensor)
        elif cp_method == "HO-RLSL":
            saved = load_saved_ho_rlsl_change_points()
            if saved is not None:
                points = saved["filtered_change_points"]
                points = points[(points >= 0) & (points < n_times)]
                intervals, _, metadata = make_paper_like_ho_rlsl_intervals(n_times, points)
                change_points = {
                    "filtered": points.astype(int).tolist(),
                    "raw": saved["raw_change_points"].astype(int).tolist(),
                    "scores": saved["change_scores"],
                    "score_times": saved["change_score_times"],
                    "modes": saved["change_point_modes"],
                }
                change_point_method = f"{metadata['method']} ({saved['sparse_solver']})"
            else:
                intervals, change_points, change_point_method = make_change_point_intervals_cached(analysis_tensor)
        else:
            # PELT, DMD, CP-Tracking
            algo_frames = load_algo_change_points(cp_method, condition, n_times)
            if algo_frames:
                intervals = make_intervals_from_change_points(n_times, algo_frames)
                change_points = algo_frames
                change_point_method = f"Precomputed {cp_method}"
            else:
                intervals, change_points, change_point_method = make_change_point_intervals_cached(analysis_tensor)

        sidebar_cp_display = normalize_change_point_display(change_points)
        interval_name = st.selectbox("Interval", list(intervals.keys()), index=1)
        matrix_source = st.radio(
            "Matrix source",
            ("FCCA consensus matrix", "FCCA mean adjacency", "Tensor interval mean"),
            index=0,
        )
        threshold_modes = ["Top weighted edges"]
        if HAS_DYCONNMAP:
            threshold_modes.insert(0, "dyconnmap OMST")
        threshold_mode = st.selectbox("Graph threshold", threshold_modes)
        
        # Hardcode graph_layout to 'Scalp electrode map' and remove selection
        graph_layout = "Scalp electrode map"
        
        edge_percent = st.slider("Top-edge percentage", 5, 100, 20, 5)
        smooth_window = st.slider("Time-course smoothing window", 1, 31, 7, 2)
        sort_heatmap = st.checkbox("Sort heatmap by community", value=True)
        show_node_labels = st.checkbox("Show node labels", value=True)

    selected_tensor = analysis_tensor
    selected_frames = intervals[interval_name]
    active_fcca_results = ho_fcca_results if low_rank_method == "Ho-RLSL saved low-rank" else results

    tabs = st.tabs(["Time Course", "Interval Network", "Subject/Time Graph", "Diagnostics", "Data"])

    with tabs[0]:
        st.subheader("Average ERN/CRN Waveform with Connectivity Change Points")
        st.pyplot(
            plot_timecourses(tensors, smooth_window, erp_traces, low_rank_method=low_rank_method),
            clear_figure=True,
        )
        if erp_error:
            st.warning(
                f"{erp_error} Falling back to z-scored mean connectivity for the red trace."
            )
        else:
            st.info(
                "The red trace uses the same ERP loading pipeline as tensor_de_v2.py "
                "(raw BIDS first, then FIF fallback). Blue lines are selected interval "
                "boundaries; red dashed lines are displayed change points."
            )

    with tabs[1]:
        st.subheader("Interval Connectivity")

        matrix = None
        labels = None
        modularity_score = None
        community_method = None
        if matrix_source != "Tensor interval mean":
            use_saved_fcca = condition == "incorrect" and low_rank_method in {
                "Tucker low-rank",
                "Ho-RLSL saved low-rank",
            } and saved_fcca_matches_current_intervals(active_fcca_results, low_rank_method)
            if use_saved_fcca:
                matrix, labels = get_fcca_interval_matrix(active_fcca_results, interval_name, matrix_source)

            if matrix is None:
                interval_result = run_fcca_interval_cached(
                    str(TENSOR_FILES[condition]),
                    tuple(int(frame) for frame in selected_frames),
                    low_rank_method,
                    interval_name,
                )
                matrix = (
                    interval_result["consensus_matrix"]
                    if matrix_source == "FCCA consensus matrix"
                    else interval_result["mean_adjacency"]
                )
                labels = interval_result["consensus_labels"]
                modularity_score = float(interval_result["consensus_modularity"])
                community_method = str(np.asarray(interval_result["community_method"]).item())
                st.info(
                    "FCCA was computed in Streamlit with the same run_fcca_interval() "
                    "function used by tensor_de_v2.py."
                )

        if matrix is None:
            matrix = interval_mean_adjacency(selected_tensor, selected_frames)
            labels, modularity_score, community_method = modularity_communities(
                matrix,
                clustering_profile=clustering_profile_for_interval(interval_name),
            )
        else:
            matrix = sanitize_adjacency(matrix)
            if labels is not None:
                labels = labels.astype(int)
                if modularity_score is None:
                    score_key = f"{interval_name}_consensus_modularity"
                    modularity_score = active_fcca_results.get(score_key, np.asarray(np.nan))
                    modularity_score = float(np.asarray(modularity_score))
                    method_key = f"{interval_name}_community_method"
                    community_method = str(np.asarray(active_fcca_results.get(method_key, "saved FCCA labels")).item())
            else:
                labels, modularity_score, community_method = modularity_communities(
                    matrix,
                    clustering_profile=clustering_profile_for_interval(interval_name),
                )

        col_a, col_b = st.columns(2)
        with col_a:
            st.pyplot(
                plot_heatmap(
                    matrix,
                    labels=labels,
                    sort_by_community=sort_heatmap,
                    title=f"{interval_name} | {matrix_source}",
                ),
                clear_figure=True,
            )
        with col_b:
            plot_network = plot_electrode_network if graph_layout == "Scalp electrode map" else plot_circular_network
            network_fig, A_thr = plot_network(
                matrix,
                labels,
                threshold_mode,
                edge_percent,
                show_node_labels,
            )
            st.pyplot(network_fig, clear_figure=True)

        metric_cols = st.columns(6)
        metric_cols[0].metric("Frames", f"{selected_frames[0]}-{selected_frames[-1]}")
        metric_cols[1].metric("Graphs in interval", selected_tensor.shape[0] * len(selected_frames))
        metric_cols[2].metric("Mean weight", f"{matrix[np.triu_indices_from(matrix, 1)].mean():.4f}")
        metric_cols[3].metric("Thresholded edges", int(np.count_nonzero(np.triu(A_thr, 1))))
        metric_cols[4].metric("Communities", len(np.unique(labels)))
        metric_cols[5].metric(
            "Modularity",
            "n/a" if not np.isfinite(modularity_score) else f"{modularity_score:.3f}",
            help=community_method,
        )

        st.dataframe(
            node_metric_table(matrix, labels, A_thr).sort_values(
                ["community", "thresholded_strength"], ascending=[True, False]
            ),
            use_container_width=True,
            hide_index=True,
        )

    with tabs[2]:
        st.subheader("Single Subject and Time Window")
        max_subject = selected_tensor.shape[0] - 1
        max_time = selected_tensor.shape[-1] - 1
        col_controls, col_time = st.columns([1, 2])

        with col_controls:
            subject_idx = st.slider("Subject index", 0, max_subject, 0)
            time_idx = st.slider("Time window", 0, max_time, int(selected_frames[0]))
            time_ms = frames_to_ms(selected_tensor.shape[-1])[time_idx]
            st.metric("Approx. time", f"{time_ms:.1f} ms")

        A_subject = sanitize_adjacency(selected_tensor[subject_idx, :, :, time_idx])
        subject_labels, subject_modularity, subject_method = modularity_communities(
            A_subject,
            clustering_profile=clustering_profile_for_interval(interval_name),
        )

        with col_time:
            st.pyplot(
                plot_heatmap(
                    A_subject,
                    labels=subject_labels,
                    sort_by_community=sort_heatmap,
                    title=f"{condition} | subject {subject_idx + 1} | window {time_idx}",
                ),
                clear_figure=True,
            )

        plot_network = plot_electrode_network if graph_layout == "Scalp electrode map" else plot_circular_network
        graph_fig, A_subject_thr = plot_network(
            A_subject,
            subject_labels,
            threshold_mode,
            edge_percent,
            show_node_labels,
        )
        st.pyplot(graph_fig, clear_figure=True)
        st.caption(
            f"Community split: {subject_method}; "
            f"modularity={'n/a' if not np.isfinite(subject_modularity) else f'{subject_modularity:.3f}'}; "
            f"communities={len(np.unique(subject_labels))}"
        )
        st.dataframe(
            node_metric_table(A_subject, subject_labels, A_subject_thr),
            use_container_width=True,
            hide_index=True,
        )

    with tabs[3]:
        st.subheader(f"Change Point Diagnostics: {cp_method}")
        plot_path = CP_PLOT_FILES.get(cp_method, {}).get(condition)
        if plot_path and plot_path.exists():
            st.image(str(plot_path), caption=f"{cp_method} optimal diagnostics ({condition})")
        else:
            st.info(f"No pre-computed optimal diagnostic plot is available for {cp_method} under {condition} condition.")

    with tabs[4]:
        st.subheader("Loaded Data")
        tensor_table = pd.DataFrame(
            [
                {
                    "condition": name,
                    "path": str(TENSOR_FILES[name]),
                    "shape": " x ".join(map(str, tensor.shape)),
                }
                for name, tensor in tensors.items()
            ]
        )
        st.dataframe(tensor_table, use_container_width=True, hide_index=True)

        if missing:
            st.warning("Missing tensor files: " + ", ".join(str(path) for path in missing))

        st.write("Tucker FCCA results file:", str(RESULTS_FILE))
        st.write("Ho-RLSL results file:", str(HO_RLSL_RESULTS_FILE))
        st.write("Ho-RLSL FCCA results file:", str(HO_RLSL_FCCA_RESULTS_FILE))
        st.write("HoSVD change-points file:", str(HOSVD_RESULTS_FILE))
        st.write("Active FCCA keys:", sorted(active_fcca_results.keys()) if active_fcca_results else "No active FCCA result file found.")
        st.write("EEG epoch directory:", str(EPOCH_DIR))
        st.write("Raw BIDS EEG directory:", str(RAW_BIDS_DIR))
        st.write("Analysis tensor:", low_rank_method)
        st.write("Change-point method:", change_point_method)
        if low_rank_method == "Ho-RLSL saved low-rank":
            st.write("Filtered Ho-RLSL change points:", sidebar_cp_display["filtered"])
            st.write("Raw Ho-RLSL change points:", sidebar_cp_display["raw"])
            score_times = np.asarray(sidebar_cp_display["score_times"], dtype=int)
            score_values = np.asarray(sidebar_cp_display["scores"], dtype=float)
            modes = np.asarray(sidebar_cp_display["modes"], dtype=int)
            if score_times.size and score_values.ndim == 2 and score_values.shape[0] == score_times.size:
                time_axis = frames_to_ms(n_times)
                filtered_set = set(int(point) for point in sidebar_cp_display["filtered"])
                raw_set = set(int(point) for point in sidebar_cp_display["raw"])
                rows = []
                for idx, frame in enumerate(score_times):
                    if frame < 0 or frame >= n_times:
                        continue
                    rows.append(
                        {
                            "frame": int(frame),
                            "time_ms": float(time_axis[frame]),
                            "score": float(score_values[idx, 0]),
                            "raw_cp": int(frame) in raw_set,
                            "filtered_cp": int(frame) in filtered_set,
                            "is_hit_25_75ms": 25.0 <= float(time_axis[frame]) <= 75.0,
                        }
                    )
                if rows:
                    score_table = pd.DataFrame(rows)
                    if modes.size == len(sidebar_cp_display["filtered"]):
                        mode_by_frame = {
                            int(frame): int(mode)
                            for frame, mode in zip(sidebar_cp_display["filtered"], modes)
                        }
                        score_table["mode"] = score_table["frame"].map(mode_by_frame)
                    st.dataframe(score_table, use_container_width=True, hide_index=True)
        if erp_traces:
            erp_table = pd.DataFrame(
                [
                    {
                        "condition": condition_name,
                        "subjects": erp["subjects"],
                        "epochs": erp["n_epochs"],
                        "channel": erp["channel"],
                        "channel type": erp["channel_type"],
                        "events used": ", ".join(erp["events"]),
                    }
                    for condition_name, erp in erp_traces.items()
                ]
            )
            st.dataframe(erp_table, use_container_width=True, hide_index=True)
        elif erp_error:
            st.warning(erp_error)

        existing_figure = FIGURE_DIR / "incorrect_correct_tensor_timecourses.png"
        if existing_figure.exists():
            st.image(str(existing_figure), caption="Saved paper-style time-course figure")


if __name__ == "__main__":
    main()
