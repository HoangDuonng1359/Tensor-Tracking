from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
from scipy.linalg import eigh
from tensorly.decomposition import tucker

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


tl.set_backend("numpy")

INPUT_FILE = Path("tensor_4d/tensor_incorrect_4d.npy")
CONDITION_FILES = {
    "incorrect": Path("tensor_4d/tensor_incorrect_4d.npy"),
    "correct": Path("tensor_4d/tensor_correct_4d.npy"),
}
OUTPUT_DIR = Path("fcca_results")
FIGURE_DIR = OUTPUT_DIR / "figures"
EPOCH_DIR = Path("01_initial_epochs")
RAW_BIDS_DIR = Path("ERN_Raw_Data_BIDS-Compatible")

TUCKER_RANK = (10, 10, 10, 40)

# Detect two change points and split the time axis into three ERN phases.
INTERVAL_NAMES = ("pre_ern", "ern", "post_ern")
N_CHANGE_POINTS = 2
MIN_INTERVAL_FRAMES = 10
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
    core, factors = tucker(X, rank=rank, init="svd")
    L = tl.tucker_to_tensor((core, factors))
    S = X - L
    return L, S, core, factors


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


def modularity_communities(A, seed=42):
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

    if "nx" not in globals() or nx is None:
        return fiedler_split(A), np.nan, "Fiedler fallback"

    G = nx.from_numpy_array(A)
    if G.number_of_edges() == 0:
        return np.zeros(n_nodes, dtype=int), np.nan, "empty graph"

    try:
        communities = nx.community.louvain_communities(G, weight="weight", seed=seed)
        method = "Louvain modularity"
    except Exception:
        communities = nx.community.greedy_modularity_communities(G, weight="weight")
        method = "Greedy modularity"

    communities = [set(community) for community in communities if len(community) > 0]
    labels = np.zeros(n_nodes, dtype=int)
    for community_idx, community in enumerate(communities):
        for node in community:
            labels[node] = community_idx

    try:
        score = nx.community.modularity(G, communities, weight="weight")
    except Exception:
        score = np.nan

    return labels, float(score), method


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

    raw_paths = sorted(Path(raw_bids_dir).glob("sub-*/eeg/*_task-ERN_eeg.set"))
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
                "source": "raw BIDS EEGLAB",
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
    boundary_frames = [
        int(frames[0])
        for frames in interval_list[1:]
    ]
    for frame in boundary_frames:
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
            y_label = "amplitude (uV)" if erp["channel_type"] == "eeg" else "baseline-corrected CSD"
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


def run_fcca_interval(X_clean, frames):
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
    consensus_labels, consensus_modularity, consensus_method = modularity_communities(consensus_matrix)
    mean_adjacency = graphs.mean(axis=0)

    return {
        "frames": np.asarray(frames),
        "graph_labels": graph_labels,
        "graph_modularity_scores": graph_modularity_scores,
        "consensus_matrix": consensus_matrix,
        "consensus_labels": consensus_labels,
        "consensus_modularity": np.asarray(consensus_modularity),
        "community_method": np.asarray(consensus_method or community_method or "unknown"),
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


def plot_circular_graph(ax, A, labels, title):
    A_thr, efficiency, threshold_label = dyconnmap_threshold(A)
    n_nodes = A_thr.shape[0]
    theta = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    xy = np.column_stack([np.cos(theta), np.sin(theta)])
    max_weight = A_thr.max() if A_thr.max() > 0 else 1.0

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if A_thr[i, j] > 0:
                ax.plot(
                    [xy[i, 0], xy[j, 0]],
                    [xy[i, 1], xy[j, 1]],
                    color="0.25",
                    alpha=0.15 + 0.65 * (A_thr[i, j] / max_weight),
                    linewidth=0.5 + 2.0 * (A_thr[i, j] / max_weight),
                )

    node_colors = np.where(labels == 0, "#1f77b4", "#d62728")
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


def save_fcca_results(results, change_points=None, change_point_method=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_dict = {}
    for interval_name, result in results.items():
        for key, value in result.items():
            save_dict[f"{interval_name}_{key}"] = value

    if change_points is not None:
        save_dict["change_points"] = np.asarray(change_points, dtype=int)
    if change_point_method is not None:
        save_dict["change_point_method"] = np.asarray(change_point_method)

    np.savez_compressed(OUTPUT_DIR / "fcca_results.npz", **save_dict)


def main():
    visualize_condition_timecourses()

    X = np.load(INPUT_FILE)

    L, S, core, factors = tucker_low_rank_decomposition(X, rank=TUCKER_RANK)
    X_clean = L

    print("Original:", X.shape)
    print("Low-rank:", L.shape)
    print("Sparse residual:", S.shape)
    print("dyconnmap visualization helpers:", "available" if HAS_DYCONNMAP else "not installed")
    print("change-point detector:", "ruptures available" if HAS_RUPTURES else "NumPy exact SSE fallback")

    n_subjects, n_nodes, _, n_times = X_clean.shape
    times_ms = frames_to_ms(n_times)
    intervals, change_points, change_point_method = make_change_point_intervals(X_clean)
    print(
        f"Detected change points with {change_point_method}: "
        f"frames={change_points} | ms={[round(float(times_ms[p]), 1) for p in change_points]}"
    )

    fcca_results = {}
    for interval_name, frames in intervals.items():
        result = run_fcca_interval(X_clean, frames)
        fcca_results[interval_name] = result
        n_communities = len(np.unique(result["consensus_labels"]))
        print(
            f"{interval_name}: frames {frames[0]}-{frames[-1]} | "
            f"ms={times_ms[frames[0]]:.1f}-{times_ms[frames[-1]]:.1f} | "
            f"graphs={n_subjects * len(frames)} | communities={n_communities} | "
            f"modularity={float(result['consensus_modularity']):.4f} | "
            f"labels={result['consensus_labels']}"
        )

    save_fcca_results(fcca_results, change_points, change_point_method)
    visualize_fcca_results(fcca_results)

    print(f"Saved FCCA arrays: {OUTPUT_DIR /  'fcca_results.npz'}")
    print(f"Saved figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
