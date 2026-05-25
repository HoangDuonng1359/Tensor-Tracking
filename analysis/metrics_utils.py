from typing import Tuple, List, Dict, Any, Union
import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity
import collections
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score

from fcca.fiedler import recursive_repartitioning
from fcca import fcca_on_lowrank_interval
from algorithms.common import convert_subject_tensor_to_stream

def extract_intervals(res: Any, zero_index: int) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    ern_start = zero_index
    ern_end = zero_index + 16
    for (start, end) in res.intervals:
        if zero_index - 5 <= start <= zero_index + 10:
            ern_start = start
            ern_end = end + 1
            break
    pre_interval = (0, ern_start - 1)
    ern_interval = (ern_start, ern_end - 1)
    post_interval = (ern_end, res.intervals[-1][1])
    return pre_interval, ern_interval, post_interval

def compute_network_metrics(W: np.ndarray, top_k_percent: float = 0.15, detailed: bool = False) -> Union[Tuple[float, float, float, float, int], Tuple[Tuple[float, float, float, float, int], Dict[str, Any]]]:
    n_nodes = W.shape[0]
    upper_tri_indices = np.triu_indices(n_nodes, 1)
    edge_weights = W[upper_tri_indices]
    k = max(1, int(top_k_percent * len(edge_weights)))
    threshold = np.sort(edge_weights)[-k]
    
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if W[i, j] >= threshold:
                G.add_edge(i, j, weight=W[i, j])
                
    try:
        communities = louvain_communities(G, weight='weight')
        mod = modularity(G, communities, weight='weight')
        n_clusters = len(communities)
        
        within_weight = 0
        between_weight = 0
        within_count = 0
        between_count = 0
        
        node_comm = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_comm[node] = i
                
        for u, v, data in G.edges(data=True):
            if node_comm.get(u) == node_comm.get(v):
                within_weight += data['weight']
                within_count += 1
            else:
                between_weight += data['weight']
                between_count += 1
                
        wb_ratio = within_weight / max(between_weight, 1e-6)
        
        cluster_sizes = [len(c) for c in communities]
        mean_within = within_weight / max(1, within_count)
        mean_between = between_weight / max(1, between_count)
        
    except Exception as e:
        mod = 0.0
        wb_ratio = 0.0
        n_clusters = 0
        cluster_sizes = []
        mean_within = 0.0
        mean_between = 0.0
        within_count = 0
        between_count = 0
        
    mean_weight = np.mean(edge_weights[edge_weights >= threshold]) if k > 0 else 0
    density = (2 * G.number_of_edges()) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
    
    res = (mod, wb_ratio, density, mean_weight, n_clusters)
    if detailed:
        return res, {
            "mean_within": mean_within,
            "mean_between": mean_between,
            "count_within": within_count,
            "count_between": between_count,
            "cluster_sizes": cluster_sizes
        }
    return res

def calculate_distance(W1: np.ndarray, W2: np.ndarray) -> float:
    return float(np.linalg.norm(W1 - W2, 'fro'))

def permutation_test_distance(interval_data_ern: np.ndarray, interval_data_crn: np.ndarray, observed_dist: float, threshold: float = 0.1, n_perms: int = 50) -> Tuple[float, float]:
    print(f"    Running Permutation Test ({n_perms} iterations)...")
    t_ern, n, _, s_ern = interval_data_ern.shape
    t_crn, _, _, s_crn = interval_data_crn.shape
    
    matrices_ern = []
    for t in range(t_ern):
        for s in range(s_ern):
            m = interval_data_ern[t, :, :, s].copy()
            m[m < threshold] = 0
            matrices_ern.append(m)
            
    matrices_crn = []
    for t in range(t_crn):
        for s in range(s_crn):
            m = interval_data_crn[t, :, :, s].copy()
            m[m < threshold] = 0
            matrices_crn.append(m)
            
    all_matrices = matrices_ern + matrices_crn
    n_ern = len(matrices_ern)
    
    all_communities = []
    for adj_matrix in all_matrices:
        all_communities.append(recursive_repartitioning(adj_matrix, list(range(n))))
        
    all_communities = np.array(all_communities, dtype=object)
    
    null_distances = []
    for p in range(n_perms):
        indices = np.random.permutation(len(all_communities))
        ern_idx = indices[:n_ern]
        crn_idx = indices[n_ern:]
        
        W_e = np.zeros((n, n), dtype=np.float64)
        for comms in all_communities[ern_idx]:
            for comm in comms:
                for i in comm:
                    for j in comm:
                        W_e[i, j] += 1.0
        W_e /= n_ern
        
        W_c = np.zeros((n, n), dtype=np.float64)
        for comms in all_communities[crn_idx]:
            for comm in comms:
                for i in comm:
                    for j in comm:
                        W_c[i, j] += 1.0
        W_c /= len(crn_idx)
        
        dist = calculate_distance(W_e, W_c)
        null_distances.append(dist)
        
    p_value = float(np.sum(np.array(null_distances) >= observed_dist) / n_perms)
    return p_value, float(np.mean(null_distances))

def compute_event_overlap(start_ms: float, end_ms: float, target_start: float = 0.0, target_end: float = 150.0) -> float:
    overlap_start = max(start_ms, target_start)
    overlap_end = min(end_ms, target_end)
    overlap = max(0, overlap_end - overlap_start)
    target_length = target_end - target_start
    return float(overlap / target_length)


def bootstrap_stability_evaluation(algorithm_runner_factory: Any, tensor: np.ndarray, base_communities: List[int], base_ern_interval: Tuple[int, int], n_boots: int = 100, algo_name: str = "") -> Tuple[Dict[int, int], float, float]:
    print(f"    Running Bootstrap Stability on Subjects ({n_boots} iterations) for {algo_name}...")
    n_times, n_nodes, _, n_subs = tensor.shape
    cp_counts = collections.defaultdict(int)
    ari_scores = []
    
    for i in tqdm(range(n_boots), desc=f"Bootstrapping {algo_name}"):
        boot_idx = np.random.choice(n_subs, size=n_subs, replace=True)
        boot_tensor = tensor[:, :, :, boot_idx]
        
        runner = algorithm_runner_factory()
        res = runner.run(convert_subject_tensor_to_stream(boot_tensor))
        
        for cp in res.change_points:
            cp_counts[cp] += 1
            
        boot_part, _ = fcca_on_lowrank_interval(res.lowrank_stream, base_ern_interval[0], base_ern_interval[1], threshold=0.1)
        ari = adjusted_rand_score(base_communities, boot_part)
        ari_scores.append(ari)
            
    return cp_counts, np.mean(ari_scores), np.std(ari_scores)
